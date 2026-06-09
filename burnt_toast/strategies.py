"""Agent loop strategies: No-Guard, Python-Guard, and Critic."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from burnt_toast.config import (
    CRITIC_MODEL,
    EXPECTED_JSON,
    MAX_AGENT_ITERATIONS,
    PYTHON_GUARD_MAX_IDENTICAL_CALLS,
)
from burnt_toast.ollama_client import OllamaClient, OllamaResponse
from burnt_toast.tools import (
    FaultySearchTool,
    PythonGuard,
    ToolCall,
    parse_final_answer,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an autonomous research agent. Your task is to locate a hidden secret code \
within the provided context document and report it in strict JSON format.

You have access to one tool:
  - search_context(query: string) — searches the context for matching text.

To call a tool, respond EXACTLY in this format:
  TOOL: search_context {"query": "your search terms"}

When you have found the secret code, respond with ONLY this JSON (no other text):
  {"secret_code": <integer>}

The secret code is a 4-digit integer hidden somewhere in the context.
Do not guess — search the context thoroughly before answering.
"""

CRITIC_SYSTEM_PROMPT = """\
You are a loop-detection critic. Given a transcript of an agent's recent actions,
determine if the agent is stuck in a repetitive loop (calling the same tool with
similar queries, or failing to make progress).

Respond with ONLY valid JSON:
  {"is_looping": true}  or  {"is_looping": false}

A loop is detected when:
- The same tool is called 3+ times with identical or near-identical arguments
- The agent receives empty results repeatedly but keeps searching identically
- No new information is introduced across turns
"""


@dataclass
class AgentRunResult:
    """Outcome of one agent strategy execution."""

    final_output: str
    iterations: int
    guard_triggered: bool = False
    responses: list[OllamaResponse] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    error_message: str = ""

    @property
    def aggregated_ttft(self) -> float:
        if not self.responses:
            return 0.0
        return sum(r.ttft_seconds for r in self.responses)

    @property
    def mean_ttft(self) -> float:
        if not self.responses:
            return 0.0
        return self.aggregated_ttft / len(self.responses)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_eval_count for r in self.responses)

    @property
    def total_completion_tokens(self) -> int:
        return sum(r.eval_count for r in self.responses)

    @property
    def mean_tokens_per_second(self) -> float | None:
        tps_values = [r.tokens_per_second for r in self.responses if r.tokens_per_second]
        if not tps_values:
            return None
        return sum(tps_values) / len(tps_values)


class BaseStrategy(ABC):
    """Abstract agent loop strategy."""

    name: str

    def __init__(self, client: OllamaClient, model: str) -> None:
        self.client = client
        self.model = model
        self.search_tool = FaultySearchTool()

    @abstractmethod
    def run(self, haystack_context: str) -> AgentRunResult:
        ...


class NoGuardStrategy(BaseStrategy):
    """
    Pure agent loop with no loop protection.

    The faulty search tool keeps returning empty results, encouraging repetition
    (the 'burnt toast' effect).
    """

    name = "No-Guard"

    def run(self, haystack_context: str) -> AgentRunResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT DOCUMENT ({len(haystack_context)} chars):\n\n"
                    f"{haystack_context}\n\n"
                    f"Find the secret code and output {EXPECTED_JSON}"
                ),
            },
        ]

        result = AgentRunResult(final_output="", iterations=0)
        transcript: list[str] = []

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            logger.info("[%s] iteration %d", self.name, iteration)
            response = self.client.chat(self.model, messages)
            result.responses.append(response)
            content = response.content.strip()
            transcript.append(f"ASSISTANT: {content}")
            result.iterations = iteration

            # Check for final JSON answer
            answer = parse_final_answer(content)
            if answer is not None:
                result.final_output = content
                result.transcript = transcript
                return result

            # Check for tool call
            tool_call = parse_tool_call(content)
            if tool_call and tool_call.name == self.search_tool.name:
                tool_result = self.search_tool.invoke(tool_call.arguments)
                tool_msg = (
                    f"TOOL RESULT ({self.search_tool.name}): "
                    f"{tool_result['message']} — results: {tool_result['results']}"
                )
                transcript.append(tool_msg)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": tool_msg})
                continue

            # Unparseable response — nudge the agent
            nudge = (
                "Please either call TOOL: search_context {\"query\": \"...\"} "
                f"or respond with JSON: {EXPECTED_JSON}"
            )
            transcript.append(f"SYSTEM: {nudge}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge})

        result.final_output = content if result.responses else ""
        result.transcript = transcript
        result.error_message = f"Max iterations ({MAX_AGENT_ITERATIONS}) reached"
        return result


class PythonGuardStrategy(BaseStrategy):
    """
    Agent loop with deterministic hash-based loop guard.

    After 3 identical tool calls, inject a system error and force final answer.
    """

    name = "Python-Guard"

    def run(self, haystack_context: str) -> AgentRunResult:
        guard = PythonGuard(max_identical=PYTHON_GUARD_MAX_IDENTICAL_CALLS)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT DOCUMENT ({len(haystack_context)} chars):\n\n"
                    f"{haystack_context}\n\n"
                    f"Find the secret code and output {EXPECTED_JSON}"
                ),
            },
        ]

        result = AgentRunResult(final_output="", iterations=0)
        transcript: list[str] = []

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            logger.info("[%s] iteration %d", self.name, iteration)
            response = self.client.chat(self.model, messages)
            result.responses.append(response)
            content = response.content.strip()
            transcript.append(f"ASSISTANT: {content}")
            result.iterations = iteration

            answer = parse_final_answer(content)
            if answer is not None:
                result.final_output = content
                result.transcript = transcript
                result.guard_triggered = guard.guard_triggered
                return result

            tool_call = parse_tool_call(content)
            if tool_call and tool_call.name == self.search_tool.name:
                if not guard.record_and_check(tool_call):
                    result.guard_triggered = True
                    guard_msg = guard.guard_message
                    transcript.append(f"SYSTEM: {guard_msg}")
                    messages.append({"role": "assistant", "content": content})
                    messages.append({"role": "user", "content": guard_msg})

                    # One final generation after guard injection
                    final_resp = self.client.chat(self.model, messages)
                    result.responses.append(final_resp)
                    result.final_output = final_resp.content.strip()
                    transcript.append(f"ASSISTANT (post-guard): {result.final_output}")
                    result.transcript = transcript
                    return result

                tool_result = self.search_tool.invoke(tool_call.arguments)
                tool_msg = (
                    f"TOOL RESULT ({self.search_tool.name}): "
                    f"{tool_result['message']} — results: {tool_result['results']}"
                )
                transcript.append(tool_msg)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": tool_msg})
                continue

            nudge = (
                "Please either call TOOL: search_context {\"query\": \"...\"} "
                f"or respond with JSON: {EXPECTED_JSON}"
            )
            transcript.append(f"SYSTEM: {nudge}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge})

        result.final_output = content if result.responses else ""
        result.transcript = transcript
        result.guard_triggered = guard.guard_triggered
        result.error_message = f"Max iterations ({MAX_AGENT_ITERATIONS}) reached"
        return result


class CriticStrategy(BaseStrategy):
    """
    Agent loop with a secondary critic model that detects logical looping.

    Uses a fast/small model to evaluate recent transcript for repetitive patterns.
    """

    name = "Critic"

    def __init__(self, client: OllamaClient, model: str, critic_model: str = CRITIC_MODEL) -> None:
        super().__init__(client, model)
        self.critic_model = critic_model

    def _detect_loop(self, transcript: list[str]) -> bool:
        if len(transcript) < 4:
            return False

        recent = "\n".join(transcript[-8:])
        messages = [
            {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
            {"role": "user", "content": f"TRANSCRIPT:\n{recent}"},
        ]

        try:
            response = self.client.chat(self.critic_model, messages, stream=False)
            import json
            import re

            text = response.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return bool(data.get("is_looping", False))
        except Exception as exc:
            logger.warning("Critic loop detection failed: %s", exc)

        return False

    def run(self, haystack_context: str) -> AgentRunResult:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"CONTEXT DOCUMENT ({len(haystack_context)} chars):\n\n"
                    f"{haystack_context}\n\n"
                    f"Find the secret code and output {EXPECTED_JSON}"
                ),
            },
        ]

        result = AgentRunResult(final_output="", iterations=0)
        transcript: list[str] = []

        for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
            logger.info("[%s] iteration %d", self.name, iteration)
            response = self.client.chat(self.model, messages)
            result.responses.append(response)
            content = response.content.strip()
            transcript.append(f"ASSISTANT: {content}")
            result.iterations = iteration

            answer = parse_final_answer(content)
            if answer is not None:
                result.final_output = content
                result.transcript = transcript
                return result

            tool_call = parse_tool_call(content)
            if tool_call and tool_call.name == self.search_tool.name:
                tool_result = self.search_tool.invoke(tool_call.arguments)
                tool_msg = (
                    f"TOOL RESULT ({self.search_tool.name}): "
                    f"{tool_result['message']} — results: {tool_result['results']}"
                )
                transcript.append(tool_msg)
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": tool_msg})

                if self._detect_loop(transcript):
                    result.guard_triggered = True
                    loop_msg = (
                        "CRITIC ALERT: Repetitive loop detected. Stop calling tools and "
                        "re-read the context document directly for the secret phrase. "
                        f"Respond with JSON: {EXPECTED_JSON}"
                    )
                    logger.warning("Critic detected loop at iteration %d", iteration)
                    transcript.append(f"SYSTEM: {loop_msg}")
                    messages.append({"role": "user", "content": loop_msg})

                    final_resp = self.client.chat(self.model, messages)
                    result.responses.append(final_resp)
                    result.final_output = final_resp.content.strip()
                    transcript.append(f"ASSISTANT (post-critic): {result.final_output}")
                    result.transcript = transcript
                    return result
                continue

            nudge = (
                "Please either call TOOL: search_context {\"query\": \"...\"} "
                f"or respond with JSON: {EXPECTED_JSON}"
            )
            transcript.append(f"SYSTEM: {nudge}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": nudge})

        result.final_output = content if result.responses else ""
        result.transcript = transcript
        result.error_message = f"Max iterations ({MAX_AGENT_ITERATIONS}) reached"
        return result


def get_strategy(
    name: str,
    client: OllamaClient,
    model: str,
    *,
    critic_model: str | None = None,
) -> BaseStrategy:
    """Factory for strategy instances."""
    strategies: dict[str, type[BaseStrategy]] = {
        "No-Guard": NoGuardStrategy,
        "Python-Guard": PythonGuardStrategy,
        "Critic": CriticStrategy,
    }
    cls = strategies.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    if cls is CriticStrategy:
        return cls(client, model, critic_model=critic_model or CRITIC_MODEL)
    return cls(client, model)
