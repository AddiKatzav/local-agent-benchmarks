"""Agent loop strategies: No-Guard, Python-Guard, and Critic."""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from burnt_toast.config import (
    BURNT_TOAST_REQUIRE_TOOL_EVIDENCE,
    CRITIC_MODEL,
    MAX_AGENT_ITERATIONS,
    PYTHON_GUARD_MAX_IDENTICAL_CALLS,
    ExperimentMode,
)
from burnt_toast.ollama_client import OllamaClient, OllamaResponse
from burnt_toast.prompts import build_initial_messages
from burnt_toast.secrets import JSON_RESPONSE_SCHEMA, RunSecret
from burnt_toast.tools import (
    FaultySearchTool,
    PythonGuard,
    parse_final_answer,
    parse_tool_call,
)

logger = logging.getLogger(__name__)

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
    tool_call_count: int = 0
    guard_triggered: bool = False
    responses: list[OllamaResponse] = field(default_factory=list)
    transcript: list[str] = field(default_factory=list)
    error_message: str = ""

    @property
    def effective_prompt_tokens(self) -> int:
        """Prompt tokens from the first turn (main context load)."""
        if not self.responses:
            return 0
        return self.responses[0].prompt_eval_count

    @property
    def mean_ttft(self) -> float:
        if not self.responses:
            return 0.0
        return sum(r.ttft_seconds for r in self.responses) / len(self.responses)

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


@dataclass
class LoopContext:
    """Shared state for the agent loop across strategies."""

    client: OllamaClient
    model: str
    mode: ExperimentMode
    num_ctx: int
    haystack_context: str
    messages: list[dict[str, str]]
    json_schema: str = JSON_RESPONSE_SCHEMA
    search_tool: FaultySearchTool = field(default_factory=FaultySearchTool)
    tool_call_count: int = 0
    tool_evidence_found: bool = False
    transcript: list[str] = field(default_factory=list)
    responses: list[OllamaResponse] = field(default_factory=list)

    def chat(self) -> OllamaResponse:
        response = self.client.chat(self.model, self.messages, num_ctx=self.num_ctx)
        self.responses.append(response)
        return response

    def record_tool_call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.tool_call_count += 1
        result = self.search_tool.invoke(arguments)
        if result.get("results"):
            self.tool_evidence_found = True
        return result

    def nudge_need_tool(self) -> str:
        return (
            "You MUST call exactly: TOOL: search_context {\"query\": \"secret agent code\"}. "
            "The document is external and not visible to you directly."
        )

    def nudge_need_answer(self) -> str:
        if self.mode == "burnt-toast":
            return (
                "No retrieved evidence is available. Do not guess. Retry exactly: "
                "TOOL: search_context {\"query\": \"secret agent code\"}"
            )
        return f"Read the context document and respond with JSON: {self.json_schema}"


def _accept_final_answer(ctx: LoopContext, answer: dict[str, Any] | None) -> bool:
    if answer is None:
        return False
    if (
        ctx.mode == "burnt-toast"
        and BURNT_TOAST_REQUIRE_TOOL_EVIDENCE
        and not ctx.tool_evidence_found
    ):
        return False
    return True


def _handle_tool_path(
    ctx: LoopContext,
    content: str,
    tool_call,
    *,
    on_before_tool: Callable[[Any], bool] | None = None,
) -> str | None:
    """
    Process a tool call. Returns 'guard_stop' if the guard halted the loop,
    or None to continue normally.
    """
    if on_before_tool is not None and not on_before_tool(tool_call):
        return "guard_stop"

    tool_result = ctx.record_tool_call(tool_call.arguments)
    tool_msg = (
        f"TOOL RESULT ({ctx.search_tool.name}): "
        f"{tool_result['message']} — results: {tool_result['results']}"
    )
    ctx.transcript.append(tool_msg)
    ctx.messages.append({"role": "assistant", "content": content})
    ctx.messages.append({"role": "user", "content": tool_msg})
    return None


def run_agent_loop(
    ctx: LoopContext,
    strategy_name: str,
    *,
    on_before_tool: Callable[[Any], bool] | None = None,
    after_tool: Callable[[LoopContext], bool] | None = None,
    on_guard_stop: Callable[[LoopContext, str], AgentRunResult] | None = None,
) -> AgentRunResult:
    """Shared multi-turn loop used by all strategies."""
    result = AgentRunResult(final_output="", iterations=0)
    content = ""

    for iteration in range(1, MAX_AGENT_ITERATIONS + 1):
        logger.info("[%s] iteration %d", strategy_name, iteration)
        response = ctx.chat()
        content = response.content.strip()
        ctx.transcript.append(f"ASSISTANT: {content}")
        result.iterations = iteration

        answer = parse_final_answer(content)
        if _accept_final_answer(ctx, answer):
            result.final_output = content
            result.tool_call_count = ctx.tool_call_count
            result.responses = ctx.responses
            result.transcript = ctx.transcript
            return result

        if answer is not None and ctx.mode == "burnt-toast" and ctx.tool_call_count == 0:
            nudge = ctx.nudge_need_tool()
            ctx.transcript.append(f"SYSTEM: {nudge}")
            ctx.messages.append({"role": "assistant", "content": content})
            ctx.messages.append({"role": "user", "content": nudge})
            continue

        tool_call = parse_tool_call(content)
        if tool_call and tool_call.name == ctx.search_tool.name:
            if ctx.mode == "needle":
                nudge = (
                    "No tools are available in this task. Read the context document "
                    f"directly and respond with JSON: {ctx.json_schema}"
                )
                ctx.transcript.append(f"SYSTEM: {nudge}")
                ctx.messages.append({"role": "assistant", "content": content})
                ctx.messages.append({"role": "user", "content": nudge})
                continue

            guard_outcome = _handle_tool_path(ctx, content, tool_call, on_before_tool=on_before_tool)
            if guard_outcome == "guard_stop" and on_guard_stop is not None:
                return on_guard_stop(ctx, content)

            if after_tool is not None and after_tool(ctx):
                if on_guard_stop is not None:
                    return on_guard_stop(ctx, content)
            continue

        if ctx.mode == "needle":
            nudge = ctx.nudge_need_answer()
        else:
            nudge = ctx.nudge_need_answer()
        ctx.transcript.append(f"SYSTEM: {nudge}")
        ctx.messages.append({"role": "assistant", "content": content})
        ctx.messages.append({"role": "user", "content": nudge})

    result.final_output = content
    result.tool_call_count = ctx.tool_call_count
    result.responses = ctx.responses
    result.transcript = ctx.transcript
    result.error_message = f"Max iterations ({MAX_AGENT_ITERATIONS}) reached"
    return result


class BaseStrategy(ABC):
    """Abstract agent loop strategy."""

    name: str

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        mode: ExperimentMode = "needle",
        num_ctx: int = 4096,
        run_secret: RunSecret | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.mode = mode
        self.num_ctx = num_ctx
        self.run_secret = run_secret

    def _make_context(self, haystack_context: str) -> LoopContext:
        return LoopContext(
            client=self.client,
            model=self.model,
            mode=self.mode,
            num_ctx=self.num_ctx,
            haystack_context=haystack_context,
            json_schema=self.run_secret.json_schema if self.run_secret else JSON_RESPONSE_SCHEMA,
            messages=build_initial_messages(self.mode, haystack_context),
        )

    @abstractmethod
    def run(self, haystack_context: str) -> AgentRunResult:
        ...


class NoGuardStrategy(BaseStrategy):
    """Pure agent loop with no loop protection."""

    name = "No-Guard"

    def run(self, haystack_context: str) -> AgentRunResult:
        ctx = self._make_context(haystack_context)
        if self.mode == "needle":
            return run_agent_loop(ctx, self.name)
        return run_agent_loop(ctx, self.name)


class PythonGuardStrategy(BaseStrategy):
    """Deterministic hash-based loop guard."""

    name = "Python-Guard"

    def run(self, haystack_context: str) -> AgentRunResult:
        guard = PythonGuard(max_identical=PYTHON_GUARD_MAX_IDENTICAL_CALLS)
        ctx = self._make_context(haystack_context)

        def on_before_tool(tool_call) -> bool:
            return guard.record_and_check(tool_call)

        def on_guard_stop(loop_ctx: LoopContext, content: str) -> AgentRunResult:
            guard_msg = guard.guard_message
            loop_ctx.transcript.append(f"SYSTEM: {guard_msg}")
            loop_ctx.messages.append({"role": "assistant", "content": content})
            loop_ctx.messages.append({"role": "user", "content": guard_msg})

            final_resp = loop_ctx.chat()
            final_output = final_resp.content.strip()
            loop_ctx.transcript.append(f"ASSISTANT (post-guard): {final_output}")
            return AgentRunResult(
                final_output=final_output,
                iterations=len(loop_ctx.responses),
                tool_call_count=loop_ctx.tool_call_count + 1,
                guard_triggered=True,
                responses=loop_ctx.responses,
                transcript=loop_ctx.transcript,
            )

        result = run_agent_loop(
            ctx,
            self.name,
            on_before_tool=on_before_tool,
            on_guard_stop=on_guard_stop,
        )
        result.guard_triggered = guard.guard_triggered
        return result


class CriticStrategy(BaseStrategy):
    """Secondary LLM critic for loop detection."""

    name = "Critic"

    def __init__(
        self,
        client: OllamaClient,
        model: str,
        *,
        mode: ExperimentMode = "needle",
        num_ctx: int = 4096,
        critic_model: str = CRITIC_MODEL,
        run_secret: RunSecret | None = None,
    ) -> None:
        super().__init__(client, model, mode=mode, num_ctx=num_ctx, run_secret=run_secret)
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
            response = self.client.chat(self.critic_model, messages, stream=False, num_ctx=4096)
            text = response.content.strip()
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return bool(data.get("is_looping", False))
        except Exception as exc:
            logger.warning("Critic loop detection failed: %s", exc)

        return False

    def run(self, haystack_context: str) -> AgentRunResult:
        ctx = self._make_context(haystack_context)

        def after_tool(loop_ctx: LoopContext) -> bool:
            return self._detect_loop(loop_ctx.transcript)

        def on_guard_stop(loop_ctx: LoopContext, content: str) -> AgentRunResult:
            schema = loop_ctx.json_schema
            if self.mode == "burnt-toast":
                loop_msg = (
                    "CRITIC ALERT: Repetitive loop detected. Stop calling tools and "
                    f"provide your best-effort JSON answer using schema: {schema}"
                )
            else:
                loop_msg = (
                    "CRITIC ALERT: Repetitive loop detected. Stop calling tools and "
                    "re-read the context document directly for the secret phrase. "
                    f"Respond with JSON: {schema}"
                )
            logger.warning("Critic detected loop")
            loop_ctx.transcript.append(f"SYSTEM: {loop_msg}")
            loop_ctx.messages.append({"role": "user", "content": loop_msg})

            final_resp = loop_ctx.chat()
            final_output = final_resp.content.strip()
            loop_ctx.transcript.append(f"ASSISTANT (post-critic): {final_output}")
            return AgentRunResult(
                final_output=final_output,
                iterations=len(loop_ctx.responses),
                tool_call_count=loop_ctx.tool_call_count,
                guard_triggered=True,
                responses=loop_ctx.responses,
                transcript=loop_ctx.transcript,
            )

        return run_agent_loop(ctx, self.name, after_tool=after_tool, on_guard_stop=on_guard_stop)


def get_strategy(
    name: str,
    client: OllamaClient,
    model: str,
    *,
    mode: ExperimentMode = "needle",
    num_ctx: int = 4096,
    critic_model: str | None = None,
    run_secret: RunSecret | None = None,
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
        return cls(
            client,
            model,
            mode=mode,
            num_ctx=num_ctx,
            critic_model=critic_model or CRITIC_MODEL,
            run_secret=run_secret,
        )
    return cls(client, model, mode=mode, num_ctx=num_ctx, run_secret=run_secret)
