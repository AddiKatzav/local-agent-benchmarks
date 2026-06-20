"""Simulated agent tools for the burnt-toast experiment."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCall:
    """Parsed tool invocation from model output."""

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw: str = ""

    def fingerprint(self) -> str:
        """Deterministic hash for loop detection."""
        payload = json.dumps(
            {"name": self.name, "arguments": self.arguments},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class FaultySearchTool:
    """
    A deliberately broken search tool that simulates the 'burnt toast' failure mode.

    Every invocation returns an empty result set, encouraging the agent to retry.
    """

    name = "search_context"

    def invoke(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query", "")
        logger.debug("search_context called with query=%r -> empty result", query)
        return {
            "status": "ok",
            "results": [],
            "message": (
                "No matches found in the indexed context window. "
                "Retry the exact same search_context query."
            ),
        }


class PythonGuard:
    """
    Deterministic wrapper that tracks tool-call fingerprints and aborts after
    PYTHON_GUARD_MAX_IDENTICAL_CALLS identical consecutive invocations.
    """

    def __init__(self, max_identical: int = 3) -> None:
        self.max_identical = max_identical
        self._call_history: list[str] = []
        self.guard_triggered = False
        self.guard_message: str = ""

    def record_and_check(self, call: ToolCall) -> bool:
        """
        Record a tool call fingerprint.

        Returns True if execution should continue, False if guard tripped.
        """
        fp = call.fingerprint()
        self._call_history.append(fp)

        # Count trailing identical fingerprints
        trailing = 0
        for past in reversed(self._call_history):
            if past == fp:
                trailing += 1
            else:
                break

        if trailing >= self.max_identical:
            self.guard_triggered = True
            self.guard_message = (
                f"SYSTEM ERROR: Execution halted — identical tool call "
                f"'{call.name}' repeated {trailing} times (hash={fp[:12]}). "
                f"Loop guard activated. Respond with your best final answer as JSON."
            )
            logger.warning("Python-Guard triggered after %d identical calls", trailing)
            return False
        return True

    def reset(self) -> None:
        self._call_history.clear()
        self.guard_triggered = False
        self.guard_message = ""


def _find_balanced_json_object(text: str, key_hint: str) -> str | None:
    """
    Find the smallest (innermost) balanced {...} substring in text whose
    content contains key_hint (e.g. '"tool"'), scanning brace depth so
    nested objects don't break matching. Prefers the innermost match so a
    hint nested inside a wrapper object (e.g. {"result": {"secret_code": 1}})
    still resolves to the specific object that actually contains it, while a
    sibling nested object (e.g. a tool call's "arguments" sub-object) that
    does NOT contain the hint is correctly excluded in favor of the outer one.
    """
    candidates: list[str] = []
    for start, ch in enumerate(text):
        if ch != "{":
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if key_hint in candidate:
                        candidates.append(candidate)
                    break
    if not candidates:
        return None
    return min(candidates, key=len)


def parse_tool_call(text: str) -> ToolCall | None:
    """
    Extract a tool call from model output.

    Accepted formats:
      TOOL: search_context {"query": "secret code"}
      ```json\n{"tool": "search_context", "arguments": {"query": "..."}}\n```
      {"tool": "search_context", "arguments": {"query": "..."}}
    """
    import re

    text = text.strip()

    # Explicit TOOL: prefix
    tool_prefix = re.search(
        r"TOOL:\s*(\w+)\s*(\{.*\})",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if tool_prefix:
        name = tool_prefix.group(1)
        try:
            args = json.loads(tool_prefix.group(2))
        except json.JSONDecodeError:
            args = {"raw": tool_prefix.group(2)}
        return ToolCall(name=name, arguments=args if isinstance(args, dict) else {}, raw=text)

    # Fenced JSON block
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []

    # Bare JSON object with "tool" key (handles a nested "arguments" object)
    bare = _find_balanced_json_object(text, '"tool"')
    if bare:
        candidates.append(bare)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if "tool" in data:
                return ToolCall(
                    name=str(data["tool"]),
                    arguments=data.get("arguments", {}),
                    raw=text,
                )
        except json.JSONDecodeError:
            continue

    return None


def parse_final_answer(text: str) -> dict[str, Any] | None:
    """Extract a JSON object containing secret_code from model output."""
    import re

    # Direct JSON object (handles a secret_code nested one level inside a
    # wrapper object, e.g. a weaker model emitting {"result": {...}})
    candidate = _find_balanced_json_object(text, "secret_code")
    if candidate:
        try:
            return json.loads(candidate.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    # Fenced JSON
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if "secret_code" in data:
                return data
        except json.JSONDecodeError:
            pass

    # Whole-text JSON
    try:
        data = json.loads(text.strip())
        if isinstance(data, dict) and "secret_code" in data:
            return data
    except json.JSONDecodeError:
        pass

    return None
