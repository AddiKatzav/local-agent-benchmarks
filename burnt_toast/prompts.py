"""Build prompts and estimate needle visibility for experiment modes."""

from __future__ import annotations

from burnt_toast.config import ExperimentMode
from burnt_toast.secrets import JSON_RESPONSE_SCHEMA


def estimate_visible_chars(effective_prompt_tokens: int, chars_per_token: float, overhead_tokens: int = 120) -> int:
    """Estimate how many haystack characters fit in the first inference call."""
    content_tokens = max(0, effective_prompt_tokens - overhead_tokens)
    return max(0, int(content_tokens * chars_per_token))


def needle_in_visible_window(
    haystack: str,
    *,
    secret_phrase: str,
    needle_position: str,
    actual_context_tokens: int,
    effective_prompt_tokens: int,
    chars_per_token: float = 4.0,
    overhead_tokens: int = 120,
    truncation_threshold: float = 0.95,
) -> bool:
    """
    Estimate whether the needle is inside the portion of the haystack the model saw.

    When truncation occurs we assume the tail of the user message is dropped first,
    which matches the observed 8K/end failures with 8K/middle successes.
    """
    if actual_context_tokens <= 0 or effective_prompt_tokens <= 0:
        return False

    if effective_prompt_tokens >= actual_context_tokens * truncation_threshold:
        return True

    visible_chars = estimate_visible_chars(effective_prompt_tokens, chars_per_token, overhead_tokens)
    needle_index = haystack.find(secret_phrase)
    if needle_index < 0:
        return False

    # Tail-truncation: only the first visible_chars of the haystack are seen.
    return needle_index < visible_chars


def is_context_truncated(
    actual_context_tokens: int,
    effective_prompt_tokens: int,
    *,
    overhead_tokens: int = 120,
    threshold: float = 0.95,
) -> bool:
    """Return True when fewer tokens were evaluated than we attempted to send."""
    if actual_context_tokens <= 0:
        return False
    evaluated_content = max(0, effective_prompt_tokens - overhead_tokens)
    return evaluated_content < actual_context_tokens * threshold


def build_needle_messages(haystack_context: str) -> list[dict[str, str]]:
    """Needle arm: full context in prompt, no tool required."""
    system = f"""\
You are a research assistant. A secret code is hidden inside the context document below.
Read the document carefully and report the code in strict JSON format.

Respond with ONLY this JSON when you have found the code (no other text):
  {JSON_RESPONSE_SCHEMA}

The secret code is a 4-digit integer embedded somewhere in the document.
Do not guess — locate the exact phrase that reveals the code.
"""
    user = (
        f"CONTEXT DOCUMENT ({len(haystack_context)} chars):\n\n"
        f"{haystack_context}\n\n"
        f"Find the secret code and output {JSON_RESPONSE_SCHEMA}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_burnt_toast_messages(haystack_context: str) -> list[dict[str, str]]:
    """Burnt-toast arm: context is external; tool use is mandatory."""
    system = f"""\
You are an autonomous research agent. A large document is stored in an external index.
You CANNOT see the document directly — you must retrieve passages with the tool below.

You have access to one tool:
  - search_context(query: string) — searches the external document index.

To call the tool, respond EXACTLY in this format:
  TOOL: search_context {{"query": "your search terms"}}

Rules:
1. Your first response MUST be exactly:
   TOOL: search_context {{"query": "secret agent code"}}
2. If the tool returns no matches, retry the exact same tool call. Do not change the query.
3. Do NOT answer from memory or guesswork.
4. Only after a tool result contains the secret phrase may you respond with ONLY:
   {JSON_RESPONSE_SCHEMA}

The secret code is a 4-digit integer hidden somewhere in the indexed document.
"""
    user = (
        f"An external document is indexed ({len(haystack_context)} chars). "
        "You do not have direct access to its contents.\n\n"
        "Search for the secret phrase containing the code. Start with the exact tool call "
        'TOOL: search_context {"query": "secret agent code"}.'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_initial_messages(mode: ExperimentMode, haystack_context: str) -> list[dict[str, str]]:
    if mode == "burnt-toast":
        return build_burnt_toast_messages(haystack_context)
    return build_needle_messages(haystack_context)
