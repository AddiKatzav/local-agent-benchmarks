"""Unit tests for burnt_toast.runner.measure_prompt_overhead_tokens.

Uses a fake OllamaClient-shaped stub (just a count_tokens method) -- no real
network calls -- to verify the wrapper-text measurement and per-(model,mode)
caching behavior in isolation.
"""

from __future__ import annotations

import unittest

import burnt_toast.runner as runner
from burnt_toast.prompts import build_initial_messages
from burnt_toast.runner import measure_prompt_overhead_tokens


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def count_tokens(self, model: str, text: str) -> int:
        self.calls.append((model, text))
        return len(text)


class TestMeasurePromptOverheadTokens(unittest.TestCase):
    def setUp(self) -> None:
        # Each test gets a clean cache -- the cache is module-level and
        # would otherwise leak state between tests / across the whole suite.
        runner._overhead_cache.clear()

    def test_counts_wrapper_text_built_from_empty_haystack(self) -> None:
        client = FakeClient()
        expected_text = "\n".join(m["content"] for m in build_initial_messages("needle", ""))

        overhead = measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "needle")

        self.assertEqual(overhead, len(expected_text))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0], ("qwen2.5:1.5b", expected_text))

    def test_second_call_same_model_mode_is_cached(self) -> None:
        client = FakeClient()

        first = measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "needle")
        second = measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "needle")

        self.assertEqual(first, second)
        self.assertEqual(len(client.calls), 1)  # only the first call hit count_tokens

    def test_different_mode_gets_independent_cache_entry(self) -> None:
        client = FakeClient()

        needle_overhead = measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "needle")
        burnt_toast_overhead = measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "burnt-toast")

        self.assertEqual(len(client.calls), 2)
        self.assertNotEqual(needle_overhead, burnt_toast_overhead)

    def test_different_model_gets_independent_cache_entry(self) -> None:
        client = FakeClient()

        measure_prompt_overhead_tokens(client, "qwen2.5:1.5b", "needle")
        measure_prompt_overhead_tokens(client, "llama3.1:8b", "needle")

        self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
