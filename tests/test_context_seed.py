"""Unit tests for burnt_toast.context.haystack_seed (no Ollama required)."""

from __future__ import annotations

import unittest

from burnt_toast.context import haystack_seed

BASE_KWARGS = dict(
    model="qwen2.5:1.5b",
    target_tokens=1000,
    needle_position="middle",
    strategy="No-Guard",
    experiment_mode="needle",
)


class TestHaystackSeed(unittest.TestCase):
    def test_deterministic_across_repeated_calls(self) -> None:
        first = haystack_seed(**BASE_KWARGS)
        second = haystack_seed(**BASE_KWARGS)
        third = haystack_seed(**BASE_KWARGS)
        self.assertEqual(first, second)
        self.assertEqual(second, third)

    def test_returns_int_in_32bit_range(self) -> None:
        seed = haystack_seed(**BASE_KWARGS)
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_varies_with_model(self) -> None:
        a = haystack_seed(**BASE_KWARGS)
        b = haystack_seed(**{**BASE_KWARGS, "model": "llama3.1:8b"})
        self.assertNotEqual(a, b)

    def test_varies_with_target_tokens(self) -> None:
        a = haystack_seed(**BASE_KWARGS)
        b = haystack_seed(**{**BASE_KWARGS, "target_tokens": 4000})
        self.assertNotEqual(a, b)

    def test_varies_with_needle_position(self) -> None:
        a = haystack_seed(**BASE_KWARGS)
        b = haystack_seed(**{**BASE_KWARGS, "needle_position": "end"})
        self.assertNotEqual(a, b)

    def test_varies_with_strategy(self) -> None:
        a = haystack_seed(**BASE_KWARGS)
        b = haystack_seed(**{**BASE_KWARGS, "strategy": "Python-Guard"})
        c = haystack_seed(**{**BASE_KWARGS, "strategy": "Critic"})
        self.assertNotEqual(a, b)
        self.assertNotEqual(a, c)
        self.assertNotEqual(b, c)

    def test_varies_with_experiment_mode(self) -> None:
        a = haystack_seed(**BASE_KWARGS)
        b = haystack_seed(**{**BASE_KWARGS, "experiment_mode": "burnt-toast"})
        self.assertNotEqual(a, b)

    def test_no_cross_field_collision_via_delimiter(self) -> None:
        # Without a delimiter, adjacent fields ("ab", "c") and ("a", "bc")
        # could collide if naively concatenated. Confirm the delimiter
        # prevents this class of bug for the two trailing (adjacent) fields.
        a = haystack_seed("m", 1, "n", "ab", "c")
        b = haystack_seed("m", 1, "n", "a", "bc")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
