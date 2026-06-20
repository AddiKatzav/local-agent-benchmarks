"""Unit tests for burnt_toast._hashing.stable_seed (no Ollama required)."""

from __future__ import annotations

import unittest

from burnt_toast._hashing import stable_seed


class TestStableSeed(unittest.TestCase):
    def test_deterministic_across_repeated_calls(self) -> None:
        a = stable_seed("model", 1000, "middle", "No-Guard", "needle")
        b = stable_seed("model", 1000, "middle", "No-Guard", "needle")
        self.assertEqual(a, b)

    def test_returns_int_in_default_mod_range(self) -> None:
        seed = stable_seed("model", 1000, "middle")
        self.assertIsInstance(seed, int)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 2**32)

    def test_respects_custom_mod(self) -> None:
        seed = stable_seed("model", 1000, mod=97)
        self.assertGreaterEqual(seed, 0)
        self.assertLess(seed, 97)

    def test_varies_per_field(self) -> None:
        base = stable_seed("a", 1, "b", "c", "d")
        variants = [
            stable_seed("z", 1, "b", "c", "d"),
            stable_seed("a", 2, "b", "c", "d"),
            stable_seed("a", 1, "z", "c", "d"),
            stable_seed("a", 1, "b", "z", "d"),
            stable_seed("a", 1, "b", "c", "z"),
        ]
        for variant in variants:
            self.assertNotEqual(base, variant)
        # All variants should also be pairwise distinct from each other.
        self.assertEqual(len(set(variants)), len(variants))

    def test_varies_with_number_of_parts(self) -> None:
        a = stable_seed("x", "y")
        b = stable_seed("x", "y", "z")
        self.assertNotEqual(a, b)

    def test_no_cross_field_collision_via_delimiter(self) -> None:
        # Without a delimiter, adjacent fields ("ab", "c") and ("a", "bc")
        # could collide if naively concatenated.
        a = stable_seed("m", 1, "ab", "c")
        b = stable_seed("m", 1, "a", "bc")
        self.assertNotEqual(a, b)


if __name__ == "__main__":
    unittest.main()
