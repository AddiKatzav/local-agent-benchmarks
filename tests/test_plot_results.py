"""Unit tests for plot_results.split_valid_failed (no Ollama/matplotlib rendering needed)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_results import load_results, split_valid_failed


def _make_df(error_messages: list) -> pd.DataFrame:
    n = len(error_messages)
    return pd.DataFrame(
        {
            "model": ["qwen2.5:1.5b"] * n,
            "context_size_tokens": [1000] * n,
            "strategy": ["No-Guard"] * n,
            "ttft_seconds": [0.0] * n,
            "error_message": error_messages,
        }
    )


class TestSplitValidFailed(unittest.TestCase):
    def test_empty_string_is_valid(self) -> None:
        df = _make_df(["", "", ""])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 3)
        self.assertEqual(len(failed), 0)

    def test_nan_is_valid(self) -> None:
        # pd.read_csv produces NaN, not "", for an empty CSV cell -- both
        # must count as "no error" or every successful real-CSV row would
        # be misclassified as failed.
        df = _make_df([float("nan"), float("nan")])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(failed), 0)

    def test_populated_error_message_is_failed(self) -> None:
        df = _make_df(["", "Max iterations (12) reached", ""])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed.iloc[0]["error_message"], "Max iterations (12) reached")

    def test_whitespace_only_error_message_is_treated_as_valid(self) -> None:
        df = _make_df(["   ", ""])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(failed), 0)

    def test_mixed_nan_and_empty_and_real_errors(self) -> None:
        df = _make_df(["", float("nan"), "timed out", "HTTP error"])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(failed), 2)

    def test_end_to_end_via_real_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            df = _make_df(["", "Read timed out", ""])
            df.to_csv(csv_path, index=False)

            loaded = load_results(csv_path)
            valid_df, failed_df = split_valid_failed(loaded)

            self.assertEqual(len(valid_df), 2)
            self.assertEqual(len(failed_df), 1)
            self.assertTrue((valid_df["error_message"].fillna("").astype(str).str.strip() == "").all())


if __name__ == "__main__":
    unittest.main()
