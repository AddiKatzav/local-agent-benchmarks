"""Unit tests for plot_results.split_valid_failed (no Ollama/matplotlib rendering needed)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from plot_results import load_results, split_valid_failed


def _make_df(rows: list[dict]) -> pd.DataFrame:
    defaults = {
        "model": "qwen2.5:1.5b",
        "context_size_tokens": 1000,
        "strategy": "No-Guard",
        "ttft_seconds": 0.0,
        "total_iterations": 1,
        "error_message": "",
    }
    return pd.DataFrame([{**defaults, **row} for row in rows])


class TestSplitValidFailed(unittest.TestCase):
    def test_successful_run_with_no_error_message_is_valid(self) -> None:
        df = _make_df([{"total_iterations": 1, "error_message": ""}])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(failed), 0)

    def test_true_crash_with_zero_iterations_is_failed(self) -> None:
        # A genuine exception (HTTP timeout, connection error) is caught
        # before the agent loop returns anything -- total_iterations stays
        # at its dataclass default of 0, alongside the populated error_message.
        df = _make_df([{
            "total_iterations": 0,
            "ttft_seconds": 0.0,
            "error_message": "HTTPConnectionPool(host='localhost', port=11434): Read timed out.",
        }])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(failed), 1)

    def test_max_iterations_reached_with_real_telemetry_is_valid_not_failed(self) -> None:
        # This is the critical case: burnt-toast mode's No-Guard strategy is
        # EXPECTED to exhaust every iteration and populate error_message with
        # "Max iterations (N) reached" -- but unlike a crash, total_iterations,
        # tool_call_count, accuracy etc. are all real, fully-formed telemetry.
        # This is the headline result the No-Guard condition exists to
        # produce; excluding it via error_message alone would silently drop
        # exactly that comparison from every burnt-toast plot.
        df = _make_df([{
            "total_iterations": 12,
            "ttft_seconds": 1.3,
            "error_message": "Max iterations (12) reached",
        }])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 1)
        self.assertEqual(len(failed), 0)

    def test_nan_total_iterations_is_treated_as_crash(self) -> None:
        df = _make_df([{"total_iterations": float("nan"), "error_message": "some error"}])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 0)
        self.assertEqual(len(failed), 1)

    def test_mixed_crashes_and_completed_runs(self) -> None:
        df = _make_df([
            {"total_iterations": 1, "error_message": ""},
            {"total_iterations": 0, "error_message": "Read timed out"},
            {"total_iterations": 12, "error_message": "Max iterations (12) reached"},
            {"total_iterations": 0, "error_message": "HTTP error"},
        ])
        valid, failed = split_valid_failed(df)
        self.assertEqual(len(valid), 2)
        self.assertEqual(len(failed), 2)

    def test_end_to_end_via_real_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            df = _make_df([
                {"total_iterations": 1, "error_message": ""},
                {"total_iterations": 0, "error_message": "Read timed out"},
                {"total_iterations": 12, "error_message": "Max iterations (12) reached"},
            ])
            df.to_csv(csv_path, index=False)

            loaded = load_results(csv_path)
            valid_df, failed_df = split_valid_failed(loaded)

            self.assertEqual(len(valid_df), 2)
            self.assertEqual(len(failed_df), 1)
            self.assertTrue((valid_df["total_iterations"] > 0).all())


if __name__ == "__main__":
    unittest.main()
