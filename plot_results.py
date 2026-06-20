#!/usr/bin/env python3
"""Plot burnt-toast benchmark results from a CSV file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

METRIC_SPECS: list[tuple[str, str, str]] = [
    ("accuracy", "Accuracy", "fraction correct"),
    ("ttft_seconds", "Time to First Token", "seconds"),
    ("tokens_per_second", "Tokens per Second", "tok/s"),
    ("peak_rss_mb", "Peak RSS Memory", "MB"),
    ("total_iterations", "Agent Iterations", "count"),
    ("tool_call_count", "Tool Calls", "count"),
]

STRATEGY_ORDER = ["No-Guard", "Python-Guard", "Critic"]
NEEDLE_STYLES = {"middle": "-", "end": "--"}
STRATEGY_COLORS = {
    "No-Guard": "#e45756",
    "Python-Guard": "#4c78a8",
    "Critic": "#54a24b",
}


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    bool_cols = ["guard_triggered", "json_valid", "accuracy"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].map(
                lambda v: str(v).strip().lower() in {"true", "1", "yes"}
                if not isinstance(v, bool)
                else v
            )

    numeric_cols = [
        "context_size_tokens",
        "actual_context_tokens",
        "ttft_seconds",
        "tokens_per_second",
        "peak_rss_mb",
        "peak_vms_mb",
        "prompt_tokens",
        "completion_tokens",
        "total_iterations",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def _title_parts(df: pd.DataFrame, csv_path: Path) -> str:
    models = ", ".join(sorted(df["model"].dropna().unique()))
    hw = ", ".join(sorted(df["hardware_env"].dropna().unique()))
    return f"{csv_path.stem}  |  {models}  |  {hw}"


def plot_results(df: pd.DataFrame, csv_path: Path, output_path: Path) -> None:
    """Render a multi-panel dashboard and save to *output_path*."""
    strategies = [s for s in STRATEGY_ORDER if s in df["strategy"].unique()]
    needles = sorted(df["needle_position"].dropna().unique())
    context_sizes = sorted(df["context_size_tokens"].dropna().unique())

    nrows, ncols = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 12), constrained_layout=True)
    axes_flat = axes.flatten()

    for ax, (column, title, ylabel) in zip(axes_flat, METRIC_SPECS):
        if column not in df.columns:
            ax.set_visible(False)
            continue
        for strategy in strategies:
            for needle in needles:
                subset = df[
                    (df["strategy"] == strategy)
                    & (df["needle_position"] == needle)
                ].sort_values("context_size_tokens")

                if subset.empty or column not in subset.columns:
                    continue

                label = f"{strategy} ({needle})"
                ax.plot(
                    subset["context_size_tokens"],
                    subset[column],
                    marker="o",
                    linestyle=NEEDLE_STYLES.get(needle, "-"),
                    color=STRATEGY_COLORS.get(strategy, None),
                    label=label,
                    linewidth=2,
                    markersize=6,
                )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Context size (tokens)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(context_sizes)
        ax.set_xticklabels([f"{int(x / 1000)}K" if x >= 1000 else str(int(x)) for x in context_sizes])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")

        if column == "accuracy":
            ax.set_ylim(-0.05, 1.05)

    fig.suptitle(_title_parts(df, csv_path), fontsize=13, fontweight="bold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot burnt-toast benchmark CSV results to a PNG dashboard.",
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to benchmark results CSV (e.g. results_qwen.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: same name as CSV with .png suffix)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_path: Path = args.csv_file.resolve()
    output_path: Path = (
        args.output.resolve()
        if args.output
        else csv_path.with_suffix(".png")
    )

    try:
        df = load_results(csv_path)
        if df.empty:
            print(f"Error: no data rows in {csv_path}", file=sys.stderr)
            return 1
        plot_results(df, csv_path, output_path)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Saved plot to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
