"""CLI entry point: python -m burnt_toast"""

from __future__ import annotations

import argparse
import logging
import sys

from burnt_toast import __version__
from burnt_toast.config import (
    CONTEXT_SIZES_TOKENS,
    HARDWARE_ENV,
    MODELS,
    NEEDLE_POSITIONS,
    OLLAMA_BASE_URL,
    RESULTS_CSV,
    STRATEGIES,
)
from burnt_toast.runner import run_benchmark


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="burnt_toast",
        description=(
            "Agent Burnt Toast Effect + Needle-in-Haystack benchmark suite. "
            "Evaluates loop-guard strategies under context inflation."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    parser.add_argument(
        "--hardware-env",
        default=HARDWARE_ENV,
        help=f"Hardware environment label (default: {HARDWARE_ENV})",
    )
    parser.add_argument(
        "--ollama-url",
        default=OLLAMA_BASE_URL,
        help=f"Ollama base URL (default: {OLLAMA_BASE_URL})",
    )
    parser.add_argument(
        "--results",
        default=str(RESULTS_CSV),
        help=f"Output CSV path (default: {RESULTS_CSV})",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="Ollama models to benchmark",
    )
    parser.add_argument(
        "--context-sizes",
        nargs="+",
        type=int,
        default=CONTEXT_SIZES_TOKENS,
        help="Context window sizes in tokens",
    )
    parser.add_argument(
        "--needle-positions",
        nargs="+",
        choices=["middle", "end"],
        default=NEEDLE_POSITIONS,
        help="Needle insertion positions",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["No-Guard", "Python-Guard", "Critic"],
        default=STRATEGIES,
        help="Agent loop guard strategies",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run matrix without executing",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke test: 1 model, 1K context, 1 strategy",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(verbose=args.verbose)

    models = args.models
    context_sizes = args.context_sizes
    strategies = args.strategies
    needle_positions = args.needle_positions

    if args.quick:
        models = [models[0]]
        context_sizes = [1_000]
        strategies = ["No-Guard"]
        needle_positions = ["middle"]

    try:
        run_benchmark(
            models=models,
            context_sizes=context_sizes,
            needle_positions=needle_positions,
            strategies=strategies,
            hardware_env=args.hardware_env,
            ollama_base_url=args.ollama_url,
            results_path=args.results,
            dry_run=args.dry_run,
        )
    except (ConnectionError, RuntimeError) as exc:
        logging.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        logging.warning("Benchmark interrupted by user")
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
