"""Experiment orchestration: runs the full benchmark matrix."""

from __future__ import annotations

import itertools
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from burnt_toast.config import (
    CONTEXT_SIZES_TOKENS,
    CRITIC_MODEL,
    HARDWARE_ENV,
    MODELS,
    NEEDLE_POSITIONS,
    RESULTS_CSV,
    STRATEGIES,
    RunConfig,
)
from burnt_toast.context import build_haystack
from burnt_toast.metrics import (
    MemoryTracker,
    ResultsWriter,
    RunMetrics,
    make_run_id,
    validate_output,
)
from burnt_toast.ollama_client import OllamaClient
from burnt_toast.strategies import get_strategy

logger = logging.getLogger(__name__)


def generate_run_matrix(
    models: list[str] | None = None,
    context_sizes: list[int] | None = None,
    needle_positions: list[str] | None = None,
    strategies: list[str] | None = None,
    hardware_env: str | None = None,
) -> list[RunConfig]:
    """Build the full cartesian product of experimental variables."""
    models = models or MODELS
    context_sizes = context_sizes or CONTEXT_SIZES_TOKENS
    needle_positions = needle_positions or NEEDLE_POSITIONS
    strategies = strategies or STRATEGIES
    hardware_env = hardware_env or HARDWARE_ENV

    configs: list[RunConfig] = []
    for idx, (model, ctx, needle, strategy) in enumerate(
        itertools.product(models, context_sizes, needle_positions, strategies)
    ):
        configs.append(
            RunConfig(
                model=model,
                context_size_tokens=ctx,
                needle_position=needle,  # type: ignore[arg-type]
                strategy=strategy,  # type: ignore[arg-type]
                hardware_env=hardware_env,
                run_index=idx,
            )
        )
    return configs


def run_single(
    config: RunConfig,
    client: OllamaClient,
    writer: ResultsWriter,
    *,
    critic_model: str | None = None,
) -> RunMetrics:
    """Execute one benchmark run and persist results."""
    run_id = make_run_id(
        config.model,
        config.context_size_tokens,
        config.needle_position,
        config.strategy,
        config.run_index,
    )
    timestamp = datetime.now(timezone.utc).isoformat()

    logger.info("=" * 72)
    logger.info(
        "START run=%s | model=%s | ctx=%d | needle=%s | strategy=%s | hw=%s",
        run_id,
        config.model,
        config.context_size_tokens,
        config.needle_position,
        config.strategy,
        config.hardware_env,
    )

    metrics = RunMetrics(
        timestamp=timestamp,
        run_id=run_id,
        hardware_env=config.hardware_env,
        model=config.model,
        context_size_tokens=config.context_size_tokens,
        actual_context_tokens=0,
        needle_position=config.needle_position,
        strategy=config.strategy,
    )

    memory_tracker = MemoryTracker()
    memory_tracker.start()
    t0 = time.perf_counter()

    try:
        haystack, actual_tokens = build_haystack(
            client,
            config.model,
            config.context_size_tokens,
            config.needle_position,
        )
        metrics.actual_context_tokens = actual_tokens
        logger.info("Haystack built: %d tokens (target %d)", actual_tokens, config.context_size_tokens)

        strategy = get_strategy(
            config.strategy,
            client,
            config.model,
            critic_model=critic_model,
        )
        agent_result = strategy.run(haystack)

        metrics.total_iterations = agent_result.iterations
        metrics.guard_triggered = agent_result.guard_triggered
        metrics.ttft_seconds = round(agent_result.mean_ttft, 6)
        if agent_result.mean_tokens_per_second is not None:
            metrics.tokens_per_second = round(agent_result.mean_tokens_per_second, 4)
        metrics.prompt_tokens = agent_result.total_prompt_tokens
        metrics.completion_tokens = agent_result.total_completion_tokens
        metrics.raw_output_snippet = agent_result.final_output[:500]
        metrics.error_message = agent_result.error_message

        json_valid, accuracy, extracted = validate_output(agent_result.final_output)
        metrics.json_valid = json_valid
        metrics.accuracy = accuracy
        metrics.extracted_secret_code = extracted

    except Exception as exc:
        logger.exception("Run %s failed: %s", run_id, exc)
        metrics.error_message = str(exc)

    elapsed = time.perf_counter() - t0
    peak_rss, peak_vms = memory_tracker.stop()
    metrics.peak_rss_mb = round(peak_rss, 2)
    metrics.peak_vms_mb = round(peak_vms, 2)

    logger.info(
        "DONE  run=%s | elapsed=%.1fs | ttft=%.3fs | tps=%s | "
        "accuracy=%s | json_valid=%s | iterations=%d | guard=%s | "
        "peak_rss=%.1fMB",
        run_id,
        elapsed,
        metrics.ttft_seconds,
        metrics.tokens_per_second,
        metrics.accuracy,
        metrics.json_valid,
        metrics.total_iterations,
        metrics.guard_triggered,
        metrics.peak_rss_mb,
    )

    writer.append(metrics)
    return metrics


def run_benchmark(
    *,
    models: list[str] | None = None,
    context_sizes: list[int] | None = None,
    needle_positions: list[str] | None = None,
    strategies: list[str] | None = None,
    hardware_env: str | None = None,
    ollama_base_url: str | None = None,
    results_path: str | None = None,
    dry_run: bool = False,
) -> list[RunMetrics]:
    """
    Run the complete benchmark matrix (or a filtered subset).

    Returns a list of RunMetrics for all completed runs.
    """
    from burnt_toast.config import OLLAMA_BASE_URL

    configs = generate_run_matrix(
        models=models,
        context_sizes=context_sizes,
        needle_positions=needle_positions,
        strategies=strategies,
        hardware_env=hardware_env,
    )

    logger.info("Benchmark matrix: %d total runs", len(configs))

    if dry_run:
        for cfg in configs:
            logger.info(
                "  [dry-run] %s | ctx=%d | needle=%s | strategy=%s",
                cfg.model,
                cfg.context_size_tokens,
                cfg.needle_position,
                cfg.strategy,
            )
        return []

    base_url = ollama_base_url or OLLAMA_BASE_URL
    client = OllamaClient(base_url)

    if not client.health_check():
        raise ConnectionError(
            f"Cannot reach Ollama at {base_url}. "
            "Ensure Ollama is running and models are pulled."
        )

    requested_models = sorted({cfg.model for cfg in configs})
    if "Critic" in {cfg.strategy for cfg in configs}:
        requested_models = sorted(set(requested_models) | {CRITIC_MODEL})
    resolved = client.validate_models(requested_models)
    logger.info("Ollama models available: %s", client.list_models())
    logger.info("Resolved benchmark models: %s", resolved)

    # Map requested config names to resolved Ollama names
    available = client.list_models()
    model_map = {
        m: client.resolve_model(m, available) or m for m in requested_models
    }

    writer = ResultsWriter(Path(results_path) if results_path else RESULTS_CSV)
    all_metrics: list[RunMetrics] = []

    for i, config in enumerate(configs, 1):
        logger.info("Progress: %d / %d", i, len(configs))
        if config.ollama_base_url != base_url:
            config.ollama_base_url = base_url
        config.model = model_map.get(config.model, config.model)
        metrics = run_single(
            config,
            client,
            writer,
            critic_model=model_map.get(CRITIC_MODEL, CRITIC_MODEL),
        )
        all_metrics.append(metrics)

    logger.info("Benchmark complete. Results saved to %s", writer.path)
    return all_metrics
