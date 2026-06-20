"""Metrics collection, parsing, and incremental CSV persistence."""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

logger = logging.getLogger(__name__)

CSV_COLUMNS: list[str] = [
    "timestamp",
    "run_id",
    "experiment_mode",
    "hardware_env",
    "model",
    "context_size_tokens",
    "actual_context_tokens",
    "effective_prompt_tokens",
    "num_ctx",
    "context_visibility",
    "context_truncated",
    "needle_in_window",
    "needle_position",
    "strategy",
    "ttft_seconds",
    "tokens_per_second",
    "peak_rss_mb",
    "peak_vms_mb",
    "prompt_tokens",
    "completion_tokens",
    "total_iterations",
    "tool_call_count",
    "guard_triggered",
    "json_valid",
    "accuracy",
    "expected_secret_code",
    "extracted_secret_code",
    "raw_output_snippet",
    "error_message",
]


@dataclass
class MemoryTracker:
    """
    Poll the *Ollama inference process'* memory usage in a background thread.

    Inference happens in the external `ollama` process (server + per-model
    runner subprocesses), not in this benchmark harness — so tracking
    `psutil.Process()` (this script) or system-wide memory conflates unrelated
    processes with the model's actual footprint. Instead this matches all
    locally-visible processes named *process_name* (default "ollama") by
    `psutil` name and sums their RSS/VMS, which on a typical Ollama install
    covers both the long-lived server and any currently-loaded model runner
    subprocess(es) — including two simultaneously-loaded models (main +
    Critic strategy's secondary model).

    A model only loads into a runner subprocess on its first inference call,
    which happens *after* `start()`, so the matching PID set is rediscovered
    periodically (every `rescan_interval_seconds`) rather than only once.
    """

    interval_seconds: float = 0.05
    rescan_interval_seconds: float = 1.0
    process_name: str = "ollama"
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    _known_pids: set[int] = field(default_factory=set, repr=False)
    _ever_found: bool = field(default=False, repr=False)
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0

    def start(self) -> None:
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self._known_pids = set()
        self._ever_found = False
        self._stop.clear()
        self._rescan()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if not self._ever_found:
            logger.warning(
                "MemoryTracker never found a process named %r — peak_rss_mb/"
                "peak_vms_mb will read 0 for this run (Ollama not local, or "
                "process name differs).",
                self.process_name,
            )
        return self.peak_rss_mb, self.peak_vms_mb

    def _rescan(self) -> None:
        """Full process-table scan; union newly matching pids into _known_pids."""
        target = self.process_name.lower()
        try:
            for proc in psutil.process_iter(["pid", "name"]):
                if (proc.info.get("name") or "").lower() == target:
                    self._known_pids.add(proc.info["pid"])
        except (psutil.Error, OSError):
            pass

    def _sample_and_update(self) -> tuple[float, float]:
        """Sum memory of all currently-known pids and fold into the running peak.

        Returns the (rss_mb, vms_mb) snapshot summed this call, for testing.
        """
        target = self.process_name.lower()
        total_rss = 0.0
        total_vms = 0.0
        for pid in list(self._known_pids):
            try:
                proc = psutil.Process(pid)
                # Guard against PID reuse: a dead pid may be recycled by an
                # unrelated process before the next rescan evicts it.
                if proc.name().lower() != target:
                    self._known_pids.discard(pid)
                    continue
                mem = proc.memory_info()
                total_rss += mem.rss / (1024 * 1024)
                total_vms += mem.vms / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                self._known_pids.discard(pid)
            except (psutil.Error, OSError):
                pass

        if self._known_pids:
            self._ever_found = True
            self.peak_rss_mb = max(self.peak_rss_mb, total_rss)
            self.peak_vms_mb = max(self.peak_vms_mb, total_vms)

        return total_rss, total_vms

    def _poll(self) -> None:
        last_scan = time.monotonic()
        while not self._stop.is_set():
            now = time.monotonic()
            if now - last_scan >= self.rescan_interval_seconds:
                self._rescan()
                last_scan = now

            self._sample_and_update()
            time.sleep(self.interval_seconds)


@dataclass
class RunMetrics:
    """All metrics for a single benchmark run."""

    timestamp: str
    run_id: str
    experiment_mode: str
    hardware_env: str
    model: str
    context_size_tokens: int
    actual_context_tokens: int
    needle_position: str
    strategy: str
    effective_prompt_tokens: int = 0
    num_ctx: int = 0
    context_visibility: str = ""
    context_truncated: bool = False
    needle_in_window: bool = False
    ttft_seconds: float = 0.0
    tokens_per_second: float | None = None
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_iterations: int = 0
    tool_call_count: int = 0
    guard_triggered: bool = False
    json_valid: bool = False
    accuracy: bool = False
    expected_secret_code: int = 0
    extracted_secret_code: int | None = None
    raw_output_snippet: str = ""
    error_message: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def validate_output(raw_output: str, expected_code: int) -> tuple[bool, bool, int | None]:
    """
    Parse model output and verify JSON validity + secret code extraction.

    Returns (json_valid, accuracy, extracted_code).
    """
    from burnt_toast.tools import parse_final_answer

    parsed = parse_final_answer(raw_output)
    if parsed is None:
        return False, False, None

    try:
        json.dumps(parsed)
        json_valid = True
    except (TypeError, ValueError):
        return False, False, None

    code = parsed.get("secret_code")
    try:
        code_int = int(code)
    except (TypeError, ValueError):
        return json_valid, False, None

    accuracy = code_int == expected_code
    return json_valid, accuracy, code_int


class ResultsWriter:
    """Thread-safe incremental CSV writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                writer.writeheader()
            logger.info("Created results file: %s", self.path)

    def append(self, metrics: RunMetrics) -> None:
        with self._lock:
            with self.path.open("a", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
                row = metrics.to_row()
                writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        logger.info(
            "Recorded run %s | model=%s ctx=%d strategy=%s accuracy=%s",
            metrics.run_id,
            metrics.model,
            metrics.context_size_tokens,
            metrics.strategy,
            metrics.accuracy,
        )


def unique_results_path(base: Path, *, when: datetime | None = None) -> Path:
    """
    Append a 5-char UUID and MM_DD_HH_MM timestamp before the file extension.

    Example: results_qwen.csv -> results_qwen_a3f2b_06_09_21_30.csv
    """
    when = when or datetime.now()
    run_tag = when.strftime("%m_%d_%H_%M")
    short_id = uuid.uuid4().hex[:5]
    return base.with_name(f"{base.stem}_{short_id}_{run_tag}{base.suffix}")


def make_run_id(
    model: str,
    context_size: int,
    needle_position: str,
    strategy: str,
    index: int,
) -> str:
    safe_model = model.replace(":", "_").replace(".", "_")
    return f"{safe_model}_{context_size}_{needle_position}_{strategy}_{index:04d}"
