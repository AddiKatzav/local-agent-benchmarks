"""Metrics collection, parsing, and incremental CSV persistence."""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from burnt_toast.config import SECRET_CODE

logger = logging.getLogger(__name__)

CSV_COLUMNS: list[str] = [
    "timestamp",
    "run_id",
    "hardware_env",
    "model",
    "context_size_tokens",
    "actual_context_tokens",
    "needle_position",
    "strategy",
    "ttft_seconds",
    "tokens_per_second",
    "peak_rss_mb",
    "peak_vms_mb",
    "prompt_tokens",
    "completion_tokens",
    "total_iterations",
    "guard_triggered",
    "json_valid",
    "accuracy",
    "extracted_secret_code",
    "raw_output_snippet",
    "error_message",
]


@dataclass
class MemoryTracker:
    """Poll system memory usage in a background thread during inference."""

    interval_seconds: float = 0.05
    _stop: threading.Event = field(default_factory=threading.Event, repr=False)
    _thread: threading.Thread | None = field(default=None, repr=False)
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0

    def start(self) -> None:
        self.peak_rss_mb = 0.0
        self.peak_vms_mb = 0.0
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float, float]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        return self.peak_rss_mb, self.peak_vms_mb

    def _poll(self) -> None:
        proc = psutil.Process()
        while not self._stop.is_set():
            try:
                mem = proc.memory_info()
                rss_mb = mem.rss / (1024 * 1024)
                vms_mb = mem.vms / (1024 * 1024)
                # Also track system-wide pressure
                vm = psutil.virtual_memory()
                sys_rss = (vm.total - vm.available) / (1024 * 1024)
                self.peak_rss_mb = max(self.peak_rss_mb, rss_mb, sys_rss)
                self.peak_vms_mb = max(self.peak_vms_mb, vms_mb)
            except (psutil.Error, OSError):
                pass
            time.sleep(self.interval_seconds)


@dataclass
class RunMetrics:
    """All metrics for a single benchmark run."""

    timestamp: str
    run_id: str
    hardware_env: str
    model: str
    context_size_tokens: int
    actual_context_tokens: int
    needle_position: str
    strategy: str
    ttft_seconds: float = 0.0
    tokens_per_second: float | None = None
    peak_rss_mb: float = 0.0
    peak_vms_mb: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_iterations: int = 0
    guard_triggered: bool = False
    json_valid: bool = False
    accuracy: bool = False
    extracted_secret_code: int | None = None
    raw_output_snippet: str = ""
    error_message: str = ""

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def validate_output(raw_output: str) -> tuple[bool, bool, int | None]:
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

    accuracy = code_int == SECRET_CODE
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


def make_run_id(
    model: str,
    context_size: int,
    needle_position: str,
    strategy: str,
    index: int,
) -> str:
    safe_model = model.replace(":", "_").replace(".", "_")
    return f"{safe_model}_{context_size}_{needle_position}_{strategy}_{index:04d}"
