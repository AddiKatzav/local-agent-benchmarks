#!/usr/bin/env python3
"""
Sanity-check that burnt_toast's measurement pipeline measures what it claims to.

Four independent tiers:
  1. Unit tests (no Ollama required) -- always run.
  2. Ollama reachability probe -- gates tiers 3-5.
  3. Live memory-tracking check -- proves MemoryTracker reads the Ollama
     process, not the harness/system (fix #2).
  4. Live haystack-divergence check -- proves different matrix cells build
     different filler text, defeating Ollama's prompt-cache reuse (fix #1).
  5. No-regression smoke check -- runs `python -m burnt_toast --quick` end to
     end for both modes and validates CSV row shape. This is a regression
     check only -- it does NOT prove either fix on its own (a buggy
     MemoryTracker can still produce a "successful," nonzero-but-wrong row).

Tiers 3-5 are skipped (not failed) when Ollama is unreachable, so this script
stays usable later when Ollama may not be running.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import psutil  # noqa: E402

from burnt_toast.config import MODELS, OLLAMA_BASE_URL  # noqa: E402
from burnt_toast.context import build_haystack, haystack_seed  # noqa: E402
from burnt_toast.metrics import CSV_COLUMNS, MemoryTracker  # noqa: E402
from burnt_toast.ollama_client import OllamaClient  # noqa: E402


class Result:
    def __init__(self, name: str, status: str, detail: str = "") -> None:
        self.name = name
        self.status = status  # "PASS" | "FAIL" | "SKIP"
        self.detail = detail


def run_unit_tests() -> Result:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return Result("unit tests", "PASS", proc.stderr.strip().splitlines()[-1] if proc.stderr else "")
    return Result("unit tests", "FAIL", proc.stdout[-2000:] + proc.stderr[-2000:])


def check_ollama_reachable(client: OllamaClient) -> bool:
    return client.health_check()


def check_memory_tracker_live() -> Result:
    tracker = MemoryTracker(interval_seconds=0.05, rescan_interval_seconds=0.2)
    tracker.start()
    time.sleep(3.0)
    peak_rss_mb, peak_vms_mb = tracker.stop()

    system_used_mb = (psutil.virtual_memory().used) / (1024 * 1024)

    if peak_rss_mb <= 0:
        return Result(
            "live memory tracker",
            "FAIL",
            f"peak_rss_mb={peak_rss_mb:.1f} -- no local 'ollama' process found "
            "(is Ollama running locally? `ollama ps` shows a loaded model?)",
        )

    if peak_rss_mb >= system_used_mb:
        return Result(
            "live memory tracker",
            "FAIL",
            f"peak_rss_mb={peak_rss_mb:.1f} >= system_used_mb={system_used_mb:.1f} "
            "-- looks like the system-wide fallback is still being used.",
        )

    return Result(
        "live memory tracker",
        "PASS",
        f"peak_rss_mb={peak_rss_mb:.1f} peak_vms_mb={peak_vms_mb:.1f} "
        f"(system_used_mb={system_used_mb:.1f}, tracked value is a small "
        f"fraction of system-wide usage, as expected)",
    )


def check_haystack_divergence_live(client: OllamaClient) -> Result:
    model = MODELS[0]
    target_tokens = 1000
    needle_position = "middle"
    secret_phrase = "The secret agent code is 4821"

    seed_a = haystack_seed(model, target_tokens, needle_position, "No-Guard", "needle")
    seed_b = haystack_seed(model, target_tokens, needle_position, "Critic", "needle")
    if seed_a == seed_b:
        return Result("live haystack divergence", "FAIL", "haystack_seed() collided across strategies")

    text_a, _ = build_haystack(client, model, target_tokens, needle_position, secret_phrase=secret_phrase, seed=seed_a)
    text_b, _ = build_haystack(client, model, target_tokens, needle_position, secret_phrase=secret_phrase, seed=seed_b)

    # Strip the (identical) embedded secret phrase before comparing, so the
    # assertion is purely about filler-text divergence.
    stripped_a = text_a.replace(secret_phrase, "")
    stripped_b = text_b.replace(secret_phrase, "")

    if stripped_a == stripped_b:
        return Result(
            "live haystack divergence",
            "FAIL",
            "filler text identical across strategies for the same (model, context, position) -- "
            "Ollama prompt-cache reuse would still corrupt TTFT/truncation metrics",
        )

    import difflib

    shared_len = min(len(stripped_a), len(stripped_b))
    first_diff = next((i for i in range(shared_len) if stripped_a[i] != stripped_b[i]), shared_len)
    similarity = difflib.SequenceMatcher(None, stripped_a, stripped_b).ratio()
    return Result(
        "live haystack divergence",
        "PASS",
        f"strategies share only a {first_diff}-char prefix (the filler corpus has just 8 distinct "
        f"paragraph templates, so a shared opening paragraph happens by chance) before diverging; "
        f"overall text similarity ratio={similarity:.3f} (1.0 would mean identical, i.e. the bug)",
    )


def run_smoke_benchmark() -> Result:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for mode in ("needle", "burnt-toast"):
            out_path = Path(tmpdir) / f"smoke_{mode}.csv"
            proc = subprocess.run(
                [
                    sys.executable, "-m", "burnt_toast",
                    "--quick", "--mode", mode,
                    "--results", str(out_path),
                    "--no-unique-suffix",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                failures.append(f"[{mode}] exit code {proc.returncode}: {proc.stderr[-1000:]}")
                continue
            if not out_path.exists():
                failures.append(f"[{mode}] no CSV produced at {out_path}")
                continue
            with out_path.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            if len(rows) != 1:
                failures.append(f"[{mode}] expected 1 row, got {len(rows)}")
                continue
            row = rows[0]
            missing = [c for c in CSV_COLUMNS if c not in row]
            if missing:
                failures.append(f"[{mode}] missing CSV columns: {missing}")

    if failures:
        return Result("no-regression smoke check", "FAIL", "; ".join(failures))
    return Result(
        "no-regression smoke check",
        "PASS",
        "both modes ran end-to-end via `--quick` and produced a well-formed CSV row "
        "(shape/regression check only -- not proof of either fix)",
    )


def main() -> int:
    results: list[Result] = [run_unit_tests()]

    client = OllamaClient(OLLAMA_BASE_URL)
    ollama_up = check_ollama_reachable(client)

    if not ollama_up:
        reason = f"Ollama unreachable at {OLLAMA_BASE_URL}"
        results.append(Result("live memory tracker", "SKIP", reason))
        results.append(Result("live haystack divergence", "SKIP", reason))
        results.append(Result("no-regression smoke check", "SKIP", reason))
    else:
        results.append(check_memory_tracker_live())
        results.append(check_haystack_divergence_live(client))
        results.append(run_smoke_benchmark())

    print("\n" + "=" * 72)
    print("PIPELINE SANITY CHECK SUMMARY")
    print("=" * 72)
    for r in results:
        print(f"[{r.status:4}] {r.name}")
        if r.detail:
            for line in r.detail.splitlines():
                print(f"        {line}")
    print("=" * 72)

    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    print(f"{passed} passed, {failed} failed, {skipped} skipped")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
