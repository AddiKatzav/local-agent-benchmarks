---
name: pipeline-sanity-check
description: >
  This skill should be used when the user wants to "sanity-check the burnt_toast pipeline",
  "verify the memory tracker", "check for KV-cache contamination", "validate burnt_toast
  measurements", "check the haystack seeding fix", or before trusting new CSV results after
  any measurement-related code change in burnt_toast/ (context.py, metrics.py, ollama_client.py,
  runner.py). Validates that burnt_toast's measurement primitives (haystack generation, memory
  tracking) produce correct, non-confounded results -- it does not validate accuracy/guard-logic
  correctness, only whether the *measurements* (TTFT, peak RSS, context truncation) mean what
  they claim to mean.
---

# Pipeline Sanity Check

burnt_toast's benchmark numbers are only as trustworthy as the primitives that produce them.
This skill runs a layered check of those primitives: pure unit tests first, then (when Ollama
is reachable) live checks against the real running instance.

It currently covers these measurement-validity classes that have bitten this repo before:

1. **Prompt-cache contamination** -- if two different matrix cells (model/context/needle-position/
   strategy/mode) build byte-identical haystack text, Ollama's automatic KV-cache prefix reuse
   silently skips re-evaluating shared prefixes on later calls, crashing TTFT and corrupting the
   `context_truncated`/`needle_in_window` heuristics without throwing any error.
2. **Wrong-process memory tracking** -- inference happens in the external `ollama` process, not
   in the benchmark harness. Tracking the wrong process (or falling back to system-wide memory)
   produces a `peak_rss_mb` that looks plausible but means "whatever the machine was doing,"
   not "how much memory this model used."
3. **Non-deterministic secret-code seeding** -- `generate_run_secret()` previously used Python's
   builtin `hash()`, which is randomized per process unless `PYTHONHASHSEED` is fixed, silently
   breaking the documented "rerunning the same config reproduces the same code" guarantee across
   process boundaries (caught only by a unit test that spawns real subprocesses).
4. **Failed runs polluting plotted averages** -- a timed-out/errored run still writes a CSV row
   with all-zero numeric fields plus a populated `error_message`; without filtering, those zeros
   get averaged into TTFT/accuracy curves right at the high-context cells most likely to fail.
5. **Unmeasured truncation-heuristic overhead** -- `is_context_truncated`/`needle_in_visible_window`
   compared two non-equivalent token counts via a hardcoded, never-measured `overhead_tokens=120`
   guess instead of the real wrapper-prose length for the given (model, mode).

This script is structured so a future measurement-validity concern can be added as a new tier
without disturbing the existing ones.

## Running it

From the repo root:

```bash
python scripts/validate_pipeline.py
```

This runs, in order:

1. **Unit tests** (`python -m unittest discover -s tests`) -- always runs, no Ollama needed.
   Covers `haystack_seed()`/`stable_seed()` determinism/divergence, `MemoryTracker`'s
   process-matching logic against a mocked `psutil` (sum-across-multiple-processes,
   late-arriving PID, mid-run disappearance, PID reuse, never-found warning), secret-code
   reproducibility (including a subprocess-spawning test that actually proves cross-process
   stability, not just same-process), `plot_results.split_valid_failed()`'s failed-row exclusion,
   `measure_prompt_overhead_tokens()`'s caching behavior against a mocked client, and
   `tools.py`'s nested-brace JSON parsing.
2. **Ollama reachability probe** -- if unreachable, tiers 3-6 below are reported as **SKIPPED**,
   not failed or passed. This is intentional: the script must stay runnable on a machine where
   Ollama isn't currently up.
3. **Live memory-tracking check** -- runs the real `MemoryTracker` for a few seconds against
   whatever Ollama processes are actually running, and asserts the reported peak is nonzero and
   clearly smaller than total system-wide used memory. This is the actual proof that memory
   tracking targets the right process.
4. **Live haystack-divergence check** -- builds two haystacks for the same (model, context size,
   needle position) but different strategies, using the real `OllamaClient`, and asserts the
   filler text differs beyond the embedded secret code. This is the actual proof that different
   matrix cells no longer share a cacheable prompt prefix.
5. **Live overhead-measurement check** -- calls `measure_prompt_overhead_tokens()` against the
   real `OllamaClient` for both experiment modes, asserting a positive, stable-across-repeated-calls
   token count -- the actual proof the truncation/needle-window heuristics use a real measurement
   instead of the old hardcoded `120` guess.
6. **No-regression smoke check** -- runs `python -m burnt_toast --quick` end-to-end for both
   `needle` and `burnt-toast` modes and checks the produced CSV row has every expected column.
   This is a shape/regression check only -- a row can look "successful" even with a broken
   memory tracker or overhead measurement, so do not treat a PASS here as proof of any other tier's claims.

## Interpreting results

The script prints a `[PASS]` / `[FAIL]` / `[SKIP]` line per tier with a one-line detail, then a
`N passed, N failed, N skipped` summary, and exits non-zero if anything failed. Report this
summary to the user plainly, including which tiers were skipped and why (almost always: Ollama
wasn't reachable at the configured base URL). Do not characterize a SKIP as a PASS.

If a live tier fails, the detail line names the specific mismatch (e.g. "peak_rss_mb >=
system_used_mb" means the system-wide fallback path has regressed back in) -- use that to point
directly at the relevant function (`MemoryTracker._sample_and_update` / `_rescan` in
`burnt_toast/metrics.py`, or `haystack_seed` / `build_haystack` in `burnt_toast/context.py`)
rather than re-deriving the diagnosis from scratch.
