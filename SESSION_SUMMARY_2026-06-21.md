# Session summary — 2026-06-21: audit, fixes, validation, and post-fix data collection

Context for whoever (human or Claude) analyzes `results_post_fix_2026-06-21/` next. This
session found and fixed 5 measurement-corrupting bugs in the `burnt_toast` benchmark
harness, validated each fix, then re-ran the experiment matrix on the fixed code. Read
this before trusting any numbers in that directory.

## TL;DR for the analyst

- Use **only** `results_post_fix_2026-06-21/results_needle.csv` and
  `results_burnt_toast.csv`. Every other CSV/PNG in the repo root (`results_qwen*.csv`,
  `smoke_bt*.csv`, `burnt_toast_results.csv`, etc.) predates these fixes and is **not
  trustworthy** — they were generated under the bugs described below.
- **Needle mode is incomplete by design**: 78/90 rows. 19 of those are genuine
  HTTP-timeout crashes (correctly excluded from the plot, kept in the CSV). 12 cells were
  never run at all (`llama3.1:8b` at 8K/16K context) — deliberately skipped once the
  timeout pattern was unambiguous, to avoid burning ~2 more hours confirming it.
- **Burnt-toast mode is complete**: 90/90 rows, 0 errors.
- `plot_results.py` already excludes true crashes from the rendered PNGs and prints an
  exclusion summary to stderr when it does. The PNGs in that directory reflect this.

## What was wrong (original audit, before this session's fixes)

A from-scratch read of every file in `burnt_toast/` turned up 5 measurement-corrupting
bugs plus 2 minor code-quality issues:

1. **Haystack cache contamination** — `build_haystack()` never varied its RNG seed
   across matrix cells, so different models/strategies/runs at the same
   (context_size, needle_position) got byte-identical filler text. Ollama's automatic
   KV-cache prefix reuse then silently skipped re-evaluating shared prefixes on later
   calls, crashing TTFT and corrupting `context_truncated`/`needle_in_window`.
2. **Wrong-process memory tracking** — `MemoryTracker` polled the benchmark harness's
   own process plus system-wide used memory, not the actual `ollama` inference process.
   `peak_rss_mb` meant "whatever the machine was doing," not "how much memory the model
   used."
3. **Non-deterministic secret codes** — `generate_run_secret()` used Python's builtin
   `hash()`, randomized per process via `PYTHONHASHSEED`, silently breaking the
   documented "rerun reproduces the same code" guarantee across process boundaries.
4. **Failed runs polluting plotted averages** — a crashed run's all-zero telemetry was
   getting averaged into TTFT/accuracy curves indistinguishably from real data.
5. **Unmeasured truncation-heuristic overhead** — `is_context_truncated`/
   `needle_in_visible_window` compared two non-equivalent token counts via a hardcoded,
   never-measured `overhead_tokens=120` guess.
6. Cleanup: dead branch in `NoGuardStrategy.run()`.
7. Cleanup: `tools.py`'s bare-JSON regexes couldn't handle nested braces (e.g. a tool
   call's `arguments` sub-object).

## What was fixed, and how each was proven

All work landed in 5 commits on `main` (newest first):

- `4cf610e` — **Fix #4-correction**: see "A bug in the bug fix" below.
- `d0f7a32` — **Fixes #3, #4, #5** + the two cleanups.
- `0c1e964` — **Fixes #1, #2**.
- `bd561e5` / `083693e` — pre-existing repo setup (not this session's work).

| # | Fix | Proof |
|---|---|---|
| 1 | `context.haystack_seed()` now derives from `(model, context_size, needle_position, strategy, experiment_mode)` via a stable sha256 hash (`burnt_toast/_hashing.py`), not the previous always-`42` default. | Live check: two haystacks built for the same model/context/position but different strategies now share only a 379-char accidental prefix (8 filler templates) before diverging; overall similarity ratio 0.103. |
| 2 | `MemoryTracker` sums RSS/VMS across all locally-visible `ollama`-named processes (server + per-model runner subprocesses), rediscovered every ~1s to catch a runner that loads after polling starts. System-wide fallback removed entirely. | Live check: `peak_rss_mb≈7-8GB` vs. `system_used_mb≈9-10GB` — a real, bounded number instead of one that tracks total machine memory; matches a manual `ps` cross-check of the same PIDs. |
| 3 | `generate_run_secret()` and `haystack_seed()` both delegate to a shared `stable_seed()` (sha256-based) instead of builtin `hash()`. | Test spawns two real subprocesses and confirms identical output — the only test shape that can catch this bug class, since `hash()` is internally consistent for a single process's whole lifetime. Manually confirmed `hash()` actually differs across processes here (3053442294 vs 911215964 for the same input). |
| 4 | `plot_results.split_valid_failed()` excludes crashed rows from plotted curves. **See correction below — the first version of this fix was itself buggy.** | See "A bug in the bug fix." |
| 5 | `runner.measure_prompt_overhead_tokens()` measures real wrapper-text token count per (model, mode), cached, instead of a hardcoded `120`. | Live measurement: needle mode = 121 tokens (the old `120` was a near-lucky guess), burnt-toast mode = 258 tokens (the old constant was off by >2x there). |
| 6/7 | `NoGuardStrategy.run()` collapsed; `tools._find_balanced_json_object()` (brace-depth scanner) replaces the regexes in both `parse_tool_call()` and `parse_final_answer()`. | Regression tests cover a tool call with nested `arguments`, and a model wrapping its answer like `{"result": {"secret_code": 4821}}` — both now parse correctly. |

### A bug in the bug fix (important — read this one)

The first version of fix #4 (`split_valid_failed()`) classified rows as "failed" purely
by whether `error_message` was non-empty. That's wrong: `error_message` is populated on
**two different paths**:

- A genuine crash (HTTP timeout, connection error) — caught in `run_single()`'s
  `except` block *before* the agent loop returns anything, so `total_iterations` stays
  at its dataclass default of `0` alongside the populated `error_message`.
- Burnt-toast mode's **No-Guard strategy reaching max iterations** — this is the
  *expected, designed* outcome (`"Max iterations (12) reached"`) with **fully valid
  telemetry** (`total_iterations=12`, real `tool_call_count`, real `accuracy`). It is
  the headline result the No-Guard condition exists to produce.

The original filter would have silently dropped all 30 No-Guard rows from every
burnt-toast plot. Caught by actually running the real post-fix burnt-toast matrix and
inspecting the output before trusting it — exactly the discipline the user asked for
("did you actually solve the problem"). Fixed in `4cf610e`: `split_valid_failed()` now
keys off `total_iterations > 0` (did the run produce telemetry at all), not
`error_message` presence. Has a dedicated regression test
(`test_max_iterations_reached_with_real_telemetry_is_valid_not_failed`).

**Lesson for the analyst**: a non-empty `error_message` does NOT always mean the row is
garbage. Check `total_iterations` — if it's `0`, the row is a crash with no usable data;
if it's `>0`, the row is real (even if `error_message` notes it hit the iteration cap).

## Validation infrastructure added this session

- `tests/` (46 unit tests total, `python -m unittest discover -s tests`, stdlib only,
  no Ollama needed): `test_hashing.py`, `test_context_seed.py`, `test_memory_tracker.py`,
  `test_secrets_seed.py` (includes the subprocess-based cross-process proof),
  `test_plot_results.py`, `test_overhead_measurement.py`, `test_tools_parsing.py`.
- `scripts/validate_pipeline.py` — 6-tier script: unit tests (always run) → Ollama
  reachability probe → live memory-tracker check → live haystack-divergence check →
  live overhead-measurement check → no-regression `--quick` smoke check (tiers 3-6
  skip gracefully if Ollama is unreachable). All tiers currently pass.
- `.claude/skills/pipeline-sanity-check/SKILL.md` — wraps the above as a reusable,
  general "does this harness measure what it claims to" skill (not a one-shot
  fix-verification script), invocable in future sessions.

## Post-fix data collection (this session, 2026-06-21)

Ran the full matrix (3 models × 5 context sizes × 2 needle positions × 3 strategies =
90 cells per mode) against the locally running Ollama (models: `qwen2.5:1.5b`,
`llama3.2:3b`, `llama3.1:8b`; `hardware_env=Laptop`).

**Needle mode** (`results_needle.csv`, 78/90 rows):
- 19 rows are genuine timeouts (`total_iterations=0`, HTTP `ReadTimeout` after 600s),
  concentrated exactly where you'd expect: `llama3.1:8b` from 4K context onward,
  `llama3.2:3b` at 8K/16K, even `qwen2.5:1.5b` hit 2 timeouts at 16K.
- The remaining 12 cells (`llama3.1:8b` × {8000, 16000} × {middle, end} × 3 strategies)
  were **never run** — killed deliberately once the timeout pattern through 4K was
  unambiguous (every `llama3.1:8b` cell at ≥4000 tokens had timed out), to avoid ~2 more
  hours confirming the obvious. This is a genuine, citable finding for the article
  ("this 8B model cannot complete a single ≥4K-token prompt evaluation within 600s on
  this hardware"), not a data gap to apologize for — just say so explicitly in the
  writeup.
- `results_needle.png` is the plotted dashboard with the 19 crash rows excluded (and
  the 12 never-run cells simply absent from the x-axis range for that model/strategy).

**Burnt-toast mode** (`results_burnt_toast.csv`, 90/90 rows, 0 errors):
- Completed in well under an hour, far faster than needle mode, because
  `build_burnt_toast_messages()` never actually injects the haystack into the prompt —
  `context_size` only sets `num_ctx` headroom, not real prompt length, so this mode is
  immune to the context-inflation timeout problem that stalled needle mode. Confirmed
  live before trusting it: runs took 5-90s regardless of context size.
- All 30 No-Guard rows correctly present with `total_iterations=12` (see "bug in the
  bug fix" above — this almost didn't make it into the plot).
- `results_burnt_toast.png` is the full, complete dashboard.

### Environmental quirk encountered (not a data-quality issue)

Early in the needle run, log timestamps showed a ~7-hour gap between one run starting
and finishing, even though that run's own internal `elapsed` timer showed only ~70s of
real work. This is almost certainly the sandbox/VM this session runs in being suspended
while idle between conversation turns, then resumed — not a benchmark bug. Wall-clock
ETAs given mid-run were unreliable for this reason; the actual recorded metrics
(`ttft_seconds`, `tokens_per_second`, etc.) are unaffected since they come from Ollama's
own `prompt_eval_duration`/`eval_duration`, not wall-clock deltas.

## Known residual limitations (documented, not fixed — out of scope)

- This Ollama install doesn't expose `/api/tokenize` (404), so `actual_context_tokens`
  falls back to a character-based estimate calibrated on a generic sample, while
  `effective_prompt_tokens` is Ollama's real token count. These can disagree by
  measurement-estimator noise alone, separate from any of the 5 bugs above — visible as
  `context_truncated=True` even at small context sizes in some early rows. Worth a
  closer look if `context_truncated`/`needle_in_window` columns matter to your analysis.
- `measure_prompt_overhead_tokens()` still undercounts by the chat-template
  special/role-marker tokens Ollama's `/api/chat` adds when actually rendering
  `messages` — no Ollama endpoint exposes a dry-run chat token count without a real
  generation call, so this residual is accepted and documented in code rather than
  chased further.
- `OLLAMA_TIMEOUT_SECONDS` is still the default 600s. Not changed this session since
  shortening it wouldn't have produced more data, only failed faster.

## Files in `results_post_fix_2026-06-21/`

| File | What it is |
|---|---|
| `results_needle.csv` | 78 rows, needle mode, post-fix. 19 rows are crashes (`total_iterations=0`). |
| `results_burnt_toast.csv` | 90 rows, burnt-toast mode, post-fix, 0 crashes. |
| `results_needle.png` | Dashboard, crash rows excluded from plotted curves. |
| `results_burnt_toast.png` | Dashboard, complete. |
| `needle_run.log` / `burnt_toast_run.log` | Full run logs (INFO level), useful for cross-checking any specific row's behavior (e.g. exact truncation warnings, guard-trigger timing). |

## Repo state

Working tree clean, all commits pushed to `origin/main` as of `4cf610e`. No outstanding
plan, no background processes running.
