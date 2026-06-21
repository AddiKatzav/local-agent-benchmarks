# Local Agent Benchmarks — Burnt Toast + Needle-in-Haystack

A reproducible benchmark suite for evaluating how **local LLM agents** behave under **context inflation** when combined with two stressors:

1. **Needle-in-a-Haystack** — a secret phrase is hidden inside a large neutral document.
2. **Agent Burnt Toast Effect** — a faulty tool always returns empty results, tempting the agent into repetitive retry loops.

The goal is to compare loop-guard strategies and measure how accuracy, latency, throughput, and memory scale as context grows — on real hardware (laptop, Raspberry Pi, etc.) using Ollama.

---

## Goal

Understand whether lightweight **loop-protection mechanisms** can prevent agents from wasting compute when tools fail silently, especially as prompt context grows from 1K to 16K tokens.

Specifically, we want to answer:

- Does **context inflation** make needle retrieval harder, and does needle **position** (middle vs. end) matter?
- Does the **burnt toast** failure mode (broken `search_context` tool) cause agents to loop instead of reading the context directly?
- Which guard strategy — **none**, **deterministic (Python-Guard)**, or **LLM-based (Critic)** — best balances **accuracy**, **latency**, and **resource use**?
- How do these trade-offs differ across **model sizes** and **hardware environments**?

---

## Method

### Task

The agent must locate a hidden secret phrase and respond with strict JSON:

```json
{"secret_code": <integer>}
```

Each run generates a **unique 4-digit code** embedded as `The secret agent code is <code>`. Prompts and nudges show only the JSON schema — never the concrete answer — so guessing `9482` or memorizing a fixed value cannot inflate accuracy.

The suite runs as **two separate experiment arms** (select with `--mode`):

| Mode | Purpose | Agent sees | Tool |
|---|---|---|---|
| **`needle`** | Context inflation / retrieval | Full haystack in prompt | Disabled (nudged away if attempted) |
| **`burnt-toast`** | Loop + guard stress test | External index only (haystack hidden) | **Mandatory** broken `search_context` (always empty) |

In `burnt-toast` mode the agent **must** call `search_context` before answering. The tool always returns an empty result set, simulating the "burnt toast" failure mode.

### Experimental Matrix

Each run varies one combination of the following variables:

| Variable | Values |
|---|---|
| **Mode** | `needle`, `burnt-toast` |
| **Models** | `qwen2.5:1.5b`, `llama3.2:3b`, `llama3.1:8b` |
| **Context size** | 1K, 2K, 4K, 8K, 16K tokens |
| **Needle position** | `middle`, `end` |
| **Guard strategy** | `No-Guard`, `Python-Guard`, `Critic` |
| **Hardware label** | e.g. `Laptop`, `Raspberry_Pi_5` (metadata only) |

**Full matrix:** 3 × 5 × 2 × 3 = **90 runs** per mode per hardware environment.

Ollama `num_ctx` is set to `context_size + 1024` headroom so 8K/16K runs are not silently truncated at 4096 tokens.

### Guard Strategies

| Strategy | Mechanism |
|---|---|
| **No-Guard** | Pure agent loop. No protection against repetitive tool calls. |
| **Python-Guard** | Deterministic SHA-256 fingerprinting of tool calls. After **3 identical calls**, execution is halted and a system error is injected, forcing a final answer. |
| **Critic** | A secondary fast model (`qwen2.5:1.5b`) reviews the recent transcript and flags logical looping. On detection, the agent is interrupted and prompted to answer directly. |

### Context Construction

1. Neutral news-style filler text is generated or loaded from `burnt_toast/data/filler_corpus.txt`.
2. The haystack is padded to the target token count (via Ollama tokenize API or a calibrated fallback).
3. The secret phrase is inserted at the **middle** or **end** of the document.

### Procedure

1. Build haystack for the target context size and needle position.
2. Run the agent loop (up to 12 iterations) with the selected guard strategy.
3. Record metrics after each run; append incrementally to CSV.
4. Plot results with `plot_results.py`.

---

## Tools

| Tool | Purpose |
|---|---|
| **[Ollama](https://ollama.com/)** | Local LLM inference server (`/api/chat`, `/api/generate`) |
| **`burnt_toast/`** | Python benchmark package — context builder, agent strategies, metrics |
| **`run_benchmark.py`** | Convenience launcher for the CLI |
| **`plot_results.py`** | CSV → PNG dashboard generator |
| **`psutil`** | Peak RAM tracking during inference |
| **`pandas` / `matplotlib`** | Results analysis and plotting |

### Test Environment — `Laptop` (`results_post_fix_2026-06-21/`)

Concrete specs for the hardware behind the `hardware_env=Laptop` label, for anyone (e.g.
an article reader) trying to interpret or reproduce the absolute latency/memory numbers
in `results_post_fix_2026-06-21/`. The `hardware_env` CSV column is metadata only — it
doesn't change runtime behavior — so this spec lives in docs rather than the data.

| Component | Spec |
|---|---|
| **Machine** | Dell Latitude 5421 laptop |
| **CPU** | Intel Core i7-11850H @ 2.50GHz, 8 cores / 16 threads |
| **RAM** | 16 GiB total (~12 GiB free at idle) |
| **GPU** | NVIDIA GeForce MX450, 2 GB VRAM |
| **OS** | Ubuntu 24.04.3 LTS under WSL2 (kernel `6.6.87.2-microsoft-standard-WSL2`) |
| **Ollama** | v0.24.0 |
| **Python** | 3.12.3 |

**GPU offload was not confirmed for these runs.** The MX450's 2 GB VRAM is small
relative to even the smallest model here (`qwen2.5:1.5b`), and no per-run GPU/CPU
processor log was retained, so whether Ollama placed any layers on GPU vs. ran fully on
CPU for a given model/context size is unknown. Treat all latency/throughput numbers as
**CPU-comparable, GPU-uncertain** — don't assume this was a GPU-accelerated run. If you
rerun this matrix, capture `ollama ps` (its `PROCESSOR` column reports `100% GPU`,
`100% CPU`, or a split) per model immediately after each run to settle this.

### Project Layout

```
local-agent-benchmarks/
├── burnt_toast/
│   ├── config.py          # models, matrix, constants
│   ├── context.py           # haystack + needle insertion
│   ├── ollama_client.py     # Ollama API + TTFT/TPS
│   ├── strategies.py        # No-Guard / Python-Guard / Critic
│   ├── tools.py             # faulty search tool + parsers
│   ├── metrics.py           # memory tracker + CSV writer
│   └── runner.py            # experiment orchestration
├── run_benchmark.py
├── plot_results.py
└── requirements.txt
```

### Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:1.5b
ollama pull llama3.2:3b
ollama pull llama3.1:8b

# Smoke test (1 run, needle mode)
python -m burnt_toast --quick

# Needle retrieval arm (context inflation test)
python -m burnt_toast --mode needle --models qwen2.5:1.5b --results results_qwen.csv

# Burnt-toast loop arm (guard stress test)
python -m burnt_toast --mode burnt-toast --models qwen2.5:1.5b --results results_qwen_bt.csv

# Full 90-run matrix
python -m burnt_toast --mode needle --hardware-env Laptop

# Plot results (use the suffixed filename printed at end of run)
python plot_results.py results_qwen_a3f2b_06_09_21_30.csv
```

Each benchmark run writes to a **uniquely suffixed CSV** by default:

`results_qwen.csv` → `results_qwen_a3f2b_06_09_21_30.csv`

Use `--no-unique-suffix` to write to the exact path given.

---

## Metrics (KPIs)

Every run records the following to CSV (`burnt_toast_results.csv` or a custom path):

| KPI | Column | Description |
|---|---|---|
| **Experiment mode** | `experiment_mode` | `needle` or `burnt-toast` |
| **Accuracy** | `accuracy` | `true` when extracted `secret_code` matches `expected_secret_code` for that run |
| **JSON validity** | `json_valid` | Output parsed as valid JSON with a `secret_code` field |
| **TTFT** | `ttft_seconds` | Time to first token (from Ollama `prompt_eval_duration` or stream delta) |
| **Throughput** | `tokens_per_second` | `eval_count / eval_duration` from Ollama |
| **Peak memory** | `peak_rss_mb` | Peak resident set size during the inference window |
| **Effective prompt** | `effective_prompt_tokens` | Prompt tokens on turn 1 (what the model actually evaluated) |
| **Context truncated** | `context_truncated` | `true` when effective prompt ≪ built haystack size |
| **Needle visible** | `needle_in_window` | Estimated whether the needle was inside the evaluated window |
| **Ollama num_ctx** | `num_ctx` | Context window size passed to Ollama for this run |
| **Tool calls** | `tool_call_count` | Number of `search_context` invocations |
| **Iterations** | `total_iterations` | Agent loop turns taken |
| **Guard triggered** | `guard_triggered` | Whether Python-Guard or Critic intervened |
| **Expected code** | `expected_secret_code` | Ground-truth code generated for this run (not shown to the model) |
| **Extracted code** | `extracted_secret_code` | Code parsed from model output |
| **Prompt tokens** | `prompt_tokens` | Total prompt tokens across all turns |
| **Completion tokens** | `completion_tokens` | Tokens generated in responses |
| **Context size** | `context_size_tokens` | Target haystack size |
| **Actual context** | `actual_context_tokens` | Measured haystack size after padding |

Metadata columns: `timestamp`, `run_id`, `hardware_env`, `model`, `needle_position`, `strategy`, `error_message`.

---

## Expected Behaviors (Hypotheses)

> These are **pre-registered predictions** — written before analyzing results. Use them as a baseline when interpreting CSV output and plots.

### Burnt toast effect (`--mode burnt-toast`)

- Agents **cannot** see the haystack directly; they must call `search_context`.
- JSON answers without a prior tool call are **rejected**.
- Under **No-Guard**, expect **`tool_call_count` ≥ 3** and rising iterations on many models.
- **Python-Guard** should show `guard_triggered=true` after 3 identical tool calls.
- **Critic** should show `guard_triggered=true` when the critic detects looping.
- Accuracy may be **low** in this arm (tool never returns the needle) — primary KPIs are `tool_call_count`, `guard_triggered`, and `total_iterations`.

### Context inflation (`--mode needle`)

- Full haystack is in the prompt; tools are disabled.
- **TTFT and peak RSS should increase** as context grows from 1K → 16K.
- With `num_ctx` fix, `context_truncated` should be `false` and `effective_prompt_tokens` ≈ `actual_context_tokens`.
- **Accuracy should degrade** at higher context when `needle_in_window=false`.
- Needle at **end** may fail more than **middle** when truncation occurs (tail dropped first).

### Guard strategies

| Strategy | Expected behavior |
|---|---|
| **No-Guard** | Highest iteration count when looping occurs. No `guard_triggered`. May waste the most tokens and time. |
| **Python-Guard** | `guard_triggered=true` after 3 identical tool calls. Should reduce wasted iterations but may force a premature or incorrect final answer if the guard fires before the agent reads the context. |
| **Critic** | `guard_triggered=true` when the critic detects looping. More adaptive than Python-Guard but adds an extra LLM call per check — may increase latency on long runs. |

### Model size

- **Smaller models** (`qwen2.5:1.5b`) should be faster (higher TPS, lower TTFT) but **less accurate** at 8K–16K contexts.
- **Larger models** (`llama3.1:8b`) should be more accurate at high context but **slower** and **more memory-hungry**.
- Mid-size (`llama3.2:3b`) may offer the best accuracy-to-latency trade-off on constrained hardware.

### Hardware environment

- The `hardware_env` label does not change runtime behavior — it tags results for cross-device comparison.
- On resource-constrained devices (e.g. Raspberry Pi 5), expect **lower TPS**, **higher TTFT**, and possible **memory pressure** (`peak_rss_mb` spikes) at 8K+ contexts.

### Success criteria for a "good" guard

A guard strategy is considered effective if it:

1. **Maintains or improves accuracy** vs. No-Guard at the same context size.
2. **Reduces iterations and completion tokens** when the burnt toast loop would otherwise occur.
3. **Does not add disproportionate latency** (TTFT/TPS) compared to the tokens saved.

---

## Analyzing Results

After running the benchmark:

```bash
python plot_results.py results_qwen.csv
```

The generated PNG contains six panels: accuracy, TTFT, tokens/sec, peak RSS, iterations, and completion tokens — each plotted against context size, with lines for strategy (color) and needle position (line style).

Compare observed curves against the hypotheses above. Pay special attention to:

- Accuracy cliffs at 8K or 16K
- Divergence between solid (middle) and dashed (end) lines
- Whether `guard_triggered` correlates with accuracy gains or losses
- Whether No-Guard shows runaway `completion_tokens` without accuracy benefit

---

## License

Internal research / benchmarking project. Adjust as needed for your deployment.
