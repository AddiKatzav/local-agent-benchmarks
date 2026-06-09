"""Experiment configuration and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Hardware environment label (override for each deployment)
# ---------------------------------------------------------------------------
HARDWARE_ENV: str = "Laptop"

# ---------------------------------------------------------------------------
# Target Ollama models
# ---------------------------------------------------------------------------
MODELS: list[str] = [
    "qwen2.5:1.5b",
    "llama3.2:3b",
    "llama3.1:8b",
]

# Smaller / faster model used by the Critic strategy for loop detection
CRITIC_MODEL: str = "qwen2.5:1.5b"

# ---------------------------------------------------------------------------
# Experimental matrix
# ---------------------------------------------------------------------------
CONTEXT_SIZES_TOKENS: list[int] = [1_000, 2_000, 4_000, 8_000, 16_000]

NeedlePosition = Literal["middle", "end"]
StrategyName = Literal["No-Guard", "Python-Guard", "Critic"]

NEEDLE_POSITIONS: list[NeedlePosition] = ["middle", "end"]
STRATEGIES: list[StrategyName] = ["No-Guard", "Python-Guard", "Critic"]

# ---------------------------------------------------------------------------
# Needle & task constants
# ---------------------------------------------------------------------------
SECRET_PHRASE: str = "The secret agent code is 9482"
SECRET_CODE: int = 9482
EXPECTED_JSON: str = '{"secret_code": 9482}'

# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------
OLLAMA_BASE_URL: str = "http://localhost:11434"
OLLAMA_TIMEOUT_SECONDS: int = 600

# ---------------------------------------------------------------------------
# Agent loop limits
# ---------------------------------------------------------------------------
MAX_AGENT_ITERATIONS: int = 12
PYTHON_GUARD_MAX_IDENTICAL_CALLS: int = 3

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_CSV: Path = Path("burnt_toast_results.csv")

# Filler corpus path (auto-generated on first run if missing)
FILLER_CORPUS_PATH: Path = Path(__file__).parent / "data" / "filler_corpus.txt"


@dataclass
class RunConfig:
    """Single experimental run parameters."""

    model: str
    context_size_tokens: int
    needle_position: NeedlePosition
    strategy: StrategyName
    hardware_env: str = field(default_factory=lambda: HARDWARE_ENV)
    ollama_base_url: str = OLLAMA_BASE_URL
    max_iterations: int = MAX_AGENT_ITERATIONS
    run_index: int = 0
