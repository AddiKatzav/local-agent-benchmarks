"""Per-run secret code generation (never leaked to prompts/nudges)."""

from __future__ import annotations

import random
from dataclasses import dataclass

# Shown to the model in system/user prompts and nudges — schema only, no answer.
JSON_RESPONSE_SCHEMA: str = '{"secret_code": <integer>}'

NEEDLE_PHRASE_TEMPLATE: str = "The secret agent code is {code}"


@dataclass(frozen=True)
class RunSecret:
    """Secret needle for a single benchmark run."""

    code: int
    phrase: str

    @property
    def json_schema(self) -> str:
        return JSON_RESPONSE_SCHEMA


def generate_run_secret(
    *,
    run_index: int,
    model: str,
    context_size_tokens: int,
    needle_position: str,
    strategy: str,
    experiment_mode: str,
    master_seed: int = 42,
) -> RunSecret:
    """
    Derive a deterministic 4-digit secret for this matrix cell.

    Re-running the same configuration reproduces the same code; each run index
    within the matrix gets a distinct value.
    """
    key = (
        master_seed,
        run_index,
        model,
        context_size_tokens,
        needle_position,
        strategy,
        experiment_mode,
    )
    rng = random.Random(hash(key) & 0xFFFFFFFF)
    code = rng.randint(1000, 9999)
    phrase = NEEDLE_PHRASE_TEMPLATE.format(code=code)
    return RunSecret(code=code, phrase=phrase)
