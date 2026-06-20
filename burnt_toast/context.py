"""Haystack context generation with precise token-length padding."""

from __future__ import annotations

import logging
import random
import textwrap
from pathlib import Path

from burnt_toast._hashing import stable_seed
from burnt_toast.config import FILLER_CORPUS_PATH
from burnt_toast.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Neutral news-style paragraphs used when no corpus file exists.
_DEFAULT_PARAGRAPHS: list[str] = [
    textwrap.dedent(
        """\
        Regional transit authorities announced a phased upgrade to signaling infrastructure
        across three metropolitan corridors. The project, budgeted at four hundred twelve
        million dollars, will replace legacy relay systems with digital interlocking modules.
        Officials expect the first segment to enter service in late autumn, pending safety
        certification from the national rail oversight board."""
    ),
    textwrap.dedent(
        """\
        A consortium of universities published findings on seasonal crop yield variability
        linked to microclimate shifts in the upper Midwest. Researchers combined satellite
        imagery with ground sensors deployed across twelve counties. The study noted that
        irrigation timing adjustments could offset projected declines in certain legume
        varieties without expanding cultivated acreage."""
    ),
    textwrap.dedent(
        """\
        Municipal water departments reported stable reservoir levels following an unusually
        dry spring. Conservation messaging reduced per-capita consumption by six percent
        compared with the prior five-year average. Engineers continue to monitor aquifer
        recharge rates and have scheduled public briefings for the next fiscal quarter."""
    ),
    textwrap.dedent(
        """\
        The national statistics office released quarterly employment figures showing modest
        growth in professional services and a slight contraction in retail trade. Analysts
        attributed the divergence to shifting consumer spending patterns and accelerated
        adoption of remote collaboration tools among small and medium enterprises."""
    ),
    textwrap.dedent(
        """\
        An international standards body ratified updated guidelines for electromagnetic
        compatibility testing in consumer electronics. Manufacturers have eighteen months
        to align certification procedures. Industry groups welcomed the clarity while noting
        supply chain lead times may complicate compliance for entry-level product lines."""
    ),
    textwrap.dedent(
        """\
        Coastal communities participated in a coordinated drill evaluating tsunami alert
        distribution channels. Participants included schools, hospitals, and harbor
        operators. Evaluators recorded average evacuation notification latency under four
        minutes in densely populated districts, an improvement over the previous exercise."""
    ),
    textwrap.dedent(
        """\
        Independent auditors reviewed procurement records for a multi-year highway
        resurfacing program. The report highlighted transparent bid evaluation but
        recommended stronger documentation for change orders. Transportation officials
        pledged to adopt the recommendations before the next contract renewal cycle."""
    ),
    textwrap.dedent(
        """\
        A public library network expanded digital lending capacity through a shared
        repository of open educational resources. Patrons accessed over two million
        document retrievals last quarter. Administrators plan to pilot assistive
        technology kiosks at four branch locations beginning next month."""
    ),
]


def ensure_filler_corpus(path: Path = FILLER_CORPUS_PATH) -> Path:
    """Create the filler corpus file if it does not exist."""
    if path.exists():
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    paragraphs: list[str] = []
    rng = random.Random(42)
    for _ in range(200):
        base = rng.choice(_DEFAULT_PARAGRAPHS)
        paragraphs.append(base)
    path.write_text("\n\n".join(paragraphs), encoding="utf-8")
    logger.info("Generated filler corpus at %s", path)
    return path


def load_filler_sentences(path: Path = FILLER_CORPUS_PATH) -> list[str]:
    ensure_filler_corpus(path)
    text = path.read_text(encoding="utf-8")
    sentences: list[str] = []
    for paragraph in text.split("\n\n"):
        paragraph = paragraph.strip()
        if paragraph:
            sentences.append(paragraph)
    return sentences


def _build_filler_block(sentences: list[str], rng: random.Random) -> str:
    shuffled = sentences.copy()
    rng.shuffle(shuffled)
    return "\n\n".join(shuffled)


def _insert_needle(filler: str, needle: str, position: str) -> str:
    if position == "end":
        return f"{filler}\n\n{needle}"
    midpoint = len(filler) // 2
    return f"{filler[:midpoint]}\n\n{needle}\n\n{filler[midpoint:]}"


def _trim_to_token_budget(
    client: OllamaClient,
    model: str,
    text: str,
    target_tokens: int,
) -> str:
    """Binary-search trim filler so total text is at most *target_tokens*."""
    tokens = client.count_tokens(model, text)
    if tokens <= target_tokens:
        return text

    # Trim by character fraction proportional to overshoot
    low, high = 0, len(text)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = text[:mid]
        count = client.count_tokens(model, candidate)
        if count <= target_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _pad_to_token_budget(
    client: OllamaClient,
    model: str,
    base_text: str,
    sentences: list[str],
    target_tokens: int,
    rng: random.Random,
) -> str:
    """Pad *base_text* with filler until token count reaches *target_tokens*."""
    current = base_text
    tokens = client.count_tokens(model, current)
    attempts = 0
    max_attempts = 500

    while tokens < target_tokens and attempts < max_attempts:
        extra = _build_filler_block(sentences, rng)
        current = f"{current}\n\n{extra}"
        tokens = client.count_tokens(model, current)
        attempts += 1

    if tokens > target_tokens:
        current = _trim_to_token_budget(client, model, current, target_tokens)

    final_tokens = client.count_tokens(model, current)
    logger.debug(
        "Context built: target=%d actual=%d chars=%d",
        target_tokens,
        final_tokens,
        len(current),
    )
    return current


def haystack_seed(
    model: str,
    target_tokens: int,
    needle_position: str,
    strategy: str,
    experiment_mode: str,
) -> int:
    """Deterministic, reproducible filler-text seed for one matrix cell.

    Uses sha256 (not builtin hash(), which is randomized per process) so that
    rerunning the same matrix cell — even in a different process, days later —
    builds byte-identical haystack text (desirable for comparing runs), while
    two DIFFERENT cells (different model/context/position/strategy/mode) get
    independent filler text. This independence is what prevents Ollama's
    KV-cache prefix reuse from corrupting cross-cell TTFT/throughput and the
    context_truncated/needle_in_window heuristics.
    """
    return stable_seed(model, target_tokens, needle_position, strategy, experiment_mode)


def build_haystack(
    client: OllamaClient,
    model: str,
    target_tokens: int,
    needle_position: str,
    *,
    secret_phrase: str,
    seed: int = 42,
) -> tuple[str, int]:
    """
    Build a haystack context padded to *target_tokens* with the secret needle inserted.

    Returns (context_text, actual_token_count).
    """
    sentences = load_filler_sentences()
    rng = random.Random(seed + target_tokens + (0 if needle_position == "middle" else 1))

    # Reserve tokens for the needle by building filler for (target - needle_tokens)
    needle_tokens = client.count_tokens(model, secret_phrase)
    filler_budget = max(target_tokens - needle_tokens - 4, 100)

    filler = _pad_to_token_budget(
        client, model, "", sentences, filler_budget, rng
    )
    context = _insert_needle(filler, secret_phrase, needle_position)
    actual = client.count_tokens(model, context)

    # Fine-tune: if we're over budget, trim filler portion only
    if actual > target_tokens:
        # Rebuild with adjusted filler budget
        overshoot = actual - target_tokens
        filler_budget = max(filler_budget - overshoot - 10, 50)
        filler = _pad_to_token_budget(
            client, model, "", sentences, filler_budget, rng
        )
        context = _insert_needle(filler, secret_phrase, needle_position)
        actual = client.count_tokens(model, context)

    if actual < target_tokens - 50:
        context = _pad_to_token_budget(
            client, model, context, sentences, target_tokens, rng
        )
        actual = client.count_tokens(model, context)

    return context, actual
