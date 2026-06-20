"""Shared stable-hash primitive for deterministic, cross-process seeds."""

from __future__ import annotations

import hashlib


def stable_seed(*parts: str | int, mod: int = 2**32) -> int:
    """Deterministic seed from arbitrary fields, stable across processes.

    Uses sha256 rather than builtin hash(), which is randomized per process
    via PYTHONHASHSEED unless explicitly fixed -- callers that need the same
    inputs to reproduce the same seed across separate process invocations
    (not just within one process) must not use hash() for this.
    """
    key = "\x1f".join(str(p) for p in parts)
    digest = hashlib.sha256(key.encode()).hexdigest()
    return int(digest, 16) % mod
