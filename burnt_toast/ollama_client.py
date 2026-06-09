"""Ollama HTTP client with streaming TTFT measurement."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class OllamaResponse:
    """Aggregated metrics from one Ollama /api/chat call."""

    content: str
    ttft_seconds: float
    tokens_per_second: float | None
    prompt_eval_count: int
    eval_count: int
    prompt_eval_duration_ns: int
    eval_duration_ns: int
    total_duration_ns: int
    done_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


_CALIBRATION_SAMPLE = "The quick brown fox jumps over the lazy dog. " * 25


class OllamaClient:
    """Thin wrapper around Ollama's REST API."""

    def __init__(self, base_url: str, timeout: int = 600) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._tokenize_supported: bool | None = None
        self._chars_per_token: float = 4.0
        self._calibrated_model: str | None = None

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health_check(self) -> bool:
        try:
            resp = self._session.get(self._url("/api/tags"), timeout=10)
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            logger.error("Ollama health check failed: %s", exc)
            return False

    def list_models(self) -> list[str]:
        resp = self._session.get(self._url("/api/tags"), timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    @staticmethod
    def resolve_model(requested: str, available: list[str]) -> str | None:
        """Match a requested model name against locally pulled Ollama models."""
        if requested in available:
            return requested
        # Ollama may store names with or without an explicit tag.
        base = requested.split(":")[0]
        for name in available:
            if name == requested or name.startswith(f"{base}:") or name.split(":")[0] == base:
                return name
        return None

    def validate_models(self, requested: list[str]) -> list[str]:
        """
        Ensure all requested models are available locally.

        Returns the resolved model names to use for API calls.
        Raises RuntimeError with pull instructions when models are missing.
        """
        available = self.list_models()
        if not available:
            pulls = "\n  ".join(f"ollama pull {m}" for m in requested)
            raise RuntimeError(
                "No Ollama models are installed. Pull the required models first:\n  "
                f"{pulls}"
            )

        resolved: list[str] = []
        missing: list[str] = []
        for model in requested:
            match = self.resolve_model(model, available)
            if match is None:
                missing.append(model)
            else:
                resolved.append(match)

        if missing:
            pulls = "\n  ".join(f"ollama pull {m}" for m in missing)
            raise RuntimeError(
                f"Missing Ollama model(s): {', '.join(missing)}\n"
                f"Available: {', '.join(available) or '(none)'}\n"
                f"Pull them with:\n  {pulls}"
            )
        return resolved

    def _probe_tokenize_support(self, model: str) -> bool:
        if self._tokenize_supported is not None:
            return self._tokenize_supported
        try:
            resp = self._session.post(
                self._url("/api/tokenize"),
                json={"model": model, "prompt": "ping"},
                timeout=30,
            )
            if resp.status_code == 404:
                self._tokenize_supported = False
                logger.warning(
                    "Ollama /api/tokenize is unavailable (404). "
                    "Falling back to calibrated character-based token estimates."
                )
                return False
            resp.raise_for_status()
            self._tokenize_supported = True
            return True
        except requests.RequestException as exc:
            logger.warning("Tokenize probe failed (%s); using fallback estimates.", exc)
            self._tokenize_supported = False
            return False

    def _count_tokens_via_generate(self, model: str, text: str) -> int:
        """Accurate token count via a zero-generation /api/generate call."""
        resp = self._session.post(
            self._url("/api/generate"),
            json={
                "model": model,
                "prompt": text,
                "stream": False,
                "options": {"num_predict": 0},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        count = resp.json().get("prompt_eval_count", 0) or 0
        return int(count)

    def calibrate_token_estimator(self, model: str) -> None:
        """Calibrate chars-per-token ratio for fallback counting."""
        if self._calibrated_model == model:
            return

        if self._probe_tokenize_support(model):
            tokens = len(self.tokenize(model, _CALIBRATION_SAMPLE))
        else:
            tokens = self._count_tokens_via_generate(model, _CALIBRATION_SAMPLE)

        if tokens > 0:
            self._chars_per_token = len(_CALIBRATION_SAMPLE) / tokens
            logger.info(
                "Token estimator calibrated for %s: %.2f chars/token",
                model,
                self._chars_per_token,
            )
        self._calibrated_model = model

    def tokenize(self, model: str, text: str) -> list[int]:
        """Return token IDs for *text* using Ollama's tokenizer."""
        resp = self._session.post(
            self._url("/api/tokenize"),
            json={"model": model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json().get("tokens", [])

    def _estimate_tokens(self, text: str) -> int:
        return max(1, int(len(text) / self._chars_per_token))

    def count_tokens(self, model: str, text: str) -> int:
        """Count tokens, preferring /api/tokenize with calibrated fallbacks."""
        if not text:
            return 0

        self.calibrate_token_estimator(model)

        if self._tokenize_supported:
            try:
                return len(self.tokenize(model, text))
            except requests.RequestException as exc:
                logger.debug("tokenize failed (%s); using estimate", exc)

        return self._estimate_tokens(text)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        stream: bool = True,
        temperature: float = 0.1,
    ) -> OllamaResponse:
        """Send a chat completion request and collect timing metrics."""
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature},
        }

        if not stream:
            return self._chat_non_streaming(payload)

        return self._chat_streaming(payload)

    def _chat_non_streaming(self, payload: dict[str, Any]) -> OllamaResponse:
        start = time.perf_counter()
        resp = self._session.post(
            self._url("/api/chat"),
            json={**payload, "stream": False},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        message = data.get("message", {})
        content = message.get("content", "")

        prompt_eval_ns = data.get("prompt_eval_duration", 0) or 0
        eval_ns = data.get("eval_duration", 0) or 0
        eval_count = data.get("eval_count", 0) or 0

        ttft = prompt_eval_ns / 1e9 if prompt_eval_ns else (time.perf_counter() - start)
        tps = (eval_count / (eval_ns / 1e9)) if eval_ns and eval_count else None

        return OllamaResponse(
            content=content,
            ttft_seconds=ttft,
            tokens_per_second=tps,
            prompt_eval_count=data.get("prompt_eval_count", 0) or 0,
            eval_count=eval_count,
            prompt_eval_duration_ns=prompt_eval_ns,
            eval_duration_ns=eval_ns,
            total_duration_ns=data.get("total_duration", 0) or 0,
            done_reason=data.get("done_reason", ""),
            raw=data,
        )

    def _chat_streaming(self, payload: dict[str, Any]) -> OllamaResponse:
        start = time.perf_counter()
        first_token_time: float | None = None
        content_parts: list[str] = []
        final_chunk: dict[str, Any] = {}

        with self._session.post(
            self._url("/api/chat"),
            json=payload,
            timeout=self.timeout,
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("message", {}).get("content"):
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    content_parts.append(chunk["message"]["content"])
                if chunk.get("done"):
                    final_chunk = chunk

        content = "".join(content_parts)
        prompt_eval_ns = final_chunk.get("prompt_eval_duration", 0) or 0
        eval_ns = final_chunk.get("eval_duration", 0) or 0
        eval_count = final_chunk.get("eval_count", 0) or 0

        if prompt_eval_ns:
            ttft = prompt_eval_ns / 1e9
        elif first_token_time is not None:
            ttft = first_token_time - start
        else:
            ttft = time.perf_counter() - start

        tps = (eval_count / (eval_ns / 1e9)) if eval_ns and eval_count else None

        return OllamaResponse(
            content=content,
            ttft_seconds=ttft,
            tokens_per_second=tps,
            prompt_eval_count=final_chunk.get("prompt_eval_count", 0) or 0,
            eval_count=eval_count,
            prompt_eval_duration_ns=prompt_eval_ns,
            eval_duration_ns=eval_ns,
            total_duration_ns=final_chunk.get("total_duration", 0) or 0,
            done_reason=final_chunk.get("done_reason", ""),
            raw=final_chunk,
        )
