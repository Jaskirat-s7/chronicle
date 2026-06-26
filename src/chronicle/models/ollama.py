"""Ollama backend — the eval judge, running locally on the Mac.

Talks to the Ollama REST API over httpx. The base URL is always taken from
config (`OLLAMA_BASE_URL`, default http://localhost:11434) — never hardcoded.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from ..tracing import trace_generation
from .base import LLMResponse


class OllamaClient:
    name = "ollama"

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Injectable client so tests can supply an httpx.MockTransport.
        self._client = client or httpx.Client(base_url=self.base_url, timeout=timeout)

    def tags(self) -> dict[str, Any]:
        """GET /api/tags — used as the local connectivity check."""
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json()

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }

        with trace_generation(
            name="ollama.complete",
            model=self.model,
            input=messages,
            metadata={"backend": "ollama"},
        ) as handle:
            start = time.perf_counter()
            resp = self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
            latency_ms = (time.perf_counter() - start) * 1000.0

            text = data["message"]["content"]
            input_tokens = int(data.get("prompt_eval_count", 0) or 0)
            output_tokens = int(data.get("eval_count", 0) or 0)
            handle.record(
                output=text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return LLMResponse(
            text=text,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            raw=data,
        )
