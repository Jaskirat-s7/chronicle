"""Gemini backend — generation (and later the agentic grade/decide steps).

Uses the `google-genai` SDK, imported lazily so the package imports without it.
Free tier; key from https://aistudio.google.com/apikey.
"""

from __future__ import annotations

import time

from ..tracing import trace_generation
from .base import LLMResponse


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str | None, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self._client = None  # built lazily on first call

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set; cannot use the Gemini generation backend."
            )
        from google import genai  # lazy import

        self._client = genai.Client(api_key=self.api_key)
        return self._client

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        from google.genai import types  # lazy import

        client = self._ensure_client()
        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        with trace_generation(
            name="gemini.complete",
            model=self.model,
            input=prompt,
            metadata={"backend": "gemini"},
        ) as handle:
            start = time.perf_counter()
            resp = client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
            latency_ms = (time.perf_counter() - start) * 1000.0

            text = resp.text or ""
            usage = getattr(resp, "usage_metadata", None)
            input_tokens = int(getattr(usage, "prompt_token_count", 0) or 0)
            output_tokens = int(getattr(usage, "candidates_token_count", 0) or 0)
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
            raw=resp,
        )
