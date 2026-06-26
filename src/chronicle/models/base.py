"""The ModelClient protocol and its response type."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: Any = None


@runtime_checkable
class ModelClient(Protocol):
    """A backend that turns a prompt into text plus token/latency accounting.

    Both the generation backend (Gemini) and the judge backend (Ollama) satisfy
    this, so the rest of the system depends only on the protocol.
    """

    name: str

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...
