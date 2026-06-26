"""Pluggable LLM access: a single ModelClient protocol, two backends.

- Generation (and later the agentic grade/decide steps) -> Gemini free tier.
- Eval judge -> local Ollama, keeping judge volume off the scarce Gemini quota.
"""

from __future__ import annotations

from ..config import Settings, get_settings
from .base import LLMResponse, ModelClient
from .gemini import GeminiClient
from .ollama import OllamaClient

__all__ = [
    "LLMResponse",
    "ModelClient",
    "GeminiClient",
    "OllamaClient",
    "get_generation_client",
    "get_judge_client",
]


def get_generation_client(settings: Settings | None = None) -> GeminiClient:
    s = settings or get_settings()
    return GeminiClient(api_key=s.google_api_key, model=s.gen_model)


def get_judge_client(settings: Settings | None = None) -> OllamaClient:
    s = settings or get_settings()
    return OllamaClient(base_url=s.ollama_base_url, model=s.judge_model)
