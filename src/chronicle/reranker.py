"""Local cross-encoder reranker: bge-reranker-v2-m3.

Non-negotiable for code retrieval. Lazy-imported (sentence-transformers).
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings, get_settings


class Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class BgeReranker:
    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        from .embeddings import _pick_device

        self.model_name = model_name
        self.device = device or _pick_device()
        self._model = CrossEncoder(model_name, device=self.device)

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        scores = self._model.predict([(query, d) for d in docs])
        return [float(s) for s in scores]


def get_reranker(settings: Settings | None = None) -> BgeReranker:
    s = settings or get_settings()
    return BgeReranker(s.reranker_model)
