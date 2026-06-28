"""Embeddings: Qwen3-Embedding-0.6B via sentence-transformers (instruction-aware).

Lazy-imported so the package and offline tests don't require torch. Output dim is
the model's native 1024 — matching the LOCKED pgvector column. No Matryoshka
truncation in the baseline.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings, get_settings

# Qwen3-Embedding wants an instruction on the *query* side only.
_QUERY_INSTRUCTION = (
    "Instruct: Given a question about a code repository's history, "
    "retrieve the most relevant code or commit context.\nQuery: "
)


class Embedder(Protocol):
    dim: int

    def encode_query(self, text: str) -> list[float]: ...
    def encode_documents(self, texts: list[str]) -> list[list[float]]: ...


def _pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.dim = dim
        self.device = device or _pick_device()
        self._model = SentenceTransformer(model_name, device=self.device)

    def encode_query(self, text: str) -> list[float]:
        vec = self._model.encode(
            _QUERY_INSTRUCTION + text,
            normalize_embeddings=True,
        )
        return vec.tolist()

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return [v.tolist() for v in vecs]


def get_embedder(settings: Settings | None = None) -> SentenceTransformerEmbedder:
    s = settings or get_settings()
    return SentenceTransformerEmbedder(s.embedding_model, s.embedding_dim)
