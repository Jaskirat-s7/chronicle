"""Hybrid retrieval: dense (pgvector) + lexical (Postgres FTS), fused with RRF,
then reranked by a local cross-encoder.

`rrf_fuse` is a pure function so the fusion logic — the defensible core — is
unit-tested independently of Postgres or any model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Hashable, Protocol, Sequence

DEFAULT_RRF_K = 60


def rrf_fuse(
    rankings: Sequence[Sequence[Hashable]],
    *,
    k: int = DEFAULT_RRF_K,
) -> list[tuple[Hashable, float]]:
    """Reciprocal Rank Fusion.

    Each input is a ranked list of ids (best first). Score(id) = sum over lists
    of 1 / (k + rank), rank being 1-based. Returns ids sorted by descending
    fused score; ties broken by first appearance for determinism.
    """
    scores: dict[Hashable, float] = {}
    first_seen: dict[Hashable, int] = {}
    order = 0
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


@dataclass
class RetrievedChunk:
    id: int
    repo: str
    file_path: str
    line_start: int
    line_end: int
    commit_sha: str
    commit_date: datetime
    content: str
    score: float


class _Store(Protocol):
    def vector_search(
        self, embedding: list[float], limit: int, repo: str | None = ...
    ) -> list[dict]: ...
    def fts_search(
        self, query: str, limit: int, repo: str | None = ...
    ) -> list[dict]: ...


class _Embedder(Protocol):
    def encode_query(self, text: str) -> list[float]: ...


class _Reranker(Protocol):
    def rerank(self, query: str, docs: list[str]) -> list[float]: ...


class HybridRetriever:
    def __init__(
        self,
        store: _Store,
        embedder: _Embedder,
        reranker: _Reranker | None = None,
        *,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        *,
        repo: str | None = None,
        top_k: int = 8,
        candidate_k: int = 50,
        rerank: bool = True,
    ) -> list[RetrievedChunk]:
        embedding = self.embedder.encode_query(query)
        dense = self.store.vector_search(embedding, candidate_k, repo)
        lexical = self.store.fts_search(query, candidate_k, repo)

        by_id: dict[int, dict] = {}
        for row in dense:
            by_id[row["id"]] = row
        for row in lexical:
            by_id.setdefault(row["id"], row)

        fused = rrf_fuse(
            [[r["id"] for r in dense], [r["id"] for r in lexical]],
            k=self.rrf_k,
        )

        # Take the fused top candidates into the (expensive) reranker.
        fused_ids = [cid for cid, _ in fused][: max(candidate_k, top_k)]
        candidates = [by_id[cid] for cid in fused_ids]

        if rerank and self.reranker is not None and candidates:
            scores = self.reranker.rerank(query, [c["content"] for c in candidates])
            ranked = sorted(zip(candidates, scores), key=lambda cs: -cs[1])
            candidates = [c for c, _ in ranked]
            final_scores = [s for _, s in ranked]
        else:
            fused_score = {cid: s for cid, s in fused}
            final_scores = [fused_score[c["id"]] for c in candidates]

        out: list[RetrievedChunk] = []
        for row, score in zip(candidates[:top_k], final_scores[:top_k]):
            out.append(
                RetrievedChunk(
                    id=row["id"],
                    repo=row["repo"],
                    file_path=row["file_path"],
                    line_start=row["line_start"],
                    line_end=row["line_end"],
                    commit_sha=row["commit_sha"],
                    commit_date=row["commit_date"],
                    content=row["content"],
                    score=float(score),
                )
            )
        return out
