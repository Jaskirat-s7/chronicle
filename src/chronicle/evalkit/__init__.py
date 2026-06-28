"""The evaluation harness: retrieval metrics, RAGAS, report, DeepEval gate.

Retrieval/answer metrics here are deterministic and fully unit-tested. RAGAS and
DeepEval are isolated behind thin adapters (lazy-imported) because they require
the live judge + extra deps.
"""

from .metrics import (
    answer_matches,
    hit_at_k,
    mrr,
    ndcg_at_k,
    relevance_flags,
)

__all__ = [
    "answer_matches",
    "hit_at_k",
    "mrr",
    "ndcg_at_k",
    "relevance_flags",
]
