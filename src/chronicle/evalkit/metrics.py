"""Deterministic retrieval + answer metrics (no judge, no model).

Relevance is defined against the git-derived ground truth: a retrieved chunk is
relevant to a question if it comes from the gold file and (when the question
targets a line) covers that line. Answer correctness is a structured check by
answer kind — e.g. the gold SHA appearing in the answer text.
"""

from __future__ import annotations

import math
import re
from typing import Any, Sequence


def relevance_flags(
    retrieved: Sequence[dict[str, Any]],
    evidence: dict[str, Any],
) -> list[bool]:
    """Per-retrieved-chunk relevance, in ranked order.

    `retrieved` items must have keys: file_path, line_start, line_end.
    `evidence` may contain `path` and optionally `line`.
    """
    gold_path = evidence.get("path")
    gold_line = evidence.get("line")
    flags: list[bool] = []
    for chunk in retrieved:
        ok = gold_path is not None and chunk["file_path"] == gold_path
        if ok and gold_line is not None:
            ok = chunk["line_start"] <= gold_line <= chunk["line_end"]
        flags.append(bool(ok))
    return flags


def hit_at_k(flags: Sequence[bool], k: int) -> float:
    """1.0 if any relevant chunk appears in the top-k, else 0.0."""
    return 1.0 if any(flags[:k]) else 0.0


def mrr(flags: Sequence[bool]) -> float:
    """Reciprocal rank of the first relevant chunk (0 if none)."""
    for i, f in enumerate(flags, start=1):
        if f:
            return 1.0 / i
    return 0.0


def ndcg_at_k(flags: Sequence[bool], k: int) -> float:
    """Binary-relevance nDCG@k. IDCG uses the relevant count present (capped)."""
    topk = list(flags[:k])
    dcg = sum(1.0 / math.log2(i + 1) for i, f in enumerate(topk, start=1) if f)
    num_rel = min(sum(1 for f in flags if f), k)
    if num_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, num_rel + 1))
    return dcg / idcg


def _shorts(sha: str) -> set[str]:
    return {sha} | {sha[:n] for n in (7, 8, 10, 12)}


def answer_matches(answer_kind: str, gold: str, answer_text: str) -> bool:
    """Structured correctness check for a generated answer against the gold label."""
    text = answer_text or ""
    if answer_kind == "commit_sha":
        low = text.lower()
        return any(s.lower() in low for s in _shorts(gold))
    if answer_kind == "date":
        return gold in text
    if answer_kind == "author":
        return gold.lower() in text.lower()
    if answer_kind == "pr_number":
        return re.search(rf"#?\b{re.escape(gold)}\b", text) is not None
    raise ValueError(f"unknown answer_kind: {answer_kind!r}")
