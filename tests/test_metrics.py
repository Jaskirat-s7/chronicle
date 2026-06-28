"""Deterministic retrieval + answer metrics."""

import math

import pytest

from chronicle.evalkit import (
    answer_matches,
    hit_at_k,
    mrr,
    ndcg_at_k,
    relevance_flags,
)


def _r(path, ls, le):
    return {"file_path": path, "line_start": ls, "line_end": le}


def test_relevance_by_file_and_line():
    retrieved = [_r("a.py", 1, 50), _r("b.py", 1, 50), _r("a.py", 51, 100)]
    flags = relevance_flags(retrieved, {"path": "a.py", "line": 60})
    assert flags == [False, False, True]


def test_relevance_file_only_when_no_line():
    retrieved = [_r("a.py", 1, 50), _r("b.py", 1, 50)]
    assert relevance_flags(retrieved, {"path": "a.py"}) == [True, False]


def test_hit_and_mrr():
    flags = [False, False, True, False]
    assert hit_at_k(flags, 2) == 0.0
    assert hit_at_k(flags, 3) == 1.0
    assert mrr(flags) == pytest.approx(1 / 3)
    assert mrr([False, False]) == 0.0


def test_ndcg():
    # relevant at rank 1 -> perfect
    assert ndcg_at_k([True, False, False], 3) == pytest.approx(1.0)
    # relevant at rank 2 only: dcg=1/log2(3), idcg=1/log2(2)=1
    assert ndcg_at_k([False, True], 5) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([False, False], 5) == 0.0


def test_answer_matches_sha():
    gold = "a" * 40
    assert answer_matches("commit_sha", gold, f"It was commit {gold[:10]} probably")
    assert answer_matches("commit_sha", gold, f"full {gold}")
    assert not answer_matches("commit_sha", gold, "no sha here")


def test_answer_matches_date_author_pr():
    assert answer_matches("date", "2021-02-09", "Changed on 2021-02-09.")
    assert answer_matches("author", "Armin Ronacher", "by armin ronacher")
    assert answer_matches("pr_number", "4190", "introduced in #4190")
    assert answer_matches("pr_number", "4190", "PR 4190")
    assert not answer_matches("pr_number", "419", "see #4190")


def test_answer_matches_unknown_kind_raises():
    with pytest.raises(ValueError):
        answer_matches("mystery", "x", "y")
