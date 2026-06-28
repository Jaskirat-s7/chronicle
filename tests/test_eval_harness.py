"""Harness aggregation + report + DeepEval gate — all with fakes (no model/db)."""

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from chronicle.evalkit import deepeval_gate
from chronicle.evalkit.harness import run_eval
from chronicle.evalkit.report import build_report
from chronicle.groundtruth.schema import EvalQuestion

DT = datetime(2021, 1, 1, tzinfo=timezone.utc)


@dataclass
class _Chunk:
    file_path: str
    line_start: int
    line_end: int


@dataclass
class _LLM:
    input_tokens: int = 10
    output_tokens: int = 5
    latency_ms: float = 100.0


@dataclass
class _Answer:
    answer: str
    retrieved: list
    contexts: list
    llm: object


def _q(qid, template, kind, gold, evidence, temporal=True, repo="r"):
    return EvalQuestion(
        id=qid, template=template, question=f"q-{qid}", answer=gold,
        answer_kind=kind, temporal=temporal, repo=repo, eval_sha="x", evidence=evidence,
    )


def test_run_eval_scores_correct_and_relevance():
    questions = [
        _q("1", "blame_commit", "commit_sha", "a" * 40, {"path": "f.py", "line": 3}),
        _q("2", "blame_author", "author", "Jane Doe", {"path": "g.py"}, temporal=False),
    ]

    def answer_fn(qtext, repo):
        if qtext == "q-1":
            # gold file retrieved at rank 1, and the answer contains the SHA
            return _Answer(
                answer=f"introduced in {'a' * 40}",
                retrieved=[_Chunk("f.py", 1, 50), _Chunk("z.py", 1, 50)],
                contexts=["ctx"],
                llm=_LLM(),
            )
        # wrong file, wrong author
        return _Answer(
            answer="someone else",
            retrieved=[_Chunk("x.py", 1, 50)],
            contexts=["ctx"],
            llm=_LLM(),
        )

    run = run_eval(questions, answer_fn, ks=(1, 5))
    r1, r2 = run.results
    assert r1.correct and r1.hits[1] == 1.0 and r1.mrr == 1.0
    assert not r2.correct and r2.hits[1] == 0.0 and r2.mrr == 0.0

    report = build_report(run)
    assert report["overall"]["n"] == 2
    assert report["overall"]["answer_accuracy"] == 0.5
    assert report["temporal"]["answer_accuracy"] == 1.0
    assert report["non_temporal"]["answer_accuracy"] == 0.0
    assert report["cost_latency_calls"]["calls_per_query"] == 1.0
    assert report["cost_latency_calls"]["total_input_tokens"] == 20


def test_gate_passes_when_equal_and_fails_on_regression():
    floor = {
        "overall": {"answer_accuracy": 0.50, "mrr": 0.40, "ndcg": 0.45},
        "temporal": {"answer_accuracy": 0.30},
        "non_temporal": {"answer_accuracy": 0.70},
    }
    # identical -> pass
    assert deepeval_gate.compare_to_floor(floor, floor).passed

    regressed = {
        "overall": {"answer_accuracy": 0.40, "mrr": 0.40, "ndcg": 0.45},  # -0.10
        "temporal": {"answer_accuracy": 0.30},
        "non_temporal": {"answer_accuracy": 0.70},
    }
    res = deepeval_gate.compare_to_floor(regressed, floor, tolerance=0.02)
    assert not res.passed
    assert any(r.metric == "overall.answer_accuracy" for r in res.regressions)


def test_gate_tolerance_absorbs_small_dip():
    floor = {"overall": {"answer_accuracy": 0.50, "mrr": 0.40, "ndcg": 0.45}}
    slightly = {"overall": {"answer_accuracy": 0.49, "mrr": 0.40, "ndcg": 0.45}}
    assert deepeval_gate.compare_to_floor(slightly, floor, tolerance=0.02).passed


def test_gate_save_load_roundtrip(tmp_path):
    report = {"overall": {"answer_accuracy": 0.5, "mrr": 0.4, "ndcg": 0.45}}
    p = tmp_path / "floor.json"
    deepeval_gate.save_floor(report, p)
    assert deepeval_gate.load_floor(p) == report
