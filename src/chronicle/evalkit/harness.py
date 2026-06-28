"""Run the eval set through the pipeline and compute per-question results.

`run_eval` takes an injectable `answer_fn(question_text, repo) -> AnswerResult`
so the aggregation logic is unit-testable with fakes, while the CLI wires the
real retrieve+generate path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

from ..groundtruth.schema import EvalQuestion
from .metrics import answer_matches, hit_at_k, mrr, ndcg_at_k, relevance_flags

DEFAULT_KS = (1, 5, 10)


class _AnswerLike(Protocol):
    answer: str
    retrieved: Sequence  # objects with file_path/line_start/line_end
    contexts: list[str]
    llm: object | None


AnswerFn = Callable[[str, str], _AnswerLike]


@dataclass
class QuestionEval:
    id: str
    template: str
    answer_kind: str
    temporal: bool
    repo: str
    gold: str
    question_text: str
    correct: bool
    mrr: float
    ndcg: float
    hits: dict[int, float]
    answer_text: str
    contexts: list[str]
    input_tokens: int
    output_tokens: int
    latency_ms: float


@dataclass
class EvalRun:
    results: list[QuestionEval] = field(default_factory=list)
    ks: tuple[int, ...] = DEFAULT_KS


def run_eval(
    questions: Sequence[EvalQuestion],
    answer_fn: AnswerFn,
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
    ndcg_k: int = 10,
) -> EvalRun:
    run = EvalRun(ks=ks)
    for q in questions:
        ar = answer_fn(q.question, q.repo)
        rows = [
            {
                "file_path": c.file_path,
                "line_start": c.line_start,
                "line_end": c.line_end,
            }
            for c in ar.retrieved
        ]
        flags = relevance_flags(rows, q.evidence)
        llm = getattr(ar, "llm", None)
        run.results.append(
            QuestionEval(
                id=q.id,
                template=q.template,
                answer_kind=q.answer_kind,
                temporal=q.temporal,
                repo=q.repo,
                gold=q.answer,
                question_text=q.question,
                correct=answer_matches(q.answer_kind, q.answer, ar.answer),
                mrr=mrr(flags),
                ndcg=ndcg_at_k(flags, ndcg_k),
                hits={k: hit_at_k(flags, k) for k in ks},
                answer_text=ar.answer,
                contexts=list(ar.contexts),
                input_tokens=getattr(llm, "input_tokens", 0) or 0,
                output_tokens=getattr(llm, "output_tokens", 0) or 0,
                latency_ms=getattr(llm, "latency_ms", 0.0) or 0.0,
            )
        )
    return run


def ragas_samples(run: EvalRun) -> list[dict]:
    """Shape per-question data for RAGAS."""
    return [
        {
            "question": r.question_text,
            "answer": r.answer_text,
            "contexts": r.contexts,
            "ground_truth": r.gold,
        }
        for r in run.results
    ]
