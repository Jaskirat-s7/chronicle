"""Aggregate an EvalRun into a baseline report (metrics + cost/latency/calls)."""

from __future__ import annotations

import statistics
from typing import Any, Sequence

from .harness import EvalRun, QuestionEval


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _percentile(xs: Sequence[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1))))
    return s[idx]


def _block(results: list[QuestionEval], ks: tuple[int, ...]) -> dict[str, Any]:
    if not results:
        return {"n": 0}
    block: dict[str, Any] = {
        "n": len(results),
        "answer_accuracy": _mean([1.0 if r.correct else 0.0 for r in results]),
        "mrr": _mean([r.mrr for r in results]),
        "ndcg": _mean([r.ndcg for r in results]),
    }
    for k in ks:
        block[f"hit@{k}"] = _mean([r.hits[k] for r in results])
    return block


def build_report(run: EvalRun, *, label: str = "baseline") -> dict[str, Any]:
    results = run.results
    ks = run.ks
    latencies = [r.latency_ms for r in results if r.latency_ms]

    by_template: dict[str, dict] = {}
    for t in sorted({r.template for r in results}):
        by_template[t] = _block([r for r in results if r.template == t], ks)

    by_repo: dict[str, dict] = {}
    for repo in sorted({r.repo for r in results}):
        by_repo[repo] = _block([r for r in results if r.repo == repo], ks)

    return {
        "label": label,
        "overall": _block(results, ks),
        "temporal": _block([r for r in results if r.temporal], ks),
        "non_temporal": _block([r for r in results if not r.temporal], ks),
        "by_template": by_template,
        "by_repo": by_repo,
        "cost_latency_calls": {
            "llm_calls": len(results),  # single-hop baseline: 1 call/question
            "calls_per_query": 1.0 if results else 0.0,
            "total_input_tokens": sum(r.input_tokens for r in results),
            "total_output_tokens": sum(r.output_tokens for r in results),
            "latency_p50_ms": _percentile(latencies, 50),
            "latency_p95_ms": _percentile(latencies, 95),
            "latency_mean_ms": _mean(latencies),
            "note": "Authoritative cost/latency/calls come from Langfuse; these "
                    "are local fallbacks computed from LLMResponse.",
        },
    }


def pretty_print(report: dict[str, Any]) -> str:
    o = report["overall"]
    lines = [
        f"=== chronicle eval report: {report['label']} ===",
        f"n={o.get('n', 0)}  answer_accuracy={o.get('answer_accuracy', 0):.3f}  "
        f"mrr={o.get('mrr', 0):.3f}  ndcg={o.get('ndcg', 0):.3f}",
        f"temporal acc={report['temporal'].get('answer_accuracy', 0):.3f}  "
        f"non-temporal acc={report['non_temporal'].get('answer_accuracy', 0):.3f}",
        "by template:",
    ]
    for t, b in report["by_template"].items():
        lines.append(
            f"  {t:16s} n={b.get('n',0):3d} acc={b.get('answer_accuracy',0):.3f} "
            f"mrr={b.get('mrr',0):.3f} ndcg={b.get('ndcg',0):.3f}"
        )
    clc = report["cost_latency_calls"]
    lines.append(
        f"calls/query={clc['calls_per_query']:.1f}  "
        f"tokens(in/out)={clc['total_input_tokens']}/{clc['total_output_tokens']}  "
        f"latency p50/p95={clc['latency_p50_ms']:.0f}/{clc['latency_p95_ms']:.0f} ms"
    )
    if "ragas" in report:
        lines.append(f"RAGAS: {report['ragas']}")
    return "\n".join(lines)
