"""RAGAS integration (faithfulness, answer relevancy, context precision/recall).

Isolated and lazy: the judge LLM runs on local Ollama (protecting the Gemini
free-tier quota), embeddings run on the local model. RAGAS + langchain deps live
in the `eval` extra. Best-effort: on any failure we return an `error` field
rather than crashing the harness — real numbers come only from a real run.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from ..logging_config import get_logger

log = get_logger("chronicle.ragas")

_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def run_ragas(samples: list[dict], settings: Settings | None = None) -> dict[str, Any]:
    """Evaluate samples with RAGAS. Each sample: question, answer, contexts, ground_truth."""
    s = settings or get_settings()
    if not samples:
        return {"error": "no samples"}
    try:
        from datasets import Dataset
        from langchain_ollama import ChatOllama
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        judge = ChatOllama(
            model=s.judge_model,
            base_url=s.ollama_base_url,
            temperature=0.0,
        )

        ds = Dataset.from_list(
            [
                {
                    "question": x["question"],
                    "answer": x["answer"],
                    "contexts": x["contexts"],
                    "ground_truth": x["ground_truth"],
                }
                for x in samples
            ]
        )
        result = evaluate(
            ds,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            llm=judge,
        )
        scores = result.to_pandas()[list(_METRIC_NAMES)].mean().to_dict()
        return {k: float(v) for k, v in scores.items()}
    except Exception as exc:  # pragma: no cover - depends on live judge + deps
        log.warning("RAGAS unavailable / failed: %s", exc)
        return {"error": str(exc)}
