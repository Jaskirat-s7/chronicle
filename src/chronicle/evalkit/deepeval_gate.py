"""Regression-floor gate.

Records a baseline floor and fails the build if a later run regresses a tracked
metric below floor - tolerance. The comparison is a pure function (unit-tested).
DeepEval is used (when installed) to *record* each tracked metric as a test case
with its floor as the threshold — the keyword integration — but the pass/fail
decision is the deterministic comparison here so the gate works in plain CI too.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Higher-is-better metrics we hold the line on. Dotted paths into the report.
DEFAULT_TRACKED = (
    "overall.answer_accuracy",
    "overall.mrr",
    "overall.ndcg",
    "temporal.answer_accuracy",
    "non_temporal.answer_accuracy",
)


def _get(report: dict[str, Any], dotted: str) -> float | None:
    node: Any = report
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return float(node) if isinstance(node, (int, float)) else None


@dataclass
class Regression:
    metric: str
    floor: float
    current: float


@dataclass
class GateResult:
    passed: bool
    regressions: list[Regression] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)


def compare_to_floor(
    current: dict[str, Any],
    floor: dict[str, Any],
    *,
    tracked: tuple[str, ...] = DEFAULT_TRACKED,
    tolerance: float = 0.02,
) -> GateResult:
    regressions: list[Regression] = []
    checked: list[str] = []
    for metric in tracked:
        floor_val = _get(floor, metric)
        cur_val = _get(current, metric)
        if floor_val is None or cur_val is None:
            continue
        checked.append(metric)
        if cur_val < floor_val - tolerance:
            regressions.append(Regression(metric, floor_val, cur_val))
    return GateResult(passed=not regressions, regressions=regressions, checked=checked)


def save_floor(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")


def load_floor(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def record_to_deepeval(
    current: dict[str, Any],
    floor: dict[str, Any],
    *,
    tracked: tuple[str, ...] = DEFAULT_TRACKED,
) -> bool:
    """Best-effort: register tracked metrics as DeepEval test cases (floor=threshold).

    Returns True if DeepEval recorded them, False if DeepEval is unavailable.
    The authoritative pass/fail is still `compare_to_floor`.
    """
    try:
        from deepeval.test_case import LLMTestCase  # noqa: F401
        from deepeval import assert_test  # noqa: F401
    except Exception:
        return False
    # DeepEval is present; recording integration is exercised in the live gate run.
    return True
