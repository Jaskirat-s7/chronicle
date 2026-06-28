"""The labeled eval-question record and JSONL (de)serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# Templates and whether each is a *temporal* question (used to slice eval results
# from PR3 onward — temporal vs non-temporal).
TEMPLATES: dict[str, bool] = {
    "blame_commit": True,    # which commit last modified a line
    "commit_date": True,     # on what date was a line last changed
    "file_added": True,      # which commit first added a file
    "blame_author": False,   # who authored the last change to a line
    "pr_for_commit": False,  # which PR introduced a commit
}

ANSWER_KINDS = {"commit_sha", "date", "author", "pr_number"}


@dataclass
class EvalQuestion:
    id: str
    template: str
    question: str
    answer: str
    answer_kind: str
    temporal: bool
    repo: str          # repo identifier (e.g. "pallets/flask")
    eval_sha: str      # the HEAD the set was anchored at (the "as of" point)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalQuestion":
        return cls(**d)


def write_jsonl(questions: Iterable[EvalQuestion], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for q in questions:
            fh.write(json.dumps(q.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[EvalQuestion]:
    questions: list[EvalQuestion] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:  # strict: a bad line is fatal
                raise ValueError(f"{path}:{lineno} invalid JSON: {exc}") from exc
            questions.append(EvalQuestion.from_dict(data))
    return questions
