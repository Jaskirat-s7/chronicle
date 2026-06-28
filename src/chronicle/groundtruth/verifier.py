"""Independently re-derive each question's answer from git and assert it matches.

The verifier uses ONLY a question's `evidence` + the repo — it does not trust the
generator's cached values. This proves each question is self-contained and truly
git-verifiable. A mismatch is fatal (strict-parse-or-crash): we would rather
crash than ship a mislabeled eval question.
"""

from __future__ import annotations

from pathlib import Path

from .. import git_ops
from .schema import EvalQuestion


class VerificationError(AssertionError):
    """Raised when a question's recomputed answer disagrees with its label."""


def _recompute(repo: Path | str, q: EvalQuestion) -> str:
    ev = q.evidence
    if q.template == "blame_commit":
        return git_ops.blame_line(repo, ev["rev"], ev["path"], ev["line"]).sha
    if q.template == "commit_date":
        bl = git_ops.blame_line(repo, ev["rev"], ev["path"], ev["line"])
        return bl.author_time.strftime("%Y-%m-%d")
    if q.template == "blame_author":
        return git_ops.blame_line(repo, ev["rev"], ev["path"], ev["line"]).author_name
    if q.template == "file_added":
        return git_ops.first_add_commit(repo, ev["path"])
    if q.template == "pr_for_commit":
        meta = git_ops.list_commits(repo, max_count=1, rev_range=ev["sha"])
        if not meta:
            raise VerificationError(f"{q.id}: commit {ev['sha']} not found")
        pr = meta[0].pr_number
        return str(pr) if pr is not None else ""
    raise VerificationError(f"{q.id}: unknown template {q.template!r}")


def verify_question(repo: Path | str, q: EvalQuestion) -> None:
    recomputed = _recompute(repo, q)
    if recomputed != q.answer:
        raise VerificationError(
            f"{q.id} [{q.template}]: label={q.answer!r} but git recomputed {recomputed!r}"
        )


def verify_all(repo: Path | str, questions: list[EvalQuestion]) -> int:
    """Verify every question. Returns the count verified; raises on first mismatch."""
    for q in questions:
        verify_question(repo, q)
    return len(questions)
