"""History ingestion: walk a repo, parse commits + diffs into structured records.

This is the strict spine PR1 promises — per change we surface file, line span,
commit SHA, date, author, message, and any linked PR/issue. The retrievable-unit
/ chunking decision is deliberately NOT made here; that lands in PR3.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import git_ops
from .diff_parser import FileDiff, parse_patch
from .git_ops import CommitMeta
from .logging_config import get_logger

log = get_logger("chronicle.ingest")


@dataclass
class CommitChange:
    """A commit plus its parsed per-file diffs."""

    meta: CommitMeta
    file_diffs: list[FileDiff] = field(default_factory=list)


@dataclass
class IngestStats:
    commits: int = 0
    merges_skipped: int = 0
    files_touched: int = 0
    hunks: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    commits_with_pr: int = 0
    renames: int = 0
    binary_files: int = 0


def walk_history(
    repo: Path | str,
    *,
    max_commits: int | None = None,
) -> list[CommitChange]:
    """Parse commit metadata and (for non-merge commits) per-file diffs."""
    commits = git_ops.list_commits(repo, max_count=max_commits)
    changes: list[CommitChange] = []
    for meta in commits:
        if meta.is_merge:
            # Combined-diff parsing for merges is out of scope; metadata only.
            changes.append(CommitChange(meta=meta, file_diffs=[]))
            continue
        patch = git_ops.commit_patch_with_renames(repo, meta.sha)
        file_diffs = parse_patch(patch)
        changes.append(CommitChange(meta=meta, file_diffs=file_diffs))
    return changes


def summarize(changes: list[CommitChange]) -> IngestStats:
    stats = IngestStats()
    for change in changes:
        stats.commits += 1
        if change.meta.is_merge:
            stats.merges_skipped += 1
        if change.meta.pr_number is not None:
            stats.commits_with_pr += 1
        for fd in change.file_diffs:
            stats.files_touched += 1
            if fd.is_rename:
                stats.renames += 1
            if fd.is_binary:
                stats.binary_files += 1
            for hunk in fd.hunks:
                stats.hunks += 1
                stats.lines_added += len(hunk.added_lines)
                stats.lines_removed += len(hunk.removed_lines)
    return stats
