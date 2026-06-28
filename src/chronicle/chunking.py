"""Snapshot-baseline chunker.

PR2 indexes only the current snapshot, so this is deliberately the *naive*
baseline: fixed-size overlapping line windows. The sophisticated retrievable-unit
decision (deduping the same function across versions, etc.) is the PR3 topic and
is intentionally NOT made here — keeping the snapshot-vs-timeline delta clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

DEFAULT_WINDOW = 50
DEFAULT_OVERLAP = 10


@dataclass
class Chunk:
    repo: str
    file_path: str
    line_start: int          # 1-based, inclusive
    line_end: int            # 1-based, inclusive
    commit_sha: str
    commit_date: datetime
    content: str


def chunk_text(
    text: str,
    *,
    repo: str,
    file_path: str,
    commit_sha: str,
    commit_date: datetime,
    window: int = DEFAULT_WINDOW,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    """Split text into overlapping line-windows.

    A line at position i appears in a chunk; windows step by (window - overlap).
    The last window is clamped to EOF so every line is covered exactly once at
    minimum (under-dedup over over-dedup).
    """
    if window <= 0:
        raise ValueError("window must be positive")
    if not 0 <= overlap < window:
        raise ValueError("overlap must satisfy 0 <= overlap < window")

    lines = text.splitlines()
    n = len(lines)
    if n == 0:
        return []

    step = window - overlap
    chunks: list[Chunk] = []
    start = 0
    while start < n:
        end = min(start + window, n)
        body = "\n".join(lines[start:end])
        chunks.append(
            Chunk(
                repo=repo,
                file_path=file_path,
                line_start=start + 1,
                line_end=end,
                commit_sha=commit_sha,
                commit_date=commit_date,
                content=body,
            )
        )
        if end == n:
            break
        start += step
    return chunks


def is_probably_text(path: str, allowed_ext: set[str] | None = None) -> bool:
    from .groundtruth.generator import ALLOWED_EXT

    exts = allowed_ext if allowed_ext is not None else ALLOWED_EXT
    return Path(path).suffix.lower() in exts
