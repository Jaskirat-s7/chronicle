"""Snapshot chunker: coverage, spans, overlap, edge cases."""

from datetime import datetime, timezone

import pytest

from chronicle.chunking import chunk_text

DT = datetime(2021, 1, 1, tzinfo=timezone.utc)


def _chunk(text, **kw):
    return chunk_text(
        text, repo="r", file_path="f.py", commit_sha="a" * 40, commit_date=DT, **kw
    )


def test_empty_text_yields_nothing():
    assert _chunk("") == []


def test_single_window_when_small():
    text = "\n".join(f"l{i}" for i in range(10))
    chunks = _chunk(text, window=50, overlap=10)
    assert len(chunks) == 1
    assert (chunks[0].line_start, chunks[0].line_end) == (1, 10)


def test_windows_overlap_and_cover_all_lines():
    text = "\n".join(f"l{i}" for i in range(1, 121))  # 120 lines
    chunks = _chunk(text, window=50, overlap=10)
    # step = 40 -> starts at 1,41,81,121(clamped). 121 > 120 so stop at end.
    assert chunks[0].line_start == 1 and chunks[0].line_end == 50
    assert chunks[1].line_start == 41 and chunks[1].line_end == 90
    assert chunks[2].line_start == 81 and chunks[2].line_end == 120
    # every line is covered by at least one chunk
    covered = set()
    for c in chunks:
        covered.update(range(c.line_start, c.line_end + 1))
    assert covered == set(range(1, 121))


def test_content_matches_span():
    text = "\n".join(f"l{i}" for i in range(1, 11))
    c = _chunk(text, window=5, overlap=0)[0]
    assert c.content == "l1\nl2\nl3\nl4\nl5"
    assert (c.line_start, c.line_end) == (1, 5)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        _chunk("a\nb", window=5, overlap=5)
    with pytest.raises(ValueError):
        _chunk("a\nb", window=0)
