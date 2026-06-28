r"""Strict unified-diff parser.

Strict-parse-or-crash: a malformed hunk header or an unexpected line prefix
raises `DiffParseError`. We never silently drop a line — a silent JSON/diff drop
has bitten this project's author before, and we would rather crash than lose data.

Each hunk header `@@ -a,b +c,d @@` tells us exactly how many old/new lines to
consume, so we read precisely that many and stop — which avoids ambiguity around
trailing blank lines and "\ No newline at end of file" markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.+?) b/(.+)$")


class DiffParseError(ValueError):
    """Raised on any malformed diff input."""


@dataclass
class Hunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    section: str
    added_lines: list[int] = field(default_factory=list)    # new-file line numbers
    removed_lines: list[int] = field(default_factory=list)  # old-file line numbers

    @property
    def new_span(self) -> tuple[int, int]:
        if self.new_count == 0:
            return (self.new_start, self.new_start)
        return (self.new_start, self.new_start + self.new_count - 1)

    @property
    def old_span(self) -> tuple[int, int]:
        if self.old_count == 0:
            return (self.old_start, self.old_start)
        return (self.old_start, self.old_start + self.old_count - 1)


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[Hunk] = field(default_factory=list)
    is_binary: bool = False
    is_rename: bool = False
    is_new: bool = False
    is_delete: bool = False

    @property
    def added(self) -> int:
        return sum(len(h.added_lines) for h in self.hunks)

    @property
    def removed(self) -> int:
        return sum(len(h.removed_lines) for h in self.hunks)


def parse_patch(text: str) -> list[FileDiff]:
    """Parse a multi-file unified diff into structured FileDiffs."""
    lines = text.splitlines()
    files: list[FileDiff] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if not line.startswith("diff --git"):
            i += 1
            continue
        m = _DIFF_HEADER_RE.match(line)
        if not m:
            raise DiffParseError(f"Malformed diff header: {line!r}")
        fd = FileDiff(old_path=m.group(1), new_path=m.group(2))
        i += 1

        # Extended header lines (modes, index, rename/copy, binary, ---/+++)
        # until the first hunk or the next file.
        while i < n and not lines[i].startswith("@@") and not lines[i].startswith("diff --git"):
            h = lines[i]
            if h.startswith("rename from "):
                fd.is_rename = True
                fd.old_path = h[len("rename from ") :]
            elif h.startswith("rename to "):
                fd.is_rename = True
                fd.new_path = h[len("rename to ") :]
            elif h.startswith("new file mode"):
                fd.is_new = True
            elif h.startswith("deleted file mode"):
                fd.is_delete = True
            elif h.startswith("Binary files") or h.startswith("GIT binary patch"):
                fd.is_binary = True
            elif h.startswith("--- "):
                pass
            elif h.startswith("+++ "):
                pass
            i += 1

        # Hunks
        while i < n and lines[i].startswith("@@"):
            i = _parse_hunk(lines, i, fd)

        files.append(fd)

    return files


def _parse_hunk(lines: list[str], i: int, fd: FileDiff) -> int:
    header = lines[i]
    m = _HUNK_RE.match(header)
    if not m:
        raise DiffParseError(f"Malformed hunk header: {header!r}")
    old_start = int(m.group(1))
    old_count = int(m.group(2)) if m.group(2) is not None else 1
    new_start = int(m.group(3))
    new_count = int(m.group(4)) if m.group(4) is not None else 1
    section = header[m.end():].lstrip()

    hunk = Hunk(
        old_start=old_start,
        old_count=old_count,
        new_start=new_start,
        new_count=new_count,
        section=section,
    )

    i += 1
    old_remaining = old_count
    new_remaining = new_count
    old_ln = old_start
    new_ln = new_start
    n = len(lines)

    while (old_remaining > 0 or new_remaining > 0):
        if i >= n:
            raise DiffParseError(
                f"Hunk truncated: {old_remaining} old / {new_remaining} new lines "
                f"unread in header {header!r}"
            )
        body = lines[i]
        if body.startswith(" "):
            # context line — git always emits a leading space, even for blank
            # source lines (so a counted hunk line is never the empty string).
            old_remaining -= 1
            new_remaining -= 1
            old_ln += 1
            new_ln += 1
        elif body.startswith("+"):
            hunk.added_lines.append(new_ln)
            new_remaining -= 1
            new_ln += 1
        elif body.startswith("-"):
            hunk.removed_lines.append(old_ln)
            old_remaining -= 1
            old_ln += 1
        elif body.startswith("\\"):
            # "\ No newline at end of file" — does not consume a line.
            pass
        else:
            raise DiffParseError(
                f"Unexpected line prefix in hunk {header!r}: {body!r}"
            )
        i += 1

        if old_remaining < 0 or new_remaining < 0:
            raise DiffParseError(
                f"Hunk overran its declared counts in header {header!r}"
            )

    fd.hunks.append(hunk)
    return i
