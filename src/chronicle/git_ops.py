"""Strict subprocess wrappers around `git`.

We use `subprocess git` (not pygit2): exact, debuggable, and a natural fit for
strict-parse-or-crash — we control the output format and raise loudly on any
non-zero exit or malformed output. No silent `except: pass`.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Field/record separators that will not appear in git metadata text.
_FS = "\x1f"  # between fields
_RS = "\x1e"  # between commit records

_LOG_FORMAT = _FS.join(["%H", "%an", "%ae", "%aI", "%P", "%s", "%b"]) + _RS

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_PR_SQUASH_RE = re.compile(r"\(#(\d+)\)")
_PR_MERGE_RE = re.compile(r"Merge pull request #(\d+)")
_ISSUE_RE = re.compile(r"#(\d+)")


class GitError(RuntimeError):
    """Raised when a git invocation fails or returns malformed output."""


@dataclass
class CommitMeta:
    sha: str
    author_name: str
    author_email: str
    author_date: datetime
    parents: list[str]
    subject: str
    body: str
    pr_number: int | None = None
    referenced_issues: list[int] = field(default_factory=list)

    @property
    def is_merge(self) -> bool:
        return len(self.parents) > 1


@dataclass
class BlameLine:
    sha: str
    author_name: str
    author_time: datetime
    summary: str


def run_git(repo: Path | str, args: list[str], *, check: bool = True) -> str:
    """Run `git -C <repo> <args>` and return stdout. Raise GitError on failure."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"`git {' '.join(args)}` failed ({proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout


def head_sha(repo: Path | str) -> str:
    sha = run_git(repo, ["rev-parse", "HEAD"]).strip()
    if not _SHA_RE.match(sha):
        raise GitError(f"rev-parse HEAD returned a non-SHA: {sha!r}")
    return sha


def short(sha: str, n: int = 10) -> str:
    return sha[:n]


def _extract_pr_and_issues(subject: str, body: str) -> tuple[int | None, list[int]]:
    pr: int | None = None
    m = _PR_SQUASH_RE.search(subject) or _PR_MERGE_RE.search(subject)
    if m:
        pr = int(m.group(1))
    issues = sorted({int(x) for x in _ISSUE_RE.findall(subject + "\n" + body)})
    return pr, issues


def list_commits(
    repo: Path | str,
    *,
    max_count: int | None = None,
    rev_range: str | None = None,
) -> list[CommitMeta]:
    """Parse commit metadata via a separator-delimited `git log`.

    Strict: every record must split into exactly 7 fields, or we crash.
    """
    args = ["log", f"--pretty=format:{_LOG_FORMAT}"]
    if max_count is not None:
        args.append(f"--max-count={max_count}")
    if rev_range is not None:
        args.append(rev_range)
    raw = run_git(repo, args)

    commits: list[CommitMeta] = []
    for record in raw.split(_RS):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split(_FS)
        if len(fields) != 7:
            raise GitError(
                f"Malformed commit record: expected 7 fields, got {len(fields)}: {record!r}"
            )
        sha, an, ae, aiso, parents_raw, subject, body = fields
        if not _SHA_RE.match(sha):
            raise GitError(f"Malformed commit SHA in log: {sha!r}")
        parents = parents_raw.split() if parents_raw.strip() else []
        pr, issues = _extract_pr_and_issues(subject, body)
        commits.append(
            CommitMeta(
                sha=sha,
                author_name=an,
                author_email=ae,
                author_date=datetime.fromisoformat(aiso),
                parents=parents,
                subject=subject,
                body=body,
                pr_number=pr,
                referenced_issues=issues,
            )
        )
    return commits


def list_files(repo: Path | str, rev: str = "HEAD") -> list[str]:
    out = run_git(repo, ["ls-tree", "-r", "--name-only", rev])
    return [line for line in out.splitlines() if line]


def show_file(repo: Path | str, rev: str, path: str) -> str:
    return run_git(repo, ["show", f"{rev}:{path}"])


def file_line_count(repo: Path | str, rev: str, path: str) -> int:
    return len(show_file(repo, rev, path).splitlines())


def blame_line(repo: Path | str, rev: str, path: str, line: int) -> BlameLine:
    """Blame a single line via porcelain output. Strict parse."""
    out = run_git(
        repo,
        ["blame", "--porcelain", "-L", f"{line},{line}", rev, "--", path],
    )
    lines = out.splitlines()
    if not lines:
        raise GitError(f"Empty blame for {path}:{line} @ {rev}")
    header = lines[0].split()
    sha = header[0]
    if not _SHA_RE.match(sha):
        raise GitError(f"Malformed blame header: {lines[0]!r}")

    author_name: str | None = None
    author_time: datetime | None = None
    summary: str | None = None
    for ln in lines[1:]:
        if ln.startswith("author "):
            author_name = ln[len("author ") :]
        elif ln.startswith("author-time "):
            author_time = datetime.fromtimestamp(int(ln.split()[1]), tz=timezone.utc)
        elif ln.startswith("summary "):
            summary = ln[len("summary ") :]
        elif ln.startswith("\t"):
            break  # reached the source line; metadata done
    if author_name is None or author_time is None or summary is None:
        raise GitError(f"Incomplete blame porcelain for {path}:{line} @ {rev}")
    return BlameLine(sha=sha, author_name=author_name, author_time=author_time, summary=summary)


def first_add_commit(repo: Path | str, path: str) -> str:
    """The earliest commit that added `path` (following renames). Strict."""
    out = run_git(
        repo,
        [
            "log",
            "--diff-filter=A",
            "--find-renames",
            "--follow",
            "--format=%H",
            "--",
            path,
        ],
    )
    shas = [s for s in out.splitlines() if s.strip()]
    if not shas:
        raise GitError(f"No add-commit found for {path!r}")
    earliest = shas[-1]
    if not _SHA_RE.match(earliest):
        raise GitError(f"Malformed add-commit SHA for {path!r}: {earliest!r}")
    return earliest


def commit_patch(repo: Path | str, sha: str) -> str:
    """Unified diff for a (non-merge) commit, patch text only."""
    return run_git(
        repo,
        ["show", sha, "--no-color", "--no-renames", "-U3", "--format=", "--"],
    )


def commit_patch_with_renames(repo: Path | str, sha: str) -> str:
    return run_git(
        repo,
        ["show", sha, "--no-color", "--find-renames", "-U3", "--format=", "--"],
    )
