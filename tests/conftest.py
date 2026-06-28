"""Shared fixtures: a real, tiny git repo built with actual git commands.

No mocking — ground truth must come from genuine git history.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest


def _run(repo: Path, *args: str, date: str | None = None) -> None:
    env = os.environ.copy()
    if date is not None:
        env["GIT_AUTHOR_DATE"] = date
        env["GIT_COMMITTER_DATE"] = date
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


@dataclass
class RepoFixture:
    path: Path
    shas: dict[str, str]  # message-subject -> full sha


@pytest.fixture
def temp_repo(tmp_path) -> RepoFixture:
    repo = tmp_path / "sample"
    repo.mkdir()
    _run(repo, "init", "-q")
    _run(repo, "config", "user.email", "tester@example.com")
    _run(repo, "config", "user.name", "Test Author")
    _run(repo, "config", "commit.gpgsign", "false")

    # c1: add alpha.py
    (repo / "alpha.py").write_text("def a():\n    return 1\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "Add alpha (#1)", date="2021-01-01T12:00:00+0000")

    # c2: add beta.py
    (repo / "beta.py").write_text("x = 1\ny = 2\nz = 3\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "Add beta module (#2)", date="2021-02-01T12:00:00+0000")

    # c3: change alpha line 2
    (repo / "alpha.py").write_text("def a():\n    return 42\n")
    _run(repo, "add", "-A")
    _run(repo, "commit", "-q", "-m", "Change alpha return (#3)", date="2021-06-15T12:00:00+0000")

    # c4: rename beta.py -> gamma.py (pure rename)
    _run(repo, "mv", "beta.py", "gamma.py")
    _run(repo, "commit", "-q", "-m", "Rename beta to gamma", date="2021-07-01T12:00:00+0000")

    log = subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%H\x1f%s"],
        check=True, capture_output=True, text=True,
    ).stdout
    shas: dict[str, str] = {}
    for line in log.splitlines():
        sha, subject = line.split("\x1f")
        shas[subject] = sha

    return RepoFixture(path=repo, shas=shas)
