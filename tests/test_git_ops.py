"""git_ops against a real temp repo."""

import pytest

from chronicle import git_ops
from chronicle.git_ops import GitError


def test_list_commits_parses_metadata(temp_repo):
    commits = git_ops.list_commits(temp_repo.path)
    assert len(commits) == 4
    subjects = [c.subject for c in commits]
    assert "Change alpha return (#3)" in subjects

    by_subject = {c.subject: c for c in commits}
    assert by_subject["Add alpha (#1)"].pr_number == 1
    assert by_subject["Change alpha return (#3)"].pr_number == 3
    assert by_subject["Rename beta to gamma"].pr_number is None
    # author date parsed as tz-aware datetime
    assert by_subject["Add alpha (#1)"].author_date.year == 2021


def test_blame_line_points_to_right_commit(temp_repo):
    c1 = temp_repo.shas["Add alpha (#1)"]
    c3 = temp_repo.shas["Change alpha return (#3)"]

    line1 = git_ops.blame_line(temp_repo.path, "HEAD", "alpha.py", 1)
    line2 = git_ops.blame_line(temp_repo.path, "HEAD", "alpha.py", 2)
    assert line1.sha == c1  # `def a():` untouched since creation
    assert line2.sha == c3  # `return 42` changed in c3
    assert line2.author_time.strftime("%Y-%m-%d") == "2021-06-15"
    assert line2.author_name == "Test Author"


def test_first_add_commit_follows_rename(temp_repo):
    c2 = temp_repo.shas["Add beta module (#2)"]
    # gamma.py was renamed from beta.py; --follow tracks the original add.
    assert git_ops.first_add_commit(temp_repo.path, "gamma.py") == c2


def test_list_files_at_head(temp_repo):
    files = set(git_ops.list_files(temp_repo.path, "HEAD"))
    assert files == {"alpha.py", "gamma.py"}


def test_bad_rev_raises(temp_repo):
    with pytest.raises(GitError):
        git_ops.blame_line(temp_repo.path, "deadbeef", "alpha.py", 1)
