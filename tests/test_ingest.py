"""Ingestion walks real history and parses diffs into stats."""

from chronicle import ingest


def test_walk_and_summarize(temp_repo):
    changes = ingest.walk_history(temp_repo.path)
    assert len(changes) == 4

    stats = ingest.summarize(changes)
    assert stats.commits == 4
    assert stats.commits_with_pr == 3      # #1, #2, #3
    assert stats.renames >= 1              # beta -> gamma
    assert stats.lines_added > 0


def test_commit3_diff_has_alpha_change(temp_repo):
    c3 = temp_repo.shas["Change alpha return (#3)"]
    changes = {c.meta.sha: c for c in ingest.walk_history(temp_repo.path)}
    fds = changes[c3].file_diffs
    paths = {fd.new_path for fd in fds}
    assert "alpha.py" in paths
    alpha = next(fd for fd in fds if fd.new_path == "alpha.py")
    assert alpha.added >= 1 and alpha.removed >= 1
