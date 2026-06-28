"""Ground-truth generation + the self-verifier, against a real temp repo."""

import pytest

from chronicle.groundtruth import (
    EvalQuestion,
    GroundTruthGenerator,
    VerificationError,
    load_jsonl,
    verify_all,
    verify_question,
    write_jsonl,
)


def test_generate_all_verify(temp_repo):
    gen = GroundTruthGenerator(temp_repo.path, "test/sample", seed=1)
    pool = gen.generate(target=12)
    assert len(pool) > 0
    # Every generated question is independently re-derivable from git.
    assert verify_all(temp_repo.path, pool) == len(pool)
    # Multiple templates represented.
    assert len({q.template for q in pool}) >= 2


def test_blame_question_answer_is_correct(temp_repo):
    """A hand-built blame question must match git's blame, and tampering fails."""
    c3 = temp_repo.shas["Change alpha return (#3)"]
    q = EvalQuestion(
        id="t",
        template="blame_commit",
        question="?",
        answer=c3,
        answer_kind="commit_sha",
        temporal=True,
        repo="test/sample",
        eval_sha="HEAD",
        evidence={"path": "alpha.py", "line": 2, "rev": "HEAD"},
    )
    verify_question(temp_repo.path, q)  # passes

    q.answer = "0" * 40
    with pytest.raises(VerificationError):
        verify_question(temp_repo.path, q)


def test_pr_question_verifies(temp_repo):
    c1 = temp_repo.shas["Add alpha (#1)"]
    q = EvalQuestion(
        id="t",
        template="pr_for_commit",
        question="?",
        answer="1",
        answer_kind="pr_number",
        temporal=False,
        repo="test/sample",
        eval_sha="HEAD",
        evidence={"sha": c1},
    )
    verify_question(temp_repo.path, q)
    q.answer = "999"
    with pytest.raises(VerificationError):
        verify_question(temp_repo.path, q)


def test_stratified_subset_covers_templates(temp_repo):
    gen = GroundTruthGenerator(temp_repo.path, "test/sample", seed=2)
    pool = gen.generate(target=12)
    dev = gen.stratified_subset(pool, size=4)
    assert 0 < len(dev) <= 4
    assert set(q.template for q in dev) <= set(q.template for q in pool)


def test_jsonl_roundtrip(tmp_path, temp_repo):
    gen = GroundTruthGenerator(temp_repo.path, "test/sample", seed=3)
    pool = gen.generate(target=8)
    path = tmp_path / "set.jsonl"
    n = write_jsonl(pool, path)
    loaded = load_jsonl(path)
    assert n == len(loaded) == len(pool)
    assert loaded[0].to_dict() == pool[0].to_dict()
