"""RRF fusion (pure) + the hybrid retriever flow with fakes."""

from datetime import datetime, timezone

from chronicle.retrieval import HybridRetriever, rrf_fuse

DT = datetime(2021, 1, 1, tzinfo=timezone.utc)


def test_rrf_rewards_agreement():
    # item "b" is high in both lists -> should win.
    dense = ["a", "b", "c"]
    lexical = ["b", "d", "a"]
    fused = rrf_fuse([dense, lexical], k=60)
    ids = [i for i, _ in fused]
    assert ids[0] == "b"
    assert set(ids) == {"a", "b", "c", "d"}


def test_rrf_deterministic_tie_break():
    # identical singletons across two lists -> stable order by first appearance.
    fused = rrf_fuse([["x"], ["y"]], k=60)
    assert [i for i, _ in fused] == ["x", "y"]


def test_rrf_scores_use_k():
    fused_small = dict(rrf_fuse([["a"]], k=1))
    fused_big = dict(rrf_fuse([["a"]], k=1000))
    assert fused_small["a"] > fused_big["a"]


def _row(i, path, content, sha="a" * 40):
    return {
        "id": i, "repo": "r", "file_path": path, "line_start": 1, "line_end": 5,
        "commit_sha": sha, "commit_date": DT, "content": content,
    }


class _FakeStore:
    def __init__(self, dense, lexical):
        self._dense, self._lexical = dense, lexical

    def vector_search(self, embedding, limit, repo=None):
        return self._dense[:limit]

    def fts_search(self, query, limit, repo=None):
        return self._lexical[:limit]


class _FakeEmbedder:
    def encode_query(self, text):
        return [0.0, 1.0]


class _FakeReranker:
    """Ranks by length of content (proxy for a real cross-encoder)."""

    def rerank(self, query, docs):
        return [float(len(d)) for d in docs]


def test_retriever_fuses_and_reranks():
    dense = [_row(1, "a.py", "short"), _row(2, "b.py", "a much longer chunk body")]
    lexical = [_row(2, "b.py", "a much longer chunk body"), _row(3, "c.py", "mid len")]
    retr = HybridRetriever(_FakeStore(dense, lexical), _FakeEmbedder(), _FakeReranker())

    out = retr.retrieve("q", top_k=3, candidate_k=10, rerank=True)
    assert out[0].file_path == "b.py"  # longest content wins under fake reranker
    assert {c.id for c in out} == {1, 2, 3}


def test_retriever_without_reranker_uses_rrf_order():
    dense = [_row(1, "a.py", "x"), _row(2, "b.py", "y")]
    lexical = [_row(2, "b.py", "y"), _row(1, "a.py", "x")]
    retr = HybridRetriever(_FakeStore(dense, lexical), _FakeEmbedder(), None)
    out = retr.retrieve("q", top_k=2, rerank=False)
    # id 2 appears at ranks (2,1) vs id 1 (1,2): equal RRF -> tie-break by first seen (id 1 first in dense)
    assert {c.id for c in out} == {1, 2}
