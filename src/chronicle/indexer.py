"""Snapshot indexing pipeline: HEAD files -> chunk -> embed -> pgvector.

PR2 indexes ONLY the current snapshot, so every chunk is tagged with the HEAD
commit + date. This is the baseline that the PR3 temporal index will beat on
temporal questions — by design, the snapshot index cannot answer "when did this
change" because it throws history away.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import git_ops
from .chunking import Chunk, chunk_text, is_probably_text
from .embeddings import Embedder
from .logging_config import get_logger
from .vector_store import VectorStore

log = get_logger("chronicle.indexer")


@dataclass
class IndexResult:
    repo: str
    head_sha: str
    files_indexed: int
    chunks_indexed: int


def build_snapshot_chunks(repo_path: Path | str, repo_name: str) -> list[Chunk]:
    head = git_ops.head_sha(repo_path)
    meta = git_ops.list_commits(repo_path, max_count=1)
    if not meta:
        raise RuntimeError("repo has no commits")
    head_date = meta[0].author_date

    chunks: list[Chunk] = []
    for path in git_ops.list_files(repo_path, head):
        if not is_probably_text(path):
            continue
        try:
            content = git_ops.show_file(repo_path, head, path)
        except git_ops.GitError:
            continue  # unreadable blob (e.g. submodule entry) — skip, not fatal
        chunks.extend(
            chunk_text(
                content,
                repo=repo_name,
                file_path=path,
                commit_sha=head,
                commit_date=head_date,
            )
        )
    return chunks


def index_snapshot(
    repo_path: Path | str,
    repo_name: str,
    embedder: Embedder,
    store: VectorStore,
    *,
    replace: bool = True,
) -> IndexResult:
    if replace:
        store.clear_repo(repo_name)

    chunks = build_snapshot_chunks(repo_path, repo_name)
    if not chunks:
        raise RuntimeError("no text/code chunks produced from snapshot")

    log.info("Embedding %d chunks...", len(chunks))
    embeddings = embedder.encode_documents([c.content for c in chunks])
    store.add(chunks, embeddings)

    files = len({c.file_path for c in chunks})
    return IndexResult(
        repo=repo_name,
        head_sha=git_ops.head_sha(repo_path),
        files_indexed=files,
        chunks_indexed=len(chunks),
    )
