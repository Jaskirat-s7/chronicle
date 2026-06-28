"""pgvector-backed store for the snapshot index: dense + lexical search.

Vectors are passed as text literals cast to `::vector` (robust across pgvector
adapter versions). Dense search uses cosine distance (`<=>`); lexical search uses
Postgres FTS over the generated `content_tsv` column with `ts_rank`.
"""

from __future__ import annotations

from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row

from .chunking import Chunk
from .config import get_settings


def _vec_literal(vec: list[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in vec) + "]"


class VectorStore:
    def __init__(self, database_url: str | None = None) -> None:
        self.database_url = database_url or get_settings().database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def clear_repo(self, repo: str) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
            n = cur.rowcount
            conn.commit()
            return n

    def count(self, repo: str | None = None) -> int:
        with self._connect() as conn, conn.cursor() as cur:
            if repo is None:
                cur.execute("SELECT count(*) AS n FROM chunks")
            else:
                cur.execute("SELECT count(*) AS n FROM chunks WHERE repo = %s", (repo,))
            return cur.fetchone()["n"]

    def add(self, chunks: Iterable[Chunk], embeddings: list[list[float]]) -> int:
        chunks = list(chunks)
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        rows = [
            (
                c.repo, c.file_path, c.line_start, c.line_end,
                c.commit_sha, c.commit_date, c.content, _vec_literal(emb),
            )
            for c, emb in zip(chunks, embeddings)
        ]
        with self._connect() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO chunks
                  (repo, file_path, line_start, line_end,
                   commit_sha, commit_date, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def vector_search(
        self, embedding: list[float], limit: int, repo: str | None = None
    ) -> list[dict[str, Any]]:
        lit = _vec_literal(embedding)
        repo_clause = "AND repo = %s" if repo else ""
        params: list[Any] = [lit]
        if repo:
            params.append(repo)
        params += [lit, limit]
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, repo, file_path, line_start, line_end,
                       commit_sha, commit_date, content,
                       1 - (embedding <=> %s::vector) AS score
                FROM chunks
                WHERE embedding IS NOT NULL {repo_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()

    def fts_search(
        self, query: str, limit: int, repo: str | None = None
    ) -> list[dict[str, Any]]:
        repo_clause = "AND repo = %s" if repo else ""
        params: list[Any] = [query]
        if repo:
            params.append(repo)
        params.append(limit)
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, repo, file_path, line_start, line_end,
                       commit_sha, commit_date, content,
                       ts_rank(content_tsv, q) AS score
                FROM chunks, websearch_to_tsquery('english', %s) AS q
                WHERE content_tsv @@ q {repo_clause}
                ORDER BY score DESC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()
