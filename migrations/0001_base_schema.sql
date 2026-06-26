-- PR0 base schema: minimal SNAPSHOT index.
--
-- Scope is deliberately the current-snapshot fields only:
--   content + commit SHA + commit date + file/line span.
-- This is exactly what the PR2 snapshot baseline needs and nothing more.
--
-- Temporal-validity columns (valid_from / valid_until, by commit + date) arrive
-- in PR3 as a SEPARATE migration, so the snapshot-vs-timeline delta stays
-- measurable. Do not add them here.
--
-- The embedding column is LOCKED at vector(1024) — Qwen3-Embedding-0.6B native
-- dim. No Matryoshka truncation in the baseline; that becomes a measured
-- experiment later.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           bigserial   PRIMARY KEY,
    repo         text        NOT NULL,
    file_path    text        NOT NULL,
    line_start   integer     NOT NULL,
    line_end     integer     NOT NULL,
    commit_sha   text        NOT NULL,
    commit_date  timestamptz NOT NULL,
    content      text        NOT NULL,
    content_tsv  tsvector    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding    vector(1024),                 -- populated in PR2
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Lexical side of hybrid retrieval (Postgres FTS).
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING gin (content_tsv);

-- Common lookup paths.
CREATE INDEX IF NOT EXISTS chunks_commit_sha_idx  ON chunks (commit_sha);
CREATE INDEX IF NOT EXISTS chunks_repo_file_idx   ON chunks (repo, file_path);

-- NOTE: the ANN index on `embedding` (HNSW, cosine) is created in PR2 once the
-- column is actually populated — building it on an empty table buys nothing.
