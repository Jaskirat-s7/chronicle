-- PR2: ANN index on the embedding column, now that PR2 populates it.
-- HNSW with cosine ops, matching the `1 - (embedding <=> q)` similarity used in
-- VectorStore.vector_search.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
