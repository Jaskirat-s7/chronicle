"""Configuration — read from environment / `.env`, never hardcoded.

Every value here is overridable by an environment variable of the same name
(upper-cased). `OLLAMA_BASE_URL` in particular is always read from env with a
localhost default, so the tool runs Mac-only without ceremony.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Postgres + pgvector ---
    database_url: str = "postgresql://chronicle:chronicle@localhost:5432/chronicle"

    # --- Embeddings / reranker (used from PR2; dim LOCKED here in PR0) ---
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_dim: int = 1024  # LOCKED: pgvector column is vector(1024).
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # --- Generation backend: Gemini free tier ---
    gen_model: str = "gemini-2.0-flash"
    google_api_key: str | None = None

    # --- Judge backend: local Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    judge_model: str = "qwen2.5:3b"

    # --- Langfuse (self-hosted v2) ---
    langfuse_host: str = "http://localhost:3000"
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def gemini_enabled(self) -> bool:
        return bool(self.google_api_key)


def get_settings() -> Settings:
    """Build a fresh Settings from env/.env.

    Deliberately uncached so tests (and a long-running process re-reading env)
    see current values.
    """
    return Settings()
