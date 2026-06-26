"""Config defaults + env overrides. These defaults are locked decisions."""

import pytest

from chronicle.config import get_settings

# Env vars that would otherwise leak host configuration into the defaults test.
_OVERRIDABLE = [
    "DATABASE_URL",
    "OLLAMA_BASE_URL",
    "JUDGE_MODEL",
    "GEN_MODEL",
    "EMBEDDING_DIM",
    "EMBEDDING_MODEL",
    "GOOGLE_API_KEY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for name in _OVERRIDABLE:
        monkeypatch.delenv(name, raising=False)
    # Run from a dir with no .env so file values don't bleed in.
    monkeypatch.chdir(tmp_path)
    return monkeypatch


def test_locked_defaults(clean_env):
    s = get_settings()
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.judge_model == "qwen2.5:3b"
    assert s.embedding_dim == 1024  # LOCKED
    assert s.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
    assert s.reranker_model == "BAAI/bge-reranker-v2-m3"


def test_disabled_flags_when_no_keys(clean_env):
    s = get_settings()
    assert s.langfuse_enabled is False
    assert s.gemini_enabled is False


def test_env_overrides(clean_env):
    clean_env.setenv("OLLAMA_BASE_URL", "http://localhost:9999")
    clean_env.setenv("JUDGE_MODEL", "qwen2.5:7b")
    clean_env.setenv("GOOGLE_API_KEY", "test-key")
    s = get_settings()
    assert s.ollama_base_url == "http://localhost:9999"
    assert s.judge_model == "qwen2.5:7b"
    assert s.gemini_enabled is True
