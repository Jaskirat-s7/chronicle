"""Migration discovery (pure) + an opt-in live migration test."""

import pytest

from chronicle import db
from chronicle.config import get_settings


def test_discover_migrations_sorted():
    migrations = db.discover_migrations()
    names = [p.name for p in migrations]
    assert "0001_base_schema.sql" in names
    assert names == sorted(names)


def test_base_schema_locks_1024_dim():
    sql = (db.MIGRATIONS_DIR / "0001_base_schema.sql").read_text()
    assert "vector(1024)" in sql
    assert "tsvector" in sql  # lexical/FTS side present


@pytest.mark.skipif(
    not db.ping(get_settings().database_url),
    reason="Postgres not reachable; bring up `docker compose up -d db` to run live.",
)
def test_migrate_is_idempotent():
    db.migrate()
    # Second run applies nothing new.
    assert db.migrate() == []
