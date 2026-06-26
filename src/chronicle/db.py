"""Postgres access + a tiny forward-only migration runner.

Migrations are plain `.sql` files in `migrations/`, applied in filename order and
recorded in `schema_migrations` so re-runs are idempotent.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

from .config import get_settings
from .logging_config import get_logger

log = get_logger("chronicle.db")

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "migrations"


def connect(database_url: str | None = None) -> psycopg.Connection:
    url = database_url or get_settings().database_url
    return psycopg.connect(url)


def ping(database_url: str | None = None) -> bool:
    """True if the database answers `SELECT 1`."""
    try:
        with connect(database_url) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone() == (1,)
    except Exception as exc:
        log.debug("DB ping failed: %s", exc)
        return False


def discover_migrations(migrations_dir: Path = MIGRATIONS_DIR) -> list[Path]:
    """All `.sql` migrations, sorted by filename (pure; no DB needed)."""
    return sorted(migrations_dir.glob("*.sql"))


def _ensure_migrations_table(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                filename   text PRIMARY KEY,
                applied_at timestamptz NOT NULL DEFAULT now()
            )
            """
        )


def _applied(conn: psycopg.Connection) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT filename FROM schema_migrations")
        return {row[0] for row in cur.fetchall()}


def migrate(
    database_url: str | None = None,
    migrations_dir: Path = MIGRATIONS_DIR,
) -> list[str]:
    """Apply pending migrations in order. Returns the filenames applied."""
    applied_now: list[str] = []
    with connect(database_url) as conn:
        _ensure_migrations_table(conn)
        already = _applied(conn)
        for path in discover_migrations(migrations_dir):
            if path.name in already:
                continue
            sql = path.read_text(encoding="utf-8")
            log.info("Applying migration %s", path.name)
            with conn.cursor() as cur:
                cur.execute(sql)
                cur.execute(
                    "INSERT INTO schema_migrations (filename) VALUES (%s)",
                    (path.name,),
                )
            conn.commit()
            applied_now.append(path.name)
    return applied_now
