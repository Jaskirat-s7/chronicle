"""`chron` CLI.

PR0 ships the skeleton: `version`, `migrate`, `doctor`, plus `index`/`ask`
stubs whose real behavior lands in later PRs. `doctor` is the PR0 gate made
runnable — it proves DB connectivity, a local Ollama judge round-trip, and a
Gemini round-trip, each traced in Langfuse.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from . import __version__
from .config import get_settings
from .logging_config import configure_logging, get_logger

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="chronicle — time-aware retrieval over a git repo's history.",
)
log = get_logger("chronicle.cli")


def _echo_ok(msg: str) -> None:
    typer.secho(f"  OK   {msg}", fg=typer.colors.GREEN)


def _echo_fail(msg: str) -> None:
    typer.secho(f" FAIL  {msg}", fg=typer.colors.RED)


def _echo_skip(msg: str) -> None:
    typer.secho(f" SKIP  {msg}", fg=typer.colors.YELLOW)


@app.command()
def version() -> None:
    """Print the chronicle version."""
    typer.echo(__version__)


@app.command()
def migrate() -> None:
    """Apply pending database migrations (base schema)."""
    from . import db

    settings = get_settings()
    configure_logging(settings.log_level)
    applied = db.migrate(settings.database_url)
    if applied:
        for name in applied:
            _echo_ok(f"applied {name}")
    else:
        typer.echo("No pending migrations.")


@app.command()
def doctor() -> None:
    """Run PR0 connectivity + traced-round-trip checks (the gate)."""
    from . import db
    from .models import get_generation_client, get_judge_client
    from .tracing import flush_traces

    settings = get_settings()
    configure_logging(settings.log_level)

    typer.echo("chronicle doctor — PR0 gate")
    typer.echo(f"  database_url     : {settings.database_url}")
    typer.echo(f"  ollama_base_url  : {settings.ollama_base_url}")
    typer.echo(f"  judge_model      : {settings.judge_model}")
    typer.echo(f"  gen_model        : {settings.gen_model}")
    typer.echo(f"  langfuse_host    : {settings.langfuse_host}")
    typer.echo(f"  langfuse_enabled : {settings.langfuse_enabled}")
    typer.echo(f"  gemini_enabled   : {settings.gemini_enabled}")
    typer.echo("")

    failures = 0

    # 1) Postgres / pgvector
    if db.ping(settings.database_url):
        _echo_ok("Postgres reachable (SELECT 1).")
    else:
        _echo_fail("Postgres not reachable — is `docker compose up -d db` running and migrated?")
        failures += 1

    # 2) Langfuse config (spans are sent during the round-trips below)
    if settings.langfuse_enabled:
        _echo_ok("Langfuse keys present — LLM calls below will be traced.")
    else:
        _echo_skip("Langfuse keys missing — round-trips run but are NOT traced.")

    # 3) Ollama connectivity + local judge round-trip (traced)
    judge = get_judge_client(settings)
    try:
        tags = judge.tags()
        names = [m.get("name") for m in tags.get("models", [])]
        _echo_ok(f"Ollama /api/tags reachable — models: {names or '(none pulled)'}")
    except Exception as exc:
        _echo_fail(f"Ollama /api/tags failed: {exc}")
        failures += 1

    try:
        resp = judge.complete("Reply with exactly one word: pong")
        _echo_ok(
            f"Judge round-trip [{resp.model}] -> {resp.text.strip()!r} "
            f"({resp.input_tokens}+{resp.output_tokens} tok, {resp.latency_ms:.0f} ms)"
        )
    except Exception as exc:
        _echo_fail(f"Judge round-trip failed: {exc}")
        failures += 1

    # 4) Gemini generation round-trip (traced)
    if settings.gemini_enabled:
        try:
            gen = get_generation_client(settings)
            resp = gen.complete("Reply with exactly one word: pong")
            _echo_ok(
                f"Gemini round-trip [{resp.model}] -> {resp.text.strip()!r} "
                f"({resp.input_tokens}+{resp.output_tokens} tok, {resp.latency_ms:.0f} ms)"
            )
        except Exception as exc:
            _echo_fail(f"Gemini round-trip failed: {exc}")
            failures += 1
    else:
        _echo_skip("Gemini round-trip — GOOGLE_API_KEY not set.")

    flush_traces()

    typer.echo("")
    if failures:
        _echo_fail(f"{failures} check(s) failed.")
        raise typer.Exit(code=1)
    typer.secho("All checks passed.", fg=typer.colors.GREEN)
    if settings.langfuse_enabled:
        typer.echo(f"View traces at {settings.langfuse_host}")


@app.command()
def index(
    repo: Path = typer.Option(..., "--repo", help="Path to the target git repository."),
) -> None:
    """Ingest a repo's history into the index. (Implemented in PR1/PR2.)"""
    typer.secho(
        f"`chron index` is a PR0 skeleton — git ingestion lands in PR1, "
        f"snapshot indexing in PR2. (requested repo: {repo})",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=0)


@app.command()
def ask(
    question: str = typer.Argument(..., help="A natural-language question about the repo's history."),
    as_of: str = typer.Option(None, "--as-of", help="Answer as of this commit/date (PR3+)."),
) -> None:
    """Ask a question about the repo's history. (Implemented in PR2+.)"""
    typer.secho(
        "`chron ask` is a PR0 skeleton — retrieval + generation land in PR2. "
        f"(question: {question!r}, as_of: {as_of!r})",
        fg=typer.colors.YELLOW,
    )
    raise typer.Exit(code=0)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
