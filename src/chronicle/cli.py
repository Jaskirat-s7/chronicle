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


def _repo_name(repo: Path) -> str:
    """Derive an 'owner/repo' identifier from the origin remote, else dir name."""
    from . import git_ops

    try:
        url = git_ops.run_git(repo, ["remote", "get-url", "origin"]).strip()
    except Exception:
        return repo.resolve().name
    url = url.removesuffix(".git")
    parts = url.replace(":", "/").split("/")
    if len(parts) >= 2:
        return "/".join(parts[-2:])
    return repo.resolve().name


@app.command()
def ingest(
    repo: Path = typer.Option(..., "--repo", help="Path to a local git repository."),
    max_commits: int = typer.Option(200, "--max-commits", help="How many commits to walk."),
) -> None:
    """Walk history and strictly parse commits + diffs; print ingestion stats."""
    from . import ingest as ingest_mod

    settings = get_settings()
    configure_logging(settings.log_level)

    changes = ingest_mod.walk_history(repo, max_commits=max_commits)
    stats = ingest_mod.summarize(changes)
    typer.echo(f"Ingested {_repo_name(repo)} (first {max_commits} commits):")
    typer.echo(f"  commits parsed     : {stats.commits}")
    typer.echo(f"  merges (meta only) : {stats.merges_skipped}")
    typer.echo(f"  commits with PR ref: {stats.commits_with_pr}")
    typer.echo(f"  files touched      : {stats.files_touched}")
    typer.echo(f"  renames            : {stats.renames}")
    typer.echo(f"  binary files       : {stats.binary_files}")
    typer.echo(f"  hunks              : {stats.hunks}")
    typer.echo(f"  lines +/-          : +{stats.lines_added} / -{stats.lines_removed}")


@app.command("eval-gen")
def eval_gen(
    repo: Path = typer.Option(..., "--repo", help="Path to a local git repository."),
    out: Path = typer.Option(Path("eval"), "--out", help="Output directory for the eval sets."),
    full_size: int = typer.Option(120, "--full-size", help="Size of the milestone/full set."),
    dev_size: int = typer.Option(30, "--dev-size", help="Size of the per-PR dev set."),
    seed: int = typer.Option(7, "--seed", help="Deterministic sampling seed."),
) -> None:
    """Auto-generate git-verifiable labeled eval questions (no model involved)."""
    import json
    from datetime import datetime, timezone

    from .groundtruth import GroundTruthGenerator, verify_all, write_jsonl

    settings = get_settings()
    configure_logging(settings.log_level)

    repo_name = _repo_name(repo)
    gen = GroundTruthGenerator(repo, repo_name, seed=seed)
    typer.echo(f"Repo: {repo_name} @ {gen.short}  (anchoring eval at this HEAD)")

    pool = gen.generate(full_size)
    if len(pool) < full_size:
        typer.secho(
            f"WARNING: only generated {len(pool)} questions (< {full_size}); "
            "repo may be small.",
            fg=typer.colors.YELLOW,
        )

    verified = verify_all(repo, pool)  # crashes on any mismatch
    _echo_ok(f"verified {verified}/{len(pool)} questions against git")

    dev = gen.stratified_subset(pool, dev_size)

    n_full = write_jsonl(pool, out / "full_set.jsonl")
    n_dev = write_jsonl(dev, out / "dev_set.jsonl")

    by_template: dict[str, int] = {}
    for q in pool:
        by_template[q.template] = by_template.get(q.template, 0) + 1
    temporal = sum(1 for q in pool if q.temporal)

    manifest = {
        "repo": repo_name,
        "eval_sha": gen.eval_sha,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "full_size": n_full,
        "dev_size": n_dev,
        "temporal_count": temporal,
        "non_temporal_count": len(pool) - temporal,
        "by_template": by_template,
        "all_verified": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    typer.echo("")
    typer.echo(f"Wrote {n_full} -> {out / 'full_set.jsonl'}")
    typer.echo(f"Wrote {n_dev} -> {out / 'dev_set.jsonl'}")
    typer.echo(f"Wrote manifest -> {out / 'manifest.json'}")
    typer.echo("")
    typer.echo("By template:")
    for t, c in sorted(by_template.items()):
        typer.echo(f"  {t:16s}: {c}")
    typer.echo(f"  {'temporal':16s}: {temporal}")
    typer.echo(f"  {'non-temporal':16s}: {len(pool) - temporal}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
