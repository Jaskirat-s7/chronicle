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
    name: str = typer.Option(None, "--name", help="Repo identifier (default: from origin remote)."),
) -> None:
    """Index a repo's current snapshot into pgvector (chunk -> embed -> store)."""
    from .embeddings import get_embedder
    from .indexer import index_snapshot
    from .tracing import flush_traces
    from .vector_store import VectorStore

    settings = get_settings()
    configure_logging(settings.log_level)
    repo_name = name or _repo_name(repo)

    embedder = get_embedder(settings)
    store = VectorStore(settings.database_url)
    typer.echo(f"Indexing snapshot of {repo_name} ...")
    res = index_snapshot(repo, repo_name, embedder, store)
    _echo_ok(
        f"indexed {res.chunks_indexed} chunks from {res.files_indexed} files "
        f"@ {res.head_sha[:10]}"
    )
    flush_traces()


@app.command()
def ask(
    question: str = typer.Argument(..., help="A natural-language question about the repo's history."),
    repo: str = typer.Option(None, "--repo", help="Repo name to scope retrieval to."),
    top_k: int = typer.Option(8, "--top-k", help="How many chunks to ground the answer in."),
    as_of: str = typer.Option(None, "--as-of", help="Answer as of this commit/date (PR3+)."),
) -> None:
    """Ask a question; retrieve (hybrid+RRF+rerank) and generate a cited answer."""
    from .answer import answer_question
    from .embeddings import get_embedder
    from .models import get_generation_client
    from .reranker import get_reranker
    from .retrieval import HybridRetriever
    from .tracing import flush_traces
    from .vector_store import VectorStore

    settings = get_settings()
    configure_logging(settings.log_level)
    if as_of:
        typer.secho("--as-of is not supported until PR3 (temporal index).", fg=typer.colors.YELLOW)

    retriever = HybridRetriever(
        VectorStore(settings.database_url),
        get_embedder(settings),
        get_reranker(settings),
    )
    result = answer_question(
        question, retriever, get_generation_client(settings), repo=repo, top_k=top_k
    )
    typer.echo(result.answer)
    typer.echo("")
    typer.secho("Sources:", fg=typer.colors.BLUE)
    for c in result.retrieved:
        typer.echo(
            f"  {c.file_path}:{c.line_start}-{c.line_end} @ {c.commit_sha[:10]} "
            f"{c.commit_date:%Y-%m-%d}"
        )
    flush_traces()


@app.command("eval-run")
def eval_run(
    eval_set: Path = typer.Option(Path("eval/dev_set.jsonl"), "--set", help="Eval set JSONL."),
    out: Path = typer.Option(Path("eval/baseline_report.json"), "--out", help="Report output."),
    top_k: int = typer.Option(8, "--top-k"),
    ragas: bool = typer.Option(False, "--ragas", help="Also compute RAGAS (needs Ollama judge)."),
    save_floor: bool = typer.Option(False, "--save-floor", help="Record this report as the DeepEval floor."),
    floor: Path = typer.Option(Path("eval/floor.json"), "--floor"),
    label: str = typer.Option("baseline", "--label"),
) -> None:
    """Run the eval set through retrieve+generate and write a metrics report."""
    import json

    from .answer import answer_question
    from .embeddings import get_embedder
    from .evalkit import deepeval_gate
    from .evalkit.harness import ragas_samples, run_eval
    from .evalkit.ragas_eval import run_ragas
    from .evalkit.report import build_report, pretty_print
    from .groundtruth import load_jsonl
    from .models import get_generation_client
    from .reranker import get_reranker
    from .retrieval import HybridRetriever
    from .tracing import flush_traces
    from .vector_store import VectorStore

    settings = get_settings()
    configure_logging(settings.log_level)

    questions = load_jsonl(eval_set)
    retriever = HybridRetriever(
        VectorStore(settings.database_url), get_embedder(settings), get_reranker(settings)
    )
    generator = get_generation_client(settings)

    def answer_fn(qtext: str, repo: str):
        return answer_question(qtext, retriever, generator, repo=repo, top_k=top_k)

    typer.echo(f"Running eval over {len(questions)} questions from {eval_set} ...")
    run = run_eval(questions, answer_fn)
    report = build_report(run, label=label)
    if ragas:
        typer.echo("Computing RAGAS (Ollama judge) ...")
        report["ragas"] = run_ragas(ragas_samples(run), settings)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    typer.echo("")
    typer.echo(pretty_print(report))
    typer.echo("")
    _echo_ok(f"wrote report -> {out}")
    if save_floor:
        deepeval_gate.save_floor(report, floor)
        _echo_ok(f"recorded DeepEval floor -> {floor}")
    flush_traces()


@app.command("eval-gate")
def eval_gate(
    report: Path = typer.Option(Path("eval/baseline_report.json"), "--report"),
    floor: Path = typer.Option(Path("eval/floor.json"), "--floor"),
    tolerance: float = typer.Option(0.02, "--tolerance"),
) -> None:
    """Fail (exit 1) if the report regresses any tracked metric below the floor."""
    from .evalkit import deepeval_gate

    cur = deepeval_gate.load_floor(report)
    fl = deepeval_gate.load_floor(floor)
    res = deepeval_gate.compare_to_floor(cur, fl, tolerance=tolerance)
    deepeval_gate.record_to_deepeval(cur, fl)

    typer.echo(f"Checked {len(res.checked)} tracked metric(s) vs floor (tol={tolerance}).")
    if res.passed:
        typer.secho("GATE PASS — no regression below floor.", fg=typer.colors.GREEN)
        return
    for r in res.regressions:
        _echo_fail(f"{r.metric}: floor={r.floor:.3f} current={r.current:.3f}")
    raise typer.Exit(code=1)


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
    repo: list[Path] = typer.Option(..., "--repo", help="Local git repo(s); repeat for multi-repo sets."),
    out: Path = typer.Option(Path("eval"), "--out", help="Output directory for the eval sets."),
    full_size: int = typer.Option(120, "--full-size", help="Total size of the full set (split across repos)."),
    dev_size: int = typer.Option(30, "--dev-size", help="Total size of the per-PR dev set."),
    seed: int = typer.Option(7, "--seed", help="Deterministic sampling seed."),
) -> None:
    """Auto-generate git-verifiable labeled eval questions (no model involved).

    Pass --repo multiple times to build a combined, multi-repo set so the harness
    is not overfit to one project's conventions.
    """
    import json
    from datetime import datetime, timezone

    from .groundtruth import GroundTruthGenerator, stratified_subset, verify_all, write_jsonl

    settings = get_settings()
    configure_logging(settings.log_level)

    repos = list(repo)
    per_repo = max(full_size // len(repos), 1)

    pool: list = []
    repo_entries: list[dict] = []
    for r in repos:
        name = _repo_name(r)
        gen = GroundTruthGenerator(r, name, seed=seed)
        typer.echo(f"Repo: {name} @ {gen.short}  (target {per_repo})")
        sub = gen.generate(per_repo)
        verified = verify_all(r, sub)  # crashes on any mismatch
        _echo_ok(f"  verified {verified}/{len(sub)} questions against git")
        pool.extend(sub)
        repo_entries.append({"repo": name, "eval_sha": gen.eval_sha, "count": len(sub)})

    dev = stratified_subset(pool, dev_size, seed=seed)

    n_full = write_jsonl(pool, out / "full_set.jsonl")
    n_dev = write_jsonl(dev, out / "dev_set.jsonl")

    by_template: dict[str, int] = {}
    by_repo: dict[str, int] = {}
    for q in pool:
        by_template[q.template] = by_template.get(q.template, 0) + 1
        by_repo[q.repo] = by_repo.get(q.repo, 0) + 1
    temporal = sum(1 for q in pool if q.temporal)

    manifest = {
        "repos": repo_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "full_size": n_full,
        "dev_size": n_dev,
        "temporal_count": temporal,
        "non_temporal_count": len(pool) - temporal,
        "by_template": by_template,
        "by_repo": by_repo,
        "all_verified": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    typer.echo("")
    typer.echo(f"Wrote {n_full} -> {out / 'full_set.jsonl'}")
    typer.echo(f"Wrote {n_dev} -> {out / 'dev_set.jsonl'}")
    typer.echo(f"Wrote manifest -> {out / 'manifest.json'}")
    typer.echo("")
    typer.echo("By repo:")
    for r, c in sorted(by_repo.items()):
        typer.echo(f"  {r:24s}: {c}")
    typer.echo("By template:")
    for t, c in sorted(by_template.items()):
        typer.echo(f"  {t:16s}: {c}")
    typer.echo(f"  {'temporal':16s}: {temporal}")
    typer.echo(f"  {'non-temporal':16s}: {len(pool) - temporal}")


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(app())
