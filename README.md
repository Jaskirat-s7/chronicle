# chronicle

Time-aware retrieval over a git repo's **history** — RAG on a timeline, not a
snapshot. Normal RAG treats a codebase as its current state; chronicle treats it
as a timeline, so it can answer *when* code changed, *why* it changed, and what
it looked like *before* — citing commit + date + PR, not just file:line.

> **Status: PR2 — baseline retrieval + RAGAS/DeepEval.** Snapshot-only retrieval
> (no temporal index yet) wired to the eval harness, so PR3+ has a measured floor
> to beat. The problem-first, headline-number README lands in PR9.

---

## What's in PR0

- Repo scaffold, `pyproject`, and the `chron` CLI skeleton (`index`, `ask`,
  `migrate`, `doctor`).
- Postgres + **pgvector** via Docker compose; base **snapshot** schema migration.
- A pluggable `ModelClient` with two backends: **Gemini** (generation) and local
  **Ollama** (eval judge), with the Ollama base URL read from `OLLAMA_BASE_URL`.
- **Langfuse** (self-hosted v2) wired so every LLM call is a traced generation
  span with token + call counts, latency, and cost.

### Locked decisions (baked into PR0)

| Decision | Choice |
| --- | --- |
| Embeddings | `Qwen/Qwen3-Embedding-0.6B`, served via sentence-transformers on MPS (PR2) |
| Vector dim | **1024, locked** — `vector(1024)`, no Matryoshka truncation in baseline |
| Lexical side | Postgres FTS (`tsvector`) |
| Reranker | `BAAI/bge-reranker-v2-m3`, local (PR2) |
| Base schema | snapshot only (content + SHA + date + file/line); temporal columns are a separate PR3 migration |
| Git access | `subprocess git`, strict-parse-or-crash (pygit2 reconsidered only at PR5) |
| Judge model | `JUDGE_MODEL`, default `qwen2.5:3b` (env-swappable to `7b` for the final report) |
| Machine | all-local on the Mac (M4 Air); `OLLAMA_BASE_URL` defaults to localhost |

---

## PR1: auto ground-truth eval set (no model involved)

Git knows which commit introduced or last changed any line, so we auto-generate
**labeled, git-verifiable** eval questions — and the generator self-checks every
question with an independent verifier before writing it (`all_verified: true` in
the manifest). Five templates:

| Template | Question | Answer | Temporal? |
| --- | --- | --- | --- |
| `blame_commit` | which commit last modified line N of a file | full SHA | yes |
| `commit_date` | on what UTC date was line N last changed | YYYY-MM-DD | yes |
| `file_added` | which commit first added a file (follows renames) | full SHA | yes |
| `blame_author` | who authored the last change to line N | author name | no |
| `pr_for_commit` | which PR introduced a commit | PR number | no |

Two tiers are produced (per the eval-set tiering decision): a small **dev set**
(per-PR DeepEval gate) and the **full set** (milestones / final report).

```bash
# clone the target repo(s) locally, then (repeat --repo for a multi-repo set):
chron eval-gen --repo /path/to/flask --repo /path/to/requests \
  --out eval --full-size 160 --dev-size 40
```

Outputs `eval/full_set.jsonl`, `eval/dev_set.jsonl`, and `eval/manifest.json`
(pins each repo + the HEAD SHA the set is anchored at, for reproducibility).
Inspect a parsed history without generating via `chron ingest --repo /path/to/flask`.

**Locked eval target: `pallets/flask` + `psf/requests`** — a multi-repo set so
the harness is not overfit to one project's commit/PR conventions. Both have rich
history, `(#NNNN)` PR links, and real renames/refactors. The committed `eval/`
set is **160 questions (80 per repo), all git-verified**. (A third candidate,
`pallets/click`, is smaller and good for fast iteration if needed.)

## PR2: baseline retrieval + eval harness

The **snapshot-only** baseline (no history yet — that's PR3): every chunk is
tagged with the HEAD commit, so "when did this change" is expected to score low.
That is the point — it's the floor the temporal index must beat.

Pipeline: `chunk` (overlapping line windows) → `embed` (Qwen3-Embedding-0.6B,
1024-d) → store in **pgvector** → **hybrid** retrieve (vector `<=>` cosine +
Postgres FTS `ts_rank`) → **RRF** fuse → **bge-reranker-v2-m3** → generate a
cited answer via **Gemini**.

Metrics (deterministic, git-derived ground truth): per-template **answer
accuracy** (e.g. does the answer contain the gold SHA), **hit@k**, **MRR**,
**nDCG**, split temporal vs non-temporal. Plus **RAGAS** (faithfulness, answer
relevancy, context precision/recall; judge on local Ollama) and
cost/latency/calls. **DeepEval** records the regression floor; later PRs fail the
gate if they drop a tracked metric below floor − tolerance.

```bash
# 0. extras for the live path
pip install -e ".[dev,embed,eval]"
docker compose up -d && chron migrate          # 0001 base + 0002 HNSW index
ollama pull qwen2.5:3b                          # RAGAS judge

# 1. index the snapshot of each locked eval repo
chron index --repo /path/to/flask
chron index --repo /path/to/requests

# 2. baseline report over the dev set, compute RAGAS, record the floor
chron eval-run --set eval/dev_set.jsonl --ragas --save-floor

# 3. the merge gate (later PRs must not regress below the floor)
chron eval-gate
```

`eval-run` writes `eval/baseline_report.json`; `--save-floor` writes
`eval/floor.json`. Use `--set eval/full_set.jsonl` for milestone runs.

## Prerequisites

- **Python 3.10–3.12** (this machine has 3.11; the default `python3` is 3.14 —
  use 3.11 for the venv).
- **Docker** (Desktop or colima) for Postgres + Langfuse.
- **Ollama** running locally with the judge model pulled:
  `ollama pull qwen2.5:3b`.
- A **Gemini API key** (free tier): https://aistudio.google.com/apikey.

## Quickstart

```bash
# 1. Virtualenv on Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Config
cp .env.example .env          # then edit: GOOGLE_API_KEY, and (after step 4) Langfuse keys

# 3. Bring up Postgres + Langfuse
docker compose up -d
chron migrate                 # applies the base schema

# 4. Langfuse keys: open http://localhost:3000, create an account + project,
#    Settings -> API Keys, paste LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY into .env

# 5. Run the gate
chron doctor
```

## The PR0 gate

`scripts/pr0_gate.sh` runs the whole thing; or check each piece:

1. **`chron --help` works** and a **smoke test passes** — `pytest`.
2. **One Gemini call and one Ollama call are traced in Langfuse** with
   cost/latency/call-count — run `chron doctor`, then open
   `http://localhost:3000` and confirm two generation spans.
3. **Local judge round-trip** — `chron doctor` hits Ollama `/api/tags` on
   localhost and runs one judge inference, traced. (The earlier hybrid Mac+PC
   network-reachability proof was dropped: everything is local now.)

## Testing

```bash
pytest            # unit tests run with no DB/network
```

The live migration test (`test_migrate_is_idempotent`) auto-skips unless
Postgres is reachable.

## Project layout

```
src/chronicle/
  cli.py            # `chron` commands: doctor, migrate, ingest, eval-gen, ...
  config.py         # env/.env settings; OLLAMA_BASE_URL, JUDGE_MODEL, locked dim
  db.py             # psycopg connection + forward-only migration runner
  tracing.py        # Langfuse generation spans (best-effort, no-op if unconfigured)
  git_ops.py        # strict subprocess git wrappers (blame, log, first-add, ...)
  diff_parser.py    # strict unified-diff parser (crash on malformed input)
  ingest.py         # walk history -> structured commits + per-file diffs
  chunking.py       # snapshot line-window chunker (baseline)
  embeddings.py     # Qwen3-Embedding via sentence-transformers (lazy)
  reranker.py       # bge-reranker-v2-m3 cross-encoder (lazy)
  vector_store.py   # pgvector dense + Postgres FTS search
  retrieval.py      # pure rrf_fuse + HybridRetriever (fuse -> rerank)
  answer.py         # retrieve -> cited prompt -> Gemini
  indexer.py        # snapshot indexing pipeline
  groundtruth/
    schema.py       # EvalQuestion + JSONL (de)serialization
    generator.py    # seeded, git-derived question generation
    verifier.py     # independent re-derivation; mismatch is fatal
  evalkit/
    metrics.py      # hit@k, MRR, nDCG, answer-match (deterministic)
    harness.py      # run eval set -> per-question results
    report.py       # aggregate metrics + cost/latency/calls
    ragas_eval.py   # RAGAS via local Ollama judge (lazy)
    deepeval_gate.py# regression-floor gate (pure comparator)
  models/
    base.py         # ModelClient protocol + LLMResponse
    gemini.py       # generation backend (lazy google-genai import)
    ollama.py       # judge backend (httpx)
migrations/0001_base_schema.sql
eval/               # committed dev_set/full_set/manifest (from pallets/flask)
docker-compose.yml  # pgvector + Langfuse v2 (+ its own Postgres)
tests/              # incl. a real temp git repo fixture (no mocking)
```

## Operational notes (single Mac)

From PR2, the embedding model, reranker, and the Ollama judge all share the
fanless M4 Air alongside Postgres + Langfuse in Docker. Eval keeps judge calls
sequential/throttled; expect memory pressure during full-set runs — that's
operational, not something the code works around.

## Roadmap

PR1 git ingestion + auto ground-truth (eval first) · PR2 baseline retrieval +
RAGAS/DeepEval · PR3 temporal index · PR4 temporal-intent router · PR5
symbol-history across renames · PR6 LangGraph agentic "why" loop · PR7 temporal
citations + contradiction handling · PR8 FastAPI + Docker · PR9 report.
