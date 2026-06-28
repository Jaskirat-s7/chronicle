# chronicle

Time-aware retrieval over a git repo's **history** — RAG on a timeline, not a
snapshot. Normal RAG treats a codebase as its current state; chronicle treats it
as a timeline, so it can answer *when* code changed, *why* it changed, and what
it looked like *before* — citing commit + date + PR, not just file:line.

> **Status: PR1 — git ingestion + auto ground-truth (eval first).** The eval
> harness ground-truth comes before any retrieval cleverness. The problem-first,
> headline-number README lands in PR9.

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
# clone a target repo somewhere local, then:
chron eval-gen --repo /path/to/flask --out eval --full-size 120 --dev-size 30
```

Outputs `eval/full_set.jsonl`, `eval/dev_set.jsonl`, and `eval/manifest.json`
(pins repo + the HEAD SHA the set is anchored at, for reproducibility). Inspect
a parsed history without generating with `chron ingest --repo /path/to/flask`.

**Candidate target repos** (rich history, GitHub `(#NNNN)` squash-merge PR links,
real renames/refactors, fast to clone):

1. **`pallets/flask`** — clean squash-merge history, well-linked PRs, ~5.5k
   commits. *Used for the committed `eval/` set in this PR.*
2. **`psf/requests`** — very rich history, classic renames/refactors.
3. **`pallets/click`** — smaller and tidy; good for fast iteration.

The committed set is from `pallets/flask` as a default — say which repo to
**lock** as the canonical eval target and I'll regenerate.

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
  groundtruth/
    schema.py       # EvalQuestion + JSONL (de)serialization
    generator.py    # seeded, git-derived question generation
    verifier.py     # independent re-derivation; mismatch is fatal
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
