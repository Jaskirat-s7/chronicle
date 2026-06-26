#!/usr/bin/env bash
# PR0 acceptance gate, end to end. Run from the repo root inside the venv.
#
#   1. `chron --help` works
#   2. test suite passes
#   3. migrations apply (needs `docker compose up -d db`)
#   4. doctor: DB + traced local-judge round-trip + traced Gemini round-trip
#      (needs Ollama running with JUDGE_MODEL pulled, GOOGLE_API_KEY set, and
#       Langfuse keys set to actually see traced spans)
set -euo pipefail

echo "== 1. chron --help =="
chron --help >/dev/null && echo "OK"

echo "== 2. pytest =="
pytest

echo "== 3. chron migrate =="
chron migrate

echo "== 4. chron doctor =="
chron doctor

echo ""
echo "PR0 gate complete. Confirm in Langfuse ($LANGFUSE_HOST) that one Gemini"
echo "and one Ollama generation appear with token + latency + call-count."
