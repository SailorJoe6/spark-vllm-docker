#!/bin/bash
#
# Runtime smoke test for the Qwen3-Embedding-8B vLLM recipe.
#
# Start the server first, for example:
#   ./run-recipe.sh qwen3-embedding-8b --solo
#
# Then run:
#   ./tests/smoke_qwen3_embedding_8b.sh

set -euo pipefail

BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8888/v1}"
MODEL="${VLLM_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
EXPECTED_DIMS="${EXPECTED_EMBEDDING_DIMS:-4096}"

payload=$(cat <<JSON
{
  "model": "$MODEL",
  "input": [
    "The capital of China is Beijing.",
    "Gravity attracts two bodies toward each other."
  ]
}
JSON
)

response="$(curl -fsS "$BASE_URL/embeddings" \
  -H "Content-Type: application/json" \
  -d "$payload")"

printf '%s' "$response" | EXPECTED_DIMS="$EXPECTED_DIMS" python3 -c '
import json
import os
import sys

response = json.load(sys.stdin)
expected = int(os.environ["EXPECTED_DIMS"])

try:
    embedding = response["data"][0]["embedding"]
except (KeyError, IndexError, TypeError) as exc:
    print(f"Invalid embeddings response shape: {exc}", file=sys.stderr)
    print(json.dumps(response)[:1000], file=sys.stderr)
    sys.exit(1)

actual = len(embedding)
if actual != expected:
    print(f"Expected {expected} dimensions, got {actual}", file=sys.stderr)
    sys.exit(1)

print(f"Qwen3-Embedding-8B smoke passed: {actual} dimensions")
'
