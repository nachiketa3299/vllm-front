#!/usr/bin/env bash
# 어디서 호출하든 vllm-front 디렉터리 기준으로 동작하도록 cd 후 실행.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8080}"

exec "$DIR/.venv/bin/python" -m uvicorn app:app \
  --host "${HOST}" \
  --port "${PORT}"
