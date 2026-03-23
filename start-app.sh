#!/usr/bin/env bash
set -euo pipefail

HOST="${APP_HOST:-0.0.0.0}"
PORT="${APP_PORT:-8080}"

exec "$(dirname "$0")/.venv/bin/python" -m uvicorn app:app \
  --host "${HOST}" \
  --port "${PORT}"
