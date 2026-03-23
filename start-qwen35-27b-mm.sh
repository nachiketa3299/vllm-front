#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/home/mnnh-ruzen/Documents/projects/vllm-server"
MODEL_DIR="/home/mnnh-ruzen/Documents/projects/qwen3.5-27b"

HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

exec "${BASE_DIR}/.venv/bin/python" -m vllm.entrypoints.cli.main serve "${MODEL_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --reasoning-parser qwen3 \
  "$@"
