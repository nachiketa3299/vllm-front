#!/usr/bin/env bash
# vllm-front 1회 부트스트랩.
# git clone 후 한 번만 실행하면 .venv 생성 + 의존성 설치.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

log() { printf '\n>>> %s\n' "$*"; }

# 1. uv 설치 확인
if ! command -v uv >/dev/null 2>&1; then
  log "uv 미설치 — 설치 진행"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env" 2>/dev/null || export PATH="${HOME}/.local/bin:${PATH}"
fi
log "uv: $(uv --version)"

# 2. .venv 생성 + 의존성
log "venv 생성 + requirements 설치"
uv venv .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 3. 자가 검증
log "self-check"
.venv/bin/python -c "from server.main import app; print('routes:', sorted(getattr(r, 'path', '') for r in app.routes))"

log "install.sh 완료. ./start-app.sh 로 시작."
