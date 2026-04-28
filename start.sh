#!/usr/bin/env bash
# vllm-front 기동 (백그라운드).
# 사용:
#   ./start.sh            # 기본 포트 8081
#   ./start.sh 8090       # 포트 지정
# 종료는 ./stop.sh
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PORT="${1:-${APP_PORT:-8081}}"
HOST="${APP_HOST:-0.0.0.0}"

mkdir -p logs
LOG="$DIR/logs/start.out"
PIDFILE="$DIR/logs/uvicorn.pid"

# 이미 떠있으면 안내하고 종료
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "이미 실행 중 (PID $(cat "$PIDFILE")). 먼저 ./stop.sh"
  exit 1
fi

# venv 점검
if [ ! -x "$DIR/.venv/bin/python" ]; then
  echo "[ERROR] .venv 없음. 먼저 ./install.sh"
  exit 1
fi

nohup "$DIR/.venv/bin/python" -m uvicorn app:app \
  --host "$HOST" --port "$PORT" \
  > "$LOG" 2>&1 &

PID=$!
echo "$PID" > "$PIDFILE"

echo ">>> vllm-front 기동 (PID $PID)"
echo "    bind: $HOST:$PORT"
echo "    log:  $LOG"
echo "    stop: ./stop.sh"
echo "    health: curl http://localhost:$PORT/health"
