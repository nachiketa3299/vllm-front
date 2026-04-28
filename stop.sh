#!/usr/bin/env bash
# vllm-front 종료. 우선 PID 파일 기반, 없으면 패턴으로 백업 종료.
set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

PIDFILE="$DIR/logs/uvicorn.pid"

stopped=0

if [ -f "$PIDFILE" ]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo ">>> SIGTERM → PID $PID"
    kill -TERM "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$PID" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$PID" 2>/dev/null; then
      echo ">>> SIGKILL → PID $PID"
      kill -KILL "$PID" 2>/dev/null || true
    fi
    stopped=1
  fi
  rm -f "$PIDFILE"
fi

# 안전망 — PID 파일 없거나 다른 인스턴스 있으면
if pgrep -f "uvicorn app:app" >/dev/null; then
  echo ">>> 추가로 발견된 uvicorn app:app 종료"
  pkill -TERM -f "uvicorn app:app" || true
  sleep 1
  pkill -KILL -f "uvicorn app:app" 2>/dev/null || true
  stopped=1
fi

if [ "$stopped" = "1" ]; then
  echo ">>> 종료 완료"
else
  echo ">>> 실행 중인 vllm-front 인스턴스 없음"
fi
