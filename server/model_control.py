import asyncio
from collections import deque
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any, Awaitable, Callable, Optional

import httpx

from .config import AppConfig, AppPaths
from .models import AppError, ModelRuntimeStatus, ProbedModelInfo


ProbeFn = Callable[[], Awaitable[Optional[ProbedModelInfo]]]
SleepFn = Callable[[float], Awaitable[None]]
LaunchFn = Callable[[list[str], dict[str, str], Path, Path], int]


class ModelControlService:
    def __init__(
        self,
        *,
        config: AppConfig,
        paths: AppPaths,
        probe_fn: ProbeFn | None = None,
        sleep_fn: SleepFn | None = None,
        launch_fn: LaunchFn | None = None,
        monotonic_fn: Callable[[], float] | None = None,
    ):
        self.config = config
        self.paths = paths
        self._probe_fn = probe_fn or self._probe_model
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._launch_fn = launch_fn or self._launch_process
        self._monotonic_fn = monotonic_fn or time.monotonic
        self._lock = asyncio.Lock()
        self._startup_task: asyncio.Task[None] | None = None
        self._shutdown_task: asyncio.Task[None] | None = None
        self._last_error_detail: str | None = None

        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)

    async def get_runtime_status(self) -> ModelRuntimeStatus:
        async with self._lock:
            return await self._collect_runtime_status_locked()

    async def ensure_generation_available(self) -> None:
        status = await self.get_runtime_status()
        if status.status in {"running_managed", "running_unmanaged"}:
            return
        raise AppError(
            503,
            "모델이 준비되지 않았습니다. 먼저 모델을 켜고 준비 완료 상태가 될 때까지 기다리세요.",
        )

    async def normalize_generation_error(self, error: AppError) -> AppError:
        if error.status_code != 502:
            return error
        status = await self.get_runtime_status()
        if status.status not in {"running_managed", "running_unmanaged"}:
            return AppError(
                503,
                "모델이 종료되었거나 아직 준비되지 않아 요청이 중단되었습니다.",
            )
        return error

    async def start(self, *, max_model_len: int) -> ModelRuntimeStatus:
        if max_model_len <= 0:
            raise AppError(400, "max_model_len must be a positive integer.")

        async with self._lock:
            current = await self._collect_runtime_status_locked()
            if not current.can_start:
                raise AppError(
                    409,
                    f"모델을 시작할 수 없는 상태입니다: {current.status}",
                )

            self._last_error_detail = None
            pid = self._launch_fn(
                self._build_command(max_model_len=max_model_len),
                self._build_child_env(),
                self.paths.base_dir,
                self._log_file_path(),
            )
            self._write_state(
                {
                    "pid": pid,
                    "pgid": pid,
                    "model": self.config.model,
                    "host": self.config.vllm_host,
                    "port": self.config.vllm_port,
                    "requested_max_model_len": max_model_len,
                    "phase": "starting",
                    "command": self._build_command(max_model_len=max_model_len),
                    "started_at": self._monotonic_fn(),
                }
            )
            self._startup_task = asyncio.create_task(self._monitor_startup(pid))
            return await self._collect_runtime_status_locked()

    async def stop(self) -> ModelRuntimeStatus:
        async with self._lock:
            current = await self._collect_runtime_status_locked()
            if current.status == "running_unmanaged":
                raise AppError(
                    409,
                    "앱이 시작한 모델 프로세스가 아니라서 웹에서 종료할 수 없습니다.",
                )
            if current.status not in {"running_managed", "starting", "error"}:
                raise AppError(
                    409,
                    f"모델을 종료할 수 없는 상태입니다: {current.status}",
                )

            state = self._read_state()
            if state is None:
                raise AppError(409, "관리 중인 모델 프로세스 정보를 찾을 수 없습니다.")

            state["phase"] = "stopping"
            self._write_state(state)
            self._signal_process_group(state["pgid"], signal.SIGTERM)
            self._shutdown_task = asyncio.create_task(
                self._monitor_shutdown(state["pid"], state["pgid"])
            )
            return await self._collect_runtime_status_locked()

    async def _collect_runtime_status_locked(self) -> ModelRuntimeStatus:
        state = self._read_state()
        probe = await self._probe_fn()

        if state is not None and not self._managed_process_matches(state):
            detail = self._with_log_tail("관리 중인 모델 프로세스가 이미 종료되었습니다.")
            self._clear_state()
            self._last_error_detail = detail
            state = None

        if state is not None:
            phase = state.get("phase")
            if phase == "stopping":
                return self._build_status(
                    status="stopping",
                    ownership="managed",
                    model=state.get("model"),
                    pid=state.get("pid"),
                    current_max_model_len=state.get("requested_max_model_len"),
                    detail="모델을 종료하는 중입니다.",
                )
            if probe is not None:
                if state.get("phase") != "running":
                    state["phase"] = "running"
                    self._write_state(state)
                self._last_error_detail = None
                return self._build_status(
                    status="running_managed",
                    ownership="managed",
                    model=probe.model or state.get("model"),
                    pid=state.get("pid"),
                    current_max_model_len=probe.max_model_len
                    or state.get("requested_max_model_len"),
                    detail="앱이 시작한 모델이 실행 중입니다.",
                )

            if phase == "starting":
                return self._build_status(
                    status="starting",
                    ownership="managed",
                    model=state.get("model"),
                    pid=state.get("pid"),
                    current_max_model_len=state.get("requested_max_model_len"),
                    detail="모델을 시작하는 중입니다.",
                )

            detail = self._with_log_tail("관리 중인 모델 프로세스가 응답하지 않습니다.")
            self._last_error_detail = detail
            return self._build_status(
                status="error",
                ownership="managed",
                model=state.get("model"),
                pid=state.get("pid"),
                current_max_model_len=state.get("requested_max_model_len"),
                detail=detail,
            )

        if probe is not None:
            self._last_error_detail = None
            return self._build_status(
                status="running_unmanaged",
                ownership="unmanaged",
                model=probe.model,
                pid=None,
                current_max_model_len=probe.max_model_len,
                detail="외부에서 시작된 모델이 실행 중입니다.",
            )

        if self._last_error_detail:
            return self._build_status(
                status="error",
                ownership="none",
                model=None,
                pid=None,
                current_max_model_len=None,
                detail=self._last_error_detail,
            )

        return self._build_status(
            status="stopped",
            ownership="none",
            model=None,
            pid=None,
            current_max_model_len=None,
            detail="모델이 꺼져 있습니다.",
        )

    async def _monitor_startup(self, pid: int) -> None:
        deadline = self._monotonic_fn() + self.config.vllm_startup_timeout_seconds
        try:
            while self._monotonic_fn() < deadline:
                async with self._lock:
                    state = self._read_state()
                    if state is None or state.get("pid") != pid:
                        return
                    if state.get("phase") != "starting":
                        return
                    if not self._managed_process_matches(state):
                        self._last_error_detail = self._with_log_tail(
                            "모델 프로세스가 시작 도중 종료되었습니다."
                        )
                        self._clear_state()
                        return

                    probe = await self._probe_fn()
                    if probe is not None:
                        state["phase"] = "running"
                        self._write_state(state)
                        self._last_error_detail = None
                        return
                await self._sleep_fn(2)

            async with self._lock:
                state = self._read_state()
                if state is None or state.get("pid") != pid:
                    return
                self._signal_process_group(state["pgid"], signal.SIGTERM)
                self._last_error_detail = self._with_log_tail(
                    "모델 시작 시간이 초과되었습니다."
                )
                self._clear_state()
        finally:
            async with self._lock:
                if self._startup_task is asyncio.current_task():
                    self._startup_task = None

    async def _monitor_shutdown(self, pid: int, pgid: int) -> None:
        deadline = self._monotonic_fn() + self.config.vllm_shutdown_grace_seconds
        try:
            while self._monotonic_fn() < deadline:
                if not self._pid_exists(pid):
                    async with self._lock:
                        state = self._read_state()
                        if state is not None and state.get("pid") == pid:
                            self._clear_state()
                        self._last_error_detail = None
                    return
                await self._sleep_fn(0.5)

            self._signal_process_group(pgid, signal.SIGKILL)
            await self._sleep_fn(0.5)
            async with self._lock:
                state = self._read_state()
                if state is not None and state.get("pid") == pid:
                    self._clear_state()
                self._last_error_detail = None
        finally:
            async with self._lock:
                if self._shutdown_task is asyncio.current_task():
                    self._shutdown_task = None

    def _build_status(
        self,
        *,
        status: str,
        ownership: str,
        model: str | None,
        pid: int | None,
        current_max_model_len: int | None,
        detail: str | None,
    ) -> ModelRuntimeStatus:
        theoretical_max_model_len = self._read_theoretical_max_model_len()
        observed_kv_cache_tokens = self._read_observed_kv_cache_tokens()
        recommended_max_model_len, recommendation_reason = self._recommend_max_model_len(
            theoretical_max_model_len=theoretical_max_model_len,
            observed_kv_cache_tokens=observed_kv_cache_tokens,
        )
        can_start = status in {"stopped", "error"}
        can_stop = status in {"starting", "running_managed"} or (
            status == "error" and ownership == "managed" and pid is not None
        )
        return ModelRuntimeStatus(
            status=status,
            ownership=ownership,
            model=model,
            pid=pid,
            current_max_model_len=current_max_model_len,
            default_max_model_len=self.config.vllm_default_max_model_len,
            theoretical_max_model_len=theoretical_max_model_len,
            observed_kv_cache_tokens=observed_kv_cache_tokens,
            recommended_max_model_len=recommended_max_model_len,
            recommended_max_model_len_reason=recommendation_reason,
            detail=detail,
            can_start=can_start,
            can_stop=can_stop,
        )

    async def _probe_model(self) -> Optional[ProbedModelInfo]:
        endpoint = f"{self.config.vllm_base_url.rstrip('/')}/models"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(endpoint)
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError):
            return None

        models = payload.get("data")
        if not isinstance(models, list) or not models:
            return ProbedModelInfo(model=None, max_model_len=None)

        first = models[0] if isinstance(models[0], dict) else {}
        raw_len = first.get("max_model_len")
        try:
            max_model_len = int(raw_len) if raw_len is not None else None
        except (TypeError, ValueError):
            max_model_len = None

        model = first.get("id")
        return ProbedModelInfo(
            model=model if isinstance(model, str) else None,
            max_model_len=max_model_len,
        )

    def _launch_process(
        self,
        command: list[str],
        env: dict[str, str],
        cwd: Path,
        log_path: Path,
    ) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(cwd),
                env=env,
                stdout=log_file,
                stderr=log_file,
                start_new_session=True,
            )
        return process.pid

    def _build_command(self, *, max_model_len: int) -> list[str]:
        python_bin = self.paths.base_dir / ".venv" / "bin" / "python"
        if not python_bin.exists():
            python_bin = Path(sys.executable)
        return [
            str(python_bin),
            "-m",
            "vllm.entrypoints.cli.main",
            "serve",
            self.config.model,
            "--host",
            self.config.vllm_host,
            "--port",
            str(self.config.vllm_port),
            "--max-model-len",
            str(max_model_len),
            "--reasoning-parser",
            self.config.vllm_reasoning_parser,
        ]

    def _build_child_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def _state_file_path(self) -> Path:
        return self.paths.runtime_dir / "managed-vllm.json"

    def _log_file_path(self) -> Path:
        return self.paths.runtime_dir / "managed-vllm.log"

    def _read_state(self) -> dict[str, Any] | None:
        path = self._state_file_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) else None

    def _write_state(self, payload: dict[str, Any]) -> None:
        self._state_file_path().write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _clear_state(self) -> None:
        with contextlib.suppress(FileNotFoundError):
            self._state_file_path().unlink()

    def _managed_process_matches(self, state: dict[str, Any]) -> bool:
        pid = state.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            return False
        if not self._pid_exists(pid):
            return False

        cmdline = self._read_cmdline(pid)
        required_parts = [
            "vllm.entrypoints.cli.main",
            "serve",
            self.config.model,
            "--host",
            self.config.vllm_host,
            "--port",
            str(self.config.vllm_port),
        ]
        return all(part in cmdline for part in required_parts)

    def _read_cmdline(self, pid: int) -> list[str]:
        try:
            raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return []
        return [part for part in raw.decode("utf-8", errors="ignore").split("\x00") if part]

    def _pid_exists(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _signal_process_group(self, pgid: int, sig: signal.Signals) -> None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(pgid, sig)

    def _with_log_tail(self, message: str) -> str:
        tail = self._read_log_tail()
        if not tail:
            return message
        return f"{message}\n\n[last log lines]\n{tail}"

    def _read_log_tail(self) -> str:
        path = self._log_file_path()
        if not path.exists():
            return ""
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return "\n".join(deque(lines, maxlen=20))

    def _read_theoretical_max_model_len(self) -> int:
        config_path = Path(self.config.model) / "config.json"
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return self.config.vllm_default_max_model_len

        text_config = payload.get("text_config")
        if isinstance(text_config, dict):
            value = text_config.get("max_position_embeddings")
            if isinstance(value, int) and value > 0:
                return value

        value = payload.get("max_position_embeddings")
        if isinstance(value, int) and value > 0:
            return value
        return self.config.vllm_default_max_model_len

    def _read_observed_kv_cache_tokens(self) -> int | None:
        path = self._log_file_path()
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

        matches = re.findall(r"GPU KV cache size:\s*([0-9,]+)\s+tokens", content)
        if not matches:
            return None
        try:
            return int(matches[-1].replace(",", ""))
        except ValueError:
            return None

    def _recommend_max_model_len(
        self,
        *,
        theoretical_max_model_len: int,
        observed_kv_cache_tokens: int | None,
    ) -> tuple[int | None, str | None]:
        if observed_kv_cache_tokens is None or observed_kv_cache_tokens <= 0:
            return None, None

        safe_budget = min(theoretical_max_model_len, int(observed_kv_cache_tokens * 0.75))
        candidates = [
            8192,
            16384,
            32768,
            65536,
            131072,
            196608,
            262144,
        ]
        recommended = None
        for candidate in candidates:
            if candidate <= safe_budget:
                recommended = candidate
        if recommended is None:
            return None, None

        return (
            recommended,
            (
                f"gx10에서 관측된 KV cache 예산이 약 {observed_kv_cache_tokens:,} 토큰이라 "
                f"`max_model_len`은 {recommended:,} 이하를 권장합니다."
            ),
        )
