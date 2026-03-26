import asyncio
import tempfile
import unittest
from pathlib import Path

from server.config import AppConfig, AppPaths
from server.model_control import ModelControlService
from server.models import ProbedModelInfo


def build_config() -> AppConfig:
    return AppConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        model="/tmp/qwen",
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_default_max_model_len=8192,
        vllm_reasoning_parser="qwen3",
        vllm_startup_timeout_seconds=2,
        vllm_shutdown_grace_seconds=1,
        max_completion_tokens=1024,
        timeout_seconds=30,
        max_image_bytes=1024,
    )


class ModelControlServiceTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.tempdir.name)
        self.paths = AppPaths(
            base_dir=self.base_dir,
            static_dir=self.base_dir / "static",
            runtime_dir=self.base_dir / "runtime",
        )
        self.paths.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.config = build_config()

    async def asyncTearDown(self) -> None:
        self.tempdir.cleanup()

    async def test_start_transitions_to_running_managed(self) -> None:
        probe_values = [
            None,
            None,
            ProbedModelInfo(model=self.config.model, max_model_len=4096),
        ]

        async def fake_probe():
            if probe_values:
                return probe_values.pop(0)
            return ProbedModelInfo(model=self.config.model, max_model_len=4096)

        async def fake_sleep(_: float) -> None:
            await asyncio.sleep(0)

        def fake_launch(command, env, cwd, log_path):
            self.assertIn("--max-model-len", command)
            self.assertEqual(str(log_path.parent), str(self.paths.runtime_dir))
            return 4242

        service = ModelControlService(
            config=self.config,
            paths=self.paths,
            probe_fn=fake_probe,
            sleep_fn=fake_sleep,
            launch_fn=fake_launch,
        )
        service._managed_process_matches = lambda state: True  # type: ignore[method-assign]

        status = await service.start(max_model_len=4096)
        self.assertEqual(status.status, "starting")

        for _ in range(10):
            await asyncio.sleep(0)

        final_status = await service.get_runtime_status()
        self.assertEqual(final_status.status, "running_managed")
        self.assertEqual(final_status.current_max_model_len, 4096)

    async def test_stop_marks_stopping_and_clears_managed_state(self) -> None:
        alive = True
        signals = []

        async def fake_probe():
            return ProbedModelInfo(model=self.config.model, max_model_len=8192) if alive else None

        async def fake_sleep(_: float) -> None:
            nonlocal alive
            alive = False
            await asyncio.sleep(0)

        service = ModelControlService(
            config=self.config,
            paths=self.paths,
            probe_fn=fake_probe,
            sleep_fn=fake_sleep,
            launch_fn=lambda *_args: 5151,
        )
        service._managed_process_matches = lambda state: alive  # type: ignore[method-assign]
        service._pid_exists = lambda pid: alive  # type: ignore[method-assign]
        service._signal_process_group = lambda pgid, sig: signals.append((pgid, sig.name))  # type: ignore[method-assign]
        service._write_state(
            {
                "pid": 5151,
                "pgid": 5151,
                "model": self.config.model,
                "host": self.config.vllm_host,
                "port": self.config.vllm_port,
                "requested_max_model_len": 8192,
                "phase": "running",
                "command": [],
                "started_at": 0.0,
            }
        )

        status = await service.stop()
        self.assertEqual(status.status, "stopping")
        self.assertEqual(signals[0], (5151, "SIGTERM"))

        for _ in range(6):
            await asyncio.sleep(0)

        final_status = await service.get_runtime_status()
        self.assertEqual(final_status.status, "stopped")

    async def test_probe_without_managed_state_is_unmanaged(self) -> None:
        async def fake_probe():
            return ProbedModelInfo(model="/external/qwen", max_model_len=16384)

        service = ModelControlService(
            config=self.config,
            paths=self.paths,
            probe_fn=fake_probe,
            sleep_fn=asyncio.sleep,
            launch_fn=lambda *_args: 11,
        )

        status = await service.get_runtime_status()
        self.assertEqual(status.status, "running_unmanaged")
        self.assertFalse(status.can_stop)
