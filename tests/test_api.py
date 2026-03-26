import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from server.api import create_router
from server.config import AppConfig, AppPaths
from server.models import AppError, ModelRuntimeStatus


def build_config() -> AppConfig:
    return AppConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        model="/tmp/qwen",
        vllm_host="127.0.0.1",
        vllm_port=8000,
        vllm_default_max_model_len=8192,
        vllm_reasoning_parser="qwen3",
        vllm_startup_timeout_seconds=10,
        vllm_shutdown_grace_seconds=5,
        max_completion_tokens=1024,
        timeout_seconds=30,
        max_image_bytes=2048,
    )


class FakeModelControl:
    def __init__(self):
        self.runtime = ModelRuntimeStatus(
            status="stopped",
            ownership="none",
            model=None,
            pid=None,
            current_max_model_len=None,
            default_max_model_len=8192,
            theoretical_max_model_len=262144,
            observed_kv_cache_tokens=210112,
            recommended_max_model_len=131072,
            recommended_max_model_len_reason="recommended",
            detail="모델이 꺼져 있습니다.",
            can_start=True,
            can_stop=False,
        )
        self.started_with = None
        self.stopped = False

    async def get_runtime_status(self):
        return self.runtime

    async def start(self, *, max_model_len: int):
        self.started_with = max_model_len
        self.runtime = ModelRuntimeStatus(
            status="starting",
            ownership="managed",
            model="/tmp/qwen",
            pid=1001,
            current_max_model_len=max_model_len,
            default_max_model_len=8192,
            theoretical_max_model_len=262144,
            observed_kv_cache_tokens=210112,
            recommended_max_model_len=131072,
            recommended_max_model_len_reason="recommended",
            detail="모델을 시작하는 중입니다.",
            can_start=False,
            can_stop=True,
        )
        return self.runtime

    async def stop(self):
        self.stopped = True
        self.runtime = ModelRuntimeStatus(
            status="stopping",
            ownership="managed",
            model="/tmp/qwen",
            pid=1001,
            current_max_model_len=8192,
            default_max_model_len=8192,
            theoretical_max_model_len=262144,
            observed_kv_cache_tokens=210112,
            recommended_max_model_len=131072,
            recommended_max_model_len_reason="recommended",
            detail="모델을 종료하는 중입니다.",
            can_start=False,
            can_stop=False,
        )
        return self.runtime


class FakeGenerationService:
    async def generate(self, *_args, **_kwargs):
        raise AppError(503, "model not ready")


class FakeTokenBudgetService:
    def estimate(self, **kwargs):
        return {
            "input_tokens": 120,
            "text_tokens": 80,
            "image_tokens": 40,
            "output_tokens": kwargs["max_completion_tokens"],
            "total_tokens": 120 + kwargs["max_completion_tokens"],
            "max_model_len": kwargs["max_model_len"],
            "remaining_tokens": kwargs["max_model_len"] - (120 + kwargs["max_completion_tokens"]),
            "exceeds_limit": False,
            "utilization_ratio": 0.5,
            "input_present": True,
        }


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base_dir = Path(self.tempdir.name)
        static_dir = base_dir / "static"
        static_dir.mkdir(parents=True, exist_ok=True)
        (static_dir / "index.html").write_text("<html></html>", encoding="utf-8")

        app = FastAPI()
        self.model_control = FakeModelControl()
        app.include_router(
            create_router(
                paths=AppPaths(
                    base_dir=base_dir,
                    static_dir=static_dir,
                    runtime_dir=base_dir / "runtime",
                ),
                config=build_config(),
                service=FakeGenerationService(),
                model_control=self.model_control,
                token_budget=FakeTokenBudgetService(),
            )
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_model_runtime_endpoint(self) -> None:
        response = self.client.get("/api/model/runtime")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "stopped")
        self.assertEqual(response.json()["theoretical_max_model_len"], 262144)
        self.assertEqual(response.json()["recommended_max_model_len"], 131072)

    def test_model_start_endpoint(self) -> None:
        response = self.client.post("/api/model/start", json={"max_model_len": 4096})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["status"], "starting")
        self.assertEqual(self.model_control.started_with, 4096)

    def test_token_budget_endpoint(self) -> None:
        response = self.client.post(
            "/api/token-budget",
            data={
                "user_request": "hello",
                "max_completion_tokens": "256",
                "max_model_len": "8192",
                "max_image_bytes": "2048",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["output_tokens"], 256)
        self.assertEqual(response.json()["max_model_len"], 8192)
