import asyncio
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

import server.api as api_module
from server.api import create_router
from server.config import AppConfig, AppPaths
from server.models import AppError, ProbedModelInfo
from server.prompt_logger import PromptQueryLogger
from server.service import GenerationService


def build_config() -> AppConfig:
    return AppConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        max_completion_tokens=1024,
        timeout_seconds=30,
        max_image_bytes=2048,
    )


class FakeClient:
    def __init__(self, info: ProbedModelInfo | None = None):
        self.info = info or ProbedModelInfo(
            model="qwen3.5-27b",
            max_model_len=8192,
            model_path="/tmp/qwen",
        )

    async def probe_model_info(self, *, force: bool = False):
        return self.info

    async def require_model_info(self) -> ProbedModelInfo:
        if self.info is None:
            raise AppError(503, "offline")
        return self.info

    async def create_chat_completion(self, **_kwargs):
        return {
            "choices": [{"message": {"content": "ok"}}],
        }


class FakeGenerationService:
    async def generate(self, *_args, **_kwargs):
        raise AppError(503, "not available")


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
        self.fake_client = FakeClient()
        app.include_router(
            create_router(
                paths=AppPaths(base_dir=base_dir, static_dir=static_dir),
                config=build_config(),
                service=FakeGenerationService(),
                client=self.fake_client,
                token_budget=FakeTokenBudgetService(),
            )
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_model_info_endpoint_online(self) -> None:
        response = self.client.get("/api/model/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["online"])
        self.assertEqual(payload["model"], "qwen3.5-27b")
        self.assertEqual(payload["max_model_len"], 8192)
        self.assertEqual(payload["base_url"], "http://127.0.0.1:8000/v1")

    def test_model_info_endpoint_offline(self) -> None:
        self.fake_client.info = None
        response = self.client.get("/api/model/info")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["online"])
        self.assertIsNone(payload["model"])
        self.assertIsNone(payload["max_model_len"])

    def test_token_budget_uses_probed_max_model_len(self) -> None:
        response = self.client.post(
            "/api/token-budget",
            data={
                "user_request": "hello",
                "max_completion_tokens": "256",
                "max_image_bytes": "2048",
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_tokens"], 256)
        self.assertEqual(payload["max_model_len"], 8192)

    def test_generate_logs_prompt_to_server_file_only(self) -> None:
        base_dir = Path(self.tempdir.name)
        static_dir = base_dir / "static"
        client_fake = FakeClient()
        service = GenerationService(
            config=build_config(),
            client=client_fake,
            prompt_logger=PromptQueryLogger(base_dir / "logs" / "request-prompts.log"),
        )

        app = FastAPI()
        app.include_router(
            create_router(
                paths=AppPaths(base_dir=base_dir, static_dir=static_dir),
                config=build_config(),
                service=service,
                client=client_fake,
                token_budget=FakeTokenBudgetService(),
            )
        )
        client = TestClient(app)

        original_wait_for_disconnect = api_module.wait_for_disconnect

        async def fake_wait_for_disconnect(_request) -> None:
            await asyncio.sleep(60)

        api_module.wait_for_disconnect = fake_wait_for_disconnect
        try:
            response = client.post("/api/generate", data={"user_request": "secret prompt"})
        finally:
            api_module.wait_for_disconnect = original_wait_for_disconnect

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["output_text"], "ok")
        self.assertNotIn("secret prompt", "\n".join(payload["logs"]))

        logged = (base_dir / "logs" / "request-prompts.log").read_text(encoding="utf-8")
        self.assertIn("secret prompt", logged)
        self.assertIn("endpoint=/api/generate", logged)
