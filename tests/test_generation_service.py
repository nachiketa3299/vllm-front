import unittest

from server.config import AppConfig
from server.models import AppError, RequestLog
from server.service import GenerationService


class FakeModelControl:
    def __init__(self, *, ready: bool = True, normalized_error: AppError | None = None):
        self.ready = ready
        self.normalized_error = normalized_error

    async def ensure_generation_available(self) -> None:
        if not self.ready:
            raise AppError(503, "model not ready")

    async def normalize_generation_error(self, error: AppError) -> AppError:
        return self.normalized_error or error


class FakeClient:
    def __init__(self, response=None, error: AppError | None = None):
        self.response = response or {
            "choices": [{"message": {"content": "ok"}}],
        }
        self.error = error

    async def create_chat_completion(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.response


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
        max_image_bytes=1024,
    )


class GenerationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_rejects_when_model_is_not_ready(self) -> None:
        service = GenerationService(
            config=build_config(),
            client=FakeClient(),
            model_control=FakeModelControl(ready=False),
        )

        with self.assertRaises(AppError) as exc:
            await service.generate(None, RequestLog(entries=[]), user_request="hello")

        self.assertEqual(exc.exception.status_code, 503)

    async def test_generate_normalizes_backend_disconnect(self) -> None:
        service = GenerationService(
            config=build_config(),
            client=FakeClient(error=AppError(502, "connect failed")),
            model_control=FakeModelControl(
                ready=True,
                normalized_error=AppError(503, "stopped during request"),
            ),
        )

        with self.assertRaises(AppError) as exc:
            await service.generate(None, RequestLog(entries=[]), user_request="hello")

        self.assertEqual(exc.exception.status_code, 503)
        self.assertEqual(exc.exception.detail, "stopped during request")
