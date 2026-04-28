import unittest
from pathlib import Path
import tempfile

from server.config import AppConfig
from server.models import AppError, ProbedModelInfo, RequestLog
from server.prompt_logger import PromptQueryLogger
from server.service import GenerationService


class FakeClient:
    def __init__(self, response=None, error: AppError | None = None):
        self.response = response or {
            "choices": [{"message": {"content": "ok"}}],
        }
        self.error = error
        self._info = ProbedModelInfo(
            model="qwen3.5-27b",
            max_model_len=8192,
            model_path="/tmp/qwen",
        )

    async def probe_model_info(self, *, force: bool = False):
        return self._info

    async def require_model_info(self) -> ProbedModelInfo:
        return self._info

    async def create_chat_completion(self, **_kwargs):
        if self.error is not None:
            raise self.error
        return self.response


def build_config() -> AppConfig:
    return AppConfig(
        vllm_base_url="http://127.0.0.1:8000/v1",
        max_completion_tokens=1024,
        timeout_seconds=30,
        max_image_bytes=1024,
    )


class GenerationServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_logger_creates_log_file_on_init(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            log_path = Path(tempdir) / "logs" / "request-prompts.log"

            PromptQueryLogger(log_path)

            self.assertTrue(log_path.parent.is_dir())
            self.assertTrue(log_path.is_file())

    async def test_generate_propagates_client_errors(self) -> None:
        service = GenerationService(
            config=build_config(),
            client=FakeClient(error=AppError(502, "connect failed")),
        )

        with self.assertRaises(AppError) as exc:
            await service.generate(None, RequestLog(entries=[]), user_request="hello")

        self.assertEqual(exc.exception.status_code, 502)

    async def test_generate_writes_prompt_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            log_path = Path(tempdir) / "logs" / "request-prompts.log"
            service = GenerationService(
                config=build_config(),
                client=FakeClient(response={"choices": [{"message": {"content": "ok"}}]}),
                prompt_logger=PromptQueryLogger(log_path),
            )

            result = await service.generate(
                None,
                RequestLog(entries=[]),
                user_request="hello\nworld",
            )

            self.assertEqual(result.text, "ok")
            logged = log_path.read_text(encoding="utf-8")
            self.assertIn("endpoint=/api/generate", logged)
            self.assertIn("prompt_chars=11", logged)
            self.assertIn("hello\nworld", logged)
