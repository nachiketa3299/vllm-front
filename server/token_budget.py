"""vLLM /tokenize 엔드포인트로 토큰 수를 정확히 받음.

클라이언트 측 토크나이저(transformers/AutoProcessor) 불필요. 머신간 운영도 가능.
"""

from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import UploadFile

from .config import AppConfig
from .image import ImagePreparer
from .models import AppError, PreparedImage, RequestLog


class TokenBudgetService:
    _TIMEOUT_SECONDS = 10.0

    def __init__(self, config: AppConfig):
        self.config = config

    async def estimate(
        self,
        *,
        upload: Optional[UploadFile],
        user_request: Optional[str],
        max_completion_tokens: int,
        max_model_len: int,
        max_image_bytes: int,
        model: str,
    ) -> dict[str, Any]:
        max_completion_tokens = self._positive_int(
            "max_completion_tokens", max_completion_tokens
        )
        max_model_len = self._positive_int("max_model_len", max_model_len)
        max_image_bytes = self._positive_int("max_image_bytes", max_image_bytes)

        note = (user_request or "").strip()
        prepared_image: Optional[PreparedImage] = None
        if upload is not None:
            prepared_image = ImagePreparer.from_upload(
                upload,
                max_image_bytes=max_image_bytes,
                log=RequestLog(entries=[]),
            )

        if not note and prepared_image is None:
            return {
                "input_tokens": 0,
                "text_tokens": 0,
                "image_tokens": 0,
                "output_tokens": max_completion_tokens,
                "total_tokens": max_completion_tokens,
                "max_model_len": max_model_len,
                "remaining_tokens": max_model_len - max_completion_tokens,
                "exceeds_limit": max_completion_tokens > max_model_len,
                "utilization_ratio": min(
                    max_completion_tokens / max_model_len, 1.0
                ),
                "input_present": False,
            }

        text_tokens = await self._tokenize(model, self._messages(note, image=None))
        if prepared_image is not None:
            input_tokens = await self._tokenize(
                model, self._messages(note, image=prepared_image)
            )
            image_tokens = max(0, input_tokens - text_tokens)
        else:
            input_tokens = text_tokens
            image_tokens = 0

        total_tokens = input_tokens + max_completion_tokens

        return {
            "input_tokens": input_tokens,
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "output_tokens": max_completion_tokens,
            "total_tokens": total_tokens,
            "max_model_len": max_model_len,
            "remaining_tokens": max_model_len - total_tokens,
            "exceeds_limit": total_tokens > max_model_len,
            "utilization_ratio": min(total_tokens / max_model_len, 1.0),
            "input_present": True,
        }

    async def _tokenize(self, model: str, messages: list[dict[str, Any]]) -> int:
        url = self._tokenize_url()
        try:
            async with httpx.AsyncClient(timeout=self._TIMEOUT_SECONDS) as client:
                response = await client.post(
                    url,
                    json={
                        "model": model,
                        "messages": messages,
                        "add_generation_prompt": True,
                    },
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise AppError(
                502,
                f"vLLM /tokenize HTTP {exc.response.status_code}: {exc.response.text}",
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(502, f"vLLM /tokenize 연결 실패: {exc}") from exc
        except ValueError as exc:
            raise AppError(502, "vLLM /tokenize 응답이 JSON 형식 아님") from exc

        count = payload.get("count")
        if not isinstance(count, int):
            raise AppError(502, "vLLM /tokenize 응답에 count 필드 없음")
        return count

    def _tokenize_url(self) -> str:
        # /tokenize 라우트는 /v1 prefix 아래가 아니라 origin 직속.
        parsed = urlparse(self.config.vllm_base_url)
        return f"{parsed.scheme}://{parsed.netloc}/tokenize"

    @staticmethod
    def _messages(
        note: str, image: Optional[PreparedImage]
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if note:
            content.append({"type": "text", "text": note})
        if image is not None:
            content.append(
                {"type": "image_url", "image_url": {"url": image.data_url}}
            )
        return [{"role": "user", "content": content}]

    @staticmethod
    def _positive_int(field_name: str, value: int) -> int:
        if value <= 0:
            raise AppError(400, f"{field_name} must be a positive integer.")
        return value
