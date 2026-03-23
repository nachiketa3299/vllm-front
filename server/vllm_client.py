from typing import Any

import httpx

from .config import AppConfig
from .models import AppError, RequestLog


class VLLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def create_payload(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_data_url: str,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": 0.0,
            "max_completion_tokens": self.config.max_completion_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                },
            ],
        }
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    async def create_chat_completion(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_data_url: str,
        response_format: dict[str, Any] | None,
        log: RequestLog,
        label: str,
    ) -> dict[str, Any]:
        log.add(f"Sending {label} request to vLLM at {self.config.vllm_base_url}")

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(
                    f"{self.config.vllm_base_url.rstrip('/')}/chat/completions",
                    json=self.create_payload(
                        system_prompt=system_prompt,
                        user_text=user_text,
                        image_data_url=image_data_url,
                        response_format=response_format,
                    ),
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            body = exc.response.text
            if "cannot identify image file" in body:
                raise AppError(
                    502,
                    (
                        "현재 실행 중인 vLLM 인스턴스가 이미지 입력을 디코드하지 못하고 있습니다. "
                        "앱 업로드 자체 문제라기보다 vLLM 설정 또는 버전 문제일 가능성이 큽니다. "
                        f"raw={body}"
                    ),
                ) from exc
            raise AppError(
                502, f"vLLM returned HTTP {exc.response.status_code}: {body}"
            ) from exc
        except httpx.RequestError as exc:
            raise AppError(
                502, f"Failed to connect to vLLM at {self.config.vllm_base_url}."
            ) from exc
        except ValueError as exc:
            raise AppError(502, "vLLM returned an invalid JSON response.") from exc

        log.add(f"Received {label} response from vLLM")
        return payload

