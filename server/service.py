from typing import AsyncIterator, Optional

from fastapi import UploadFile

from .config import AppConfig
from .image import ImagePreparer
from .models import AppError, GeneratedPayload, RequestLog
from .parsing import ResponseParser
from .prompt_logger import PromptQueryLogger
from .vllm_client import VLLMClient


class GenerationService:
    def __init__(
        self,
        *,
        config: AppConfig,
        client: VLLMClient,
        prompt_logger: PromptQueryLogger | None = None,
    ):
        self.config = config
        self.client = client
        self.prompt_logger = prompt_logger

    async def generate(
        self,
        upload: Optional[UploadFile],
        log: RequestLog,
        user_request: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        max_image_bytes: Optional[int] = None,
        json_output: Optional[bool] = None,
        enable_thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> GeneratedPayload:
        note = (user_request or "").strip()
        if self.prompt_logger is not None:
            self.prompt_logger.log_generate_request(
                prompt_text=note,
                has_image=upload is not None,
            )

        resolved_max_completion_tokens = self._resolve_positive_int(
            field_name="max_completion_tokens",
            override=max_completion_tokens,
            default=self.config.max_completion_tokens,
        )
        resolved_max_image_bytes = self._resolve_positive_int(
            field_name="max_image_bytes",
            override=max_image_bytes,
            default=self.config.max_image_bytes,
        )
        resolved_json_output = bool(json_output)
        resolved_enable_thinking = bool(enable_thinking)
        resolved_temperature = self._resolve_temperature(temperature)

        prepared_image = None
        if upload is not None:
            prepared_image = ImagePreparer.from_upload(
                upload,
                max_image_bytes=resolved_max_image_bytes,
                log=log,
            )

        if prepared_image is None and not note:
            raise AppError(400, "Either an image or prompt text is required.")

        if prepared_image is not None:
            log.add("Using image input")
        else:
            log.add("No image provided")
        if note:
            log.add(f"Using prompt text ({len(note)} chars)")
        else:
            log.add("No prompt text provided")
        log.add(
            "Using request config: "
            f"max_completion_tokens={resolved_max_completion_tokens}, "
            f"max_image_bytes={resolved_max_image_bytes}, "
            f"json_output={resolved_json_output}, "
            f"enable_thinking={resolved_enable_thinking}, "
            f"temperature={resolved_temperature}"
        )

        output_text, reasoning_text = await self._generate_text(
            image_data_url=prepared_image.data_url if prepared_image is not None else None,
            user_request=note,
            max_completion_tokens=resolved_max_completion_tokens,
            json_output=resolved_json_output,
            enable_thinking=resolved_enable_thinking,
            temperature=resolved_temperature,
            log=log,
        )

        return GeneratedPayload(text=output_text, reasoning=reasoning_text)

    async def _generate_text(
        self,
        *,
        image_data_url: Optional[str],
        user_request: str,
        max_completion_tokens: int,
        json_output: bool,
        enable_thinking: bool,
        temperature: float,
        log: RequestLog,
    ) -> tuple[str, str]:
        response = await self.client.create_chat_completion(
            system_prompt=None,
            user_text=user_request or None,
            image_data_url=image_data_url,
            response_format={"type": "json_object"} if json_output else None,
            max_completion_tokens=max_completion_tokens,
            enable_thinking=enable_thinking,
            temperature=temperature,
            log=log,
            label="generation",
        )
        log.add(f"generation raw preview: {ResponseParser.preview_content(response)}")
        output_text = ResponseParser.extract_normalized_content(response)
        if not output_text:
            raise AppError(502, "vLLM returned an empty response.")
        reasoning_text = ResponseParser.extract_reasoning_content(response)
        log.add("Captured raw text output")
        return output_text, reasoning_text

    @staticmethod
    def _resolve_positive_int(*, field_name: str, override: Optional[int], default: int) -> int:
        value = default if override is None else override
        if value <= 0:
            raise AppError(400, f"{field_name} must be a positive integer.")
        return value

    @staticmethod
    def _resolve_temperature(override: Optional[float]) -> float:
        value = 0.0 if override is None else float(override)
        if value < 0:
            raise AppError(400, "temperature must be >= 0.")
        return value

    async def generate_stream(
        self,
        upload: Optional[UploadFile],
        log: RequestLog,
        user_request: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        max_image_bytes: Optional[int] = None,
        json_output: Optional[bool] = None,
        enable_thinking: Optional[bool] = None,
        temperature: Optional[float] = None,
    ) -> AsyncIterator[dict[str, str]]:
        """Streaming 버전. chunk dict 를 yield. 끝에 done 표시는 없음 — 호출자가
        generator 종료를 done 으로 해석."""
        note = (user_request or "").strip()
        if self.prompt_logger is not None:
            self.prompt_logger.log_generate_request(
                prompt_text=note,
                has_image=upload is not None,
            )

        resolved_max_completion_tokens = self._resolve_positive_int(
            field_name="max_completion_tokens",
            override=max_completion_tokens,
            default=self.config.max_completion_tokens,
        )
        resolved_max_image_bytes = self._resolve_positive_int(
            field_name="max_image_bytes",
            override=max_image_bytes,
            default=self.config.max_image_bytes,
        )
        resolved_json_output = bool(json_output)
        resolved_enable_thinking = bool(enable_thinking)
        resolved_temperature = self._resolve_temperature(temperature)

        prepared_image = None
        if upload is not None:
            prepared_image = ImagePreparer.from_upload(
                upload,
                max_image_bytes=resolved_max_image_bytes,
                log=log,
            )

        if prepared_image is None and not note:
            raise AppError(400, "Either an image or prompt text is required.")

        if prepared_image is not None:
            log.add("Using image input")
        if note:
            log.add(f"Using prompt text ({len(note)} chars)")
        log.add(
            "Using request config: "
            f"max_completion_tokens={resolved_max_completion_tokens}, "
            f"json_output={resolved_json_output}, "
            f"enable_thinking={resolved_enable_thinking}, "
            f"temperature={resolved_temperature}"
        )

        async for chunk in self.client.stream_chat_completion(
            system_prompt=None,
            user_text=note or None,
            image_data_url=(
                prepared_image.data_url if prepared_image is not None else None
            ),
            response_format={"type": "json_object"} if resolved_json_output else None,
            max_completion_tokens=resolved_max_completion_tokens,
            enable_thinking=resolved_enable_thinking,
            temperature=resolved_temperature,
            log=log,
        ):
            # vLLM 의 qwen3 reasoning_parser 가 streaming 모드에서
            # enable_thinking=false 케이스를 제대로 처리하지 못하고 모든 토큰을
            # delta.reasoning 으로 보내는 동작이 있음. 사용자가 사고를 끄면
            # 사고 패널 대신 본문으로 매핑.
            if not resolved_enable_thinking and "reasoning" in chunk and "content" not in chunk:
                chunk = {"content": chunk["reasoning"]}
            yield chunk

        log.add("Stream finished")
