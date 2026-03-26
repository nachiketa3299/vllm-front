from typing import Optional

from fastapi import UploadFile

from .config import AppConfig
from .image import ImagePreparer
from .model_control import ModelControlService
from .models import AppError, GeneratedPayload, RequestLog
from .parsing import ResponseParser
from .vllm_client import VLLMClient


class GenerationService:
    def __init__(
        self,
        *,
        config: AppConfig,
        client: VLLMClient,
        model_control: ModelControlService,
    ):
        self.config = config
        self.client = client
        self.model_control = model_control

    async def generate(
        self,
        upload: Optional[UploadFile],
        log: RequestLog,
        user_request: Optional[str] = None,
        max_completion_tokens: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
        max_image_bytes: Optional[int] = None,
        json_output: Optional[bool] = None,
    ) -> GeneratedPayload:
        await self.model_control.ensure_generation_available()

        resolved_max_completion_tokens = self._resolve_positive_int(
            field_name="max_completion_tokens",
            override=max_completion_tokens,
            default=self.config.max_completion_tokens,
        )
        resolved_timeout_seconds = self._resolve_positive_int(
            field_name="timeout_seconds",
            override=timeout_seconds,
            default=self.config.timeout_seconds,
        )
        resolved_max_image_bytes = self._resolve_positive_int(
            field_name="max_image_bytes",
            override=max_image_bytes,
            default=self.config.max_image_bytes,
        )
        resolved_json_output = bool(json_output)

        note = (user_request or "").strip()
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
            f"timeout_seconds={resolved_timeout_seconds}, "
            f"max_image_bytes={resolved_max_image_bytes}, "
            f"json_output={resolved_json_output}"
        )

        output_text = await self._generate_text(
            image_data_url=prepared_image.data_url if prepared_image is not None else None,
            user_request=note,
            max_completion_tokens=resolved_max_completion_tokens,
            timeout_seconds=resolved_timeout_seconds,
            json_output=resolved_json_output,
            log=log,
        )

        return GeneratedPayload(text=output_text)

    async def _generate_text(
        self,
        *,
        image_data_url: Optional[str],
        user_request: str,
        max_completion_tokens: int,
        timeout_seconds: int,
        json_output: bool,
        log: RequestLog,
    ) -> str:
        try:
            response = await self.client.create_chat_completion(
                system_prompt=None,
                user_text=user_request or None,
                image_data_url=image_data_url,
                response_format={"type": "json_object"} if json_output else None,
                max_completion_tokens=max_completion_tokens,
                timeout_seconds=timeout_seconds,
                log=log,
                label="generation",
            )
        except AppError as exc:
            raise await self.model_control.normalize_generation_error(exc) from exc
        log.add(f"generation raw preview: {ResponseParser.preview_content(response)}")
        output_text = ResponseParser.extract_normalized_content(response)
        if not output_text:
            raise AppError(502, "vLLM returned an empty response.")
        log.add("Captured raw text output")
        return output_text

    @staticmethod
    def _resolve_positive_int(*, field_name: str, override: Optional[int], default: int) -> int:
        value = default if override is None else override
        if value <= 0:
            raise AppError(400, f"{field_name} must be a positive integer.")
        return value
