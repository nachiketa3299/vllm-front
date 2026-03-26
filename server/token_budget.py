import io
from typing import Any, Optional

from fastapi import UploadFile
from PIL import Image
from transformers import AutoProcessor

from .config import AppConfig
from .image import ImagePreparer
from .models import AppError, PreparedImage, RequestLog


class TokenBudgetService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._processor = None

    def estimate(
        self,
        *,
        upload: Optional[UploadFile],
        user_request: Optional[str],
        max_completion_tokens: int,
        max_model_len: int,
        max_image_bytes: int,
    ) -> dict[str, Any]:
        resolved_output_tokens = self._resolve_positive_int(
            field_name="max_completion_tokens",
            value=max_completion_tokens,
        )
        resolved_max_model_len = self._resolve_positive_int(
            field_name="max_model_len",
            value=max_model_len,
        )
        resolved_max_image_bytes = self._resolve_positive_int(
            field_name="max_image_bytes",
            value=max_image_bytes,
        )

        note = (user_request or "").strip()
        prepared_image = None
        if upload is not None:
            prepared_image = ImagePreparer.from_upload(
                upload,
                max_image_bytes=resolved_max_image_bytes,
                log=RequestLog(entries=[]),
            )

        input_tokens, text_tokens = self._estimate_input_tokens(
            user_request=note,
            prepared_image=prepared_image,
        )
        image_tokens = max(0, input_tokens - text_tokens)
        total_tokens = input_tokens + resolved_output_tokens
        remaining_tokens = resolved_max_model_len - total_tokens

        return {
            "input_tokens": input_tokens,
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "output_tokens": resolved_output_tokens,
            "total_tokens": total_tokens,
            "max_model_len": resolved_max_model_len,
            "remaining_tokens": remaining_tokens,
            "exceeds_limit": total_tokens > resolved_max_model_len,
            "utilization_ratio": min(total_tokens / resolved_max_model_len, 1.0),
            "input_present": bool(note or prepared_image is not None),
        }

    def _estimate_input_tokens(
        self,
        *,
        user_request: str,
        prepared_image: PreparedImage | None,
    ) -> tuple[int, int]:
        processor = self._get_processor()
        text_messages = self._build_messages(user_request=user_request, include_image=False)
        text_template = processor.apply_chat_template(
            text_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        text_encoding = processor(text=[text_template], padding=False, return_tensors="pt")
        text_tokens = int(text_encoding["input_ids"].shape[-1])

        if prepared_image is None:
            return text_tokens, text_tokens

        full_messages = self._build_messages(user_request=user_request, include_image=True)
        full_template = processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image = self._open_image(prepared_image)
        full_encoding = processor(
            text=[full_template],
            images=[image],
            padding=False,
            return_tensors="pt",
        )
        return int(full_encoding["input_ids"].shape[-1]), text_tokens

    def _build_messages(self, *, user_request: str, include_image: bool) -> list[dict[str, Any]]:
        user_content: list[dict[str, Any]] = []
        if user_request:
            user_content.append({"type": "text", "text": user_request})
        if include_image:
            user_content.append({"type": "image"})
        return [{"role": "user", "content": user_content}]

    def _open_image(self, prepared_image: PreparedImage) -> Image.Image:
        try:
            image = Image.open(io.BytesIO(prepared_image.bytes_data))
            return image.convert("RGB")
        except Exception as exc:
            raise AppError(400, "이미지 토큰 추정을 위해 업로드 이미지를 열 수 없습니다.") from exc

    def _get_processor(self):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.config.model,
                trust_remote_code=True,
            )
        return self._processor

    @staticmethod
    def _resolve_positive_int(*, field_name: str, value: int) -> int:
        if value <= 0:
            raise AppError(400, f"{field_name} must be a positive integer.")
        return value
