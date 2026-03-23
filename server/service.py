import base64
import io
import json
import zipfile
from typing import Any

from fastapi import UploadFile

from .config import AppConfig
from .image import ImagePreparer
from .models import GeneratedPayload, RequestLog
from .parsing import ResponseParser
from .prompting import PromptBuilder
from .transform import TemplateValueCodec
from .validation import TemplateValidator
from .vllm_client import VLLMClient


class GenerationService:
    def __init__(
        self,
        *,
        config: AppConfig,
        client: VLLMClient,
        codec: TemplateValueCodec,
        prompt_builder: PromptBuilder,
        validator: TemplateValidator,
    ):
        self.config = config
        self.client = client
        self.codec = codec
        self.prompt_builder = prompt_builder
        self.validator = validator

    async def generate(self, upload: UploadFile, log: RequestLog) -> GeneratedPayload:
        prepared_image = ImagePreparer.from_upload(
            upload,
            max_image_bytes=self.config.max_image_bytes,
            log=log,
        )

        struct_response = await self.client.create_chat_completion(
            system_prompt=self.prompt_builder.build_struct_prompt(),
            user_text=(
                "Analyze the garment image and return only the ordered struct values array "
                "for ChatGarment."
            ),
            image_data_url=prepared_image.data_url,
            response_format=self.prompt_builder.build_struct_response_format(),
            log=log,
            label="struct",
        )
        log.add(f"struct raw preview: {ResponseParser.preview_content(struct_response)}")
        struct_values = ResponseParser.parse_values_chat_completion(struct_response, "struct")
        log.add("Parsed struct values response as JSON")
        struct_values, struct_note = self.codec.normalize_struct_values(struct_values)
        if struct_note:
            log.add(struct_note)
        struct_payload = self.codec.build_struct_from_values(struct_values)
        log.add("Reconstructed struct JSON from ordered values")
        struct_payload = self.validator.validate_struct(struct_payload)
        log.add("Validated struct keys against template")

        floats_payload: dict[str, Any] = {}
        for section_name in self.codec.floats_sections:
            floats_response = await self.client.create_chat_completion(
                system_prompt=self.prompt_builder.build_floats_section_prompt(
                    section_name,
                    struct_payload,
                ),
                user_text=(
                    "Analyze the garment image and return only the ordered floats values array "
                    f"for ChatGarment section floats.{section_name}. "
                    "Use the image and the provided struct context."
                ),
                image_data_url=prepared_image.data_url,
                response_format=self.prompt_builder.build_floats_section_response_format(
                    section_name
                ),
                log=log,
                label=f"floats.{section_name}",
            )
            log.add(
                f"floats.{section_name} raw preview: "
                f"{ResponseParser.preview_content(floats_response)}"
            )
            floats_values = ResponseParser.parse_values_chat_completion(
                floats_response,
                f"floats.{section_name}",
            )
            log.add(f"Parsed floats.{section_name} values response as JSON")
            floats_values, floats_note = self.codec.normalize_floats_section_values(
                section_name,
                floats_values,
            )
            if floats_note:
                log.add(floats_note)
            floats_payload[section_name] = self.codec.build_floats_section_from_values(
                section_name,
                floats_values,
            )
            log.add(f"Reconstructed floats.{section_name} JSON from ordered values")

        floats_payload = self.validator.validate_floats(floats_payload)
        log.add("Validated floats keys against template")

        zip_bytes = io.BytesIO()
        with zipfile.ZipFile(zip_bytes, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "struct.json",
                json.dumps(struct_payload, ensure_ascii=False, indent=2),
            )
            archive.writestr(
                "floats.json",
                json.dumps(floats_payload, ensure_ascii=False, indent=2),
            )
        log.add("Built ZIP payload")

        return GeneratedPayload(
            struct=struct_payload,
            floats=floats_payload,
            zip_base64=base64.b64encode(zip_bytes.getvalue()).decode("ascii"),
        )
