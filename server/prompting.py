import json
from typing import Any

from .assets import AssetStore
from .transform import LeafField, TemplateValueCodec


class PromptBuilder:
    def __init__(self, assets: AssetStore, codec: TemplateValueCodec):
        self.assets = assets
        self.codec = codec
        self._struct_field_instructions = self._format_struct_field_instructions(
            codec.struct_leaf_fields
        )
        self._floats_section_field_instructions = {
            name: self._format_floats_field_instructions(fields)
            for name, fields in codec.floats_section_leaf_fields.items()
        }
        self._struct_response_format = self._build_values_response_format(
            name="struct_values",
            fields=codec.struct_leaf_fields,
        )
        self._floats_section_response_formats = {
            name: self._build_values_response_format(
                name=f"floats_{name}_values",
                fields=fields,
                number_only=True,
            )
            for name, fields in codec.floats_section_leaf_fields.items()
        }

    def build_struct_prompt(self) -> str:
        return "\n".join(
            [
                self.assets.get_prompt_text(),
                "Return exactly one JSON array and nothing else. No markdown.",
                (
                    f"Return {len(self.codec.struct_leaf_fields)} values in this exact order. "
                    "Do not include field names."
                ),
                f"The array must contain exactly {len(self.codec.struct_leaf_fields)} elements.",
                "Do not add any leading or trailing values.",
                "Field order and types:",
                self._struct_field_instructions,
            ]
        )

    def build_floats_section_prompt(
        self,
        section_name: str,
        struct_payload: dict[str, Any],
    ) -> str:
        struct_context = json.dumps(struct_payload, ensure_ascii=False, separators=(",", ":"))
        fields = self.codec.floats_section_leaf_fields[section_name]
        return "\n".join(
            [
                self.assets.get_prompt_text(),
                "Return exactly one JSON array and nothing else. No markdown.",
                (
                    f"Return {len(fields)} values in this exact order for floats.{section_name}. "
                    "Do not include field names."
                ),
                f"The array must contain exactly {len(fields)} elements.",
                "Do not add any leading or trailing values.",
                f"struct context: {struct_context}",
                "Field order:",
                self._floats_section_field_instructions[section_name],
            ]
        )

    @staticmethod
    def _format_struct_field_instructions(fields: list[LeafField]) -> str:
        return ";".join(
            f"{field.path}[{field.type_label}]"
            for field in fields
        )

    @staticmethod
    def _format_floats_field_instructions(fields: list[LeafField]) -> str:
        return ";".join(field.path for field in fields)

    @staticmethod
    def _build_values_response_format(
        *,
        name: str,
        fields: list[LeafField],
        number_only: bool = False,
    ) -> dict[str, Any]:
        if number_only:
            item_schema: dict[str, Any] = {"type": "number"}
        else:
            schemas = sorted(
                [
                    TemplateValueCodec.json_schema_type(field.type_label)
                    for field in fields
                ],
                key=lambda item: json.dumps(item, sort_keys=True),
            )
            deduped: list[dict[str, Any]] = []
            seen: set[str] = set()
            for schema in schemas:
                key = json.dumps(schema, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    deduped.append(schema)
            item_schema = {"anyOf": deduped}

        return {
            "type": "json_schema",
            "json_schema": {
                "name": name,
                "schema": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": item_schema,
                            "minItems": len(fields),
                            "maxItems": len(fields),
                        }
                    },
                    "required": ["values"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }

    def build_struct_response_format(self) -> dict[str, Any]:
        return self._struct_response_format

    def build_floats_section_response_format(self, section_name: str) -> dict[str, Any]:
        return self._floats_section_response_formats[section_name]

