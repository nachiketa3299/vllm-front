import json
from dataclasses import dataclass
from typing import Any

from .models import AppError


@dataclass(frozen=True)
class LeafField:
    path: str
    type_label: str


class TemplateValueCodec:
    def __init__(self, struct_template: Any, floats_template: Any):
        self.struct_template = struct_template
        self.floats_template = floats_template
        self.floats_sections = dict(self.floats_template)
        self.struct_leaf_fields = self._build_leaf_fields(self.struct_template)
        self.floats_leaf_fields = self._build_leaf_fields(self.floats_template)
        self.struct_leaf_defaults = self._collect_leaf_defaults(self.struct_template)
        self.floats_leaf_defaults = self._collect_leaf_defaults(self.floats_template)
        self.floats_section_leaf_fields = {
            name: self._build_leaf_fields(section_template, name)
            for name, section_template in self.floats_sections.items()
        }
        self.floats_section_leaf_defaults = {
            name: self._collect_leaf_defaults(section_template)
            for name, section_template in self.floats_sections.items()
        }

    @classmethod
    def _build_leaf_fields(cls, template: Any, path: str = "") -> list[LeafField]:
        if isinstance(template, dict):
            fields: list[LeafField] = []
            for key, value in template.items():
                next_path = f"{path}.{key}" if path else key
                fields.extend(cls._build_leaf_fields(value, next_path))
            return fields

        return [LeafField(path=path, type_label=cls._type_label(template))]

    @staticmethod
    def _type_label(template: Any) -> str:
        if isinstance(template, bool):
            return "boolean"
        if template is None:
            return "null_or_string"
        if isinstance(template, (int, float)) and not isinstance(template, bool):
            return "number"
        if isinstance(template, str):
            return "string"
        if isinstance(template, list):
            return "array"
        return "value"

    @staticmethod
    def json_schema_type(type_label: str) -> dict[str, Any]:
        if type_label == "boolean":
            return {"type": "boolean"}
        if type_label == "null_or_string":
            return {"anyOf": [{"type": "null"}, {"type": "string"}]}
        if type_label == "number":
            return {"type": "number"}
        if type_label == "string":
            return {"type": "string"}
        if type_label == "array":
            return {"type": "array"}
        return {}

    @classmethod
    def _collect_leaf_defaults(cls, template: Any) -> list[Any]:
        if isinstance(template, dict):
            defaults: list[Any] = []
            for value in template.values():
                defaults.extend(cls._collect_leaf_defaults(value))
            return defaults
        return [template]

    def build_struct_from_values(self, values: list[Any]) -> dict[str, Any]:
        return self._build_from_values(
            template=self.struct_template,
            values=values,
            field_count=len(self.struct_leaf_fields),
            target_name="struct",
        )

    def build_floats_section_from_values(self, section_name: str, values: list[Any]) -> Any:
        section_template = self.floats_sections[section_name]
        section_fields = self.floats_section_leaf_fields[section_name]
        return self._build_from_values(
            template=section_template,
            values=values,
            field_count=len(section_fields),
            target_name=f"floats.{section_name}",
        )

    @staticmethod
    def _build_from_values(
        *,
        template: Any,
        values: list[Any],
        field_count: int,
        target_name: str,
    ) -> Any:
        if len(values) != field_count:
            preview_values = json.dumps(values[:8], ensure_ascii=False)
            raise AppError(
                502,
                (
                    f"{target_name} value count mismatch: expected {field_count}, got {len(values)}. "
                    f"leading_values={preview_values}"
                ),
            )

        index = 0

        def fill(current_template: Any) -> Any:
            nonlocal index
            if isinstance(current_template, dict):
                return {key: fill(value) for key, value in current_template.items()}

            value = values[index]
            index += 1
            return value

        return fill(template)

    def normalize_struct_values(self, values: list[Any]) -> tuple[list[Any], str | None]:
        return self._normalize_value_count(
            values=values,
            defaults=self.struct_leaf_defaults,
            target_name="struct",
        )

    def normalize_floats_section_values(
        self,
        section_name: str,
        values: list[Any],
    ) -> tuple[list[Any], str | None]:
        return self._normalize_value_count(
            values=values,
            defaults=self.floats_section_leaf_defaults[section_name],
            target_name=f"floats.{section_name}",
        )

    @staticmethod
    def _normalize_value_count(
        *,
        values: list[Any],
        defaults: list[Any],
        target_name: str,
    ) -> tuple[list[Any], str | None]:
        expected = len(defaults)
        actual = len(values)

        if actual == expected:
            return values, None

        if actual > expected:
            return (
                values[:expected],
                f"{target_name} returned {actual} values; truncated to {expected}.",
            )

        padded = values + defaults[actual:]
        return (
            padded,
            f"{target_name} returned {actual} values; filled missing {expected - actual} values from template defaults.",
        )

