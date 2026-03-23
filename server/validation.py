from typing import Any

from .models import AppError


class TemplateValidator:
    def __init__(self, struct_template: Any, floats_template: Any):
        self.struct_template = struct_template
        self.floats_template = floats_template

    def validate_struct(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._validate_value(payload, self.struct_template, path="struct")

    def validate_floats(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._validate_value(payload, self.floats_template, path="floats")

    def _validate_value(self, value: Any, template: Any, *, path: str) -> Any:
        if isinstance(template, dict):
            return self._validate_object(value, template, path=path)
        if isinstance(template, list):
            if not isinstance(value, list):
                raise AppError(502, f"{path} must be an array.")
            return value
        if isinstance(template, bool):
            if not isinstance(value, bool):
                raise AppError(502, f"{path} must be a boolean.")
            return value
        if template is None:
            if value is not None and not isinstance(value, str):
                raise AppError(502, f"{path} must be null or string.")
            return value
        if isinstance(template, (int, float)) and not isinstance(template, bool):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise AppError(502, f"{path} must be a number.")
            return value
        if isinstance(template, str):
            if not isinstance(value, str):
                raise AppError(502, f"{path} must be a string.")
            return value
        return value

    def _validate_object(
        self, value: Any, template: dict[str, Any], *, path: str
    ) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise AppError(502, f"{path or 'root'} must be an object.")

        expected_keys = set(template.keys())
        actual_keys = set(value.keys())
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)

        if missing_keys:
            raise AppError(
                502,
                f"{path or 'root'} is missing keys: {', '.join(missing_keys)}",
            )
        if extra_keys:
            raise AppError(
                502,
                f"{path or 'root'} has unexpected keys: {', '.join(extra_keys)}",
            )

        return {
            key: self._validate_value(
                value[key],
                template[key],
                path=f"{path}.{key}" if path else key,
            )
            for key in template
        }

