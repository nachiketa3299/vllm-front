import json
from typing import Any

from .models import AppError


class ResponseParser:
    @staticmethod
    def parse_values_chat_completion(response: dict[str, Any], target_name: str) -> list[Any]:
        normalized = ResponseParser.extract_normalized_content(response)
        for candidate in ResponseParser._candidate_json_payloads(normalized):
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue

            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and isinstance(parsed.get(target_name), list):
                return parsed[target_name]
            if isinstance(parsed, dict) and isinstance(parsed.get("values"), list):
                return parsed["values"]

        raise AppError(502, f"{target_name} response must be a JSON array of values.")

    @staticmethod
    def preview_content(response: dict[str, Any], limit: int = 400) -> str:
        normalized = ResponseParser.extract_normalized_content(response)
        preview = normalized[:limit]
        if len(normalized) > limit:
            preview += "...(truncated)"
        return preview

    @staticmethod
    def extract_normalized_content(response: dict[str, Any]) -> str:
        content = ResponseParser._extract_message_content(response)
        return ResponseParser._strip_markdown_fence(content)

    @staticmethod
    def _candidate_json_payloads(normalized: str) -> list[str]:
        candidates = [normalized]
        stripped = normalized.strip()

        array_start = stripped.find("[")
        if array_start == -1:
            return candidates

        array_end = stripped.rfind("]")
        if array_end != -1:
            candidates.append(stripped[array_start : array_end + 1])
        else:
            candidates.append(stripped[array_start:] + "]")

        repaired: list[str] = []
        for candidate in candidates:
            fixed = candidate.replace(",]", "]")
            if fixed not in repaired:
                repaired.append(fixed)
        return repaired

    @staticmethod
    def _extract_message_content(response: dict[str, Any]) -> str:
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AppError(502, "vLLM response did not contain a chat message.") from exc

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
            content = "\n".join(text_parts)

        if not isinstance(content, str) or not content.strip():
            raise AppError(502, "vLLM returned an empty response.")

        return content

    @staticmethod
    def _strip_markdown_fence(content: str) -> str:
        normalized = content.strip()
        if not normalized.startswith("```"):
            return normalized

        lines = normalized.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()

