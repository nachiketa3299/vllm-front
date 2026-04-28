from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class PromptQueryLogger:
    def __init__(self, path: Path):
        self.path = path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)

    def log_generate_request(self, *, prompt_text: str, has_image: bool) -> None:
        normalized_prompt = prompt_text.strip()
        if not normalized_prompt and not has_image:
            return

        timestamp = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        header = (
            f"[{timestamp}] endpoint=/api/generate "
            f"prompt_chars={len(normalized_prompt)} image={'yes' if has_image else 'no'}"
        )
        body = normalized_prompt or "<empty>"
        entry = f"{header}\n{body}\n\n"

        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(entry)
