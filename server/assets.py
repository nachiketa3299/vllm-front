import json
import threading
from pathlib import Path
from typing import Any

from .config import AppPaths


class AssetStore:
    def __init__(self, paths: AppPaths):
        self.paths = paths
        self.struct_template = self._load_json(paths.struct_template_path)
        self.floats_template = self._load_json(paths.floats_template_path)
        self._lock = threading.Lock()
        self._cached_prompt_mtime: float | None = None
        self._cached_prompt_text: str | None = None

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def get_prompt_text(self) -> str:
        prompt_mtime = self.paths.prompt_path.stat().st_mtime
        with self._lock:
            if (
                self._cached_prompt_text is not None
                and self._cached_prompt_mtime == prompt_mtime
            ):
                return self._cached_prompt_text

            with self.paths.prompt_path.open("r", encoding="utf-8") as file:
                prompt_text = file.read().strip()
            self._cached_prompt_mtime = prompt_mtime
            self._cached_prompt_text = prompt_text
            return prompt_text

