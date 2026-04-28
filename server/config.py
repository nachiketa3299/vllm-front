from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    static_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "AppPaths":
        return cls(
            base_dir=base_dir,
            static_dir=base_dir / "static",
        )


@dataclass(frozen=True)
class AppConfig:
    vllm_base_url: str
    max_completion_tokens: int
    max_image_bytes: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            max_completion_tokens=int(
                os.environ.get("VLLM_MAX_COMPLETION_TOKENS", "20000")
            ),
            max_image_bytes=15 * 1024 * 1024,
        )


BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = AppPaths.from_base_dir(BASE_DIR)
CONFIG = AppConfig.from_env()
