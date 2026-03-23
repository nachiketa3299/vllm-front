from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    static_dir: Path
    prompt_path: Path
    struct_template_path: Path
    floats_template_path: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "AppPaths":
        templates_dir = base_dir / "templates"
        prompts_dir = base_dir / "prompts"
        return cls(
            base_dir=base_dir,
            static_dir=base_dir / "static",
            prompt_path=prompts_dir / "chatgarment_prompt.txt",
            struct_template_path=templates_dir / "struct.json",
            floats_template_path=templates_dir / "floats.json",
        )


@dataclass(frozen=True)
class AppConfig:
    vllm_base_url: str
    model: str
    max_completion_tokens: int
    timeout_seconds: int
    zip_filename: str
    max_image_bytes: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get(
                "VLLM_MODEL",
                "/home/mnnh-ruzen/Documents/projects/qwen3.5-27b",
            ),
            max_completion_tokens=int(
                os.environ.get("VLLM_MAX_COMPLETION_TOKENS", "1024")
            ),
            timeout_seconds=int(os.environ.get("VLLM_TIMEOUT_SECONDS", "600")),
            zip_filename="chatgarment_result.zip",
            max_image_bytes=15 * 1024 * 1024,
        )


BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = AppPaths.from_base_dir(BASE_DIR)
CONFIG = AppConfig.from_env()

