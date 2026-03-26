from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import hashlib


@dataclass(frozen=True)
class AppPaths:
    base_dir: Path
    static_dir: Path
    runtime_dir: Path

    @classmethod
    def from_base_dir(cls, base_dir: Path) -> "AppPaths":
        digest = hashlib.sha1(str(base_dir).encode("utf-8")).hexdigest()[:10]
        return cls(
            base_dir=base_dir,
            static_dir=base_dir / "static",
            runtime_dir=Path(tempfile.gettempdir()) / f"vllm-server-runtime-{digest}",
        )


@dataclass(frozen=True)
class AppConfig:
    vllm_base_url: str
    model: str
    vllm_host: str
    vllm_port: int
    vllm_default_max_model_len: int
    vllm_reasoning_parser: str
    vllm_startup_timeout_seconds: int
    vllm_shutdown_grace_seconds: int
    max_completion_tokens: int
    timeout_seconds: int
    max_image_bytes: int

    @classmethod
    def from_env(cls) -> "AppConfig":
        return cls(
            vllm_base_url=os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
            model=os.environ.get(
                "VLLM_MODEL",
                "/home/mnnh-ruzen/Documents/projects/qwen3.5-27b",
            ),
            vllm_host=os.environ.get("VLLM_HOST", "127.0.0.1"),
            vllm_port=int(os.environ.get("VLLM_PORT", "8000")),
            vllm_default_max_model_len=int(
                os.environ.get("VLLM_DEFAULT_MAX_MODEL_LEN", "8192")
            ),
            vllm_reasoning_parser=os.environ.get("VLLM_REASONING_PARSER", "qwen3"),
            vllm_startup_timeout_seconds=int(
                os.environ.get("VLLM_STARTUP_TIMEOUT_SECONDS", "900")
            ),
            vllm_shutdown_grace_seconds=int(
                os.environ.get("VLLM_SHUTDOWN_GRACE_SECONDS", "15")
            ),
            max_completion_tokens=int(
                os.environ.get("VLLM_MAX_COMPLETION_TOKENS", "1024")
            ),
            timeout_seconds=int(os.environ.get("VLLM_TIMEOUT_SECONDS", "600")),
            max_image_bytes=15 * 1024 * 1024,
        )


BASE_DIR = Path(__file__).resolve().parent.parent
PATHS = AppPaths.from_base_dir(BASE_DIR)
CONFIG = AppConfig.from_env()
