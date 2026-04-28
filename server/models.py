import base64
from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestLog:
    entries: list[str]

    def add(self, message: str) -> None:
        self.entries.append(message)


@dataclass(frozen=True)
class PreparedImage:
    bytes_data: bytes
    mime_type: str

    @property
    def data_url(self) -> str:
        encoded = base64.b64encode(self.bytes_data).decode("ascii")
        return f"data:{self.mime_type};base64,{encoded}"


@dataclass(frozen=True)
class GeneratedPayload:
    text: str
    reasoning: str = ""


@dataclass(frozen=True)
class ProbedModelInfo:
    model: str
    max_model_len: Optional[int]


class AppError(Exception):
    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
