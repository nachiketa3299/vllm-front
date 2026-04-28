import asyncio
import contextlib
import json
from typing import Optional

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import AppConfig, AppPaths
from .models import AppError, RequestLog
from .service import GenerationService
from .token_budget import TokenBudgetService
from .vllm_client import VLLMClient


def error_response(status_code: int, detail: str, logs: list[str]) -> JSONResponse:
    logs.append(f"[ERROR] Request failed (HTTP {status_code}): {detail}")
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "logs": logs},
    )


async def wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


def create_router(
    *,
    paths: AppPaths,
    config: AppConfig,
    service: GenerationService,
    client: VLLMClient,
    token_budget: TokenBudgetService,
) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def index() -> FileResponse:
        return FileResponse(paths.static_dir / "index.html")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/api/config")
    def app_config() -> dict[str, list[dict[str, object]]]:
        return {
            "entries": [
                {
                    "key": "max_completion_tokens",
                    "label": "기본 출력 토큰",
                    "description": "한 응답에서 생성할 최대 토큰 수. 크게 두면 긴 답변이 가능하지만 컨텍스트 예산을 더 소모합니다.",
                    "value": str(config.max_completion_tokens),
                    "editable": True,
                    "control": "number",
                    "min": "1",
                    "step": "1",
                },
                {
                    "key": "temperature",
                    "label": "온도",
                    "description": "샘플링 온도 (temperature). 0.0이면 결정론적(매번 같은 답), 1.0이면 다양한 답. 보통 0.0~1.0 범위. 자동화/정확성엔 0.0, 창의적 작업엔 0.7~1.0.",
                    "value": "0.0",
                    "editable": True,
                    "control": "number",
                    "min": "0",
                    "step": "0.1",
                },
                {
                    "key": "json_output",
                    "label": "JSON 출력 강제 옵션",
                    "description": "켜면 모델이 JSON 객체로만 응답하도록 서버가 강제합니다.",
                    "value": False,
                    "editable": True,
                    "control": "checkbox",
                },
                {
                    "key": "enable_thinking",
                    "label": "사고 과정 사용",
                    "description": "켜면 답변 전에 추론 토큰(<think>)을 생성합니다. 복잡한 판단에는 품질이 올라가지만 응답이 느려집니다.",
                    "value": False,
                    "editable": True,
                    "control": "checkbox",
                },
            ],
            "info_entries": [
                {
                    "key": "max_image_bytes",
                    "label": "최대 이미지 크기 (KB)",
                    "description": "vllm-front 가 업로드를 거부하는 한도 (킬로바이트, 1 KB = 1024 B). vLLM 에는 전송되지 않음.",
                    "value": f"{config.max_image_bytes // 1024} KB",
                },
            ],
        }

    @router.get("/api/runtime")
    def runtime_info() -> dict[str, object]:
        """시작 시 설정된 정적 런타임 정보. vLLM 호출 안 함."""
        return {
            "vllm_base_url": config.vllm_base_url,
        }

    @router.get("/api/model/info")
    async def model_info() -> dict[str, object]:
        info = await client.probe_model_info(force=True)
        if info is None:
            return {
                "online": False,
                "model": None,
                "max_model_len": None,
                "base_url": config.vllm_base_url,
            }
        return {
            "online": True,
            "model": info.model,
            "max_model_len": info.max_model_len,
            "base_url": config.vllm_base_url,
        }

    @router.post("/api/token-budget")
    async def token_budget_preview(
        image: Optional[UploadFile] = File(default=None),
        user_request: Optional[str] = Form(default=None),
        max_completion_tokens: Optional[int] = Form(default=None),
        max_image_bytes: Optional[int] = Form(default=None),
    ) -> dict[str, object]:
        info = await client.require_model_info()
        if info.max_model_len is None:
            raise AppError(503, "vLLM 서버가 max_model_len을 보고하지 않아서 토큰 예산을 계산할 수 없습니다.")

        try:
            return await token_budget.estimate(
                upload=image,
                user_request=user_request,
                max_completion_tokens=max_completion_tokens or config.max_completion_tokens,
                max_model_len=info.max_model_len,
                max_image_bytes=max_image_bytes or config.max_image_bytes,
                model=info.model,
            )
        except AppError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

    @router.post("/api/generate")
    async def generate(
        image: Optional[UploadFile] = File(default=None),
        user_request: Optional[str] = Form(default=None),
        max_completion_tokens: Optional[int] = Form(default=None),
        max_image_bytes: Optional[int] = Form(default=None),
        json_output: Optional[bool] = Form(default=None),
        enable_thinking: Optional[bool] = Form(default=None),
        temperature: Optional[float] = Form(default=None),
    ) -> StreamingResponse:
        """SSE 스트리밍. 각 이벤트 형식:
            data: {"reasoning": "..."}        # <think> 안쪽 토큰 chunk
            data: {"content": "..."}          # 본문 토큰 chunk
            data: {"done": true, "logs": [..]}  # 정상 종료
            data: {"error": "...", "status": 502, "logs": [..]}  # 실패
        """
        log = RequestLog(entries=[])

        async def event_stream():
            try:
                async for chunk in service.generate_stream(
                    image,
                    log,
                    user_request=user_request,
                    max_completion_tokens=max_completion_tokens,
                    max_image_bytes=max_image_bytes,
                    json_output=json_output,
                    enable_thinking=enable_thinking,
                    temperature=temperature,
                ):
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                yield (
                    "data: "
                    + json.dumps({"done": True, "logs": log.entries}, ensure_ascii=False)
                    + "\n\n"
                )
            except AppError as exc:
                log.add(f"[ERROR] {exc.detail}")
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "error": exc.detail,
                            "status": exc.status_code,
                            "logs": log.entries,
                        },
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            except asyncio.CancelledError:
                log.add("Request cancelled")
                raise
            except Exception as exc:
                detail = f"Unhandled server error: {type(exc).__name__}: {exc}"
                log.add(f"[ERROR] {detail}")
                yield (
                    "data: "
                    + json.dumps(
                        {"error": detail, "status": 500, "logs": log.entries},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return router
