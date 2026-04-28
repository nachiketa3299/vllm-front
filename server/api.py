import asyncio
import contextlib
from typing import Optional

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

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
                },
                {
                    "key": "timeout_seconds",
                    "label": "응답 타임아웃 (초)",
                    "description": "서버 응답을 기다리는 최대 시간(초). 짧게 두면 긴 출력이 완료 직전 끊길 수 있습니다.",
                    "value": str(config.timeout_seconds),
                    "editable": True,
                },
                {
                    "key": "max_image_bytes",
                    "label": "최대 이미지 크기 (바이트)",
                    "description": "업로드 이미지 한 장의 최대 파일 크기(바이트). 초과하면 요청이 거부됩니다.",
                    "value": str(config.max_image_bytes),
                    "editable": True,
                    "control": "number",
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
            ]
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
        if not info.model_path:
            raise AppError(503, "vLLM 서버가 모델 경로를 보고하지 않아서 토큰 예산을 계산할 수 없습니다.")

        try:
            return token_budget.estimate(
                upload=image,
                user_request=user_request,
                max_completion_tokens=max_completion_tokens or config.max_completion_tokens,
                max_model_len=info.max_model_len,
                max_image_bytes=max_image_bytes or config.max_image_bytes,
                model_path=info.model_path,
            )
        except AppError as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail},
            )

    @router.post("/api/generate")
    async def generate(
        request: Request,
        image: Optional[UploadFile] = File(default=None),
        user_request: Optional[str] = Form(default=None),
        max_completion_tokens: Optional[int] = Form(default=None),
        timeout_seconds: Optional[int] = Form(default=None),
        max_image_bytes: Optional[int] = Form(default=None),
        json_output: Optional[bool] = Form(default=None),
        enable_thinking: Optional[bool] = Form(default=None),
    ) -> Response:
        log = RequestLog(entries=[])
        generation_task = asyncio.create_task(
            service.generate(
                image,
                log,
                user_request=user_request,
                max_completion_tokens=max_completion_tokens,
                timeout_seconds=timeout_seconds,
                max_image_bytes=max_image_bytes,
                json_output=json_output,
                enable_thinking=enable_thinking,
            )
        )
        disconnect_task = asyncio.create_task(wait_for_disconnect(request))

        try:
            done, _ = await asyncio.wait(
                {generation_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if disconnect_task in done:
                log.add("Client disconnected; cancelling vLLM request")
                generation_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await generation_task
                return Response(status_code=499)

            disconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await disconnect_task

            generated = await generation_task
            return JSONResponse(
                {
                    "output_text": generated.text,
                    "reasoning_text": generated.reasoning,
                    "logs": log.entries,
                }
            )
        except AppError as exc:
            return error_response(exc.status_code, exc.detail, log.entries)
        except asyncio.CancelledError:
            log.add("Request cancelled")
            return Response(status_code=499)
        except Exception as exc:
            detail = f"Unhandled server error: {type(exc).__name__}: {exc}"
            return error_response(500, detail, log.entries)
        finally:
            for task in (generation_task, disconnect_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task

    return router
