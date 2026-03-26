import asyncio
import contextlib
from typing import Optional

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import AppConfig, AppPaths
from .model_control import ModelControlService
from .models import AppError, ModelRuntimeStatus, RequestLog
from .service import GenerationService
from .token_budget import TokenBudgetService


def error_response(status_code: int, detail: str, logs: list[str]) -> JSONResponse:
    logs.append(f"Request failed: {detail}")
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "logs": logs},
    )


async def wait_for_disconnect(request: Request) -> None:
    while not await request.is_disconnected():
        await asyncio.sleep(0.25)


class StartModelRequest(BaseModel):
    max_model_len: int


def serialize_runtime(status: ModelRuntimeStatus) -> dict[str, object]:
    return {
        "status": status.status,
        "ownership": status.ownership,
        "model": status.model,
        "pid": status.pid,
        "current_max_model_len": status.current_max_model_len,
        "default_max_model_len": status.default_max_model_len,
        "theoretical_max_model_len": status.theoretical_max_model_len,
        "observed_kv_cache_tokens": status.observed_kv_cache_tokens,
        "recommended_max_model_len": status.recommended_max_model_len,
        "recommended_max_model_len_reason": status.recommended_max_model_len_reason,
        "detail": status.detail,
        "can_start": status.can_start,
        "can_stop": status.can_stop,
    }


def create_router(
    *,
    paths: AppPaths,
    config: AppConfig,
    service: GenerationService,
    model_control: ModelControlService,
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
                    "label": "기본 출력 토큰 (max_completion_tokens)",
                    "value": str(config.max_completion_tokens),
                    "editable": True,
                },
                {
                    "key": "timeout_seconds",
                    "label": "타임아웃 초 (timeout_seconds)",
                    "value": str(config.timeout_seconds),
                    "editable": True,
                },
                {
                    "key": "max_image_bytes",
                    "label": "최대 이미지 바이트 (max_image_bytes)",
                    "value": str(config.max_image_bytes),
                    "editable": True,
                    "control": "number",
                },
                {
                    "key": "json_output",
                    "label": "JSON 출력 (json_output)",
                    "value": False,
                    "editable": True,
                    "control": "checkbox",
                },
            ]
        }

    @router.get("/api/model/runtime")
    async def model_runtime() -> dict[str, object]:
        return serialize_runtime(await model_control.get_runtime_status())

    @router.post("/api/model/start", status_code=202)
    async def start_model(payload: StartModelRequest) -> dict[str, object]:
        runtime = await model_control.start(max_model_len=payload.max_model_len)
        return serialize_runtime(runtime)

    @router.post("/api/model/stop", status_code=202)
    async def stop_model() -> dict[str, object]:
        runtime = await model_control.stop()
        return serialize_runtime(runtime)

    @router.post("/api/token-budget")
    async def token_budget_preview(
        image: Optional[UploadFile] = File(default=None),
        user_request: Optional[str] = Form(default=None),
        max_completion_tokens: Optional[int] = Form(default=None),
        max_model_len: Optional[int] = Form(default=None),
        max_image_bytes: Optional[int] = Form(default=None),
    ) -> dict[str, object]:
        runtime = await model_control.get_runtime_status()
        resolved_max_model_len = max_model_len
        if resolved_max_model_len is None:
            resolved_max_model_len = (
                runtime.current_max_model_len or runtime.default_max_model_len
            )

        return token_budget.estimate(
            upload=image,
            user_request=user_request,
            max_completion_tokens=max_completion_tokens or config.max_completion_tokens,
            max_model_len=resolved_max_model_len,
            max_image_bytes=max_image_bytes or config.max_image_bytes,
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
