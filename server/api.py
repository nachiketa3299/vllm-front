import asyncio
import contextlib

from fastapi import APIRouter, File, Form, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from .config import AppConfig, AppPaths
from .models import AppError, RequestLog
from .service import GenerationService


def error_response(status_code: int, detail: str, logs: list[str]) -> JSONResponse:
    logs.append(f"Request failed: {detail}")
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
) -> APIRouter:
    router = APIRouter()

    @router.get("/")
    def index() -> FileResponse:
        return FileResponse(paths.static_dir / "index.html")

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @router.get("/api/prompt")
    def prompt() -> dict[str, str]:
        return {"prompt_text": service.prompt_builder.assets.get_prompt_text()}

    @router.post("/api/generate")
    async def generate(
        request: Request,
        image: UploadFile = File(...),
        prompt_text: str | None = Form(default=None),
    ) -> Response:
        log = RequestLog(entries=[])
        generation_task = asyncio.create_task(service.generate(image, log, prompt_text))
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
                    "filename": config.zip_filename,
                    "struct": generated.struct,
                    "floats": generated.floats,
                    "zip_base64": generated.zip_base64,
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
