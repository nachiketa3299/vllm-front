from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .config import CONFIG, PATHS
from .model_control import ModelControlService
from .service import GenerationService
from .token_budget import TokenBudgetService
from .vllm_client import VLLMClient


MODEL_CONTROL_SERVICE = ModelControlService(
    config=CONFIG,
    paths=PATHS,
)
TOKEN_BUDGET_SERVICE = TokenBudgetService(CONFIG)
VLLM_CLIENT = VLLMClient(CONFIG)
GENERATION_SERVICE = GenerationService(
    config=CONFIG,
    client=VLLM_CLIENT,
    model_control=MODEL_CONTROL_SERVICE,
)


app = FastAPI(title="ChatGarment vLLM Tool")
app.mount("/static", StaticFiles(directory=PATHS.static_dir), name="static")
app.include_router(
    create_router(
        paths=PATHS,
        config=CONFIG,
        service=GENERATION_SERVICE,
        model_control=MODEL_CONTROL_SERVICE,
        token_budget=TOKEN_BUDGET_SERVICE,
    )
)
