from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .config import CONFIG, PATHS
from .prompt_logger import PromptQueryLogger
from .service import GenerationService
from .token_budget import TokenBudgetService
from .vllm_client import VLLMClient


TOKEN_BUDGET_SERVICE = TokenBudgetService(CONFIG)
VLLM_CLIENT = VLLMClient(CONFIG)
PROMPT_QUERY_LOGGER = PromptQueryLogger(PATHS.base_dir / "logs" / "request-prompts.log")
GENERATION_SERVICE = GenerationService(
    config=CONFIG,
    client=VLLM_CLIENT,
    prompt_logger=PROMPT_QUERY_LOGGER,
)


app = FastAPI(title="ChatGarment vLLM Tool")
app.mount("/static", StaticFiles(directory=PATHS.static_dir), name="static")
app.include_router(
    create_router(
        paths=PATHS,
        config=CONFIG,
        service=GENERATION_SERVICE,
        client=VLLM_CLIENT,
        token_budget=TOKEN_BUDGET_SERVICE,
    )
)
