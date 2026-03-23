from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import create_router
from .assets import AssetStore
from .config import CONFIG, PATHS
from .prompting import PromptBuilder
from .service import GenerationService
from .transform import TemplateValueCodec
from .validation import TemplateValidator
from .vllm_client import VLLMClient


ASSETS = AssetStore(PATHS)
CODEC = TemplateValueCodec(ASSETS.struct_template, ASSETS.floats_template)
PROMPT_BUILDER = PromptBuilder(ASSETS, CODEC)
VLLM_CLIENT = VLLMClient(CONFIG)
VALIDATOR = TemplateValidator(ASSETS.struct_template, ASSETS.floats_template)
GENERATION_SERVICE = GenerationService(
    config=CONFIG,
    client=VLLM_CLIENT,
    codec=CODEC,
    prompt_builder=PROMPT_BUILDER,
    validator=VALIDATOR,
)


app = FastAPI(title="ChatGarment vLLM Tool")
app.mount("/static", StaticFiles(directory=PATHS.static_dir), name="static")
app.include_router(
    create_router(
        paths=PATHS,
        config=CONFIG,
        service=GENERATION_SERVICE,
    )
)
