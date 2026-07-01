"""FastAPI entrypoint for StyleScribe."""

from fastapi import FastAPI

from backend.app.models.request_models import ArticleGenerationRequest
from backend.app.models.response_models import (
    ArticleGenerationStubResponse,
    HealthResponse,
)
from backend.app.services.generation_service import build_stub_generation_response

app = FastAPI(title="StyleScribe API")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service health."""

    return HealthResponse(status="ok", service="stylescribe-api")


@app.post("/generate/article", response_model=ArticleGenerationStubResponse)
def generate_article(
    request: ArticleGenerationRequest,
) -> ArticleGenerationStubResponse:
    """Return a realistic stub for the future article generation pipeline."""

    return build_stub_generation_response(request)
