"""Response models for StyleScribe endpoints."""

from pydantic import BaseModel

from backend.app.models.request_models import ModelName


class HealthResponse(BaseModel):
    """Health endpoint response."""

    status: str
    service: str


class ArticleGenerationStubResponse(BaseModel):
    """Stub response for the article generation endpoint."""

    request_id: str
    status: str
    message: str
    selected_models: list[ModelName]
    target_language: str
    pipeline_steps: list[str]
