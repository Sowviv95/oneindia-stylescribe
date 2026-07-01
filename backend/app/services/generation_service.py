"""Stub generation service for Sprint 1."""

from uuid import uuid4

from backend.app.models.request_models import ArticleGenerationRequest
from backend.app.models.response_models import ArticleGenerationStubResponse

PIPELINE_STEPS = [
    "source_processing",
    "grounded_brief_generation",
    "author_style_retrieval",
    "multi_model_generation",
    "qc_evaluation",
]


def build_stub_generation_response(
    request: ArticleGenerationRequest,
) -> ArticleGenerationStubResponse:
    """Build a deterministic-shape stub response without invoking any LLMs."""

    return ArticleGenerationStubResponse(
        request_id=str(uuid4()),
        status="stub",
        message="Article generation pipeline is not implemented yet.",
        selected_models=request.models,
        target_language=request.target_language,
        pipeline_steps=PIPELINE_STEPS,
    )
