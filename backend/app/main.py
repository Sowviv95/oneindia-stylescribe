"""FastAPI entrypoint for StyleScribe."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.app.db.repository import StyleScribeRepository
from backend.app.models.article_draft_models import (
    ArticleDraftRequest,
    ArticleDraftResponse,
)
from backend.app.models.grounded_brief_models import (
    GroundedBriefRequest,
    GroundedBriefResponse,
)
from backend.app.models.ingestion_models import (
    ArticleListResponseItem,
    AuthorIngestionRequest,
    IngestionSummary,
)
from backend.app.models.request_models import ArticleGenerationRequest
from backend.app.models.response_models import (
    ArticleGenerationStubResponse,
    HealthResponse,
)
from backend.app.models.style_models import AuthorStyleSnapshotResponse
from backend.app.models.style_profile_models import AuthorStyleProfileResponse
from backend.app.services.article_generation_service import (
    ArticleGenerationError,
    generate_article_draft,
    get_article_draft,
)
from backend.app.services.author_ingestion_service import ingest_author_samples
from backend.app.services.author_style_profile_service import (
    AuthorStyleProfileError,
    generate_author_style_profile,
    get_latest_author_style_profile,
)
from backend.app.services.generation_service import build_stub_generation_response
from backend.app.services.grounded_brief_service import (
    GroundedBriefError,
    generate_grounded_brief,
    get_grounded_brief,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError
from backend.app.services.source_processor import SourceProcessingError
from backend.app.services.style_snapshot_service import (
    AuthorStyleSnapshotError,
    build_author_style_snapshot,
    get_latest_author_style_snapshot,
)

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


@app.post("/authors/ingest-local", response_model=IngestionSummary)
def ingest_local_author_samples(request: AuthorIngestionRequest) -> IngestionSummary:
    """Ingest local author sample DOCX files and optional metadata."""

    metadata_path = Path(request.metadata_path) if request.metadata_path else None
    return ingest_author_samples(
        author_id=request.author_id,
        display_name=request.display_name,
        language=request.language,
        articles_dir=Path(request.articles_dir),
        metadata_path=metadata_path,
    )


@app.get(
    "/authors/{author_id}/articles",
    response_model=list[ArticleListResponseItem],
)
def list_author_articles(author_id: str) -> list[ArticleListResponseItem]:
    """Return lightweight article records for an author."""

    repository = StyleScribeRepository()
    repository.initialize_schema()
    return [
        ArticleListResponseItem(
            article_id=article.article_id,
            filename=article.filename,
            title=article.title,
            heading=article.heading,
            category=article.category,
            text_char_count=article.text_char_count,
            url=article.url,
        )
        for article in repository.list_articles_for_author(author_id)
    ]


@app.post(
    "/authors/{author_id}/style-snapshot",
    response_model=AuthorStyleSnapshotResponse,
)
def create_author_style_snapshot(author_id: str) -> AuthorStyleSnapshotResponse:
    """Build and store a deterministic style snapshot for an author."""

    try:
        return build_author_style_snapshot(author_id)
    except AuthorStyleSnapshotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/authors/{author_id}/style-snapshot/latest",
    response_model=AuthorStyleSnapshotResponse,
)
def latest_author_style_snapshot(author_id: str) -> AuthorStyleSnapshotResponse:
    """Return the latest deterministic style snapshot for an author."""

    try:
        return get_latest_author_style_snapshot(author_id)
    except AuthorStyleSnapshotError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post(
    "/authors/{author_id}/style-profile",
    response_model=AuthorStyleProfileResponse,
)
def create_author_style_profile(author_id: str) -> AuthorStyleProfileResponse:
    """Generate and store an OpenAI-based author style profile."""

    try:
        return generate_author_style_profile(author_id)
    except AuthorStyleProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OpenAIClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get(
    "/authors/{author_id}/style-profile/latest",
    response_model=AuthorStyleProfileResponse,
)
def latest_author_style_profile(author_id: str) -> AuthorStyleProfileResponse:
    """Return the latest saved OpenAI-based author style profile."""

    try:
        return get_latest_author_style_profile(author_id)
    except AuthorStyleProfileError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/briefs/grounded", response_model=GroundedBriefResponse)
def create_grounded_brief(request: GroundedBriefRequest) -> GroundedBriefResponse:
    """Generate and save a grounded factual brief."""

    try:
        return generate_grounded_brief(
            source_type=request.source_type,
            source_input=request.source_input,
            target_language=request.target_language,
        )
    except SourceProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except GroundedBriefError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OpenAIClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/briefs/{brief_id}", response_model=GroundedBriefResponse)
def read_grounded_brief(brief_id: str) -> GroundedBriefResponse:
    """Return a saved grounded factual brief."""

    try:
        return get_grounded_brief(brief_id)
    except GroundedBriefError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/drafts/article", response_model=ArticleDraftResponse)
def create_article_draft(request: ArticleDraftRequest) -> ArticleDraftResponse:
    """Generate and save a controlled Tamil article draft."""

    try:
        return generate_article_draft(
            author_id=request.author_id,
            brief_id=request.brief_id,
            author_instruction=request.author_instruction,
            target_language=request.target_language,
            article_type=request.article_type,
            desired_word_count=request.desired_word_count,
            tone_override=request.tone_override,
            include_seo=request.include_seo,
        )
    except ArticleGenerationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except OpenAIClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/drafts/{draft_id}", response_model=ArticleDraftResponse)
def read_article_draft(draft_id: str) -> ArticleDraftResponse:
    """Return a saved article draft."""

    try:
        return get_article_draft(draft_id)
    except ArticleGenerationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
