"""FastAPI entrypoint for StyleScribe."""

from pathlib import Path

from fastapi import FastAPI, HTTPException

from backend.app.db.repository import StyleScribeRepository
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
from backend.app.services.author_ingestion_service import ingest_author_samples
from backend.app.services.generation_service import build_stub_generation_response
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
