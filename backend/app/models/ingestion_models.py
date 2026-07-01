"""Models for local author sample ingestion."""

from pydantic import BaseModel, Field


class AuthorIngestionRequest(BaseModel):
    author_id: str = Field(..., min_length=1)
    display_name: str = Field(..., min_length=1)
    language: str = Field(..., min_length=1)
    articles_dir: str = Field(..., min_length=1)
    metadata_path: str | None = Field(default=None)


class IngestionSummary(BaseModel):
    run_id: str
    author_id: str
    status: str
    articles_seen: int
    articles_ingested: int
    articles_failed: int
    metadata_rows_seen: int
    warnings: list[str]


class ArticleListResponseItem(BaseModel):
    article_id: str
    filename: str
    title: str | None
    heading: str | None
    category: str | None
    text_char_count: int
    url: str | None
