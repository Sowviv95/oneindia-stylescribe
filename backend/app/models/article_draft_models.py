"""Models for generated Tamil article drafts."""

from typing import Any

from pydantic import BaseModel, Field


class ArticleDraftRequest(BaseModel):
    author_id: str = Field(..., min_length=1)
    brief_id: str = Field(..., min_length=1)
    author_instruction: str | None = Field(default=None)
    target_language: str = Field(default="ta", min_length=1)
    article_type: str | None = Field(default=None)
    desired_word_count: int | None = Field(default=None, ge=250, le=1200)
    tone_override: str | None = Field(default=None)
    include_seo: bool = Field(default=True)


class ArticleDraftSummary(BaseModel):
    draft_id: str
    author_id: str
    profile_id: str
    brief_id: str
    target_language: str
    model_provider: str
    model_name: str
    status: str
    article_type: str | None = None
    desired_word_count: int | None = None
    tone_override: str | None = None
    include_seo: bool = True
    created_at: str


class ArticleDraftResponse(ArticleDraftSummary):
    draft: dict[str, Any]
    warnings: list[str]
