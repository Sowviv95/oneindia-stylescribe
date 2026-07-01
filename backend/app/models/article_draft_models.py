"""Models for generated Tamil article drafts."""

from typing import Any

from pydantic import BaseModel, Field


class ArticleDraftRequest(BaseModel):
    author_id: str = Field(..., min_length=1)
    brief_id: str = Field(..., min_length=1)
    author_instruction: str | None = Field(default=None)
    target_language: str = Field(default="ta", min_length=1)


class ArticleDraftSummary(BaseModel):
    draft_id: str
    author_id: str
    profile_id: str
    brief_id: str
    target_language: str
    model_provider: str
    model_name: str
    status: str
    created_at: str


class ArticleDraftResponse(ArticleDraftSummary):
    draft: dict[str, Any]
    warnings: list[str]
