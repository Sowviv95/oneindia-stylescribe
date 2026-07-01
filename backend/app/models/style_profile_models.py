"""Models for LLM-generated author style profiles."""

from typing import Any

from pydantic import BaseModel


class AuthorStyleProfileSummary(BaseModel):
    profile_id: str
    author_id: str
    snapshot_id: str
    language: str
    model_provider: str
    model_name: str
    status: str
    created_at: str


class AuthorStyleProfileResponse(AuthorStyleProfileSummary):
    profile: dict[str, Any]
    source_excerpt_refs: list[dict[str, Any]]
    warnings: list[str]
