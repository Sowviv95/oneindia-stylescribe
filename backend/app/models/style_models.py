"""Models for deterministic author style snapshots."""

from typing import Any

from pydantic import BaseModel


class AuthorStyleSnapshotSummary(BaseModel):
    snapshot_id: str
    author_id: str
    article_count: int
    language: str
    status: str
    created_at: str


class AuthorStyleSnapshotResponse(AuthorStyleSnapshotSummary):
    stats: dict[str, Any]
    excerpt_pack: dict[str, Any]
    warnings: list[str]
