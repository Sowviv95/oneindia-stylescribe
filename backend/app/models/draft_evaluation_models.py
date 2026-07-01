"""Models for draft grounding evaluations."""

from typing import Any

from pydantic import BaseModel


class DraftEvaluationResponse(BaseModel):
    evaluation_id: str
    draft_id: str
    brief_id: str
    author_id: str
    model_provider: str
    model_name: str
    status: str
    evaluation: dict[str, Any]
    warnings: list[str]
    created_at: str


class DraftEvaluationSummary(BaseModel):
    evaluation_id: str
    draft_id: str
    brief_id: str
    author_id: str
    model_provider: str
    model_name: str
    status: str
    overall_risk: str | None
    editorial_readiness: str | None
    created_at: str
