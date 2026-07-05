"""Models for multi-author article comparison workflows."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from backend.app.models.pasted_text_workflow_models import (
    SourceCleanupSummary,
    WorkflowBriefSummary,
    WorkflowEvaluationSummary,
    WorkflowMode,
)

RecommendationValue = Literal[
    "author_a",
    "author_b",
    "no_clear_recommendation",
]


class MultiAuthorComparisonRequest(BaseModel):
    source_text: str = Field(..., min_length=1)
    author_id_a: str = Field(..., min_length=1)
    author_id_b: str = Field(..., min_length=1)
    author_instruction: str | None = Field(default=None)
    target_language: str = Field(default="ta", min_length=1)
    article_type: str = Field(default="news", min_length=1)
    desired_word_count: int = Field(default=600, ge=250, le=1200)
    tone_override: str | None = Field(default=None)
    workflow_mode: WorkflowMode = "standard"
    generation_model: str | None = Field(default=None)


class AuthorComparisonOutput(BaseModel):
    author_id: str
    role: Literal["author_a", "author_b"]
    profile_id: str
    draft_id: str
    evaluation_id: str | None = None
    plan_id: str | None = None
    generated_headline: str | None = None
    generated_subheadline: str | None = None
    article_body: str
    word_count: int
    grounding_score: int | None = None
    final_readiness: str | None = None
    blockers: list[Any] = Field(default_factory=list)
    warnings: list[Any] = Field(default_factory=list)
    evaluation_summary: WorkflowEvaluationSummary | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)


class MultiAuthorComparisonSummary(BaseModel):
    factual_faithfulness_comparison: str
    author_style_difference: str
    readability_difference: str
    recommended_draft: RecommendationValue
    recommendation_rationale: str


class SharedGroundedBriefMetadata(BaseModel):
    brief_id: str
    source_language: str
    target_language: str
    model_provider: str
    model_name: str
    status: str
    source_text_excerpt: str


class MultiAuthorComparisonResponse(BaseModel):
    workflow_id: str
    workflow_completed: bool
    status: str
    desired_word_count: int
    target_min_word_count: int | None = None
    target_max_word_count: int | None = None
    workflow_mode: WorkflowMode
    source_cleanup: SourceCleanupSummary
    brief_summary: WorkflowBriefSummary
    shared_grounded_brief: SharedGroundedBriefMetadata
    author_a: AuthorComparisonOutput
    author_b: AuthorComparisonOutput
    comparison_summary: MultiAuthorComparisonSummary
    warnings: list[str] = Field(default_factory=list)
    aggregate_runtime_seconds: float | None = None
    aggregate_token_usage: dict[str, Any] = Field(default_factory=dict)
    aggregate_estimated_cost_usd: float | None = None
    telemetry: dict[str, Any] = Field(default_factory=dict)
