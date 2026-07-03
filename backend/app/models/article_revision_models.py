"""Models for grounded article revisions."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ExportFormat = Literal["md", "markdown", "html"]


class ArticleRevisionRequest(BaseModel):
    evaluation_id: str | None = Field(default=None)
    run_final_evaluation: bool = False
    export_review: bool = False
    export_format: ExportFormat = "html"


class ArticleRevisionResponse(BaseModel):
    revision_id: str
    draft_id: str
    evaluation_id: str
    author_id: str
    model_provider: str
    model_name: str
    revised_draft: dict[str, Any]
    revision_summary: str
    removed_or_softened_claims: list[Any]
    revision_mode: str | None = None
    revision_patch_count: int = 0
    revision_patches_applied_count: int = 0
    revision_patches_skipped_count: int = 0
    revision_patch_skipped_reasons: list[Any] = Field(default_factory=list)
    revision_input_word_count: int | None = None
    revision_output_word_count: int | None = None
    revision_delta_word_count: int | None = None
    revision_rejected_for_length_collapse: bool = False
    revision_rejected_reason: str | None = None
    revised_article_source: str | None = None
    unsupported_claim_findings_count: int = 0
    unsupported_claim_patch_count: int = 0
    unsupported_claim_patches_applied_count: int = 0
    unsupported_claim_patches_skipped_count: int = 0
    unsupported_claim_patch_skipped_reasons: list[Any] = Field(default_factory=list)
    unsupported_claims_unresolved_count: int = 0
    unsupported_claims_cleared_by_patch: bool = False
    token_usage: dict[str, Any]
    created_at: str
    warnings: list[str]


class ArticleRevisionWorkflowResponse(BaseModel):
    revision: ArticleRevisionResponse
    initial_evaluation: dict[str, Any]
    final_evaluation_id: str | None = None
    final_evaluation: dict[str, Any] | None = None
    export_paths: list[str] = []
