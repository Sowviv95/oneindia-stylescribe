"""Models for pasted website text to article draft workflow."""

from typing import Any, Literal

from pydantic import BaseModel, Field

ExportFormat = Literal["md", "markdown", "html"]
WorkflowMode = Literal["standard", "fast_review", "full_quality"]


class PastedTextWorkflowRequest(BaseModel):
    author_id: str = Field(..., min_length=1)
    source_text: str = Field(..., min_length=1)
    author_instruction: str | None = Field(default=None)
    target_language: str = Field(default="ta", min_length=1)
    article_type: str = Field(default="news", min_length=1)
    desired_word_count: int = Field(default=600, ge=250, le=1200)
    tone_override: str | None = Field(default=None)
    run_grounding_evaluation: bool = True
    run_auto_revision: bool = False
    run_final_evaluation: bool = False
    export_review: bool = False
    export_format: ExportFormat = "html"
    workflow_mode: WorkflowMode = "standard"


class SourceCleanupSummary(BaseModel):
    original_char_count: int
    cleaned_char_count: int
    removed_line_count: int
    warnings: list[str]


class WorkflowBriefSummary(BaseModel):
    topic: str
    one_line_summary: str
    confirmed_facts: list[Any]
    claims_to_avoid: list[Any]


class WorkflowDraftSummary(BaseModel):
    headline: str
    subheadline: str
    seo_title: str
    tags: list[Any]


class WorkflowEvaluationSummary(BaseModel):
    grounding_score: int | None = None
    claim_safety_score: int | None = None
    overall_risk: str | None = None
    editorial_readiness: str | None = None


class PastedTextWorkflowResponse(BaseModel):
    workflow_id: str
    workflow_run_id: str | None = None
    status: str
    author_id: str
    brief_id: str
    draft_id: str
    evaluation_id: str | None
    initial_evaluation_id: str | None = None
    revision_id: str | None = None
    final_evaluation_id: str | None = None
    generated_headline: str | None = None
    generated_subheadline: str | None = None
    source_cleanup: SourceCleanupSummary
    brief_summary: WorkflowBriefSummary
    draft_summary: WorkflowDraftSummary
    evaluation_summary: WorkflowEvaluationSummary | None
    initial_readiness: str | None = None
    initial_readiness_reasons: list[Any] = Field(default_factory=list)
    final_readiness: str | None = None
    final_readiness_reasons: list[Any] = Field(default_factory=list)
    readiness_decision_source: str | None = None
    final_publication_blockers: list[Any] = Field(default_factory=list)
    final_publication_warnings: list[Any] = Field(default_factory=list)
    publication_ready_completeness_passed: bool = False
    final_evaluation_summary: WorkflowEvaluationSummary | None = None
    article_plan_used: bool = False
    plan_id: str | None = None
    desired_word_count: int | None = None
    target_min_word_count: int | None = None
    target_max_word_count: int | None = None
    generation_mode_used: str | None = None
    generated_section_count: int = 0
    assembled_section_count: int = 0
    section_assembled_article_word_count: int | None = None
    section_assembled_article_paragraph_count: int | None = None
    original_draft_source: str | None = None
    original_draft_word_count_after_assignment: int | None = None
    original_draft_matches_section_assembly: bool | None = None
    revision_input_word_count: int | None = None
    revision_mode: str | None = None
    revision_patch_count: int = 0
    revision_patches_applied_count: int = 0
    revision_patches_skipped_count: int = 0
    revision_patch_skipped_reasons: list[Any] = Field(default_factory=list)
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
    length_recovery_skipped_reason: str | None = None
    length_recovery_input_word_count: int | None = None
    final_article_source_stage: str | None = None
    final_word_count_ratio: float | None = None
    section_generation_trace: list[dict[str, Any]] = Field(default_factory=list)
    max_concurrent_section_calls: int | None = None
    generation_section_group_size: int | None = None
    generation_group_call_count: int = 0
    generation_single_section_fallback_count: int = 0
    generation_context_pack_tokens: int | None = None
    generation_context_pack_chars: int | None = None
    original_generation_context_chars: int | None = None
    compressed_generation_context_chars: int | None = None
    generation_context_compression_ratio: float | None = None
    planned_section_count: int = 0
    planned_target_word_count: int | None = None
    planned_min_word_count: int | None = None
    planned_max_word_count: int | None = None
    section_coverage_status: str | None = None
    section_coverage_warnings: list[str] = Field(default_factory=list)
    tamil_quality_status: str | None = None
    tamil_quality_issues_count: int = 0
    tamil_quality_warnings: list[str] = Field(default_factory=list)
    requested_word_count: int | None = None
    original_draft_word_count: int | None = None
    revised_word_count_before_expansion: int | None = None
    final_article_word_count: int | None = None
    length_status: str | None = None
    length_warning_reason: str | None = None
    final_article_word_count_ratio: float | None = None
    length_recovery_required: bool = False
    length_recovery_attempted: bool = False
    length_recovery_succeeded: bool = False
    length_recovery_failed: bool = False
    short_output_invalid: bool = False
    expansion_items_available: int = 0
    expansion_items_used: list[Any] = Field(default_factory=list)
    export_paths: list[str]
    warnings: list[str]
    workflow_mode: WorkflowMode = "standard"
    total_runtime_seconds: float | None = None
    llm_call_count_total: int = 0
    llm_call_count_by_stage: dict[str, Any] = Field(default_factory=dict)
    runtime_by_stage: dict[str, Any] = Field(default_factory=dict)
    slowest_stage: str | None = None
    planning_runtime_seconds: float | None = None
    section_generation_runtime_seconds: float | None = None
    section_generation_call_count: int = 0
    section_retry_call_count: int = 0
    generation_runtime_seconds: float | None = None
    initial_evaluation_runtime_seconds: float | None = None
    revision_runtime_seconds: float | None = None
    final_evaluation_runtime_seconds: float | None = None
    length_recovery_runtime_seconds: float | None = None
    export_runtime_seconds: float | None = None
    model_used_by_stage: dict[str, Any] = Field(default_factory=dict)
    planning_model_used: str | None = None
    generation_model_used: str | None = None
    revision_model_used: str | None = None
    evaluation_model_used: str | None = None
    length_recovery_model_used: str | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    cached_prompt_tokens_total: int = 0
    uncached_prompt_tokens_total: int = 0
    prompt_cache_hit_ratio: float | None = None
    token_usage_by_stage: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens_by_stage: dict[str, Any] = Field(default_factory=dict)
    completion_tokens_by_stage: dict[str, Any] = Field(default_factory=dict)
    cached_prompt_tokens_by_stage: dict[str, Any] = Field(default_factory=dict)
    uncached_prompt_tokens_by_stage: dict[str, Any] = Field(default_factory=dict)
    prompt_cache_hit_ratio_by_stage: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_total_usd: float | None = None
    estimated_cost_by_stage_usd: dict[str, Any] = Field(default_factory=dict)
    estimated_cost_model_breakdown: dict[str, Any] = Field(default_factory=dict)
    cost_estimation_available: bool = False
    cost_estimation_notes: list[str] = Field(default_factory=list)
    highest_cost_stage: str | None = None
