"""End-to-end workflow for pasted website text to Tamil article draft."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Protocol, TypeVar
from uuid import uuid4

from backend.app.db.repository import StyleScribeRepository, WorkflowRunRecord
from backend.app.models.article_revision_models import ArticleRevisionResponse
from backend.app.models.pasted_text_workflow_models import (
    PastedTextWorkflowResponse,
    SourceCleanupSummary,
    WorkflowBriefSummary,
    WorkflowDraftSummary,
    WorkflowEvaluationSummary,
)
from backend.app.scripts.review_article_draft import TAMIL_FONT_STACK
from backend.app.services.article_generation_service import generate_article_draft
from backend.app.services.article_length_recovery_service import (
    assess_length_recovery_need,
    count_expansion_items,
    expand_article_to_target_length,
)
from backend.app.services.article_plan_service import generate_article_plan
from backend.app.services.article_revision_service import (
    export_revision_review,
    get_latest_article_revision,
    revise_article_grounding,
)
from backend.app.services.draft_grounding_evaluation_service import (
    evaluate_draft_grounding,
    evaluate_revision_grounding,
)
from backend.app.services.grounded_brief_service import generate_grounded_brief
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)
from backend.app.services.source_processor import ProcessedSource, process_source
from backend.app.services.tamil_quality_scanner import (
    approximate_tamil_word_count,
    scan_tamil_quality,
)
from backend.app.services.workflow_telemetry import (
    WorkflowTelemetry,
    resolve_stage_model,
)

WORKFLOW_TYPE = "pasted_text_to_draft"
REVIEW_OUTPUT_DIR = Path("review_outputs")
SOURCE_REVIEW_EXCERPT_CHARS = 1000
LOGGER = logging.getLogger(__name__)
T = TypeVar("T")
GROUNDING_READY_THRESHOLD = 90
GROUNDING_REVIEW_THRESHOLD = 75


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class PastedTextWorkflowError(RuntimeError):
    """Raised when the pasted text workflow cannot complete."""


@dataclass(frozen=True)
class ReadinessDecision:
    readiness: str | None
    reasons: list[str]
    source: str
    blockers: list[str]
    warnings: list[str]
    publication_ready_completeness_passed: bool


def run_pasted_text_to_draft_workflow(
    author_id: str,
    source_text: str,
    author_instruction: str | None = None,
    target_language: str = "ta",
    article_type: str = "news",
    desired_word_count: int = 600,
    tone_override: str | None = None,
    run_grounding_evaluation: bool = True,
    run_auto_revision: bool = False,
    run_final_evaluation: bool = False,
    export_review: bool = False,
    export_format: str = "html",
    workflow_mode: str = "standard",
    repository: StyleScribeRepository | None = None,
    brief_model_client: StructuredJsonClient | None = None,
    plan_model_client: StructuredJsonClient | None = None,
    draft_model_client: StructuredJsonClient | None = None,
    evaluation_model_client: StructuredJsonClient | None = None,
    revision_model_client: StructuredJsonClient | None = None,
    length_recovery_model_client: StructuredJsonClient | None = None,
) -> PastedTextWorkflowResponse:
    """Run the pasted website text cleanup, brief, draft, and evaluation flow."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    workflow_id = str(uuid4())
    workflow_started = perf_counter()
    telemetry = WorkflowTelemetry(started_at=workflow_started)
    processed = _run_stage(
        workflow_id,
        "pasted text cleanup",
        lambda: process_source("text", source_text, "pasted_web_text"),
        telemetry=telemetry,
        telemetry_stage="source_cleanup",
    )

    brief_response = _run_stage(
        workflow_id,
        "grounded brief generation",
        lambda: generate_grounded_brief(
            source_type="text",
            source_input=processed.cleaned_text,
            target_language=target_language,
            source_input_mode="plain_text",
            repository=repo,
            model_client=brief_model_client
            or _stage_client(
                "planning",
                "OPENAI_API_KEY is required for grounded briefs.",
            ),
        ),
        telemetry=telemetry,
        telemetry_stage="brief_generation",
    )
    telemetry.record_model("brief_generation", brief_response.model_name)
    telemetry.record_calls("brief_generation", 1)
    telemetry.record_tokens(
        "brief_generation",
        _dict_value(brief_response.brief.get("token_usage")),
    )
    plan_response = None
    if desired_word_count:
        plan_response = _run_stage(
            workflow_id,
            "grounded article plan generation",
            lambda: generate_article_plan(
                brief_id=brief_response.brief_id,
                author_id=author_id,
                article_type=article_type,
                desired_word_count=desired_word_count,
                target_language=target_language,
                tone_override=tone_override,
                author_instruction=author_instruction,
                repository=repo,
                model_client=plan_model_client
                or _stage_client(
                    "planning",
                    "OPENAI_API_KEY is required for article planning.",
                ),
            ),
            telemetry=telemetry,
            telemetry_stage="planning",
        )
        telemetry.record_model("planning", plan_response.model_name)
        telemetry.record_calls("planning", 1)
        telemetry.record_tokens("planning", plan_response.token_usage)
    draft_response = _run_stage(
        workflow_id,
        "initial draft generation",
        lambda: generate_article_draft(
            author_id=author_id,
            brief_id=brief_response.brief_id,
            author_instruction=author_instruction,
            target_language=target_language,
            article_type=article_type,
            desired_word_count=desired_word_count,
            tone_override=tone_override,
            plan_id=plan_response.plan_id if plan_response else None,
            repository=repo,
            model_client=draft_model_client
            or _stage_client(
                "generation",
                "OPENAI_API_KEY is required for article draft generation.",
            ),
        ),
        telemetry=telemetry,
        telemetry_stage="generation",
    )
    draft_generation_metadata = _draft_generation_metadata(draft_response.draft)
    telemetry.record_model("generation", draft_response.model_name)
    section_trace = _section_generation_trace(draft_response.draft)
    section_retry_count = sum(
        1 for trace in section_trace if trace.get("retry_attempted")
    )
    section_call_count = len(section_trace) + section_retry_count
    telemetry.record_calls("generation", 1 + section_call_count)
    telemetry.record_calls("section_generation", len(section_trace))
    telemetry.record_calls("section_retries", section_retry_count)
    telemetry.record_tokens("generation", _generation_token_usage(draft_response.draft))

    evaluation_response = None
    if run_grounding_evaluation or run_auto_revision:
        evaluation_response = _run_stage(
            workflow_id,
            "initial grounding evaluation",
            lambda: evaluate_draft_grounding(
                draft_response.draft_id,
                repository=repo,
                model_client=evaluation_model_client
                or _stage_client(
                    "evaluation",
                    "OPENAI_API_KEY is required for draft evaluation.",
                ),
            ),
            telemetry=telemetry,
            telemetry_stage="initial_evaluation",
        )
        telemetry.record_model("initial_evaluation", evaluation_response.model_name)
        telemetry.record_calls("initial_evaluation", 1)
        telemetry.record_tokens(
            "initial_evaluation",
            _dict_value(evaluation_response.evaluation.get("token_usage")),
        )

    revision_response = None
    expansion_response = None
    final_evaluation_response = None
    revised_word_count_before_expansion = None
    length_recovery_required = False
    length_recovery_attempted = False
    length_recovery_succeeded = False
    length_recovery_failed = False
    short_output_invalid = False
    expansion_items_used: list[object] = []
    revision_input_word_count = None
    revision_mode = None
    revision_patch_count = 0
    revision_patches_applied_count = 0
    revision_patches_skipped_count = 0
    revision_patch_skipped_reasons: list[object] = []
    revision_output_word_count = None
    revision_delta_word_count = None
    revision_rejected_for_length_collapse = False
    revision_rejected_reason = None
    revised_article_source = None
    unsupported_claim_findings_count = 0
    unsupported_claim_patch_count = 0
    unsupported_claim_patches_applied_count = 0
    unsupported_claim_patches_skipped_count = 0
    unsupported_claim_patch_skipped_reasons: list[object] = []
    unsupported_claims_unresolved_count = 0
    unsupported_claims_cleared_by_patch = False
    if run_auto_revision:
        if evaluation_response is None:
            raise PastedTextWorkflowError(
                "Auto revision requires grounding evaluation."
            )
        revision_response = _run_stage(
            workflow_id,
            "auto revision",
            lambda: revise_article_grounding(
                draft_response.draft_id,
                evaluation_id=evaluation_response.evaluation_id,
                repository=repo,
                model_client=revision_model_client
                or _stage_client(
                    "revision",
                    "OPENAI_API_KEY is required for article revision.",
                ),
            ),
            telemetry=telemetry,
            telemetry_stage="revision",
        )
        telemetry.record_model("revision", revision_response.model_name)
        telemetry.record_calls("revision", 1)
        telemetry.record_tokens("revision", _dict_value(revision_response.token_usage))
        revised_word_count_before_expansion = approximate_tamil_word_count(
            str(revision_response.revised_draft.get("article_body") or "")
        )
        revision_guardrail = _revision_guardrail_metadata(
            revision_response.token_usage
        )
        revision_input_word_count = _optional_int(
            revision_guardrail.get("revision_input_word_count")
        )
        revision_mode = _optional_str(revision_guardrail.get("revision_mode"))
        revision_patch_count = _optional_int(
            revision_guardrail.get("revision_patch_count")
        ) or 0
        revision_patches_applied_count = _optional_int(
            revision_guardrail.get("revision_patches_applied_count")
        ) or 0
        revision_patches_skipped_count = _optional_int(
            revision_guardrail.get("revision_patches_skipped_count")
        ) or 0
        revision_patch_skipped_reasons = _list_value(
            revision_guardrail.get("revision_patch_skipped_reasons")
        )
        revision_output_word_count = _optional_int(
            revision_guardrail.get("revision_output_word_count")
        )
        revision_delta_word_count = _optional_int(
            revision_guardrail.get("revision_delta_word_count")
        )
        revision_rejected_for_length_collapse = bool(
            revision_guardrail.get("revision_rejected_for_length_collapse")
        )
        revision_rejected_reason = _optional_str(
            revision_guardrail.get("revision_rejected_reason")
        )
        revised_article_source = _optional_str(
            revision_guardrail.get("revised_article_source")
        )
        unsupported_claim_findings_count = _optional_int(
            revision_guardrail.get("unsupported_claim_findings_count")
        ) or 0
        unsupported_claim_patch_count = _optional_int(
            revision_guardrail.get("unsupported_claim_patch_count")
        ) or 0
        unsupported_claim_patches_applied_count = _optional_int(
            revision_guardrail.get("unsupported_claim_patches_applied_count")
        ) or 0
        unsupported_claim_patches_skipped_count = _optional_int(
            revision_guardrail.get("unsupported_claim_patches_skipped_count")
        ) or 0
        unsupported_claim_patch_skipped_reasons = _list_value(
            revision_guardrail.get("unsupported_claim_patch_skipped_reasons")
        )
        unsupported_claims_unresolved_count = _optional_int(
            revision_guardrail.get("unsupported_claims_unresolved_count")
        ) or 0
        unsupported_claims_cleared_by_patch = bool(
            revision_guardrail.get("unsupported_claims_cleared_by_patch")
        )
        recovery_decision = assess_length_recovery_need(
            revision_response.revised_draft,
            brief_response.brief,
            desired_word_count,
        )
        length_recovery_required = recovery_decision.length_recovery_required
        short_output_invalid = recovery_decision.short_output_invalid
        if length_recovery_required:
            draft_record = repo.fetch_article_draft(draft_response.draft_id)
            brief_record = repo.fetch_grounded_brief(brief_response.brief_id)
            revision_record = repo.fetch_article_revision(revision_response.revision_id)
            evaluation_record = repo.fetch_draft_evaluation(
                evaluation_response.evaluation_id
            )
            if (
                draft_record is None
                or brief_record is None
                or revision_record is None
                or evaluation_record is None
            ):
                raise PastedTextWorkflowError(
                    "Length recovery requires saved draft, brief, revision, and "
                    "evaluation records."
                )
            profile_record = repo.fetch_author_style_profile(draft_record.profile_id)
            if profile_record is None:
                raise PastedTextWorkflowError(
                    "Length recovery requires saved author style profile."
                )
            length_recovery_attempted = True
            expansion_response = _run_stage(
                workflow_id,
                "grounded length recovery",
                lambda: expand_article_to_target_length(
                    current_revision=revision_record,
                    brief=brief_record,
                    evaluation=evaluation_record,
                    profile=profile_record,
                    desired_word_count=desired_word_count,
                    article_type=article_type,
                    target_language=target_language,
                    tone_override=tone_override,
                    repository=repo,
                    model_client=length_recovery_model_client
                    or _stage_client(
                        "length_recovery",
                        "OPENAI_API_KEY is required for length recovery.",
                    ),
                ),
                telemetry=telemetry,
                telemetry_stage="length_recovery",
            )
            telemetry.record_model("length_recovery", expansion_response.model_name)
            telemetry.record_calls("length_recovery", 1)
            telemetry.record_tokens("length_recovery", expansion_response.token_usage)
            expansion_items_used = expansion_response.expansion_items_used
            revision_response = _revision_response_from_expansion(
                expansion_response.revision_id,
                repo,
            )
            expanded_decision = assess_length_recovery_need(
                revision_response.revised_draft,
                brief_response.brief,
                desired_word_count,
            )
            length_recovery_succeeded = not expanded_decision.short_output_invalid
            length_recovery_failed = expanded_decision.short_output_invalid
        if run_final_evaluation:
            final_evaluation_response = _run_stage(
                workflow_id,
                "final grounding evaluation",
                lambda: evaluate_revision_grounding(
                    revision_response.revision_id,
                    repository=repo,
                    model_client=evaluation_model_client
                    or _stage_client(
                        "evaluation",
                        "OPENAI_API_KEY is required for draft evaluation.",
                    ),
                ),
                telemetry=telemetry,
                telemetry_stage="final_evaluation",
            )
            telemetry.record_model(
                "final_evaluation",
                final_evaluation_response.model_name,
            )
            telemetry.record_calls("final_evaluation", 1)
            telemetry.record_tokens(
                "final_evaluation",
                _dict_value(final_evaluation_response.evaluation.get("token_usage")),
            )

    cleanup_summary = _cleanup_summary(processed)
    brief_summary = _brief_summary(brief_response.brief)
    draft_summary = _draft_summary(draft_response.draft)
    evaluation_summary = (
        _evaluation_summary(evaluation_response.evaluation)
        if evaluation_response
        else None
    )
    final_evaluation_summary = (
        _evaluation_summary(final_evaluation_response.evaluation)
        if final_evaluation_response
        else None
    )
    final_article = (
        revision_response.revised_draft if revision_response else draft_response.draft
    )
    final_article_source_stage = (
        "length_recovery"
        if expansion_response
        else "revision"
        if revision_response
        else str(draft_generation_metadata.get("original_draft_source") or "draft")
    )
    original_draft_word_count = approximate_tamil_word_count(
        str(draft_response.draft.get("article_body") or "")
    )
    quality_result = scan_tamil_quality(final_article, desired_word_count)
    section_coverage_status, section_coverage_warnings = _section_coverage(
        plan_sections=plan_response.planned_sections if plan_response else [],
        final_article=final_article,
        final_word_count=quality_result.final_article_word_count,
        planned_min_word_count=(
            plan_response.target_min_word_count if plan_response else None
        ),
    )
    length_warning_reason = _length_warning_reason(
        brief=brief_response.brief,
        requested_word_count=quality_result.requested_word_count,
        original_draft_word_count=original_draft_word_count,
        final_article_word_count=quality_result.final_article_word_count,
        scanner_reason=quality_result.length_warning_reason,
        length_recovery_failed=length_recovery_failed,
    )
    if final_evaluation_response is not None:
        unsupported_claims_unresolved_count = len(
            _list_value(
                final_evaluation_response.evaluation.get("unsupported_claims")
            )
        )
        unsupported_claims_cleared_by_patch = (
            unsupported_claim_findings_count > 0
            and unsupported_claims_unresolved_count == 0
        )
    initial_readiness_reasons = _evaluation_readiness_reasons(
        evaluation_response.evaluation if evaluation_response else None
    )
    readiness_decision = _final_readiness_decision(
        final_evaluation=(
            final_evaluation_response.evaluation if final_evaluation_response else None
        ),
        initial_evaluation=(
            evaluation_response.evaluation if evaluation_response else None
        ),
        final_article_word_count=quality_result.final_article_word_count,
        target_min_word_count=_target_min(desired_word_count),
        target_max_word_count=_target_max(desired_word_count),
        tamil_quality_status=quality_result.tamil_quality_status,
        tamil_quality_warnings=quality_result.tamil_quality_warnings,
        length_status=quality_result.length_status,
        section_coverage_status=section_coverage_status,
        revision_patch_skipped_reasons=revision_patch_skipped_reasons,
        revision_rejected_for_length_collapse=revision_rejected_for_length_collapse,
        revision_rejected_reason=revision_rejected_reason,
        run_final_evaluation=run_final_evaluation,
        unsupported_claim_patch_skipped_reasons=(
            unsupported_claim_patch_skipped_reasons
        ),
    )
    warnings = _merge_warnings(
        cleanup_summary.warnings,
        brief_response.warnings,
        draft_response.warnings,
        evaluation_response.warnings if evaluation_response else [],
        revision_response.warnings if revision_response else [],
        final_evaluation_response.warnings if final_evaluation_response else [],
        expansion_response.warnings if expansion_response else [],
    )
    telemetry_fields = _response_telemetry_fields(
        telemetry,
        perf_counter() - workflow_started,
    )
    export_paths = (
        _run_stage(
            workflow_id,
            "review export",
            lambda: export_revision_review(
                workflow_id=workflow_id,
                export_format=export_format,
                cleaned_source_excerpt=_excerpt(processed.cleaned_text),
                brief=brief_response.brief,
                original_draft=draft_response.draft,
                initial_evaluation=(
                    evaluation_response.evaluation if evaluation_response else None
                ),
                revised_draft=(
                    revision_response.revised_draft if revision_response else None
                ),
                revision_summary=(
                    revision_response.revision_summary if revision_response else None
                ),
                final_evaluation=(
                    final_evaluation_response.evaluation
                    if final_evaluation_response
                    else None
                ),
                tamil_quality_status=quality_result.tamil_quality_status,
                tamil_quality_warnings=quality_result.tamil_quality_warnings,
                requested_word_count=quality_result.requested_word_count,
                original_draft_word_count=original_draft_word_count,
                revised_word_count_before_expansion=(
                    revised_word_count_before_expansion
                ),
                final_article_word_count=quality_result.final_article_word_count,
                length_status=quality_result.length_status,
                length_warning_reason=length_warning_reason,
                final_article_word_count_ratio=(
                    quality_result.final_article_word_count_ratio
                ),
                length_recovery_required=length_recovery_required,
                length_recovery_attempted=length_recovery_attempted,
                length_recovery_succeeded=length_recovery_succeeded,
                length_recovery_failed=length_recovery_failed,
                expansion_items_available=count_expansion_items(brief_response.brief),
                expansion_items_used=expansion_items_used,
                article_plan=(plan_response.__dict__ if plan_response else None),
                generation_metadata={
                    **draft_generation_metadata,
                    "desired_word_count": desired_word_count,
                    "target_min_word_count": _target_min(desired_word_count),
                    "target_max_word_count": _target_max(desired_word_count),
                    "revision_input_word_count": revision_input_word_count,
                    "revision_mode": revision_mode,
                    "revision_patch_count": revision_patch_count,
                    "revision_patches_applied_count": (
                        revision_patches_applied_count
                    ),
                    "revision_patches_skipped_count": (
                        revision_patches_skipped_count
                    ),
                    "revision_patch_skipped_reasons": (
                        revision_patch_skipped_reasons
                    ),
                    "revision_output_word_count": revision_output_word_count,
                    "revision_delta_word_count": revision_delta_word_count,
                    "revision_rejected_for_length_collapse": (
                        revision_rejected_for_length_collapse
                    ),
                    "revision_rejected_reason": revision_rejected_reason,
                    "revised_article_source": revised_article_source,
                    "unsupported_claim_findings_count": (
                        unsupported_claim_findings_count
                    ),
                    "unsupported_claim_patch_count": unsupported_claim_patch_count,
                    "unsupported_claim_patches_applied_count": (
                        unsupported_claim_patches_applied_count
                    ),
                    "unsupported_claim_patches_skipped_count": (
                        unsupported_claim_patches_skipped_count
                    ),
                    "unsupported_claim_patch_skipped_reasons": (
                        unsupported_claim_patch_skipped_reasons
                    ),
                    "unsupported_claims_unresolved_count": (
                        unsupported_claims_unresolved_count
                    ),
                    "unsupported_claims_cleared_by_patch": (
                        unsupported_claims_cleared_by_patch
                    ),
                    "length_recovery_skipped_reason": draft_response.draft.get(
                        "length_recovery_skipped_reason"
                    ),
                    "length_recovery_input_word_count": draft_response.draft.get(
                        "length_recovery_input_word_count"
                    ),
                    "final_article_source_stage": final_article_source_stage,
                    "initial_readiness": (
                        evaluation_summary.editorial_readiness
                        if evaluation_summary
                        else None
                    ),
                    "initial_readiness_reasons": initial_readiness_reasons,
                    "final_readiness": readiness_decision.readiness,
                    "final_readiness_reasons": readiness_decision.reasons,
                    "readiness_decision_source": readiness_decision.source,
                    "final_publication_blockers": readiness_decision.blockers,
                    "final_publication_warnings": readiness_decision.warnings,
                    "publication_ready_completeness_passed": (
                        readiness_decision.publication_ready_completeness_passed
                    ),
                    **telemetry_fields,
                },
                section_generation_trace=_section_generation_trace(
                    draft_response.draft
                ),
                section_coverage_status=section_coverage_status,
                section_coverage_warnings=section_coverage_warnings,
                readiness_metadata={
                    "initial_readiness": (
                        evaluation_summary.editorial_readiness
                        if evaluation_summary
                        else None
                    ),
                    "initial_readiness_reasons": initial_readiness_reasons,
                    "final_readiness": readiness_decision.readiness,
                    "final_readiness_reasons": readiness_decision.reasons,
                    "readiness_decision_source": readiness_decision.source,
                    "final_publication_blockers": readiness_decision.blockers,
                    "final_publication_warnings": readiness_decision.warnings,
                    "publication_ready_completeness_passed": (
                        readiness_decision.publication_ready_completeness_passed
                    ),
                    "allowed_english_terms_block_readiness": (
                        "no"
                        if "allowed_english_terms_only"
                        in readiness_decision.warnings
                        and "tamil_quality_blocker"
                        not in readiness_decision.blockers
                        else "not_present"
                    ),
                    "skipped_patch_impact": _skipped_patch_impact(
                        readiness_decision
                    ),
                },
            ),
            telemetry=telemetry,
            telemetry_stage="export",
        )
        if export_review
        else []
    )

    response = PastedTextWorkflowResponse(
        workflow_id=workflow_id,
        workflow_run_id=workflow_id,
        status="completed",
        author_id=author_id,
        brief_id=brief_response.brief_id,
        draft_id=draft_response.draft_id,
        evaluation_id=(
            evaluation_response.evaluation_id if evaluation_response else None
        ),
        initial_evaluation_id=(
            evaluation_response.evaluation_id if evaluation_response else None
        ),
        revision_id=revision_response.revision_id if revision_response else None,
        final_evaluation_id=(
            final_evaluation_response.evaluation_id
            if final_evaluation_response
            else None
        ),
        source_cleanup=cleanup_summary,
        brief_summary=brief_summary,
        draft_summary=draft_summary,
        evaluation_summary=evaluation_summary,
        initial_readiness=(
            evaluation_summary.editorial_readiness if evaluation_summary else None
        ),
        initial_readiness_reasons=initial_readiness_reasons,
        final_readiness=readiness_decision.readiness,
        final_readiness_reasons=readiness_decision.reasons,
        readiness_decision_source=readiness_decision.source,
        final_publication_blockers=readiness_decision.blockers,
        final_publication_warnings=readiness_decision.warnings,
        publication_ready_completeness_passed=(
            readiness_decision.publication_ready_completeness_passed
        ),
        final_evaluation_summary=final_evaluation_summary,
        article_plan_used=plan_response is not None,
        plan_id=plan_response.plan_id if plan_response else None,
        desired_word_count=desired_word_count,
        target_min_word_count=_target_min(desired_word_count),
        target_max_word_count=_target_max(desired_word_count),
        generation_mode_used=_optional_str(
            draft_generation_metadata.get("generation_mode_used")
        ),
        generated_section_count=_int_from_metadata(
            draft_generation_metadata,
            "generated_section_count",
        ),
        assembled_section_count=_int_from_metadata(
            draft_generation_metadata,
            "assembled_section_count",
        ),
        section_assembled_article_word_count=_optional_int(
            draft_generation_metadata.get("section_assembled_article_word_count")
        ),
        section_assembled_article_paragraph_count=_optional_int(
            draft_generation_metadata.get("section_assembled_article_paragraph_count")
        ),
        original_draft_source=_optional_str(
            draft_generation_metadata.get("original_draft_source")
        ),
        original_draft_word_count_after_assignment=_optional_int(
            draft_generation_metadata.get("original_draft_word_count_after_assignment")
        ),
        original_draft_matches_section_assembly=_optional_bool(
            draft_generation_metadata.get("original_draft_matches_section_assembly")
        ),
        revision_input_word_count=revision_input_word_count,
        revision_mode=revision_mode,
        revision_patch_count=revision_patch_count,
        revision_patches_applied_count=revision_patches_applied_count,
        revision_patches_skipped_count=revision_patches_skipped_count,
        revision_patch_skipped_reasons=revision_patch_skipped_reasons,
        revision_output_word_count=revision_output_word_count,
        revision_delta_word_count=revision_delta_word_count,
        revision_rejected_for_length_collapse=revision_rejected_for_length_collapse,
        revision_rejected_reason=revision_rejected_reason,
        revised_article_source=revised_article_source,
        unsupported_claim_findings_count=unsupported_claim_findings_count,
        unsupported_claim_patch_count=unsupported_claim_patch_count,
        unsupported_claim_patches_applied_count=(
            unsupported_claim_patches_applied_count
        ),
        unsupported_claim_patches_skipped_count=(
            unsupported_claim_patches_skipped_count
        ),
        unsupported_claim_patch_skipped_reasons=(
            unsupported_claim_patch_skipped_reasons
        ),
        unsupported_claims_unresolved_count=unsupported_claims_unresolved_count,
        unsupported_claims_cleared_by_patch=unsupported_claims_cleared_by_patch,
        length_recovery_skipped_reason=_optional_str(
            draft_response.draft.get("length_recovery_skipped_reason")
        ),
        length_recovery_input_word_count=_optional_int(
            draft_response.draft.get("length_recovery_input_word_count")
        ),
        final_article_source_stage=final_article_source_stage,
        final_word_count_ratio=quality_result.final_article_word_count_ratio,
        section_generation_trace=_section_generation_trace(draft_response.draft),
        max_concurrent_section_calls=_optional_int(
            draft_generation_metadata.get("max_concurrent_section_calls")
        ),
        planned_section_count=(
            _int_from_metadata(draft_generation_metadata, "planned_section_count")
            or (len(plan_response.planned_sections) if plan_response else 0)
        ),
        planned_target_word_count=(
            plan_response.target_word_count if plan_response else None
        ),
        planned_min_word_count=(
            plan_response.target_min_word_count if plan_response else None
        ),
        planned_max_word_count=(
            plan_response.target_max_word_count if plan_response else None
        ),
        section_coverage_status=section_coverage_status,
        section_coverage_warnings=section_coverage_warnings,
        tamil_quality_status=quality_result.tamil_quality_status,
        tamil_quality_issues_count=quality_result.tamil_quality_issues_count,
        tamil_quality_warnings=quality_result.tamil_quality_warnings,
        requested_word_count=quality_result.requested_word_count,
        original_draft_word_count=original_draft_word_count,
        revised_word_count_before_expansion=revised_word_count_before_expansion,
        final_article_word_count=quality_result.final_article_word_count,
        length_status=quality_result.length_status,
        length_warning_reason=length_warning_reason,
        final_article_word_count_ratio=quality_result.final_article_word_count_ratio,
        length_recovery_required=length_recovery_required,
        length_recovery_attempted=length_recovery_attempted,
        length_recovery_succeeded=length_recovery_succeeded,
        length_recovery_failed=length_recovery_failed,
        short_output_invalid=short_output_invalid,
        expansion_items_available=count_expansion_items(brief_response.brief),
        expansion_items_used=expansion_items_used,
        export_paths=export_paths,
        warnings=warnings,
        workflow_mode=workflow_mode,
        **_response_telemetry_fields(telemetry, perf_counter() - workflow_started),
    )
    _save_workflow_run(repo, response, source_text, article_type, desired_word_count)
    return response


def _run_stage(
    workflow_id: str,
    stage_name: str,
    operation: Callable[[], T],
    telemetry: WorkflowTelemetry | None = None,
    telemetry_stage: str | None = None,
) -> T:
    started = perf_counter()
    LOGGER.info(
        "workflow_id=%s stage=%s status=started",
        workflow_id,
        stage_name,
    )
    try:
        result = operation()
    except Exception as exc:
        elapsed = perf_counter() - started
        LOGGER.exception(
            "workflow_id=%s stage=%s status=failed elapsed_seconds=%.2f",
            workflow_id,
            stage_name,
            elapsed,
        )
        if isinstance(exc, OpenAIClientError):
            raise OpenAIClientError(
                "Workflow stage "
                f"'{stage_name}' failed after {elapsed:.2f}s: {exc}"
            ) from exc
        if exc.__class__.__name__ == "SourceProcessingError":
            raise
        raise PastedTextWorkflowError(
            f"Workflow stage '{stage_name}' failed after {elapsed:.2f}s: {exc}"
        ) from exc
    elapsed = perf_counter() - started
    if telemetry and telemetry_stage:
        telemetry.record_runtime(telemetry_stage, elapsed)
    LOGGER.info(
        "workflow_id=%s stage=%s status=completed elapsed_seconds=%.2f",
        workflow_id,
        stage_name,
        elapsed,
    )
    return result


def _stage_client(stage: str, missing_key_message: str) -> OpenAIJsonClient:
    return OpenAIJsonClient(
        model_name=resolve_stage_model(stage),
        missing_key_message=missing_key_message,
    )


def _cleanup_summary(processed: ProcessedSource) -> SourceCleanupSummary:
    return SourceCleanupSummary(
        original_char_count=processed.original_char_count,
        cleaned_char_count=processed.cleaned_char_count,
        removed_line_count=processed.removed_line_count,
        warnings=processed.warnings,
    )


def _brief_summary(brief: dict[str, object]) -> WorkflowBriefSummary:
    return WorkflowBriefSummary(
        topic=str(brief.get("topic") or ""),
        one_line_summary=str(brief.get("one_line_summary") or ""),
        confirmed_facts=_list_value(brief.get("confirmed_facts")),
        claims_to_avoid=_list_value(brief.get("claims_to_avoid")),
    )


def _draft_summary(draft: dict[str, object]) -> WorkflowDraftSummary:
    return WorkflowDraftSummary(
        headline=str(draft.get("headline") or ""),
        subheadline=str(draft.get("subheadline") or ""),
        seo_title=str(draft.get("seo_title") or ""),
        tags=_list_value(draft.get("suggested_tags")),
    )


def _evaluation_summary(
    evaluation: dict[str, object],
) -> WorkflowEvaluationSummary:
    return WorkflowEvaluationSummary(
        grounding_score=_optional_int(evaluation.get("grounding_score")),
        claim_safety_score=_optional_int(evaluation.get("claim_safety_score")),
        overall_risk=_optional_str(evaluation.get("overall_risk")),
        editorial_readiness=_optional_str(evaluation.get("editorial_readiness")),
    )


def _evaluation_readiness_reasons(
    evaluation: dict[str, object] | None,
) -> list[str]:
    if not evaluation:
        return []
    reasons: list[str] = []
    score = _optional_int(evaluation.get("grounding_score"))
    if score is not None:
        if score < GROUNDING_REVIEW_THRESHOLD:
            reasons.append("grounding_score_below_threshold")
        elif score < GROUNDING_READY_THRESHOLD:
            reasons.append("grounding_score_review_band")
        else:
            reasons.append("grounding_score_ready_band")
    if _list_value(evaluation.get("unsupported_claims")):
        reasons.append("unsupported_claims_remaining")
    if _list_value(evaluation.get("invented_facts")):
        reasons.append("invented_facts_remaining")
    if _list_value(evaluation.get("contradictions")):
        reasons.append("contradictions_remaining")
    if _list_value(evaluation.get("claims_to_avoid_violations")):
        reasons.append("claims_to_avoid_violations_remaining")
    if _list_value(evaluation.get("overclaim_phrases")):
        reasons.append("overclaim_phrases_remaining")
    if not reasons:
        reasons.append("no_evaluation_blockers_detected")
    return reasons


def _final_readiness_decision(
    *,
    final_evaluation: dict[str, object] | None,
    initial_evaluation: dict[str, object] | None,
    final_article_word_count: int,
    target_min_word_count: int | None,
    target_max_word_count: int | None,
    tamil_quality_status: str | None,
    tamil_quality_warnings: list[str],
    length_status: str | None,
    section_coverage_status: str | None,
    revision_patch_skipped_reasons: list[object],
    revision_rejected_for_length_collapse: bool,
    revision_rejected_reason: str | None,
    run_final_evaluation: bool,
    unsupported_claim_patch_skipped_reasons: list[object] | None = None,
) -> ReadinessDecision:
    evaluation = final_evaluation or initial_evaluation
    source = "final_article_state"
    if final_evaluation is None:
        source = (
            "initial_evaluation_no_final_evaluation"
            if initial_evaluation and run_final_evaluation is False
            else "local_final_article_state_no_evaluation"
        )

    blockers: list[str] = []
    warnings: list[str] = []
    reasons: list[str] = []
    completeness_passed = _publication_ready_completeness_passed(
        final_article_word_count=final_article_word_count,
        target_min_word_count=target_min_word_count,
        target_max_word_count=target_max_word_count,
        section_coverage_status=section_coverage_status,
    )

    score = _optional_int(evaluation.get("grounding_score")) if evaluation else None
    if score is None:
        warnings.append("grounding_score_missing")
    elif score < GROUNDING_REVIEW_THRESHOLD:
        blockers.append("grounding_score_below_threshold")
    elif score < GROUNDING_READY_THRESHOLD:
        warnings.append("grounding_score_review_band")

    if evaluation:
        _extend_if_present(
            blockers,
            evaluation,
            "unsupported_claims",
            "unsupported_claims_remaining",
        )
        _extend_if_present(
            blockers,
            evaluation,
            "invented_facts",
            "invented_facts_remaining",
        )
        _extend_if_present(
            blockers,
            evaluation,
            "contradictions",
            "contradictions_remaining",
        )
        _extend_if_present(
            blockers,
            evaluation,
            "claims_to_avoid_violations",
            "claims_to_avoid_violations_remaining",
        )
        _extend_if_present(
            warnings,
            evaluation,
            "overclaim_phrases",
            "overclaim_phrases_remaining",
        )
        _extend_if_present(
            warnings,
            evaluation,
            "missing_key_facts",
            "missing_key_facts_remaining",
        )

    if (
        target_min_word_count is not None
        and final_article_word_count < target_min_word_count
    ):
        blockers.append("length_below_minimum")
    if (
        target_max_word_count is not None
        and final_article_word_count > target_max_word_count
    ):
        blockers.append("length_above_maximum")
    if length_status == "warning" and not completeness_passed:
        blockers.append("length_warning")
    if section_coverage_status == "warning":
        warnings.append("section_coverage_warning")

    if tamil_quality_status == "fail":
        blockers.append("tamil_quality_blocker")
    elif tamil_quality_status == "warning":
        warnings.extend(_tamil_quality_warning_reasons(tamil_quality_warnings))

    skipped_reasons = [str(reason) for reason in revision_patch_skipped_reasons]
    unsupported_skips = [
        str(reason) for reason in unsupported_claim_patch_skipped_reasons or []
    ]
    if unsupported_skips and _list_value(
        evaluation.get("unsupported_claims") if evaluation else None
    ):
        blockers.append("revision_patch_pending")
    elif skipped_reasons:
        warnings.append("revision_patch_skipped_non_critical")

    if revision_rejected_for_length_collapse:
        warnings.append("revision_rejected_for_length_collapse")
    if revision_rejected_reason:
        blockers.append(str(revision_rejected_reason))

    blockers = _unique_strings(blockers)
    warnings = _unique_strings(warnings)
    if blockers:
        readiness = "revision_required"
        reasons.extend(blockers)
    elif warnings:
        readiness = "review_required"
        reasons.extend(warnings)
    else:
        readiness = "safe_to_review"
        reasons.append("no_blockers_detected")

    if completeness_passed and "length_warning" not in blockers:
        reasons.append("publication_ready_completeness_passed")

    return ReadinessDecision(
        readiness=readiness,
        reasons=_unique_strings(reasons),
        source=source,
        blockers=blockers,
        warnings=warnings,
        publication_ready_completeness_passed=completeness_passed,
    )


def _publication_ready_completeness_passed(
    *,
    final_article_word_count: int,
    target_min_word_count: int | None,
    target_max_word_count: int | None,
    section_coverage_status: str | None,
) -> bool:
    if target_min_word_count is None or target_max_word_count is None:
        return section_coverage_status != "warning"
    return (
        target_min_word_count <= final_article_word_count <= target_max_word_count
        and section_coverage_status != "warning"
    )


def _extend_if_present(
    target: list[str],
    evaluation: dict[str, object],
    key: str,
    reason: str,
) -> None:
    if _list_value(evaluation.get(key)):
        target.append(reason)


def _tamil_quality_warning_reasons(warnings: list[str]) -> list[str]:
    if not warnings:
        return ["tamil_quality_warning"]
    reasons: list[str] = []
    for warning in warnings:
        if warning.startswith("Allowed English term(s) present:"):
            reasons.append("allowed_english_terms_only")
        elif "materially shorter" in warning:
            reasons.append("length_below_minimum")
        else:
            reasons.append("tamil_quality_warning")
    return _unique_strings(reasons)


def _skipped_patch_impact(decision: ReadinessDecision) -> str:
    if "revision_patch_pending" in decision.blockers:
        return "blocking"
    if "revision_patch_skipped_non_critical" in decision.warnings:
        return "non_blocking_warning"
    return "none"


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique


def _length_warning_reason(
    brief: dict[str, object],
    requested_word_count: int | None,
    original_draft_word_count: int,
    final_article_word_count: int,
    scanner_reason: str | None,
    length_recovery_failed: bool = False,
) -> str | None:
    if not requested_word_count or scanner_reason is None:
        return scanner_reason
    minimum_expected = int(requested_word_count * 0.75)
    if final_article_word_count >= minimum_expected:
        return None
    if length_recovery_failed:
        return (
            "Length recovery failed despite sufficient grounded expansion material."
        )

    richness_count = (
        len(_list_value(brief.get("confirmed_facts")))
        + len(_list_value(brief.get("numbers_and_statistics")))
        + len(_list_value(brief.get("quotes")))
        + len(_list_value(brief.get("affected_groups")))
        + len(_list_value(brief.get("policy_or_legal_context")))
        + len(_list_value(brief.get("background_from_source")))
    )
    if richness_count < 5:
        return (
            f"{scanner_reason} Grounded brief appears too limited "
            f"({richness_count} expansion items) to safely support the target."
        )
    if original_draft_word_count < minimum_expected:
        return (
            f"{scanner_reason} Grounded brief has {richness_count} expansion "
            "items, so this indicates generation and revision failed to meet "
            "the length target rather than source thinness."
        )
    return (
        f"{scanner_reason} Original draft was {original_draft_word_count} words, "
        "so this indicates the revision step failed to preserve article length."
    )


def _section_coverage(
    plan_sections: list[object],
    final_article: dict[str, object],
    final_word_count: int,
    planned_min_word_count: int | None,
) -> tuple[str | None, list[str]]:
    if not plan_sections:
        return None, []
    warnings: list[str] = []
    if len(plan_sections) < 6:
        warnings.append(
            f"Plan has only {len(plan_sections)} sections; 6-8 is preferred."
        )
    if planned_min_word_count and final_word_count < planned_min_word_count:
        warnings.append(
            "Final article word count is below the planned minimum "
            f"({final_word_count}/{planned_min_word_count})."
        )
    article_body = str(final_article.get("article_body") or "")
    paragraph_count = len(
        [part for part in article_body.split("\n") if part.strip()]
    )
    if paragraph_count < min(len(plan_sections), 6):
        warnings.append(
            f"Final article has {paragraph_count} paragraphs for "
            f"{len(plan_sections)} planned sections."
        )
    return ("warning" if warnings else "pass"), warnings


def _revision_response_from_expansion(
    revision_id: str,
    repo: StyleScribeRepository,
) -> ArticleRevisionResponse:
    record = repo.fetch_article_revision(revision_id)
    if record is None:
        raise PastedTextWorkflowError("Expanded article revision was not saved.")
    return get_latest_article_revision(record.draft_id, repository=repo)


def _save_workflow_run(
    repo: StyleScribeRepository,
    response: PastedTextWorkflowResponse,
    source_text: str,
    article_type: str,
    desired_word_count: int,
) -> None:
    created_at = datetime.now(UTC).isoformat()
    input_summary = {
        "source_input_mode": "pasted_web_text",
        "source_original_char_count": len(source_text),
        "article_type": article_type,
        "desired_word_count": desired_word_count,
        "run_auto_revision": response.revision_id is not None,
        "article_plan_used": response.article_plan_used,
    }
    output_summary = {
        "brief_id": response.brief_id,
        "draft_id": response.draft_id,
        "evaluation_id": response.evaluation_id,
        "initial_evaluation_id": response.initial_evaluation_id,
        "revision_id": response.revision_id,
        "final_evaluation_id": response.final_evaluation_id,
        "export_paths": response.export_paths,
        "plan_id": response.plan_id,
        "generation_mode_used": response.generation_mode_used,
        "original_draft_source": response.original_draft_source,
        "section_assembled_article_word_count": (
            response.section_assembled_article_word_count
        ),
        "final_article_word_count": response.final_article_word_count,
        "revision_rejected_for_length_collapse": (
            response.revision_rejected_for_length_collapse
        ),
        "revision_mode": response.revision_mode,
        "revision_patch_count": response.revision_patch_count,
        "revision_patches_applied_count": response.revision_patches_applied_count,
        "revision_patches_skipped_count": response.revision_patches_skipped_count,
        "final_readiness": response.final_readiness,
        "readiness_decision_source": response.readiness_decision_source,
        "final_publication_blockers": response.final_publication_blockers,
        "final_publication_warnings": response.final_publication_warnings,
        "publication_ready_completeness_passed": (
            response.publication_ready_completeness_passed
        ),
        "unsupported_claim_findings_count": response.unsupported_claim_findings_count,
        "unsupported_claim_patch_count": response.unsupported_claim_patch_count,
        "unsupported_claim_patches_applied_count": (
            response.unsupported_claim_patches_applied_count
        ),
        "unsupported_claim_patches_skipped_count": (
            response.unsupported_claim_patches_skipped_count
        ),
        "unsupported_claims_unresolved_count": (
            response.unsupported_claims_unresolved_count
        ),
    }
    repo.save_workflow_run(
        WorkflowRunRecord(
            workflow_id=response.workflow_id,
            workflow_type=WORKFLOW_TYPE,
            author_id=response.author_id,
            brief_id=response.brief_id,
            draft_id=response.draft_id,
            evaluation_id=response.evaluation_id,
            status=response.status,
            input_summary_json=StyleScribeRepository.encode_json(input_summary),
            output_summary_json=StyleScribeRepository.encode_json(output_summary),
            warnings_json=StyleScribeRepository.encode_warnings(response.warnings),
            created_at=created_at,
        )
    )


def _export_review(
    workflow_id: str,
    export_format: str,
    cleaned_source_excerpt: str,
    brief: dict[str, object],
    draft: dict[str, object],
    evaluation: dict[str, object] | None,
) -> list[str]:
    REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "html" if export_format == "html" else "md"
    output_path = REVIEW_OUTPUT_DIR / f"{workflow_id}_pasted_text_review.{suffix}"
    markdown = _render_markdown(cleaned_source_excerpt, brief, draft, evaluation)
    content = _render_html(markdown) if export_format == "html" else markdown
    output_path.write_text(content, encoding="utf-8")
    return [str(output_path)]


def _render_markdown(
    cleaned_source_excerpt: str,
    brief: dict[str, object],
    draft: dict[str, object],
    evaluation: dict[str, object] | None,
) -> str:
    lines = [
        "# Pasted Text Workflow Review",
        "",
        "## Cleaned Source Excerpt",
        "",
        cleaned_source_excerpt,
        "",
        "## Grounded Brief",
        "",
        f"- Topic: {brief.get('topic')}",
        f"- One-line summary: {brief.get('one_line_summary')}",
        "- Confirmed facts:",
    ]
    lines.extend(f"  - {fact}" for fact in _list_value(brief.get("confirmed_facts")))
    lines.extend(["- Claims to avoid:"])
    lines.extend(f"  - {claim}" for claim in _list_value(brief.get("claims_to_avoid")))
    lines.extend(
        [
            "",
            "## Generated Draft",
            "",
            f"### Headline\n\n{draft.get('headline') or ''}",
            f"### Subheadline\n\n{draft.get('subheadline') or ''}",
            f"### Article Body\n\n{draft.get('article_body') or ''}",
            f"### SEO Title\n\n{draft.get('seo_title') or ''}",
            f"### Meta Description\n\n{draft.get('meta_description') or ''}",
            "### Tags",
        ]
    )
    lines.extend(f"- {tag}" for tag in _list_value(draft.get("suggested_tags")))
    if evaluation:
        lines.extend(
            [
                "",
                "## Grounding Evaluation",
                "",
                f"- Grounding score: {evaluation.get('grounding_score')}",
                f"- Claim safety score: {evaluation.get('claim_safety_score')}",
                f"- Overall risk: {evaluation.get('overall_risk')}",
                f"- Editorial readiness: {evaluation.get('editorial_readiness')}",
                "- Unsupported claims:",
            ]
        )
        lines.extend(
            f"  - {item}" for item in _list_value(evaluation.get("unsupported_claims"))
        )
        lines.extend(["- Overclaim phrases:"])
        lines.extend(
            f"  - {item}" for item in _list_value(evaluation.get("overclaim_phrases"))
        )
        lines.extend(["- Rewrite guidance:"])
        lines.extend(
            f"  - {item}" for item in _list_value(evaluation.get("rewrite_guidance"))
        )
    return "\n".join(lines) + "\n"


def _render_html(markdown: str) -> str:
    body_lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("### "):
            body_lines.append(f"<h3>{escape(line[4:])}</h3>")
        elif line:
            body_lines.append(f"<p>{escape(line)}</p>")
        else:
            body_lines.append("")
    body = "\n".join(body_lines)
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>Pasted Text Workflow Review</title>
  <style>
    body {{
      font-family: {TAMIL_FONT_STACK};
      line-height: 1.65;
      margin: 32px;
      max-width: 980px;
    }}
    h1, h2, h3 {{ line-height: 1.3; }}
    p {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _merge_warnings(*warning_groups: list[str]) -> list[str]:
    warnings: list[str] = []
    for group in warning_groups:
        warnings.extend(group)
    return warnings


def _response_telemetry_fields(
    telemetry: WorkflowTelemetry,
    total_runtime_seconds: float,
) -> dict[str, object]:
    summary = telemetry.summary(total_runtime_seconds)
    runtime_by_stage = summary["runtime_by_stage"]
    call_counts = summary["llm_call_count_by_stage"]
    model_by_stage = summary["model_used_by_stage"]
    return {
        **summary,
        "planning_runtime_seconds": _dict_number(runtime_by_stage, "planning"),
        "section_generation_runtime_seconds": _dict_number(
            runtime_by_stage,
            "generation",
        ),
        "generation_runtime_seconds": _dict_number(runtime_by_stage, "generation"),
        "initial_evaluation_runtime_seconds": _dict_number(
            runtime_by_stage,
            "initial_evaluation",
        ),
        "revision_runtime_seconds": _dict_number(runtime_by_stage, "revision"),
        "final_evaluation_runtime_seconds": _dict_number(
            runtime_by_stage,
            "final_evaluation",
        ),
        "length_recovery_runtime_seconds": _dict_number(
            runtime_by_stage,
            "length_recovery",
        ),
        "export_runtime_seconds": _dict_number(runtime_by_stage, "export"),
        "section_generation_call_count": _dict_int(call_counts, "section_generation"),
        "section_retry_call_count": _dict_int(call_counts, "section_retries"),
        "planning_model_used": _dict_str(model_by_stage, "planning"),
        "generation_model_used": _dict_str(model_by_stage, "generation"),
        "revision_model_used": _dict_str(model_by_stage, "revision"),
        "evaluation_model_used": (
            _dict_str(model_by_stage, "final_evaluation")
            or _dict_str(model_by_stage, "initial_evaluation")
        ),
        "length_recovery_model_used": _dict_str(model_by_stage, "length_recovery"),
    }


def _generation_token_usage(draft: dict[str, object]) -> dict[str, object] | None:
    usage = draft.get("token_usage")
    return usage if isinstance(usage, dict) else None


def _excerpt(text: str) -> str:
    if len(text) <= SOURCE_REVIEW_EXCERPT_CHARS:
        return text
    return text[: SOURCE_REVIEW_EXCERPT_CHARS - 3].rstrip() + "..."


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _dict_number(values: object, key: str) -> float | None:
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    if isinstance(value, int | float):
        return float(value)
    return None


def _dict_int(values: object, key: str) -> int:
    if not isinstance(values, dict):
        return 0
    value = values.get(key)
    return value if isinstance(value, int) else 0


def _dict_str(values: object, key: str) -> str | None:
    if not isinstance(values, dict):
        return None
    value = values.get(key)
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _target_min(desired_word_count: int | None) -> int | None:
    return round(desired_word_count * 0.75) if desired_word_count else None


def _target_max(desired_word_count: int | None) -> int | None:
    return round(desired_word_count * 1.15) if desired_word_count else None


def _draft_generation_metadata(draft: dict[str, object]) -> dict[str, object]:
    keys = (
        "generation_mode_used",
        "article_plan_used",
        "planned_section_count",
        "generated_section_count",
        "assembled_section_count",
        "section_assembled_article_word_count",
        "section_assembled_article_paragraph_count",
        "original_draft_source",
        "original_draft_word_count_after_assignment",
        "original_draft_matches_section_assembly",
        "max_concurrent_section_calls",
    )
    return {key: draft.get(key) for key in keys if key in draft}


def _section_generation_trace(draft: dict[str, object]) -> list[dict[str, object]]:
    trace = draft.get("section_generation_trace")
    if not isinstance(trace, list):
        return []
    return [item for item in trace if isinstance(item, dict)]


def _revision_guardrail_metadata(token_usage: dict[str, object]) -> dict[str, object]:
    metadata = token_usage.get("revision_length_guardrail")
    return metadata if isinstance(metadata, dict) else {}


def _int_from_metadata(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    return value if isinstance(value, int) else 0
