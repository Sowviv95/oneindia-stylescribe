"""Side-by-side multi-author article comparison workflow."""

from __future__ import annotations

from time import perf_counter
from typing import Literal, Protocol, cast
from uuid import uuid4

from backend.app.db.repository import StyleScribeRepository
from backend.app.models.multi_author_comparison_models import (
    AuthorComparisonOutput,
    EditorAttentionItem,
    MultiAuthorComparisonResponse,
    MultiAuthorComparisonSummary,
    RecommendationValue,
    SharedGroundedBriefMetadata,
)
from backend.app.models.pasted_text_workflow_models import WorkflowMode
from backend.app.services.article_generation_service import generate_article_draft
from backend.app.services.article_length_recovery_service import count_expansion_items
from backend.app.services.article_plan_service import generate_article_plan
from backend.app.services.draft_grounding_evaluation_service import (
    evaluate_draft_grounding,
)
from backend.app.services.grounded_brief_service import generate_grounded_brief
from backend.app.services.pasted_text_workflow_service import (
    _brief_summary,
    _cleanup_summary,
    _dict_value,
    _display_provider,
    _draft_generation_metadata,
    _evaluation_readiness_reasons,
    _evaluation_summary,
    _final_readiness_decision,
    _int_from_metadata,
    _list_value,
    _optional_int,
    _response_telemetry_fields,
    _run_stage,
    _section_coverage,
    _section_generation_trace,
    _stage_client,
    _target_max,
    _target_min,
)
from backend.app.services.source_processor import process_source
from backend.app.services.tamil_quality_scanner import (
    scan_tamil_quality,
)
from backend.app.services.workflow_telemetry import WorkflowTelemetry


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


class MultiAuthorComparisonError(RuntimeError):
    """Raised when multi-author comparison cannot complete."""


def run_multi_author_comparison_workflow(
    *,
    source_text: str,
    author_id_a: str,
    author_id_b: str,
    author_instruction: str | None = None,
    target_language: str = "ta",
    article_type: str = "news",
    desired_word_count: int = 600,
    tone_override: str | None = None,
    workflow_mode: WorkflowMode = "standard",
    repository: StyleScribeRepository | None = None,
    brief_model_client: StructuredJsonClient | None = None,
    plan_model_client: StructuredJsonClient | None = None,
    draft_model_client: StructuredJsonClient | None = None,
    evaluation_model_client: StructuredJsonClient | None = None,
) -> MultiAuthorComparisonResponse:
    """Generate two author-specific drafts from one shared grounded brief."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    _validate_latest_profile(repo, author_id_a)
    _validate_latest_profile(repo, author_id_b)

    workflow_id = str(uuid4())
    workflow_started = perf_counter()
    telemetry = WorkflowTelemetry(started_at=workflow_started)

    processed = _run_stage(
        workflow_id,
        "comparison pasted text cleanup",
        lambda: process_source("text", source_text, "pasted_web_text"),
        telemetry=telemetry,
        telemetry_stage="source_cleanup",
    )
    brief_response = _run_stage(
        workflow_id,
        "comparison shared grounded brief generation",
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

    author_a = _run_author_branch(
        role="author_a",
        workflow_id=workflow_id,
        author_id=author_id_a,
        brief_id=brief_response.brief_id,
        brief=brief_response.brief,
        author_instruction=author_instruction,
        target_language=target_language,
        article_type=article_type,
        desired_word_count=desired_word_count,
        tone_override=tone_override,
        repository=repo,
        telemetry=telemetry,
        plan_model_client=plan_model_client,
        draft_model_client=draft_model_client,
        evaluation_model_client=evaluation_model_client,
    )
    author_b = _run_author_branch(
        role="author_b",
        workflow_id=workflow_id,
        author_id=author_id_b,
        brief_id=brief_response.brief_id,
        brief=brief_response.brief,
        author_instruction=author_instruction,
        target_language=target_language,
        article_type=article_type,
        desired_word_count=desired_word_count,
        tone_override=tone_override,
        repository=repo,
        telemetry=telemetry,
        plan_model_client=plan_model_client,
        draft_model_client=draft_model_client,
        evaluation_model_client=evaluation_model_client,
    )

    total_runtime = perf_counter() - workflow_started
    telemetry_fields = _response_telemetry_fields(telemetry, total_runtime)
    warnings = [
        *processed.warnings,
        *brief_response.warnings,
        *[str(warning) for warning in author_a.warnings],
        *[str(warning) for warning in author_b.warnings],
    ]
    return MultiAuthorComparisonResponse(
        workflow_id=workflow_id,
        workflow_completed=True,
        status="completed",
        desired_word_count=desired_word_count,
        target_min_word_count=_target_min(desired_word_count),
        target_max_word_count=_target_max(desired_word_count),
        workflow_mode=workflow_mode,
        source_cleanup=_cleanup_summary(processed),
        brief_summary=_brief_summary(brief_response.brief),
        shared_grounded_brief=SharedGroundedBriefMetadata(
            brief_id=brief_response.brief_id,
            source_language=brief_response.source_language,
            target_language=brief_response.target_language,
            model_provider=brief_response.model_provider,
            model_name=brief_response.model_name,
            status=brief_response.status,
            source_text_excerpt=brief_response.source_text_excerpt,
        ),
        author_a=author_a,
        author_b=author_b,
        comparison_summary=_comparison_summary(author_a, author_b),
        warnings=_unique_strings(warnings),
        aggregate_runtime_seconds=telemetry_fields.get("total_runtime_seconds"),
        aggregate_token_usage={
            "total_prompt_tokens": telemetry_fields.get("total_prompt_tokens"),
            "total_completion_tokens": telemetry_fields.get(
                "total_completion_tokens"
            ),
            "total_tokens": telemetry_fields.get("total_tokens"),
            "cached_prompt_tokens_total": telemetry_fields.get(
                "cached_prompt_tokens_total"
            ),
            "token_usage_by_stage": telemetry_fields.get("token_usage_by_stage"),
        },
        aggregate_estimated_cost_usd=telemetry_fields.get(
            "estimated_cost_total_usd"
        ),
        telemetry=telemetry_fields,
    )


def _run_author_branch(
    *,
    role: str,
    workflow_id: str,
    author_id: str,
    brief_id: str,
    brief: dict[str, object],
    author_instruction: str | None,
    target_language: str,
    article_type: str,
    desired_word_count: int,
    tone_override: str | None,
    repository: StyleScribeRepository,
    telemetry: WorkflowTelemetry,
    plan_model_client: StructuredJsonClient | None,
    draft_model_client: StructuredJsonClient | None,
    evaluation_model_client: StructuredJsonClient | None,
) -> AuthorComparisonOutput:
    profile = repository.fetch_latest_author_style_profile(author_id)
    if profile is None:
        raise MultiAuthorComparisonError(
            f"No author style profile found for author_id: {author_id}"
        )

    plan_response = _run_stage(
        workflow_id,
        f"{role} grounded article plan generation",
        lambda: generate_article_plan(
            brief_id=brief_id,
            author_id=author_id,
            article_type=article_type,
            desired_word_count=desired_word_count,
            target_language=target_language,
            tone_override=tone_override,
            author_instruction=author_instruction,
            repository=repository,
            model_client=plan_model_client
            or _stage_client(
                "planning",
                "OPENAI_API_KEY is required for article planning.",
            ),
        ),
        telemetry=telemetry,
        telemetry_stage=f"{role}_planning",
    )
    telemetry.record_model(f"{role}_planning", plan_response.model_name)
    telemetry.record_calls(f"{role}_planning", 1)
    telemetry.record_tokens(f"{role}_planning", plan_response.token_usage)

    draft_response = _run_stage(
        workflow_id,
        f"{role} draft generation",
        lambda: generate_article_draft(
            author_id=author_id,
            brief_id=brief_id,
            author_instruction=author_instruction,
            target_language=target_language,
            article_type=article_type,
            desired_word_count=desired_word_count,
            tone_override=tone_override,
            plan_id=plan_response.plan_id,
            repository=repository,
            model_client=draft_model_client
            or _stage_client(
                "generation",
                "OPENAI_API_KEY is required for article draft generation.",
            ),
        ),
        telemetry=telemetry,
        telemetry_stage=f"{role}_generation",
    )
    draft_generation_metadata = _draft_generation_metadata(draft_response.draft)
    telemetry.record_model(f"{role}_generation", draft_response.model_name)
    telemetry.record_calls(
        f"{role}_generation",
        _generation_call_count(draft_response.draft),
    )
    telemetry.record_tokens(
        f"{role}_generation",
        _dict_value(draft_response.draft.get("token_usage")),
    )

    evaluation_response = _run_stage(
        workflow_id,
        f"{role} grounding evaluation",
        lambda: evaluate_draft_grounding(
            draft_response.draft_id,
            repository=repository,
            model_client=evaluation_model_client
            or _stage_client(
                "evaluation",
                "OPENAI_API_KEY is required for draft evaluation.",
            ),
        ),
        telemetry=telemetry,
        telemetry_stage=f"{role}_evaluation",
    )
    telemetry.record_model(f"{role}_evaluation", evaluation_response.model_name)
    telemetry.record_calls(f"{role}_evaluation", 1)
    telemetry.record_tokens(
        f"{role}_evaluation",
        _dict_value(evaluation_response.evaluation.get("token_usage")),
    )

    quality = scan_tamil_quality(draft_response.draft, desired_word_count)
    section_coverage_status, section_coverage_warnings = _section_coverage(
        plan_response.planned_sections,
        draft_response.draft,
        quality.final_article_word_count,
        plan_response.target_min_word_count,
    )
    readiness = _final_readiness_decision(
        final_evaluation=None,
        initial_evaluation=evaluation_response.evaluation,
        final_article_word_count=quality.final_article_word_count,
        target_min_word_count=_target_min(desired_word_count),
        target_max_word_count=_target_max(desired_word_count),
        tamil_quality_status=quality.tamil_quality_status,
        tamil_quality_warnings=quality.tamil_quality_warnings,
        length_status=quality.length_status,
        section_coverage_status=section_coverage_status,
        revision_patch_skipped_reasons=[],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=False,
    )
    evaluation_summary = _evaluation_summary(evaluation_response.evaluation)
    branch_telemetry = {
        "plan_model": plan_response.model_name,
        "generation_provider": _display_provider(draft_response.model_provider),
        "generation_model": draft_response.model_name,
        "evaluation_model": evaluation_response.model_name,
        "plan_token_usage": plan_response.token_usage,
        "generation_token_usage": _dict_value(draft_response.draft.get("token_usage")),
        "evaluation_token_usage": _dict_value(
            evaluation_response.evaluation.get("token_usage")
        ),
        "generation_mode_used": draft_generation_metadata.get(
            "generation_mode_used"
        ),
        "generated_section_count": _int_from_metadata(
            draft_generation_metadata,
            "generated_section_count",
        ),
        "assembled_section_count": _int_from_metadata(
            draft_generation_metadata,
            "assembled_section_count",
        ),
        "generation_group_call_count": _int_from_metadata(
            draft_generation_metadata,
            "generation_group_call_count",
        ),
        "generation_single_section_fallback_count": _int_from_metadata(
            draft_generation_metadata,
            "generation_single_section_fallback_count",
        ),
        "section_generation_trace_count": len(
            _section_generation_trace(draft_response.draft)
        ),
        "expansion_items_available": count_expansion_items(brief),
        "tamil_quality_status": quality.tamil_quality_status,
        "tamil_quality_warnings": quality.tamil_quality_warnings,
        "length_status": quality.length_status,
        "length_warning_reason": quality.length_warning_reason,
        "final_article_word_count_ratio": quality.final_article_word_count_ratio,
        "section_coverage_status": section_coverage_status,
        "section_coverage_warnings": section_coverage_warnings,
        "initial_readiness_reasons": _evaluation_readiness_reasons(
            evaluation_response.evaluation
        ),
        "final_readiness_reasons": readiness.reasons,
        "readiness_decision_source": readiness.source,
        "publication_ready_completeness_passed": (
            readiness.publication_ready_completeness_passed
        ),
    }
    return AuthorComparisonOutput(
        author_id=author_id,
        role=cast(Literal["author_a", "author_b"], role),
        profile_id=profile.profile_id,
        draft_id=draft_response.draft_id,
        evaluation_id=evaluation_response.evaluation_id,
        plan_id=plan_response.plan_id,
        generated_headline=_optional_str(draft_response.draft.get("headline")),
        generated_subheadline=_optional_str(
            draft_response.draft.get("subheadline")
        ),
        article_body=str(draft_response.draft.get("article_body") or ""),
        word_count=quality.final_article_word_count,
        generation_provider_used=_display_provider(draft_response.model_provider),
        generation_model_used=draft_response.model_name,
        grounding_score=evaluation_summary.grounding_score,
        final_readiness=readiness.readiness,
        blockers=readiness.blockers,
        warnings=[
            *draft_response.warnings,
            *evaluation_response.warnings,
            *quality.tamil_quality_warnings,
            *section_coverage_warnings,
            *readiness.warnings,
        ],
        editor_attention_items=_editor_attention_items(
            evaluation=evaluation_response.evaluation,
            blockers=readiness.blockers,
            warnings=readiness.warnings,
            article_body=draft_response.draft.get("article_body"),
            claims_to_avoid=brief.get("claims_to_avoid"),
        ),
        evaluation_summary=evaluation_summary,
        telemetry=branch_telemetry,
    )


def _comparison_summary(
    author_a: AuthorComparisonOutput,
    author_b: AuthorComparisonOutput,
) -> MultiAuthorComparisonSummary:
    score_a = author_a.grounding_score
    score_b = author_b.grounding_score
    factual = _score_comparison(
        score_a,
        score_b,
        "Both drafts need factual review because grounding scores are unavailable.",
        "Both drafts have similar factual faithfulness scores.",
        "Author A has the stronger factual faithfulness score.",
        "Author B has the stronger factual faithfulness score.",
    )
    readability = _word_count_comparison(author_a, author_b)
    style = (
        "The drafts use separate author profiles and author-specific plans; compare "
        "headline framing, paragraph rhythm, and emphasis before choosing."
    )
    recommendation = _recommendation(author_a, author_b)
    rationale = _recommendation_rationale(author_a, author_b, recommendation)
    return MultiAuthorComparisonSummary(
        factual_faithfulness_comparison=factual,
        author_style_difference=style,
        readability_difference=readability,
        recommended_draft=recommendation,
        recommendation_rationale=rationale,
    )


def _recommendation(
    author_a: AuthorComparisonOutput,
    author_b: AuthorComparisonOutput,
) -> RecommendationValue:
    rank_a = _recommendation_rank(author_a)
    rank_b = _recommendation_rank(author_b)
    if rank_a > rank_b:
        return "author_a"
    if rank_b > rank_a:
        return "author_b"
    return "no_clear_recommendation"


def _recommendation_rank(output: AuthorComparisonOutput) -> tuple[int, int, int]:
    readiness_score = {
        "safe_to_review": 3,
        "review_required": 2,
        "revision_required": 1,
    }.get(output.final_readiness or "", 0)
    blocker_score = -len(output.blockers)
    grounding_score = output.grounding_score or 0
    return (readiness_score, blocker_score, grounding_score)


def _recommendation_rationale(
    author_a: AuthorComparisonOutput,
    author_b: AuthorComparisonOutput,
    recommendation: RecommendationValue,
) -> str:
    if recommendation == "author_a":
        return _winning_rationale(author_a, author_b)
    if recommendation == "author_b":
        return _winning_rationale(author_b, author_a)
    return (
        "No clear recommendation: readiness, blocker count, and grounding score "
        "do not clearly separate the drafts."
    )


def _winning_rationale(
    winner: AuthorComparisonOutput,
    other: AuthorComparisonOutput,
) -> str:
    return (
        f"{winner.author_id} is preferred for editor review because it has "
        f"readiness={winner.final_readiness}, blockers={len(winner.blockers)}, "
        f"and grounding_score={winner.grounding_score}; the other draft has "
        f"readiness={other.final_readiness}, blockers={len(other.blockers)}, "
        f"and grounding_score={other.grounding_score}."
    )


def _score_comparison(
    score_a: int | None,
    score_b: int | None,
    missing: str,
    similar: str,
    a_better: str,
    b_better: str,
) -> str:
    if score_a is None or score_b is None:
        return missing
    delta = score_a - score_b
    if abs(delta) < 5:
        return f"{similar} A={score_a}, B={score_b}."
    if delta > 0:
        return f"{a_better} A={score_a}, B={score_b}."
    return f"{b_better} A={score_a}, B={score_b}."


def _word_count_comparison(
    author_a: AuthorComparisonOutput,
    author_b: AuthorComparisonOutput,
) -> str:
    delta = author_a.word_count - author_b.word_count
    if abs(delta) <= 30:
        return (
            "Both drafts are similar in article length. "
            f"A={author_a.word_count} words, B={author_b.word_count} words."
        )
    if delta > 0:
        return (
            "Author A produced the longer draft. "
            f"A={author_a.word_count} words, B={author_b.word_count} words."
        )
    return (
        "Author B produced the longer draft. "
        f"A={author_a.word_count} words, B={author_b.word_count} words."
    )


def _generation_call_count(draft: dict[str, object]) -> int:
    trace = _section_generation_trace(draft)
    single_section_retry_count = sum(
        1
        for item in trace
        if item.get("retry_attempted") and not item.get("group_generation_used")
    )
    section_single_call_count = sum(
        1 for item in trace if not item.get("group_generation_used")
    )
    metadata = _draft_generation_metadata(draft)
    return (
        1
        + _int_from_metadata(metadata, "generation_group_call_count")
        + section_single_call_count
        + single_section_retry_count
    )


def _editor_attention_items(
    *,
    evaluation: dict[str, object],
    blockers: list[str],
    warnings: list[str],
    article_body: object,
    claims_to_avoid: object,
) -> list[EditorAttentionItem]:
    article_text = str(article_body or "")
    avoid_rules = [str(rule) for rule in _list_value(claims_to_avoid) if str(rule)]
    items: list[EditorAttentionItem] = []

    unsupported_count = _add_evaluation_items(
        items,
        category="unsupported_claim",
        severity="blocker",
        label="Unsupported claim",
        values=_list_value(evaluation.get("unsupported_claims")),
        text_keys=("claim", "text"),
        article_text=article_text,
        editor_action=(
            "Verify this claim against the grounded brief, then remove it or rewrite "
            "it using only supported facts."
        ),
    )
    claims_to_avoid_count = _add_evaluation_items(
        items,
        category="claims_to_avoid_violation",
        severity="blocker",
        label="Claims-to-avoid violation",
        values=_list_value(evaluation.get("claims_to_avoid_violations")),
        text_keys=("claim", "text", "phrase"),
        article_text=article_text,
        editor_action=(
            "Remove or neutralize this article text so it does not make a claim the "
            "grounded brief marked as unsafe or unavailable."
        ),
        avoid_rules=avoid_rules,
    )
    overclaim_count = _add_evaluation_items(
        items,
        category="overclaim_phrase",
        severity="warning",
        label="Overclaim phrase",
        values=_list_value(evaluation.get("overclaim_phrases")),
        text_keys=("phrase", "claim", "text"),
        article_text=article_text,
        editor_action=(
            "Tone down this wording unless the grounded brief explicitly supports "
            "the implied impact or benefit."
        ),
    )

    if "unsupported_claims_remaining" in blockers and unsupported_count == 0:
        items.append(
            _fallback_attention_item(
                category="unsupported_claim",
                severity="blocker",
                label="Unsupported claims remain",
                reason=(
                    "The grounding evaluation reported unsupported claims, but exact "
                    "claim text was not available in the comparison response."
                ),
                editor_action=(
                    "Review the generated article against the grounded brief before "
                    "publication."
                ),
            )
        )
    if (
        "claims_to_avoid_violations_remaining" in blockers
        and claims_to_avoid_count == 0
    ):
        items.append(
            _fallback_attention_item(
                category="claims_to_avoid_violation",
                severity="blocker",
                label="Claims-to-avoid violations remain",
                avoid_rule=_single_or_joined_value(avoid_rules),
                reason=(
                    "The grounding evaluation reported a claims-to-avoid violation, "
                    "but exact matched article text was not available."
                ),
                editor_action=(
                    "Compare the article against the listed claims-to-avoid and "
                    "remove unsupported specificity."
                ),
            )
        )
    if "overclaim_phrases_remaining" in warnings and overclaim_count == 0:
        items.append(
            _fallback_attention_item(
                category="overclaim_phrase",
                severity="warning",
                label="Overclaim phrases remain",
                reason=(
                    "The grounding evaluation reported overclaim wording, but exact "
                    "phrase text was not available."
                ),
                editor_action=(
                    "Review impact, benefit, assurance, and certainty language for "
                    "unsupported emphasis."
                ),
            )
        )

    for code in (*blockers, *warnings):
        grounding_item = _grounding_code_attention_item(code, evaluation)
        if grounding_item is not None:
            items.append(grounding_item)

    return items


def _add_evaluation_items(
    items: list[EditorAttentionItem],
    *,
    category: str,
    severity: str,
    label: str,
    values: list[object],
    text_keys: tuple[str, ...],
    article_text: str,
    editor_action: str,
    avoid_rules: list[str] | None = None,
) -> int:
    added = 0
    for value in values:
        text = _attention_text(value, text_keys)
        reason = _attention_text(value, ("reason", "explanation"))
        suggested_action = _attention_text(
            value,
            ("suggested_fix", "suggested_action", "editor_action"),
        )
        avoid_rule = _avoid_rule(value, avoid_rules or [])
        if not text and not reason and not avoid_rule:
            continue
        items.append(
            EditorAttentionItem(
                category=category,
                severity=severity,
                label=label,
                claim_text=text,
                matched_article_text=text if text and text in article_text else None,
                avoid_rule=avoid_rule,
                reason=reason,
                editor_action=suggested_action or editor_action,
            )
        )
        added += 1
    return added


def _fallback_attention_item(
    *,
    category: str,
    severity: str,
    label: str,
    reason: str,
    editor_action: str,
    avoid_rule: str | None = None,
) -> EditorAttentionItem:
    return EditorAttentionItem(
        category=category,
        severity=severity,
        label=label,
        avoid_rule=avoid_rule,
        reason=reason,
        editor_action=editor_action,
    )


def _grounding_code_attention_item(
    code: str,
    evaluation: dict[str, object],
) -> EditorAttentionItem | None:
    score = _optional_int(evaluation.get("grounding_score"))
    if code == "grounding_score_below_threshold":
        return EditorAttentionItem(
            category="grounding_issue",
            severity="blocker",
            label="Grounding score below threshold",
            reason=f"Grounding score is {score}." if score is not None else None,
            editor_action=(
                "Treat this draft as requiring editor revision before publication."
            ),
        )
    if code == "grounding_score_review_band":
        return EditorAttentionItem(
            category="grounding_issue",
            severity="warning",
            label="Grounding score in review band",
            reason=f"Grounding score is {score}." if score is not None else None,
            editor_action="Review factual support before publication.",
        )
    if code == "grounding_score_missing":
        return EditorAttentionItem(
            category="grounding_issue",
            severity="warning",
            label="Grounding score missing",
            reason="No grounding score was available from the evaluation result.",
            editor_action="Run or inspect grounding evaluation before publication.",
        )
    return None


def _attention_text(value: object, keys: tuple[str, ...]) -> str | None:
    if isinstance(value, dict):
        for key in keys:
            text = value.get(key)
            if text:
                return str(text)
        return None
    if isinstance(value, str) and value:
        return value
    return None


def _avoid_rule(value: object, avoid_rules: list[str]) -> str | None:
    if isinstance(value, dict):
        for key in ("avoid_rule", "violated_rule", "rule", "claim_to_avoid"):
            rule = value.get(key)
            if rule:
                return str(rule)
    if len(avoid_rules) == 1:
        return avoid_rules[0]
    return None


def _single_or_joined_value(values: list[str]) -> str | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return "; ".join(values)


def _validate_latest_profile(repo: StyleScribeRepository, author_id: str) -> None:
    if repo.fetch_latest_author_style_profile(author_id) is None:
        raise MultiAuthorComparisonError(
            f"No author style profile found for author_id: {author_id}"
        )


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _unique_strings(values: list[str]) -> list[str]:
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return unique
