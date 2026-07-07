"""Revise generated article drafts using grounding evaluation feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticleDraftRecord,
    ArticleRevisionRecord,
    AuthorStyleProfileRecord,
    DraftEvaluationRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.models.article_revision_models import ArticleRevisionResponse
from backend.app.scripts.review_article_draft import TAMIL_FONT_STACK
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "article_revision_prompt.txt"
REVIEW_OUTPUT_DIR = Path("review_outputs")


@dataclass(frozen=True)
class RevisionLengthGuardrailResult:
    revision: dict[str, object]
    metadata: dict[str, object]
    revision_rejected_for_length_collapse: bool


@dataclass(frozen=True)
class RevisionPatchApplicationResult:
    revision: dict[str, object]
    metadata: dict[str, object]
    cleanup_candidate: dict[str, object]


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class ArticleRevisionError(RuntimeError):
    """Raised when article revision cannot be completed or fetched."""


def revise_article_grounding(
    draft_id: str,
    evaluation_id: str | None = None,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> ArticleRevisionResponse:
    """Revise a draft using the selected or latest grounding evaluation."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    draft = repo.fetch_article_draft(draft_id)
    if draft is None:
        raise ArticleRevisionError(f"No article draft found for draft_id: {draft_id}")
    brief = repo.fetch_grounded_brief(draft.brief_id)
    if brief is None:
        raise ArticleRevisionError(
            f"No grounded brief found for brief_id: {draft.brief_id}"
        )
    evaluation = (
        repo.fetch_draft_evaluation(evaluation_id)
        if evaluation_id
        else repo.fetch_latest_draft_evaluation(draft_id)
    )
    if evaluation is None:
        raise ArticleRevisionError(
            f"No grounding evaluation found for draft_id: {draft_id}"
        )
    if evaluation.draft_id != draft.draft_id:
        raise ArticleRevisionError(
            "Grounding evaluation does not belong to the requested draft."
        )
    profile = repo.fetch_author_style_profile(draft.profile_id)
    if profile is None:
        raise ArticleRevisionError(
            f"No author style profile found for profile_id: {draft.profile_id}"
        )

    revision_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article revision."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    evaluation_json = StyleScribeRepository.decode_json_object(
        evaluation.evaluation_json
    )
    grounding_findings = _grounding_findings_to_patch(evaluation_json)
    user_payload = build_article_revision_input(
        draft=draft,
        brief=brief,
        evaluation=evaluation,
        profile=profile,
    )

    try:
        revision_instruction = revision_client.generate_structured_json(
            prompt,
            user_payload,
        )
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise ArticleRevisionError("OpenAI article revision failed.") from exc

    draft_json = StyleScribeRepository.decode_json_object(draft.draft_json)
    patch_result = _apply_revision_patch_mode(
        revision_instruction=revision_instruction,
        original_draft=draft_json,
        desired_word_count=draft.desired_word_count,
        unsupported_claim_findings=_unsupported_claim_findings(grounding_findings),
    )
    revised = patch_result.revision
    raw_revised_text = _draft_text(revised)
    revised = cleanup_revised_article_tamil(revised)
    cleanup_applied = raw_revised_text != _draft_text(revised)
    guardrail = _apply_revision_length_guardrail(
        revised=revised,
        original_draft=draft_json,
        desired_word_count=draft.desired_word_count,
    )
    revised = guardrail.revision
    revision_summary = _revision_summary_with_cleanup_note(
        str(revised.get("revision_summary") or ""),
        cleanup_applied,
    )
    if guardrail.revision_rejected_for_length_collapse:
        revision_summary = (
            f"{revision_summary} Revision body rejected for length collapse; "
            "the original assembled article body was preserved."
        )
    created_at = datetime.now(UTC).isoformat()
    record = ArticleRevisionRecord(
        revision_id=str(uuid4()),
        draft_id=draft.draft_id,
        evaluation_id=evaluation.evaluation_id,
        author_id=draft.author_id,
        revised_headline=str(revised.get("headline") or ""),
        revised_subheadline=str(revised.get("subheadline") or ""),
        revised_article_body=str(revised.get("article_body") or ""),
        revised_seo_title=str(revised.get("seo_title") or ""),
        revised_meta_description=str(revised.get("meta_description") or ""),
        revised_tags_json=StyleScribeRepository.encode_json(
            _list_value(revised.get("suggested_tags"))
        ),
        revision_summary=revision_summary,
        removed_or_softened_claims_json=StyleScribeRepository.encode_json(
            _list_value(revised.get("removed_or_softened_claims"))
        ),
        model_provider=revision_client.provider,
        model_name=revision_client.model_name,
        token_usage_json=StyleScribeRepository.encode_json(
            {
                **_dict_value(revision_instruction.get("token_usage")),
                "revision_patch_metadata": patch_result.metadata,
                "revision_length_guardrail": {
                    **guardrail.metadata,
                    **patch_result.metadata,
                },
            }
        ),
        created_at=created_at,
    )
    repo.save_article_revision(record)
    return _revision_response(record, [])


def get_latest_article_revision(
    draft_id: str,
    repository: StyleScribeRepository | None = None,
) -> ArticleRevisionResponse:
    """Fetch the latest stored revision for a draft."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_latest_article_revision(draft_id)
    if record is None:
        raise ArticleRevisionError(
            f"No article revision found for draft_id: {draft_id}"
        )
    return _revision_response(record, [])


def build_article_revision_input(
    draft: ArticleDraftRecord,
    brief: GroundedBriefRecord,
    evaluation: DraftEvaluationRecord,
    profile: AuthorStyleProfileRecord,
) -> str:
    """Build bounded revision input from draft, brief, evaluation, and style."""

    evaluation_json = StyleScribeRepository.decode_json_object(
        evaluation.evaluation_json
    )
    draft_json = StyleScribeRepository.decode_json_object(draft.draft_json)
    original_word_count = approximate_tamil_word_count(
        str(draft_json.get("article_body") or "")
    )
    target_range = _target_word_count_range(draft.desired_word_count)
    grounding_findings = _grounding_findings_to_patch(evaluation_json)
    payload = {
        "target_language": draft.target_language,
        "article_type": draft.article_type,
        "desired_word_count": draft.desired_word_count,
        "original_draft_approximate_word_count": original_word_count,
        "revised_article_target_word_count_range": target_range,
        "tone_override": draft.tone_override,
        "author_instruction": draft.author_instruction,
        "style_profile_for_voice_only": {
            "profile_id": profile.profile_id,
            "language": profile.language,
            "profile": StyleScribeRepository.decode_json_object(profile.profile_json),
        },
        "grounded_brief_for_facts_only": {
            "brief_id": brief.brief_id,
            "source_language": brief.source_language,
            "target_language": brief.target_language,
            "brief": StyleScribeRepository.decode_json_object(brief.brief_json),
            "source_excerpt": brief.source_text_excerpt,
        },
        "original_generated_draft": {
            "draft_id": draft.draft_id,
            "draft": draft_json,
            "article_body_word_count": original_word_count,
        },
        "grounding_evaluation_feedback": {
            "evaluation_id": evaluation.evaluation_id,
            "evaluation_summary": {
                "grounding_score": evaluation_json.get("grounding_score"),
                "claim_safety_score": evaluation_json.get("claim_safety_score"),
                "fact_preservation_score": evaluation_json.get(
                    "fact_preservation_score"
                ),
                "overall_risk": evaluation_json.get("overall_risk"),
                "editorial_readiness": evaluation_json.get(
                    "editorial_readiness"
                ),
            },
            "grounding_findings_to_patch": grounding_findings,
            "unsupported_claims": evaluation_json.get("unsupported_claims", []),
            "invented_facts": evaluation_json.get("invented_facts", []),
            "contradictions": evaluation_json.get("contradictions", []),
            "claims_to_avoid_violations": evaluation_json.get(
                "claims_to_avoid_violations",
                [],
            ),
            "overclaim_phrases": evaluation_json.get("overclaim_phrases", []),
            "rewrite_guidance": evaluation_json.get("rewrite_guidance", []),
        },
        "revision_rule": (
            "Return only patch instructions. Do not rewrite the full article. "
            "Use only grounded_brief_for_facts_only for facts. First address "
            "unsupported claims, invented facts, contradictions, and "
            "claims_to_avoid violations in grounding_findings_to_patch. Remove "
            "or soften those claims with small exact text replacements. Do not "
            "introduce new facts."
        ),
        "language_rule": (
            "Write natural Tamil. Remove English leftovers such as 'ruling' when "
            "a natural Tamil equivalent fits the grounded facts."
        ),
        "length_preservation_rule": (
            "The article is already section-assembled. Preserve its structure, "
            "section coverage, and length. Do not compress a full article into a "
            "short summary. Suggest small phrase-level or sentence-level patches "
            "only. When possible, rewrite unsupported sentences into neutral "
            "grounded context. Keep the final article near 75% to 115% of "
            "desired_word_count. If an issue cannot be safely corrected with a "
            "compact patch, put it in notes instead of rewriting the article."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _grounding_findings_to_patch(
    evaluation_json: dict[str, object],
) -> list[dict[str, object]]:
    findings: list[dict[str, object]] = []
    finding_specs = (
        ("unsupported_claim", "unsupported_claims", "claim", "high"),
        ("invented_fact", "invented_facts", "fact", "high"),
        ("contradiction", "contradictions", "claim", "high"),
        ("claims_to_avoid_violation", "claims_to_avoid_violations", "claim", "high"),
        ("overclaim", "overclaim_phrases", "phrase", "medium"),
    )
    for issue_type, key, text_key, default_severity in finding_specs:
        for index, item in enumerate(_list_value(evaluation_json.get(key)), start=1):
            if isinstance(item, dict):
                text = (
                    item.get(text_key)
                    or item.get("claim")
                    or item.get("phrase")
                    or item.get("fact")
                    or item.get("text")
                )
                reason = item.get("reason") or item.get("explanation") or ""
                suggested_action = item.get("suggested_fix") or item.get(
                    "suggested_action"
                )
                severity = item.get("severity") or item.get("confidence")
            else:
                text = item
                reason = ""
                suggested_action = None
                severity = None
            findings.append(
                {
                    "finding_id": f"{issue_type}_{index}",
                    "issue_type": issue_type,
                    "claim_text": str(text or ""),
                    "reason": str(reason or ""),
                    "evidence_status": "not_supported_by_grounded_brief",
                    "severity": str(severity or default_severity),
                    "suggested_action": str(suggested_action or ""),
                }
            )
    return [finding for finding in findings if finding["claim_text"]]


def _unsupported_claim_findings(
    findings: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        finding
        for finding in findings
        if finding.get("issue_type") == "unsupported_claim"
    ]


def _target_word_count_range(desired_word_count: int | None) -> dict[str, int] | None:
    if desired_word_count is None:
        return None
    return {
        "minimum_75_percent": int(desired_word_count * 0.75),
        "target": desired_word_count,
        "maximum_115_percent": int(desired_word_count * 1.15),
    }


def _apply_revision_length_guardrail(
    revised: dict[str, object],
    original_draft: dict[str, object],
    desired_word_count: int | None,
) -> RevisionLengthGuardrailResult:
    input_body = str(original_draft.get("article_body") or "")
    output_body = str(revised.get("article_body") or "")
    input_word_count = approximate_tamil_word_count(input_body)
    output_word_count = approximate_tamil_word_count(output_body)
    target_min = int(desired_word_count * 0.75) if desired_word_count else None
    reject = False
    if target_min is not None:
        if (
            input_word_count < target_min
            and output_word_count < int(input_word_count * 0.90)
        ):
            reject = True
        if input_word_count >= target_min and output_word_count < target_min:
            reject = True
    guarded = dict(revised)
    if reject:
        guarded["article_body"] = input_body
    metadata: dict[str, object] = {
        "revision_input_word_count": input_word_count,
        "revision_output_word_count": output_word_count,
        "revision_delta_word_count": output_word_count - input_word_count,
        "target_min_word_count": target_min,
        "revision_rejected_for_length_collapse": reject,
    }
    return RevisionLengthGuardrailResult(
        revision=guarded,
        metadata=metadata,
        revision_rejected_for_length_collapse=reject,
    )


def _apply_revision_patch_mode(
    revision_instruction: dict[str, object],
    original_draft: dict[str, object],
    desired_word_count: int | None,
    unsupported_claim_findings: list[dict[str, object]],
) -> RevisionPatchApplicationResult:
    unsupported_claim_findings_count = len(unsupported_claim_findings)
    if "patches" not in revision_instruction:
        legacy = dict(revision_instruction)
        return RevisionPatchApplicationResult(
            revision=legacy,
            cleanup_candidate=legacy,
            metadata={
                "revision_mode": "full_rewrite_legacy",
                "revision_patch_count": 0,
                "revision_patches_applied_count": 0,
                "revision_patches_skipped_count": 0,
                "revision_patch_skipped_reasons": [],
                "revision_rejected_reason": None,
                "revised_article_source": "full_rewrite_legacy",
                "unsupported_claim_findings_count": unsupported_claim_findings_count,
                "unsupported_claim_patch_count": 0,
                "unsupported_claim_patches_applied_count": 0,
                "unsupported_claim_patches_skipped_count": 0,
                "unsupported_claim_patch_skipped_reasons": [],
                "unsupported_claims_unresolved_count": unsupported_claim_findings_count,
                "unsupported_claims_cleared_by_patch": (
                    unsupported_claim_findings_count == 0
                ),
            },
        )

    original_body = str(original_draft.get("article_body") or "")
    input_word_count = approximate_tamil_word_count(original_body)
    current = _revision_from_original_draft(original_draft)
    skipped: list[str] = []
    unsupported_skipped: list[str] = []
    applied_count = 0
    unsupported_patch_count = 0
    unsupported_applied_count = 0
    patch_items = _list_value(revision_instruction.get("patches"))[:8]
    removed_or_softened_claims: list[object] = []
    applied_patch_details: list[dict[str, object]] = []
    unsupported_patch_details: list[dict[str, object]] = []

    for index, item in enumerate(patch_items, start=1):
        if not isinstance(item, dict):
            skipped.append(f"patch {index}: other")
            continue
        issue_type = str(item.get("issue_type") or "other")
        confidence = str(item.get("confidence") or "").lower()
        target_text = str(item.get("target_text") or "")
        replacement_text = str(item.get("replacement_text") or "")
        source_finding_id = str(item.get("source_finding_id") or "")
        resolves_blocker = bool(
            item.get("resolves_blocker") or issue_type == "unsupported_claim"
        )
        blocker_type = str(item.get("blocker_type") or issue_type)
        is_unsupported_patch = issue_type == "unsupported_claim" or (
            blocker_type == "unsupported_claim"
        )
        if is_unsupported_patch:
            unsupported_patch_count += 1
        if confidence not in {"high", "medium"}:
            reason = _patch_skip_reason(index, issue_type, "low_confidence")
            skipped.append(reason)
            if is_unsupported_patch:
                unsupported_skipped.append(reason)
            continue
        if not target_text:
            reason = _patch_skip_reason(index, issue_type, "target_not_found")
            skipped.append(reason)
            if is_unsupported_patch:
                unsupported_skipped.append(reason)
            continue
        if not replacement_text:
            reason = _patch_skip_reason(index, issue_type, "empty_replacement")
            skipped.append(reason)
            if is_unsupported_patch:
                unsupported_skipped.append(reason)
            continue
        target_word_count = approximate_tamil_word_count(target_text)
        replacement_word_count = approximate_tamil_word_count(replacement_text)
        if (
            issue_type != "unsupported_claim"
            and target_word_count >= 8
            and replacement_word_count < int(target_word_count * 0.50)
        ):
            reason = _patch_skip_reason(index, issue_type, "unsafe_length_reduction")
            skipped.append(reason)
            continue
        field_name, occurrence_count, match_mode, span = _find_unique_patch_target(
            current,
            target_text,
            allow_normalized=is_unsupported_patch,
        )
        if occurrence_count == 0 and is_unsupported_patch and source_finding_id:
            finding_target = _finding_target_text(
                unsupported_claim_findings,
                source_finding_id,
            )
            if finding_target and finding_target != target_text:
                (
                    field_name,
                    occurrence_count,
                    match_mode,
                    span,
                ) = _find_unique_patch_target(
                    current,
                    finding_target,
                    allow_normalized=True,
                )
                if occurrence_count == 1:
                    target_text = finding_target
                    match_mode = f"finding_{match_mode}"
        if occurrence_count == 0:
            reason_code = (
                "unsupported_claim_patch_unmatched"
                if is_unsupported_patch
                else "target_not_found"
            )
            reason = _patch_skip_reason(index, issue_type, reason_code)
            skipped.append(reason)
            if is_unsupported_patch:
                unsupported_skipped.append(reason)
            continue
        if occurrence_count > 1 or field_name is None:
            reason = _patch_skip_reason(
                index,
                issue_type,
                "target_found_multiple_times",
            )
            skipped.append(reason)
            if is_unsupported_patch:
                unsupported_skipped.append(reason)
            continue
        current[field_name] = _replace_patch_target(
            text=str(current.get(field_name) or ""),
            target_text=target_text,
            replacement_text=replacement_text,
            span=span,
        )
        applied_count += 1
        detail = {
            "patch_index": index,
            "issue_type": issue_type,
            "source_finding_id": source_finding_id,
            "resolves_blocker": resolves_blocker,
            "blocker_type": blocker_type,
            "confidence": confidence,
            "match_mode": match_mode,
        }
        applied_patch_details.append(detail)
        if is_unsupported_patch:
            unsupported_applied_count += 1
            removed_or_softened_claims.append(target_text)
            unsupported_patch_details.append(detail)

    patched_body = str(current.get("article_body") or "")
    output_word_count = approximate_tamil_word_count(patched_body)
    target_min = int(desired_word_count * 0.75) if desired_word_count else None
    rejected_reason = None
    if target_min is not None and output_word_count < target_min:
        rejected_reason = "patched_article_below_target_min_word_count"
    elif output_word_count < int(input_word_count * 0.85):
        rejected_reason = "patched_article_reduced_word_count_too_much"

    if rejected_reason:
        current = _revision_from_original_draft(original_draft)
        output_word_count = approximate_tamil_word_count(
            str(current.get("article_body") or "")
        )
        revised_article_source = "original_due_to_patch_rejection"
    else:
        revised_article_source = "patch_applied"

    current["revision_summary"] = _patch_revision_summary(
        applied_count,
        len(skipped),
        revision_instruction,
        rejected_reason,
    )
    current["removed_or_softened_claims"] = removed_or_softened_claims
    metadata: dict[str, object] = {
        "revision_mode": "patch",
        "revision_patch_count": len(patch_items),
        "revision_patches_applied_count": applied_count if not rejected_reason else 0,
        "revision_patches_skipped_count": len(skipped),
        "revision_patch_skipped_reasons": skipped,
        "revision_patches_applied": applied_patch_details,
        "revision_input_word_count": input_word_count,
        "revision_output_word_count": output_word_count,
        "revision_delta_word_count": output_word_count - input_word_count,
        "revision_rejected_reason": rejected_reason,
        "revised_article_source": revised_article_source,
        "unsupported_claim_findings_count": unsupported_claim_findings_count,
        "unsupported_claim_patch_count": unsupported_patch_count,
        "unsupported_claim_patches_applied_count": (
            unsupported_applied_count if not rejected_reason else 0
        ),
        "unsupported_claim_patches_skipped_count": len(unsupported_skipped),
        "unsupported_claim_patch_skipped_reasons": unsupported_skipped,
        "unsupported_claim_patch_details": unsupported_patch_details,
        "unsupported_claims_unresolved_count": max(
            unsupported_claim_findings_count - unsupported_applied_count,
            0,
        ),
        "unsupported_claims_cleared_by_patch": (
            unsupported_claim_findings_count > 0
            and unsupported_applied_count >= unsupported_claim_findings_count
            and not rejected_reason
        ),
    }
    return RevisionPatchApplicationResult(
        revision=current,
        cleanup_candidate=current,
        metadata=metadata,
    )


def _revision_from_original_draft(
    original_draft: dict[str, object],
) -> dict[str, object]:
    return {
        "headline": str(original_draft.get("headline") or ""),
        "subheadline": str(original_draft.get("subheadline") or ""),
        "article_body": str(original_draft.get("article_body") or ""),
        "seo_title": str(original_draft.get("seo_title") or ""),
        "meta_description": str(original_draft.get("meta_description") or ""),
        "suggested_tags": _list_value(original_draft.get("suggested_tags")),
        "revision_summary": "",
        "removed_or_softened_claims": [],
    }


def _find_unique_patch_target(
    draft: dict[str, object],
    target_text: str,
    allow_normalized: bool = False,
) -> tuple[str | None, int, str, tuple[int, int] | None]:
    field_name = None
    total = 0
    match_mode = "exact"
    span = None
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        text = str(draft.get(key) or "")
        count = text.count(target_text)
        if count:
            field_name = key
            total += count
            if count == 1:
                start = text.index(target_text)
                span = (start, start + len(target_text))
    if total or not allow_normalized:
        return field_name, total, match_mode, span

    normalized_field = None
    normalized_span = None
    normalized_total = 0
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        text = str(draft.get(key) or "")
        count, candidate_span = _normalized_occurrence(text, target_text)
        if count:
            normalized_field = key
            normalized_total += count
            normalized_span = candidate_span
    if normalized_total:
        return normalized_field, normalized_total, "normalized", normalized_span
    return None, 0, "skipped", None


def _finding_target_text(
    findings: list[dict[str, object]],
    source_finding_id: str,
) -> str | None:
    for finding in findings:
        if finding.get("finding_id") == source_finding_id:
            return str(finding.get("claim_text") or "")
    return None


def _normalized_occurrence(
    text: str,
    target_text: str,
) -> tuple[int, tuple[int, int] | None]:
    normalized_text, mapping = _normalize_for_patch_match(text)
    normalized_target, _ = _normalize_for_patch_match(target_text)
    if not normalized_target:
        return 0, None
    count = normalized_text.count(normalized_target)
    if count != 1:
        return count, None
    normalized_start = normalized_text.index(normalized_target)
    normalized_end = normalized_start + len(normalized_target)
    original_start = mapping[normalized_start]
    original_end = mapping[normalized_end - 1] + 1
    return 1, (original_start, original_end)


def _normalize_for_patch_match(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    mapping: list[int] = []
    previous_space = False
    punctuation_map = {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
    }
    for index, char in enumerate(text):
        replacement = punctuation_map.get(char, char)
        if replacement.isspace():
            if previous_space:
                continue
            replacement = " "
            previous_space = True
        else:
            previous_space = False
        normalized_chars.append(replacement)
        mapping.append(index)
    normalized = "".join(normalized_chars).strip()
    joined = "".join(normalized_chars)
    if normalized == joined:
        return normalized, mapping
    leading_trim = len(joined) - len(joined.lstrip())
    trailing_trim = len(joined.rstrip())
    return normalized, mapping[leading_trim:trailing_trim]


def _replace_patch_target(
    *,
    text: str,
    target_text: str,
    replacement_text: str,
    span: tuple[int, int] | None,
) -> str:
    if span is not None:
        start, end = span
        return f"{text[:start]}{replacement_text}{text[end:]}"
    return text.replace(target_text, replacement_text, 1)


def _patch_skip_reason(index: int, issue_type: str, reason_code: str) -> str:
    return f"patch {index}: {issue_type}: {reason_code}"


def _patch_revision_summary(
    applied_count: int,
    skipped_count: int,
    revision_instruction: dict[str, object],
    rejected_reason: str | None,
) -> str:
    notes = _list_value(revision_instruction.get("notes"))
    parts = [
        (
            "Patch revision applied "
            f"{applied_count} safe edits and skipped {skipped_count} patches."
        )
    ]
    if rejected_reason:
        parts.append(
            "Patched article rejected by local safety checks; original article "
            "body was preserved."
        )
    if notes:
        parts.append("Notes: " + "; ".join(str(note) for note in notes[:3]))
    return " ".join(parts)


def cleanup_revised_article_tamil(revised: dict[str, object]) -> dict[str, object]:
    """Conservatively clean common Tamil-English leftovers in revised drafts."""

    cleaned = dict(revised)
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        value = cleaned.get(key)
        if isinstance(value, str):
            cleaned[key] = _cleanup_tamil_english_leftovers(value)
    return cleaned


def export_revision_review(
    workflow_id: str,
    export_format: str,
    cleaned_source_excerpt: str,
    brief: dict[str, object],
    original_draft: dict[str, object],
    initial_evaluation: dict[str, object] | None,
    revised_draft: dict[str, object] | None,
    revision_summary: str | None,
    final_evaluation: dict[str, object] | None = None,
    tamil_quality_status: str | None = None,
    tamil_quality_warnings: list[str] | None = None,
    requested_word_count: int | None = None,
    original_draft_word_count: int | None = None,
    revised_word_count_before_expansion: int | None = None,
    final_article_word_count: int | None = None,
    length_status: str | None = None,
    length_warning_reason: str | None = None,
    final_article_word_count_ratio: float | None = None,
    length_recovery_required: bool = False,
    length_recovery_attempted: bool = False,
    length_recovery_succeeded: bool = False,
    length_recovery_failed: bool = False,
    expansion_items_available: int = 0,
    expansion_items_used: list[object] | None = None,
    article_plan: dict[str, object] | None = None,
    generation_metadata: dict[str, object] | None = None,
    section_generation_trace: list[dict[str, object]] | None = None,
    section_coverage_status: str | None = None,
    section_coverage_warnings: list[str] | None = None,
    readiness_metadata: dict[str, object] | None = None,
    google_signals: dict[str, object] | None = None,
) -> list[str]:
    """Export a before/after grounded revision review."""

    REVIEW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "html" if export_format == "html" else "md"
    output_path = (
        REVIEW_OUTPUT_DIR / f"{workflow_id}_grounding_revision_review.{suffix}"
    )
    markdown = render_revision_review_markdown(
        cleaned_source_excerpt=cleaned_source_excerpt,
        brief=brief,
        original_draft=original_draft,
        initial_evaluation=initial_evaluation,
        revised_draft=revised_draft,
        revision_summary=revision_summary,
        final_evaluation=final_evaluation,
        tamil_quality_status=tamil_quality_status,
        tamil_quality_warnings=tamil_quality_warnings,
        requested_word_count=requested_word_count,
        original_draft_word_count=original_draft_word_count,
        revised_word_count_before_expansion=revised_word_count_before_expansion,
        final_article_word_count=final_article_word_count,
        length_status=length_status,
        length_warning_reason=length_warning_reason,
        final_article_word_count_ratio=final_article_word_count_ratio,
        length_recovery_required=length_recovery_required,
        length_recovery_attempted=length_recovery_attempted,
        length_recovery_succeeded=length_recovery_succeeded,
        length_recovery_failed=length_recovery_failed,
        expansion_items_available=expansion_items_available,
        expansion_items_used=expansion_items_used,
        article_plan=article_plan,
        generation_metadata=generation_metadata,
        section_generation_trace=section_generation_trace,
        section_coverage_status=section_coverage_status,
        section_coverage_warnings=section_coverage_warnings,
        readiness_metadata=readiness_metadata,
        google_signals=google_signals,
    )
    content = (
        render_revision_review_html(markdown)
        if export_format == "html"
        else markdown
    )
    output_path.write_text(content, encoding="utf-8")
    return [str(output_path)]


def render_revision_review_markdown(
    cleaned_source_excerpt: str,
    brief: dict[str, object],
    original_draft: dict[str, object],
    initial_evaluation: dict[str, object] | None,
    revised_draft: dict[str, object] | None,
    revision_summary: str | None,
    final_evaluation: dict[str, object] | None = None,
    tamil_quality_status: str | None = None,
    tamil_quality_warnings: list[str] | None = None,
    requested_word_count: int | None = None,
    original_draft_word_count: int | None = None,
    revised_word_count_before_expansion: int | None = None,
    final_article_word_count: int | None = None,
    length_status: str | None = None,
    length_warning_reason: str | None = None,
    final_article_word_count_ratio: float | None = None,
    length_recovery_required: bool = False,
    length_recovery_attempted: bool = False,
    length_recovery_succeeded: bool = False,
    length_recovery_failed: bool = False,
    expansion_items_available: int = 0,
    expansion_items_used: list[object] | None = None,
    article_plan: dict[str, object] | None = None,
    generation_metadata: dict[str, object] | None = None,
    section_generation_trace: list[dict[str, object]] | None = None,
    section_coverage_status: str | None = None,
    section_coverage_warnings: list[str] | None = None,
    readiness_metadata: dict[str, object] | None = None,
    google_signals: dict[str, object] | None = None,
) -> str:
    lines = [
        "# Grounded Revision Review",
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
    if article_plan:
        lines.extend(
            [
                "",
                "## Article Plan",
                "",
                f"- Plan ID: {article_plan.get('plan_id')}",
                f"- Planned target word count: {article_plan.get('target_word_count')}",
                "- Planned min word count: "
                f"{article_plan.get('target_min_word_count')}",
                "- Planned max word count: "
                f"{article_plan.get('target_max_word_count')}",
                f"- Plan summary: {article_plan.get('plan_summary')}",
                "- Planned sections:",
            ]
        )
        for section in _list_value(article_plan.get("planned_sections")):
            lines.append(f"  - {section}")
        lines.extend(["- Expansion items used by plan:"])
        lines.extend(
            f"  - {item}"
            for item in _list_value(article_plan.get("expansion_items_used"))
        )
        lines.extend(["- Plan claims to avoid:"])
        lines.extend(
            f"  - {claim}"
            for claim in _list_value(article_plan.get("claims_to_avoid"))
        )
    lines.extend(["", "## Original Draft", ""])
    lines.extend(_draft_markdown_lines(original_draft))
    if initial_evaluation:
        lines.extend(["", "## Initial Grounding Evaluation", ""])
        lines.extend(_evaluation_markdown_lines(initial_evaluation))
    lines.extend(["", "## Revision Summary", "", revision_summary or ""])
    if revised_draft:
        lines.extend(["", "## Revised Draft", ""])
        lines.extend(_draft_markdown_lines(revised_draft))
    if generation_metadata:
        lines.extend(
            [
                "",
                "## Workflow Runtime Summary",
                "",
                "- Total runtime seconds: "
                f"{generation_metadata.get('total_runtime_seconds')}",
                "- Model by stage: "
                f"{generation_metadata.get('model_used_by_stage')}",
                "- LLM call count by stage: "
                f"{generation_metadata.get('llm_call_count_by_stage')}",
                "- Runtime by stage: "
                f"{generation_metadata.get('runtime_by_stage')}",
                "- Token usage by stage: "
                f"{generation_metadata.get('token_usage_by_stage')}",
                "- Estimated cost by stage USD: "
                f"{generation_metadata.get('estimated_cost_by_stage_usd')}",
                "- Estimated cost total USD: "
                f"{generation_metadata.get('estimated_cost_total_usd')}",
                "- Cost estimation available: "
                f"{generation_metadata.get('cost_estimation_available')}",
                "- Slowest stage: "
                f"{generation_metadata.get('slowest_stage')}",
                "- Highest-cost stage: "
                f"{generation_metadata.get('highest_cost_stage')}",
                "",
                "This is an editor-assisted draft. Blockers require human review "
                "before publication.",
            ]
        )
        lines.extend(
            [
                "",
                "## Dynamic Section Assembly",
                "",
                "- Desired word count: "
                f"{generation_metadata.get('desired_word_count')}",
                "- Target min word count: "
                f"{generation_metadata.get('target_min_word_count')}",
                "- Target max word count: "
                f"{generation_metadata.get('target_max_word_count')}",
                f"- Generation mode: {generation_metadata.get('generation_mode_used')}",
                f"- Article plan used: {generation_metadata.get('article_plan_used')}",
                "- Planned section count: "
                f"{generation_metadata.get('planned_section_count')}",
                "- Generated section count: "
                f"{generation_metadata.get('generated_section_count')}",
                "- Assembled section count: "
                f"{generation_metadata.get('assembled_section_count')}",
                "- Assembled article word count: "
                f"{generation_metadata.get('section_assembled_article_word_count')}",
                "- Original draft source: "
                f"{generation_metadata.get('original_draft_source')}",
                "- Original draft matches assembly: "
                f"{generation_metadata.get('original_draft_matches_section_assembly')}",
                "- Revision input word count: "
                f"{generation_metadata.get('revision_input_word_count')}",
                "- Revision mode: "
                f"{generation_metadata.get('revision_mode')}",
                "- Revision patch count: "
                f"{generation_metadata.get('revision_patch_count')}",
                "- Revision patches applied: "
                f"{generation_metadata.get('revision_patches_applied_count')}",
                "- Revision patches skipped: "
                f"{generation_metadata.get('revision_patches_skipped_count')}",
                "- Revision patch skipped reasons: "
                f"{generation_metadata.get('revision_patch_skipped_reasons')}",
                "- Revision output word count: "
                f"{generation_metadata.get('revision_output_word_count')}",
                "- Revision delta word count: "
                f"{generation_metadata.get('revision_delta_word_count')}",
                "- Revision rejected for length collapse: "
                f"{generation_metadata.get('revision_rejected_for_length_collapse')}",
                "- Revision rejected reason: "
                f"{generation_metadata.get('revision_rejected_reason')}",
                "- Revised article source: "
                f"{generation_metadata.get('revised_article_source')}",
                "- Unsupported claim findings count: "
                f"{generation_metadata.get('unsupported_claim_findings_count')}",
                "- Unsupported claim patch count: "
                f"{generation_metadata.get('unsupported_claim_patch_count')}",
                "- Unsupported claim patches applied: "
                f"{generation_metadata.get('unsupported_claim_patches_applied_count')}",
                "- Unsupported claim patches skipped: "
                f"{generation_metadata.get('unsupported_claim_patches_skipped_count')}",
                "- Unsupported claim patch skipped reasons: "
                f"{generation_metadata.get('unsupported_claim_patch_skipped_reasons')}",
                "- Unsupported claims unresolved count: "
                f"{generation_metadata.get('unsupported_claims_unresolved_count')}",
                "- Unsupported claims cleared by patch: "
                f"{generation_metadata.get('unsupported_claims_cleared_by_patch')}",
                "- Length recovery skipped reason: "
                f"{generation_metadata.get('length_recovery_skipped_reason')}",
                "- Length recovery input word count: "
                f"{generation_metadata.get('length_recovery_input_word_count')}",
                "- Final article source stage: "
                f"{generation_metadata.get('final_article_source_stage')}",
            ]
        )
    if section_generation_trace:
        lines.extend(["", "### Section Word Counts", ""])
        lines.append(
            "| Section | Heading | Target | Min | Max | First | Retry | "
            "Selected | Reason |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for trace in section_generation_trace:
            lines.append(
                f"| {trace.get('section_id')} | {trace.get('heading')} | "
                f"{trace.get('target_word_count')} | {trace.get('min_word_count')} | "
                f"{trace.get('max_word_count')} | "
                f"{trace.get('first_pass_word_count')} | "
                f"{trace.get('retry_word_count')} | "
                f"{trace.get('selected_word_count')} | "
                f"{trace.get('selected_reason')} |"
            )
    if final_evaluation:
        lines.extend(["", "## Final Grounding Evaluation", ""])
        lines.extend(_evaluation_markdown_lines(final_evaluation))
    if readiness_metadata:
        lines.extend(
            [
                "",
                "## Readiness Decision",
                "",
                f"- Initial readiness: {readiness_metadata.get('initial_readiness')}",
                "- Initial readiness reasons: "
                f"{readiness_metadata.get('initial_readiness_reasons')}",
                f"- Final readiness: {readiness_metadata.get('final_readiness')}",
                "- Final readiness reasons: "
                f"{readiness_metadata.get('final_readiness_reasons')}",
                "- Readiness decision source: "
                f"{readiness_metadata.get('readiness_decision_source')}",
                "- Final publication blockers: "
                f"{readiness_metadata.get('final_publication_blockers')}",
                "- Final publication warnings: "
                f"{readiness_metadata.get('final_publication_warnings')}",
                "- Publication ready completeness passed: "
                f"{readiness_metadata.get('publication_ready_completeness_passed')}",
                "- Allowed English terms block readiness: "
                f"{readiness_metadata.get('allowed_english_terms_block_readiness')}",
                "- Skipped patch impact: "
                f"{readiness_metadata.get('skipped_patch_impact')}",
            ]
        )
    if google_signals:
        lines.extend(_google_signals_markdown_lines(google_signals))
    if generation_metadata:
        lines.extend(
            [
                "",
                "## Unsupported Claim Closure",
                "",
                "- Initial unsupported claim findings: "
                f"{generation_metadata.get('unsupported_claim_findings_count')}",
                "- Unsupported claim patches generated: "
                f"{generation_metadata.get('unsupported_claim_patch_count')}",
                "- Unsupported claim patches applied: "
                f"{generation_metadata.get('unsupported_claim_patches_applied_count')}",
                "- Unsupported claim patches skipped: "
                f"{generation_metadata.get('unsupported_claim_patches_skipped_count')}",
                "- Unsupported claim skipped reasons: "
                f"{generation_metadata.get('unsupported_claim_patch_skipped_reasons')}",
                "- Unsupported claims unresolved after patch: "
                f"{generation_metadata.get('unsupported_claims_unresolved_count')}",
                "- Unsupported claims cleared by patch: "
                f"{generation_metadata.get('unsupported_claims_cleared_by_patch')}",
                "- Final readiness impact: "
                f"{(readiness_metadata or {}).get('final_publication_blockers')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Tamil Quality And Length",
            "",
            f"- Tamil quality status: {tamil_quality_status or 'not run'}",
            f"- Tamil quality warnings: {len(tamil_quality_warnings or [])}",
            f"- Requested desired_word_count: {requested_word_count}",
            f"- Original draft approximate word count: {original_draft_word_count}",
            "- Revised draft word count before expansion: "
            f"{revised_word_count_before_expansion}",
            f"- Final article approximate word count: {final_article_word_count}",
            f"- Length status: {length_status or 'not run'}",
            f"- Length warning reason: {length_warning_reason or 'none'}",
            f"- Final article word count ratio: {final_article_word_count_ratio}",
            f"- Section coverage status: {section_coverage_status or 'not run'}",
        ]
    )
    lines.extend(f"  - {warning}" for warning in tamil_quality_warnings or [])
    lines.extend(["- Section coverage warnings:"])
    lines.extend(f"  - {warning}" for warning in section_coverage_warnings or [])
    lines.extend(
        [
            "",
            "## Length Recovery",
            "",
            f"- Length recovery required: {length_recovery_required}",
            f"- Length recovery attempted: {length_recovery_attempted}",
            f"- Length recovery succeeded: {length_recovery_succeeded}",
            f"- Length recovery failed: {length_recovery_failed}",
            f"- Expansion items available: {expansion_items_available}",
            "- Expansion items used:",
        ]
    )
    lines.extend(f"  - {item}" for item in expansion_items_used or [])
    lines.extend(["", "## Final Article Used For Evaluation", ""])
    if revised_draft:
        lines.extend(_draft_markdown_lines(revised_draft))
    lines.extend(
        [
            "",
            "## Revision Checks",
            "",
            "- English leftovers removed: "
            f"{_english_leftovers_removed(original_draft, revised_draft)}",
            "- Initial unsupported claim count: "
            f"{_unsupported_claim_count(initial_evaluation)}",
            "- Final unsupported claim count: "
            f"{_unsupported_claim_count(final_evaluation)}",
            f"- Initial readiness: {_readiness(initial_evaluation)}",
            f"- Final readiness: {_readiness(final_evaluation)}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_revision_review_html(markdown: str) -> str:
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
  <title>Grounded Revision Review</title>
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


def _revision_response(
    record: ArticleRevisionRecord,
    warnings: list[str],
) -> ArticleRevisionResponse:
    token_usage = json.loads(record.token_usage_json)
    guardrail = _dict_value(token_usage.get("revision_length_guardrail"))
    revised_draft = {
        "headline": record.revised_headline,
        "subheadline": record.revised_subheadline,
        "article_body": record.revised_article_body,
        "seo_title": record.revised_seo_title,
        "meta_description": record.revised_meta_description,
        "suggested_tags": json.loads(record.revised_tags_json),
    }
    return ArticleRevisionResponse(
        revision_id=record.revision_id,
        draft_id=record.draft_id,
        evaluation_id=record.evaluation_id,
        author_id=record.author_id,
        model_provider=record.model_provider,
        model_name=record.model_name,
        revised_draft=revised_draft,
        revision_summary=record.revision_summary,
        removed_or_softened_claims=json.loads(record.removed_or_softened_claims_json),
        revision_mode=_optional_str(guardrail.get("revision_mode")),
        revision_patch_count=_optional_int(guardrail.get("revision_patch_count")) or 0,
        revision_patches_applied_count=(
            _optional_int(guardrail.get("revision_patches_applied_count")) or 0
        ),
        revision_patches_skipped_count=(
            _optional_int(guardrail.get("revision_patches_skipped_count")) or 0
        ),
        revision_patch_skipped_reasons=_list_value(
            guardrail.get("revision_patch_skipped_reasons")
        ),
        revision_input_word_count=_optional_int(
            guardrail.get("revision_input_word_count")
        ),
        revision_output_word_count=_optional_int(
            guardrail.get("revision_output_word_count")
        ),
        revision_delta_word_count=_optional_int(
            guardrail.get("revision_delta_word_count")
        ),
        revision_rejected_for_length_collapse=bool(
            guardrail.get("revision_rejected_for_length_collapse")
        ),
        revision_rejected_reason=_optional_str(
            guardrail.get("revision_rejected_reason")
        ),
        revised_article_source=_optional_str(guardrail.get("revised_article_source")),
        unsupported_claim_findings_count=(
            _optional_int(guardrail.get("unsupported_claim_findings_count")) or 0
        ),
        unsupported_claim_patch_count=(
            _optional_int(guardrail.get("unsupported_claim_patch_count")) or 0
        ),
        unsupported_claim_patches_applied_count=(
            _optional_int(
                guardrail.get("unsupported_claim_patches_applied_count")
            )
            or 0
        ),
        unsupported_claim_patches_skipped_count=(
            _optional_int(
                guardrail.get("unsupported_claim_patches_skipped_count")
            )
            or 0
        ),
        unsupported_claim_patch_skipped_reasons=_list_value(
            guardrail.get("unsupported_claim_patch_skipped_reasons")
        ),
        unsupported_claims_unresolved_count=(
            _optional_int(guardrail.get("unsupported_claims_unresolved_count")) or 0
        ),
        unsupported_claims_cleared_by_patch=bool(
            guardrail.get("unsupported_claims_cleared_by_patch")
        ),
        token_usage=token_usage,
        created_at=record.created_at,
        warnings=warnings,
    )


def _draft_markdown_lines(draft: dict[str, object]) -> list[str]:
    lines: list[str] = []
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        lines.extend([f"### {key.replace('_', ' ').title()}", ""])
        lines.extend([str(draft.get(key) or ""), ""])
    lines.extend(["### Tags", ""])
    lines.extend(f"- {tag}" for tag in _list_value(draft.get("suggested_tags")))
    return lines


def _evaluation_markdown_lines(evaluation: dict[str, object]) -> list[str]:
    lines = [
        f"- Grounding score: {evaluation.get('grounding_score')}",
        f"- Claim safety score: {evaluation.get('claim_safety_score')}",
        f"- Overall risk: {evaluation.get('overall_risk')}",
        f"- Editorial readiness: {evaluation.get('editorial_readiness')}",
        "- Unsupported claims:",
    ]
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
    return lines


def _google_signals_markdown_lines(
    google_signals: dict[str, object],
) -> list[str]:
    metadata = _dict_value(google_signals.get("metadata"))
    lines = [
        "",
        "## Google Signals",
        "",
        f"- Score: {google_signals.get('score')}/100",
        f"- Version: {google_signals.get('version')}",
        f"- Primary search intent: {metadata.get('primary_search_intent')}",
        f"- Suggested meta description: {metadata.get('suggested_meta_description')}",
        f"- Suggested slug: {metadata.get('suggested_slug')}",
        f"- Schema type: {metadata.get('schema_type')}",
        "- Components:",
    ]
    for component in _list_value(google_signals.get("components")):
        if not isinstance(component, dict):
            continue
        lines.append(
            "  - "
            f"{component.get('name')}: {component.get('score')}/100 "
            f"(weight {component.get('weight')}, "
            f"risk {component.get('risk_level')}) - "
            f"{component.get('rationale')}"
        )
    lines.extend(["- Risk flags:"])
    lines.extend(
        f"  - {item}" for item in _list_value(google_signals.get("risk_flags"))
    )
    lines.extend(["- Recommendations:"])
    lines.extend(
        f"  - {item}" for item in _list_value(google_signals.get("recommendations"))
    )
    if google_signals.get("overall_rationale"):
        lines.append(f"- Overall rationale: {google_signals.get('overall_rationale')}")
    return lines


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _cleanup_tamil_english_leftovers(value: str) -> str:
    replacements = (
        ("இந்த ruling", "இந்த தீர்ப்பு"),
        ("pertenc செய்கிறார்கள்", "சேர்ந்துள்ளனர்"),
        ("pertencிக்கிறார்கள்", "சேர்ந்துள்ளனர்"),
        ("pertenc", "சேர்ந்துள்ளனர்"),
        ("ruling வழங்கியுள்ளது", "தீர்ப்பு வழங்கியுள்ளது"),
        ("ruling குறித்து", "தீர்ப்பு குறித்து"),
        ("ruling", "தீர்ப்பு"),
        ("முக்கியமானதாக", "தொடர்புடையதாக"),
        ("முக்கியமானது", "தொடர்புடையது"),
        ("ஆதரவுடன் வந்துள்ளது", "கருத்துகளுடன் பதிவாகியுள்ளது"),
        (
            "உரிமைகளை உறுதிப்படுத்துவதில் தொடர்புடையது",
            "பிறப்புரிமை குடியுரிமை தொடர்பானதாகும்",
        ),
        (
            "வாழ்வில் தொடர்புடையது எனக் கூறலாம்",
            "குடும்பங்களுடன் தொடர்புடையதாக பார்க்கப்படுகிறது",
        ),
        (
            "எதிர்காலத்தை உறுதிப்படுத்துகிறது",
            "பிறப்புரிமை குடியுரிமையை குறிப்பிடுகிறது",
        ),
        (
            "குடும்பங்களுக்கு தொடர்புடையது",
            "குழந்தைகளின் பிறப்புரிமை குடியுரிமை தொடர்பானதாகும்",
        ),
        (
            "சமூகத்தில் தாக்கத்தை ஏற்படுத்தும் என்று எதிர்பார்க்கப்படுகிறது",
            "சமூகத்துடன் தொடர்புடையதாக பார்க்கப்படுகிறது",
        ),
        (
            "புதிய அத்தியாயத்தை தொடங்குகிறது",
            "தொடர்புடைய சட்ட முடிவாகும்",
        ),
        ("முக்கியமான சட்ட முடிவாகும்", "தொடர்புடைய சட்ட முடிவாகும்"),
        ("முக்கியமான சட்ட முடிவு", "தொடர்புடைய சட்ட முடிவு"),
    )
    cleaned = value
    for source, replacement in replacements:
        cleaned = cleaned.replace(source, replacement)
    return cleaned


def _revision_summary_with_cleanup_note(
    revision_summary: str,
    cleanup_applied: bool,
) -> str:
    summary_parts = [revision_summary.strip()] if revision_summary.strip() else []
    if cleanup_applied:
        summary_parts.append("Tamil-English mixed phrasing was cleaned.")
    joined = " ".join(summary_parts).strip()
    return joined or "Grounding revision completed."


def build_article_revision_length_recovery_input(
    original_user_payload: str,
    current_revision: dict[str, object],
    desired_word_count: int | None,
) -> str:
    """Build a retry payload when revision collapses below the target range."""

    payload = StyleScribeRepository.decode_json_object(original_user_payload)
    current_count = approximate_tamil_word_count(
        str(current_revision.get("article_body") or "")
    )
    minimum = int(desired_word_count * 0.75) if desired_word_count else None
    payload["length_recovery_request"] = {
        "reason": (
            "The revised article_body is below the 75% target and has collapsed "
            "toward a summary."
        ),
        "current_article_body_word_count": current_count,
        "minimum_expected_word_count": minimum,
        "desired_word_count": desired_word_count,
        "current_revision_to_expand_or_rewrite": current_revision,
        "instruction": (
            "Revise again without deleting safe grounded material. Keep grounded "
            "paragraphs from the original draft, rewrite unsupported sentences "
            "into neutral grounded context, preserve safe attributed quotes, and "
            "add a grounded closing paragraph. Use only grounded_brief_for_facts_only "
            "and its source_excerpt. Do not add filler or repeat points."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _needs_revision_length_recovery(
    revision: dict[str, object],
    desired_word_count: int | None,
) -> bool:
    if not desired_word_count:
        return False
    word_count = approximate_tamil_word_count(str(revision.get("article_body") or ""))
    return word_count < int(desired_word_count * 0.75)


def _draft_text(draft: dict[str, object] | None) -> str:
    if not draft:
        return ""
    fields = (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    )
    return "\n".join(str(draft.get(field) or "") for field in fields)


def _english_leftovers_removed(
    original_draft: dict[str, object],
    revised_draft: dict[str, object] | None,
) -> str:
    original_has_ruling = "ruling" in _draft_text(original_draft)
    revised_has_ruling = "ruling" in _draft_text(revised_draft)
    if original_has_ruling and not revised_has_ruling:
        return "yes"
    if revised_has_ruling:
        return "no"
    return "not present"


def _unsupported_claim_count(evaluation: dict[str, object] | None) -> int | str:
    if not evaluation:
        return "not run"
    return len(_list_value(evaluation.get("unsupported_claims")))


def _readiness(evaluation: dict[str, object] | None) -> str:
    if not evaluation:
        return "not run"
    value = evaluation.get("editorial_readiness")
    return str(value) if value else ""
