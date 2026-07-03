"""Generate and persist controlled Tamil article drafts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticleDraftRecord,
    ArticlePlanRecord,
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.models.article_draft_models import ArticleDraftResponse
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "article_generation_prompt.txt"
SECTION_PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "article_section_generation_prompt.txt"
)
MIN_EXPECTED_SECTION_OUTPUT_WORDS = 45


@dataclass(frozen=True)
class DynamicLengthTargets:
    desired_word_count: int
    target_min_word_count: int
    target_max_word_count: int
    section_count: int
    section_target_word_count: int
    section_min_word_count: int
    section_max_word_count: int


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class ArticleGenerationError(RuntimeError):
    """Raised when an article draft cannot be generated or fetched."""


def generate_article_draft(
    author_id: str,
    brief_id: str,
    author_instruction: str | None = None,
    target_language: str = "ta",
    article_type: str | None = None,
    desired_word_count: int | None = None,
    tone_override: str | None = None,
    include_seo: bool = True,
    plan_id: str | None = None,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> ArticleDraftResponse:
    """Generate, save, and return a controlled Tamil article draft."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    profile_record = repo.fetch_latest_author_style_profile(author_id)
    if profile_record is None:
        raise ArticleGenerationError(
            "No author style profile found. Generate style profile first."
        )

    brief_record = repo.fetch_grounded_brief(brief_id)
    if brief_record is None:
        message = f"No grounded brief found for brief_id: {brief_id}"
        raise ArticleGenerationError(message)
    plan_record = repo.fetch_article_plan(plan_id) if plan_id else None
    if plan_id and plan_record is None:
        raise ArticleGenerationError(f"No article plan found for plan_id: {plan_id}")

    resolved_article_type = article_type or "news"
    resolved_word_count = desired_word_count or 600
    warnings = _build_warnings(
        profile_record=profile_record,
        brief_record=brief_record,
        target_language=target_language,
        desired_word_count=resolved_word_count,
    )
    draft_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article draft generation."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_article_generation_input(
        profile_record=profile_record,
        brief_record=brief_record,
        author_instruction=author_instruction,
        target_language=target_language,
        article_type=resolved_article_type,
        desired_word_count=resolved_word_count,
        tone_override=tone_override,
        include_seo=include_seo,
        plan_record=plan_record,
    )
    dynamic_targets = derive_dynamic_length_targets(resolved_word_count)

    try:
        base_draft = draft_client.generate_structured_json(prompt, user_payload)
        section_draft = _generate_dynamic_article_sections(
            draft_client=draft_client,
            base_draft=base_draft,
            original_user_payload=user_payload,
            plan_record=plan_record,
            targets=dynamic_targets,
        )
        draft = section_draft or _normalize_plan_based_draft(base_draft, plan_record)
        draft_word_count = approximate_tamil_word_count(
            str(draft.get("article_body") or "")
        )
        skipped_reason = (
            "section assembly reached target minimum"
            if draft_word_count >= dynamic_targets.target_min_word_count
            else "initial draft preserves section assembly; workflow fallback may run"
        )
        draft = _annotate_length_recovery(
            draft,
            attempted=False,
            skipped_reason=skipped_reason,
            input_word_count=draft_word_count,
        )
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise ArticleGenerationError("OpenAI article draft generation failed.") from exc

    created_at = datetime.now(UTC).isoformat()
    record = ArticleDraftRecord(
        draft_id=str(uuid4()),
        author_id=author_id,
        profile_id=profile_record.profile_id,
        brief_id=brief_id,
        target_language=target_language,
        model_provider=draft_client.provider,
        model_name=draft_client.model_name,
        status="completed",
        author_instruction=author_instruction,
        article_type=resolved_article_type,
        desired_word_count=resolved_word_count,
        tone_override=tone_override,
        include_seo=include_seo,
        draft_json=StyleScribeRepository.encode_json(draft),
        warnings_json=StyleScribeRepository.encode_warnings(warnings),
        created_at=created_at,
    )
    repo.save_article_draft(record)
    return _draft_response(record, draft, warnings)


def get_article_draft(
    draft_id: str,
    repository: StyleScribeRepository | None = None,
) -> ArticleDraftResponse:
    """Fetch a saved article draft."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_article_draft(draft_id)
    if record is None:
        raise ArticleGenerationError(f"No article draft found for draft_id: {draft_id}")
    return _draft_response(
        record,
        StyleScribeRepository.decode_json_object(record.draft_json),
        StyleScribeRepository.decode_json_list(record.warnings_json),
    )


def build_article_generation_input(
    profile_record: AuthorStyleProfileRecord,
    brief_record: GroundedBriefRecord,
    author_instruction: str | None,
    target_language: str,
    article_type: str = "news",
    desired_word_count: int = 600,
    tone_override: str | None = None,
    include_seo: bool = True,
    plan_record: ArticlePlanRecord | None = None,
) -> str:
    """Build explicitly separated style/fact input for the model."""

    payload = {
        "target_language": target_language,
        "article_type": article_type,
        "desired_word_count": desired_word_count,
        "article_body_target_word_count_range": {
            "minimum_75_percent": int(desired_word_count * 0.75),
            "target": desired_word_count,
            "maximum_115_percent": int(desired_word_count * 1.15),
        },
        "tone_override": tone_override,
        "include_seo": include_seo,
        "author_instruction": author_instruction,
        "style_profile_for_voice_only": {
            "profile_id": profile_record.profile_id,
            "language": profile_record.language,
            "profile": StyleScribeRepository.decode_json_object(
                profile_record.profile_json
            ),
        },
        "grounded_brief_for_facts_only": {
            "brief_id": brief_record.brief_id,
            "source_language": brief_record.source_language,
            "target_language": brief_record.target_language,
            "brief": StyleScribeRepository.decode_json_object(
                brief_record.brief_json
            ),
        },
        "separation_rule": (
            "Use style_profile_for_voice_only only for tone and writing style. "
            "Use grounded_brief_for_facts_only as the only factual source."
        ),
        "grounding_rule": (
            "Facts must come only from grounded_brief_for_facts_only. Do not use "
            "outside knowledge or facts from author samples/style profile."
        ),
        "style_adaptation_rule": (
            "Use the author style profile as writing influence, not a fixed "
            "emotional template. Adapt tone to article_type and topic."
        ),
        "length_rule": (
            "Treat desired_word_count as the article_body target. Aim for the "
            "article_body_target_word_count_range when the grounded brief has "
            "enough confirmed facts, entities, numbers, quotes, background, "
            "timeline, affected groups, or legal/policy context. Do not return "
            "a short summary unless the brief is too thin for the requested length."
        ),
    }
    if plan_record is not None:
        plan = plan_record
        payload["grounded_article_plan"] = {
            "plan_id": plan.plan_id,
            "target_min_word_count": plan.target_min_word_count,
            "target_max_word_count": plan.target_max_word_count,
            "planned_sections": StyleScribeRepository.decode_json_list(
                plan.planned_sections_json
            ),
            "expansion_items_used": StyleScribeRepository.decode_json_list(
                plan.expansion_items_used_json
            ),
            "claims_to_avoid": StyleScribeRepository.decode_json_list(
                plan.claims_to_avoid_json
            ),
            "plan_summary": plan.plan_summary,
        }
        payload["plan_generation_rule"] = (
            "Write according to grounded_article_plan. Cover planned sections "
            "and section target words. Do not skip planned sections unless the "
            "section is unsupported by the grounded brief. Produce a complete "
            "Tamil article, not a short summary. Return article_sections with "
            "section_name, target_words, section_text, and grounded_facts_used "
            "for each supported planned section. The combined section_text "
            "content must meet the target range when the brief supports it."
        )
    return json.dumps(payload, ensure_ascii=False)


def build_article_generation_length_recovery_input(
    original_user_payload: str,
    current_draft: dict[str, object],
    desired_word_count: int,
) -> str:
    """Build a bounded retry payload when the first draft is too short."""

    payload = StyleScribeRepository.decode_json_object(original_user_payload)
    current_count = approximate_tamil_word_count(
        str(current_draft.get("article_body") or "")
    )
    recovery_request = {
        "reason": (
            "The first article_body was below 75% of desired_word_count and "
            "read like a short summary."
        ),
        "current_article_body_word_count": current_count,
        "minimum_expected_word_count": int(desired_word_count * 0.75),
        "desired_word_count": desired_word_count,
        "current_draft_to_expand_or_rewrite": current_draft,
        "instruction": (
            "Regenerate a complete grounded Tamil article_body within the target "
            "range if the brief supports it. Use confirmed facts, key entities, "
            "numbers/statistics, affected groups, quotes, timeline, background, "
            "and legal/policy context from grounded_brief_for_facts_only. Do not "
            "add filler, repeat points, or invent unsupported facts."
        ),
    }
    if payload.get("grounded_article_plan"):
        recovery_request["plan_section_instruction"] = (
            "Return article_sections with one section_text per supported planned "
            "section. Each section_text must be developed Tamil news prose near "
            "that section target_words. The backend will assemble article_body "
            "from article_sections, so do not leave sections as notes."
        )
    payload["length_recovery_request"] = recovery_request
    return json.dumps(payload, ensure_ascii=False)


def _needs_length_recovery(
    draft: dict[str, object],
    desired_word_count: int | None,
) -> bool:
    if not desired_word_count:
        return False
    word_count = approximate_tamil_word_count(str(draft.get("article_body") or ""))
    return word_count < int(desired_word_count * 0.75)


def derive_dynamic_length_targets(desired_word_count: int) -> DynamicLengthTargets:
    """Derive article and section word-count targets from requested length."""

    section_count = _section_count_for_desired_words(desired_word_count)
    section_target = round(desired_word_count / section_count)
    return DynamicLengthTargets(
        desired_word_count=desired_word_count,
        target_min_word_count=round(desired_word_count * 0.75),
        target_max_word_count=round(desired_word_count * 1.15),
        section_count=section_count,
        section_target_word_count=section_target,
        section_min_word_count=round(section_target * 0.80),
        section_max_word_count=round(section_target * 1.25),
    )


def _section_count_for_desired_words(desired_word_count: int) -> int:
    if desired_word_count <= 350:
        suggested_count = 3
    elif desired_word_count <= 500:
        suggested_count = 4
    elif desired_word_count <= 700:
        suggested_count = 6
    elif desired_word_count <= 900:
        suggested_count = 7
    elif desired_word_count <= 1200:
        suggested_count = 8
    else:
        suggested_count = 10
    minimum_viable_count = _ceiling_division(
        round(desired_word_count * 0.75),
        MIN_EXPECTED_SECTION_OUTPUT_WORDS,
    )
    return min(10, max(suggested_count, minimum_viable_count))


def _ceiling_division(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _normalize_plan_based_draft(
    draft: dict[str, object],
    plan_record: ArticlePlanRecord | None,
) -> dict[str, object]:
    if plan_record is None:
        return draft
    article_sections = draft.get("article_sections")
    if not isinstance(article_sections, list):
        return draft

    section_texts: list[str] = []
    normalized_sections: list[dict[str, object]] = []
    for section in article_sections:
        if not isinstance(section, dict):
            continue
        text = str(
            section.get("section_text")
            or section.get("text")
            or section.get("article_text")
            or ""
        ).strip()
        if not text:
            continue
        normalized = dict(section)
        normalized["section_text"] = text
        normalized_sections.append(normalized)
        section_texts.append(text)

    if not section_texts:
        return draft

    normalized_draft = dict(draft)
    normalized_draft["article_sections"] = normalized_sections
    normalized_draft["article_body"] = "\n\n".join(section_texts)
    notes = normalized_draft.get("fact_usage_notes")
    if isinstance(notes, list):
        normalized_draft["fact_usage_notes"] = [
            *notes,
            "Article body assembled from grounded article plan sections.",
        ]
    return normalized_draft


def _generate_dynamic_article_sections(
    draft_client: StructuredJsonClient,
    base_draft: dict[str, object],
    original_user_payload: str,
    plan_record: ArticlePlanRecord | None,
    targets: DynamicLengthTargets,
) -> dict[str, object] | None:
    payload = StyleScribeRepository.decode_json_object(original_user_payload)
    planned_sections = _dynamic_planned_sections(plan_record, targets)
    if not planned_sections:
        return None

    prompt = SECTION_PROMPT_PATH.read_text(encoding="utf-8")
    generated_sections: list[dict[str, object]] = []
    section_trace: list[dict[str, object]] = []
    for index, section in enumerate(planned_sections, start=1):
        generated = _generate_one_dynamic_section(
            draft_client=draft_client,
            prompt=prompt,
            base_payload=payload,
            section=section,
            section_index=index,
            targets=targets,
        )
        generated_sections.append(generated["section"])
        section_trace.append(generated["trace"])

    section_texts = [
        str(section.get("section_text") or "").strip()
        for section in generated_sections
        if str(section.get("section_text") or "").strip()
    ]
    if not section_texts:
        return None

    assembled_body = "\n\n".join(section_texts)
    assembled_word_count = approximate_tamil_word_count(assembled_body)
    assembled_paragraph_count = len(section_texts)
    target_payload = _targets_payload(targets)
    draft = dict(base_draft)
    draft["article_body"] = assembled_body
    draft["article_sections"] = generated_sections
    draft["dynamic_length_targets"] = target_payload
    draft["generation_mode_used"] = "section_assembled"
    draft["article_plan_used"] = plan_record is not None
    draft["planned_section_count"] = len(planned_sections)
    draft["generated_section_count"] = len(generated_sections)
    draft["assembled_section_count"] = len(section_texts)
    draft["section_assembled_article_word_count"] = assembled_word_count
    draft["section_assembled_article_paragraph_count"] = assembled_paragraph_count
    draft["original_draft_source"] = "section_assembled"
    draft["original_draft_word_count_after_assignment"] = assembled_word_count
    draft["original_draft_matches_section_assembly"] = (
        str(draft.get("article_body") or "") == assembled_body
    )
    draft["section_generation_trace"] = section_trace
    draft["token_usage"] = _sum_token_usage(
        [_dict_value(base_draft.get("token_usage"))]
        + [
            _dict_value(trace.get("first_pass_token_usage"))
            for trace in section_trace
        ]
        + [
            _dict_value(trace.get("retry_token_usage"))
            for trace in section_trace
        ]
    )
    draft["section_coverage_status"] = _section_generation_coverage_status(
        section_trace,
        targets,
    )
    draft["section_coverage_warnings"] = _section_generation_warnings(
        section_trace,
        targets,
    )
    notes = _list_value(draft.get("fact_usage_notes"))
    draft["fact_usage_notes"] = [
        *notes,
        "Article body generated section-by-section and assembled by backend.",
    ]
    return draft


def _generate_one_dynamic_section(
    draft_client: StructuredJsonClient,
    prompt: str,
    base_payload: dict[str, object],
    section: dict[str, object],
    section_index: int,
    targets: DynamicLengthTargets,
) -> dict[str, dict[str, object]]:
    section_payload = _dynamic_section_payload(
        base_payload=base_payload,
        section=section,
        section_index=section_index,
        targets=targets,
    )
    first_pass = draft_client.generate_structured_json(
        prompt,
        json.dumps(section_payload, ensure_ascii=False),
    )
    first_text = str(first_pass.get("section_text") or "").strip()
    first_word_count = approximate_tamil_word_count(first_text)
    selected = first_pass
    selected_text = first_text
    selected_reason = "first_pass_met_minimum"
    retry_attempted = first_word_count < targets.section_min_word_count
    retry_word_count: int | None = None

    if retry_attempted:
        retry_payload = dict(section_payload)
        retry_payload["section_retry_request"] = {
            "reason": "The generated section was below the minimum section length.",
            "current_section_word_count": first_word_count,
            "minimum_section_words": targets.section_min_word_count,
            "current_section_text": first_text,
            "instruction": (
                f"You wrote only {first_word_count} words. Expand this same "
                f"section to at least {targets.section_min_word_count} Tamil "
                "words using only the provided brief. Do not summarize. Do not "
                "restart. Preserve the same section focus."
            ),
        }
        retried = draft_client.generate_structured_json(
            prompt,
            json.dumps(retry_payload, ensure_ascii=False),
        )
        retried_text = str(retried.get("section_text") or "").strip()
        retry_word_count = approximate_tamil_word_count(retried_text)
        if retry_word_count >= first_word_count:
            selected = retried
            selected_text = retried_text
            selected_reason = "retry_selected_longer_or_equal"
        else:
            selected_reason = "first_pass_selected_retry_shorter"

    selected_word_count = approximate_tamil_word_count(selected_text)
    section_record = {
        "section_id": str(
            section.get("section_id") or section.get("section_name") or section_index
        ),
        "heading": str(section.get("heading") or section.get("section_name") or ""),
        "target_word_count": targets.section_target_word_count,
        "min_word_count": targets.section_min_word_count,
        "max_word_count": targets.section_max_word_count,
        "section_text": selected_text,
        "grounded_facts_used": _list_value(selected.get("grounded_facts_used")),
    }
    trace = {
        "section_id": section_record["section_id"],
        "heading": section_record["heading"],
        "target_word_count": targets.section_target_word_count,
        "min_word_count": targets.section_min_word_count,
        "max_word_count": targets.section_max_word_count,
        "first_pass_word_count": first_word_count,
        "retry_attempted": retry_attempted,
        "retry_word_count": retry_word_count,
        "selected_word_count": selected_word_count,
        "selected_reason": selected_reason,
        "first_pass_token_usage": _dict_value(first_pass.get("token_usage")),
        "retry_token_usage": _dict_value(retried.get("token_usage"))
        if retry_attempted
        else {},
    }
    return {"section": section_record, "trace": trace}


def _dynamic_section_payload(
    base_payload: dict[str, object],
    section: dict[str, object],
    section_index: int,
    targets: DynamicLengthTargets,
) -> dict[str, object]:
    heading = str(section.get("heading") or section.get("section_name") or "")
    purpose = str(section.get("purpose") or section.get("section_purpose") or "")
    grounded_article_plan = base_payload.get("grounded_article_plan")
    plan_payload = (
        grounded_article_plan if isinstance(grounded_article_plan, dict) else {}
    )
    return {
        "target_language": base_payload.get("target_language"),
        "article_type": base_payload.get("article_type"),
        "tone_override": base_payload.get("tone_override"),
        "author_instruction": base_payload.get("author_instruction"),
        "style_profile_for_voice_only": base_payload.get(
            "style_profile_for_voice_only"
        ),
        "grounded_brief_for_facts_only": base_payload.get(
            "grounded_brief_for_facts_only"
        ),
        "grounded_article_plan_summary": plan_payload.get("plan_summary"),
        "planned_section": section,
        "section_index": section_index,
        "dynamic_length_targets": _targets_payload(targets),
        "section_heading": heading,
        "section_purpose": purpose,
        "claims_to_avoid": plan_payload.get("claims_to_avoid"),
        "section_instruction": (
            "Generate only this section body. Output strict JSON. The section "
            f"target is {targets.section_target_word_count} Tamil words, minimum "
            f"{targets.section_min_word_count}, maximum "
            f"{targets.section_max_word_count}. Use only grounded facts from the "
            "brief. Neutral connective context is allowed only if it does not "
            "introduce new factual claims. Do not summarize too aggressively."
        ),
    }


def _dynamic_planned_sections(
    plan_record: ArticlePlanRecord | None,
    targets: DynamicLengthTargets,
) -> list[dict[str, object]]:
    raw_sections = (
        _decode_json_list_objects(plan_record.planned_sections_json)
        if plan_record is not None
        else []
    )
    sections: list[dict[str, object]] = []
    for index, raw_section in enumerate(raw_sections, start=1):
        if isinstance(raw_section, dict):
            section = dict(raw_section)
        else:
            section = {"section_name": str(raw_section)}
        section["target_words"] = targets.section_target_word_count
        section["min_words"] = targets.section_min_word_count
        section["max_words"] = targets.section_max_word_count
        section["section_id"] = section.get("section_id") or f"section_{index}"
        sections.append(section)
    if len(sections) >= targets.section_count:
        return sections[: targets.section_count]

    fallback_angles = [
        ("opening", "Opening news hook and core development"),
        ("background", "Background and why the development matters"),
        ("key_details", "Key details, entities, dates, and places"),
        ("numbers", "Numbers and statistics from the brief"),
        ("affected_groups", "Affected groups and reader relevance"),
        ("quotes", "Attributed quotes or source-supported statements"),
        ("policy_context", "Legal, policy, or civic context"),
        ("closing", "Cautious grounded closing"),
        ("next_steps", "Timeline and known next steps"),
        ("summary_close", "Final grounded synthesis"),
    ]
    for index in range(len(sections) + 1, targets.section_count + 1):
        key, purpose = fallback_angles[(index - 1) % len(fallback_angles)]
        sections.append(
            {
                "section_id": f"section_{index}",
                "section_name": key,
                "heading": purpose,
                "purpose": purpose,
                "target_words": targets.section_target_word_count,
                "min_words": targets.section_min_word_count,
                "max_words": targets.section_max_word_count,
            }
        )
    return sections


def _decode_json_list_objects(value: str) -> list[object]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []


def _targets_payload(targets: DynamicLengthTargets) -> dict[str, int]:
    return {
        "desired_word_count": targets.desired_word_count,
        "target_min_word_count": targets.target_min_word_count,
        "target_max_word_count": targets.target_max_word_count,
        "section_count": targets.section_count,
        "section_target_word_count": targets.section_target_word_count,
        "section_min_word_count": targets.section_min_word_count,
        "section_max_word_count": targets.section_max_word_count,
    }


def _section_generation_coverage_status(
    section_trace: list[dict[str, object]],
    targets: DynamicLengthTargets,
) -> str:
    return (
        "pass"
        if not _section_generation_warnings(section_trace, targets)
        else "warning"
    )


def _section_generation_warnings(
    section_trace: list[dict[str, object]],
    targets: DynamicLengthTargets,
) -> list[str]:
    warnings: list[str] = []
    if len(section_trace) != targets.section_count:
        warnings.append(
            f"Generated {len(section_trace)} sections for target "
            f"{targets.section_count}."
        )
    for trace in section_trace:
        selected_word_count = _coerce_int(trace.get("selected_word_count"))
        if selected_word_count < targets.section_min_word_count:
            warnings.append(
                f"Section {trace.get('section_id')} remained below minimum "
                f"({selected_word_count}/{targets.section_min_word_count})."
            )
        if selected_word_count > targets.section_max_word_count:
            warnings.append(
                f"Section {trace.get('section_id')} exceeded maximum "
                f"({selected_word_count}/{targets.section_max_word_count})."
            )
    return warnings


def _annotate_length_recovery(
    draft: dict[str, object],
    attempted: bool,
    skipped_reason: str | None,
    input_word_count: int | None,
) -> dict[str, object]:
    annotated = dict(draft)
    annotated["length_recovery_attempted"] = attempted
    annotated["length_recovery_skipped_reason"] = skipped_reason
    annotated["length_recovery_input_word_count"] = input_word_count
    return annotated


def _generate_article_sections_from_plan(
    draft_client: StructuredJsonClient,
    base_draft: dict[str, object],
    original_user_payload: str,
    plan_record: ArticlePlanRecord,
) -> dict[str, object]:
    payload = StyleScribeRepository.decode_json_object(original_user_payload)
    planned_sections = StyleScribeRepository.decode_json_list(
        plan_record.planned_sections_json
    )
    if not planned_sections:
        return base_draft

    prompt = SECTION_PROMPT_PATH.read_text(encoding="utf-8")
    generated_sections: list[dict[str, object]] = []
    for section in planned_sections:
        if not isinstance(section, dict):
            continue
        section_payload = {
            "target_language": payload.get("target_language"),
            "article_type": payload.get("article_type"),
            "tone_override": payload.get("tone_override"),
            "author_instruction": payload.get("author_instruction"),
            "style_profile_for_voice_only": payload.get(
                "style_profile_for_voice_only"
            ),
            "grounded_brief_for_facts_only": payload.get(
                "grounded_brief_for_facts_only"
            ),
            "grounded_article_plan_summary": (
                payload.get("grounded_article_plan") or {}
            ).get("plan_summary"),
            "planned_section": section,
            "claims_to_avoid": (payload.get("grounded_article_plan") or {}).get(
                "claims_to_avoid"
            ),
            "section_instruction": (
                "Write this planned section as developed Tamil news prose. "
                "Stay near target_words when supported, and do not write a "
                "single-sentence summary."
            ),
        }
        generated = draft_client.generate_structured_json(
            prompt,
            json.dumps(section_payload, ensure_ascii=False),
        )
        generated = _retry_short_section_if_needed(
            draft_client=draft_client,
            prompt=prompt,
            section_payload=section_payload,
            generated=generated,
            planned_section=section,
        )
        section_text = str(generated.get("section_text") or "").strip()
        if not section_text:
            continue
        generated_sections.append(
            {
                "section_name": str(
                    generated.get("section_name")
                    or section.get("section_name")
                    or "section"
                ),
                "target_words": generated.get("target_words")
                or section.get("target_words")
                or 0,
                "section_text": section_text,
                "grounded_facts_used": generated.get("grounded_facts_used") or [],
                "warnings": generated.get("warnings") or [],
            }
        )

    if not generated_sections:
        return base_draft

    assembled = dict(base_draft)
    assembled["article_sections"] = generated_sections
    assembled["article_body"] = "\n\n".join(
        str(section["section_text"]) for section in generated_sections
    )
    notes = assembled.get("fact_usage_notes")
    if isinstance(notes, list):
        assembled["fact_usage_notes"] = [
            *notes,
            "Article body generated section-by-section from grounded article plan.",
        ]
    return assembled


def _retry_short_section_if_needed(
    draft_client: StructuredJsonClient,
    prompt: str,
    section_payload: dict[str, object],
    generated: dict[str, object],
    planned_section: dict[object, object],
) -> dict[str, object]:
    target_words = _coerce_int(planned_section.get("target_words"))
    if target_words <= 0:
        return generated
    minimum_words = max(40, int(target_words * 0.75))
    section_text = str(generated.get("section_text") or "")
    current_words = approximate_tamil_word_count(section_text)
    if current_words >= minimum_words:
        return generated

    retry_payload = dict(section_payload)
    retry_payload["section_retry_request"] = {
        "reason": "The generated section was below 75% of target_words.",
        "current_section_word_count": current_words,
        "minimum_section_words": minimum_words,
        "target_words": target_words,
        "current_section_text": section_text,
        "instruction": (
            "Rewrite this section as fuller grounded Tamil news prose using only "
            "the planned facts, quotes, attributions, numbers, and context. Do "
            "not add filler or unsupported claims."
        ),
    }
    retried = draft_client.generate_structured_json(
        prompt,
        json.dumps(retry_payload, ensure_ascii=False),
    )
    retried_words = approximate_tamil_word_count(
        str(retried.get("section_text") or "")
    )
    if retried_words > current_words:
        return retried
    return generated


def _coerce_int(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return 0
    return 0


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _sum_token_usage(usages: list[dict[str, object]]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _build_warnings(
    profile_record: AuthorStyleProfileRecord,
    brief_record: GroundedBriefRecord,
    target_language: str,
    desired_word_count: int,
) -> list[str]:
    warnings: list[str] = []
    profile_warnings = StyleScribeRepository.decode_json_list(
        profile_record.warnings_json
    )
    brief_warnings = StyleScribeRepository.decode_json_list(brief_record.warnings_json)
    warnings.extend(f"Style profile warning: {warning}" for warning in profile_warnings)
    warnings.extend(f"Grounded brief warning: {warning}" for warning in brief_warnings)
    if target_language != "ta":
        warnings.append(
            "MVP article draft generation is Tamil-focused; target_language is not ta."
        )
    brief = StyleScribeRepository.decode_json_object(brief_record.brief_json)
    confirmed_facts = brief.get("confirmed_facts")
    fact_count = len(confirmed_facts) if isinstance(confirmed_facts, list) else 0
    if desired_word_count >= 800 and fact_count < 5:
        warnings.append(
            "Grounded brief may be too thin to support a long article without padding."
        )
    return warnings


def _draft_response(
    record: ArticleDraftRecord,
    draft: dict[str, object],
    warnings: list[str],
) -> ArticleDraftResponse:
    return ArticleDraftResponse(
        draft_id=record.draft_id,
        author_id=record.author_id,
        profile_id=record.profile_id,
        brief_id=record.brief_id,
        target_language=record.target_language,
        model_provider=record.model_provider,
        model_name=record.model_name,
        status=record.status,
        article_type=record.article_type,
        desired_word_count=record.desired_word_count,
        tone_override=record.tone_override,
        include_seo=record.include_seo,
        draft=draft,
        warnings=warnings,
        created_at=record.created_at,
    )
