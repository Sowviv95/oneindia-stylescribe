"""Generate and persist controlled Tamil article drafts."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.config import get_settings
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
NEWSROOM_V1_PROMPT_DIR = Path(__file__).parents[1] / "prompts" / "newsroom_v1"
NEWSROOM_V1_PROMPT_PATH = NEWSROOM_V1_PROMPT_DIR / "runtime_prompt.txt"
NEWSROOM_V1_METADATA_PATH = NEWSROOM_V1_PROMPT_DIR / "prompt_metadata.json"
NEWSROOM_PROMPT_VERSION_PATHS = {
    "oneindia_newsroom_v1.0": (
        NEWSROOM_V1_PROMPT_PATH,
        NEWSROOM_V1_METADATA_PATH,
    ),
    "oneindia_newsroom_v1.1_length_calibrated": (
        Path(__file__).parents[1]
        / "prompts"
        / "newsroom_v1_1_length_calibrated"
        / "runtime_prompt.txt",
        Path(__file__).parents[1]
        / "prompts"
        / "newsroom_v1_1_length_calibrated"
        / "prompt_metadata.json",
    ),
}
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
        prompt_cache_key: str | None = None,
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
        base_draft = _generate_structured_json(
            draft_client,
            prompt,
            user_payload,
            prompt_cache_key=_prompt_cache_key("article", user_payload),
        )
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


def generate_newsroom_article_draft(
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
    input_identifier: str | None = None,
    git_commit: str | None = None,
    newsroom_prompt_version: str = "oneindia_newsroom_v1.0",
) -> ArticleDraftResponse:
    """Generate a generic newsroom draft without loading an author profile."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    brief_record = repo.fetch_grounded_brief(brief_id)
    if brief_record is None:
        message = f"No grounded brief found for brief_id: {brief_id}"
        raise ArticleGenerationError(message)
    plan_record = repo.fetch_article_plan(plan_id) if plan_id else None
    if plan_id and plan_record is None:
        raise ArticleGenerationError(f"No article plan found for plan_id: {plan_id}")

    resolved_article_type = article_type or "news"
    resolved_word_count = desired_word_count or 600
    warnings = _build_newsroom_warnings(
        brief_record=brief_record,
        target_language=target_language,
        desired_word_count=resolved_word_count,
    )
    draft_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article draft generation."
    )
    prompt_path, _metadata_path = _newsroom_prompt_paths(newsroom_prompt_version)
    prompt = prompt_path.read_text(encoding="utf-8")
    metadata = _newsroom_prompt_metadata(newsroom_prompt_version)
    user_payload = build_newsroom_generation_input(
        brief_record=brief_record,
        author_instruction=author_instruction,
        target_language=target_language,
        article_type=resolved_article_type,
        desired_word_count=resolved_word_count,
        tone_override=tone_override,
        include_seo=include_seo,
        plan_record=plan_record,
        prompt_metadata=metadata,
    )

    try:
        draft = _generate_structured_json(
            draft_client,
            prompt,
            user_payload,
            prompt_cache_key=_prompt_cache_key("newsroom-v1", user_payload),
        )
        draft = _normalize_plan_based_draft(draft, plan_record)
        draft = _annotate_newsroom_generation_metadata(
            draft,
            prompt_metadata=metadata,
            provider=draft_client.provider,
            model=draft_client.model_name,
            input_identifier=input_identifier,
            git_commit=git_commit,
        )
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise ArticleGenerationError(
            "Newsroom article draft generation failed."
        ) from exc

    created_at = datetime.now(UTC).isoformat()
    record = ArticleDraftRecord(
        draft_id=str(uuid4()),
        author_id=author_id,
        profile_id=str(metadata["newsroom_profile_version"]),
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


def build_newsroom_generation_input(
    brief_record: GroundedBriefRecord,
    author_instruction: str | None,
    target_language: str,
    article_type: str = "news",
    desired_word_count: int = 600,
    tone_override: str | None = None,
    include_seo: bool = True,
    plan_record: ArticlePlanRecord | None = None,
    prompt_metadata: dict[str, object] | None = None,
) -> str:
    """Build separated fact/editorial input for generic newsroom generation."""

    metadata = prompt_metadata or _newsroom_prompt_metadata()
    payload: dict[str, object] = {
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
        "prompt_version": metadata["prompt_version"],
        "newsroom_profile_version": metadata["newsroom_profile_version"],
        "factual_source_brief": {
            "brief_id": brief_record.brief_id,
            "source_language": brief_record.source_language,
            "target_language": brief_record.target_language,
            "brief": StyleScribeRepository.decode_json_object(
                brief_record.brief_json
            ),
        },
        "generic_newsroom_editorial_rules": {
            "voice": (
                "Natural Oneindia Tamil newsroom voice; no individual author imitation."
            ),
            "opening": (
                "Use a fact-dense opening lede when supported. Do not treat "
                "source headline candidates as confirmed editorial headlines."
            ),
            "sequencing": (
                "Order the main fact first, then supporting details, attribution, "
                "context/background and reader relevance when supported."
            ),
            "paragraphs": "Use compact paragraphs; avoid one-paragraph summaries.",
            "attribution": (
                "Use reported speech and direct quotations only when present in "
                "the factual source brief."
            ),
            "caution": (
                "Use cautious phrasing only for source-supported uncertainty, "
                "expectation or prediction."
            ),
            "phrase_bank_policy": (
                "Frequent newsroom constructions are optional, not required. "
                "Do not force phrase-bank wording into the article."
            ),
        },
        "length_control": {
            "requested_target_words": desired_word_count,
            "target_range": {
                "minimum": int(desired_word_count * 0.75),
                "target": desired_word_count,
                "maximum": int(desired_word_count * 1.15),
            },
            "rule": (
                "Use the target range when source-supported facts, attribution, "
                "timeline, affected groups, context or plan sections provide "
                "enough material. Never add filler or unsupported context."
            ),
        },
        "optional_topic_guidance": {
            "rule": (
                "Use topic guidance only when source facts clearly support the "
                "topic; otherwise write as general news."
            )
        },
        "output_schema": {
            "required_fields": [
                "headline",
                "subheadline",
                "article_body",
                "article_sections",
                "seo_title",
                "meta_description",
                "suggested_tags",
                "fact_usage_notes",
                "style_usage_notes",
            ]
        },
        "prohibited_behaviours": [
            "Do not invent facts, dates, numbers, quotes, names or context.",
            "Do not imitate any individual corpus author.",
            "Do not add retrieval, external references or unsupported background.",
            "Do not force place-led openings, quotations or stock phrases.",
            "Do not write literal English-to-Tamil translation prose.",
            "Do not use fact-bearing phrase-bank terms unless present in the brief.",
        ],
        "separation_rule": (
            "factual_source_brief is the only factual layer. Editorial rules "
            "control structure and tone only."
        ),
    }
    if plan_record is not None:
        payload["grounded_article_plan"] = {
            "plan_id": plan_record.plan_id,
            "target_min_word_count": plan_record.target_min_word_count,
            "target_max_word_count": plan_record.target_max_word_count,
            "planned_sections": StyleScribeRepository.decode_json_list(
                plan_record.planned_sections_json
            ),
            "expansion_items_used": StyleScribeRepository.decode_json_list(
                plan_record.expansion_items_used_json
            ),
            "claims_to_avoid": StyleScribeRepository.decode_json_list(
                plan_record.claims_to_avoid_json
            ),
            "plan_summary": plan_record.plan_summary,
        }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


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
    full_stable_payload = _dynamic_section_stable_payload(
        payload,
        targets,
        use_compact_context=False,
    )
    stable_payload = _dynamic_section_stable_payload(
        payload,
        targets,
        use_compact_context=True,
    )
    context_metrics = _generation_context_metrics(full_stable_payload, stable_payload)
    stable_cache_key = _prompt_cache_key(
        "section",
        json.dumps(stable_payload, ensure_ascii=False),
    )
    settings = get_settings()
    group_size = min(max(settings.generation_section_group_size, 1), 3)
    section_groups = _section_groups(planned_sections, group_size)
    max_workers = min(
        max(settings.max_concurrent_section_calls, 1),
        len(section_groups),
    )
    indexed_results: dict[int, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _generate_dynamic_section_group,
                draft_client,
                prompt,
                stable_payload,
                group,
                group_index,
                targets,
                stable_cache_key,
                group_size,
            ): group_index
            for group_index, group in enumerate(section_groups, start=1)
        }
        for future in as_completed(futures):
            group_index = futures[future]
            indexed_results[group_index] = future.result()

    generated_sections: list[dict[str, object]] = []
    section_trace: list[dict[str, object]] = []
    group_call_count = 0
    fallback_count = 0
    for group_index in range(1, len(section_groups) + 1):
        group_result = indexed_results[group_index]
        generated_sections.extend(_list_of_dicts(group_result.get("sections")))
        section_trace.extend(_list_of_dicts(group_result.get("traces")))
        group_call_count += _coerce_int(group_result.get("group_call_count"))
        fallback_count += _coerce_int(group_result.get("fallback_count"))

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
    draft["max_concurrent_section_calls"] = max_workers
    draft["generation_section_group_size"] = group_size
    draft["generation_group_call_count"] = group_call_count
    draft["generation_single_section_fallback_count"] = fallback_count
    draft.update(context_metrics)
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


def _generate_dynamic_section_group(
    draft_client: StructuredJsonClient,
    prompt: str,
    stable_payload: dict[str, object],
    group: list[tuple[int, dict[str, object]]],
    group_index: int,
    targets: DynamicLengthTargets,
    prompt_cache_key: str,
    configured_group_size: int,
) -> dict[str, object]:
    if configured_group_size <= 1 or len(group) == 1:
        generated = [
            _generate_one_dynamic_section(
                draft_client,
                prompt,
                stable_payload,
                section,
                section_index,
                targets,
                prompt_cache_key,
            )
            for section_index, section in group
        ]
        return {
            "sections": [item["section"] for item in generated],
            "traces": [item["trace"] for item in generated],
            "group_call_count": 0,
            "fallback_count": 0,
        }

    group_payload = _dynamic_section_group_payload(
        stable_payload=stable_payload,
        group=group,
        group_index=group_index,
        targets=targets,
    )
    group_response = _generate_structured_json(
        draft_client,
        prompt,
        json.dumps(group_payload, ensure_ascii=False),
        prompt_cache_key=prompt_cache_key,
    )
    normalized = _normalize_group_response(group_response, group, targets)
    if normalized is not None:
        group_call_count = 1
        if _group_needs_retry(normalized["traces"], targets):
            retry_payload = dict(group_payload)
            retry_payload["section_retry_request"] = {
                "reason": (
                    "At least one generated section in this group was below "
                    "the minimum section length."
                ),
                "minimum_section_words": targets.section_min_word_count,
                "current_sections": [
                    {
                        "section_id": section.get("section_id"),
                        "selected_word_count": trace.get("selected_word_count"),
                        "section_text": section.get("section_text"),
                    }
                    for section, trace in zip(
                        normalized["sections"],
                        normalized["traces"],
                        strict=True,
                    )
                ],
                "instruction": (
                    "Rewrite the same section_group as fuller grounded Tamil "
                    "prose. Keep one article_sections item per requested section "
                    "in the same order. Do not merge sections."
                ),
            }
            retry_response = _generate_structured_json(
                draft_client,
                prompt,
                json.dumps(retry_payload, ensure_ascii=False),
                prompt_cache_key=prompt_cache_key,
            )
            retry_normalized = _normalize_group_response(
                retry_response,
                group,
                targets,
            )
            group_call_count += 1
            if retry_normalized is not None:
                normalized = _select_group_retry(normalized, retry_normalized)
        return {
            "sections": normalized["sections"],
            "traces": normalized["traces"],
            "group_call_count": group_call_count,
            "fallback_count": 0,
        }

    fallback_generated = [
        _generate_one_dynamic_section(
            draft_client,
            prompt,
            stable_payload,
            section,
            section_index,
            targets,
            prompt_cache_key,
        )
        for section_index, section in group
    ]
    return {
        "sections": [item["section"] for item in fallback_generated],
        "traces": [
            {
                **item["trace"],
                "group_fallback_used": True,
                "group_index": group_index,
            }
            for item in fallback_generated
        ],
        "group_call_count": 1,
        "fallback_count": len(group),
    }


def _generate_one_dynamic_section(
    draft_client: StructuredJsonClient,
    prompt: str,
    stable_payload: dict[str, object],
    section: dict[str, object],
    section_index: int,
    targets: DynamicLengthTargets,
    prompt_cache_key: str,
) -> dict[str, dict[str, object]]:
    section_payload = _dynamic_section_payload(
        stable_payload=stable_payload,
        section=section,
        section_index=section_index,
        targets=targets,
    )
    first_pass = _generate_structured_json(
        draft_client,
        prompt,
        json.dumps(section_payload, ensure_ascii=False),
        prompt_cache_key=prompt_cache_key,
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
        retried = _generate_structured_json(
            draft_client,
            prompt,
            json.dumps(retry_payload, ensure_ascii=False),
            prompt_cache_key=prompt_cache_key,
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


def _dynamic_section_group_payload(
    stable_payload: dict[str, object],
    group: list[tuple[int, dict[str, object]]],
    group_index: int,
    targets: DynamicLengthTargets,
) -> dict[str, object]:
    return {
        **stable_payload,
        "section_group": [
            {
                "section_index": section_index,
                "section_heading": str(
                    section.get("heading") or section.get("section_name") or ""
                ),
                "section_purpose": str(
                    section.get("purpose") or section.get("section_purpose") or ""
                ),
                "planned_section": section,
            }
            for section_index, section in group
        ],
        "section_group_index": group_index,
        "section_instruction": (
            "Generate the requested adjacent section_group. Return strict JSON "
            "with article_sections containing one item per requested section, in "
            "the same order. Each section_text should be publication-ready Tamil "
            f"near {targets.section_target_word_count} words, minimum "
            f"{targets.section_min_word_count}, maximum "
            f"{targets.section_max_word_count}. Do not merge sections and do not "
            "write the full article."
        ),
    }


def _normalize_group_response(
    response: dict[str, object],
    group: list[tuple[int, dict[str, object]]],
    targets: DynamicLengthTargets,
) -> dict[str, list[dict[str, object]]] | None:
    raw_sections = response.get("article_sections")
    if not isinstance(raw_sections, list) or len(raw_sections) != len(group):
        return None

    normalized_sections: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []
    for offset, ((section_index, planned_section), raw_section) in enumerate(zip(
        group,
        raw_sections,
        strict=True,
    )):
        if not isinstance(raw_section, dict):
            return None
        text = str(raw_section.get("section_text") or "").strip()
        if not text:
            return None
        word_count = approximate_tamil_word_count(text)
        section_id = str(
            planned_section.get("section_id")
            or planned_section.get("section_name")
            or section_index
        )
        heading = str(
            planned_section.get("heading") or planned_section.get("section_name") or ""
        )
        normalized_sections.append(
            {
                "section_id": section_id,
                "heading": heading,
                "target_word_count": targets.section_target_word_count,
                "min_word_count": targets.section_min_word_count,
                "max_word_count": targets.section_max_word_count,
                "section_text": text,
                "grounded_facts_used": _list_value(
                    raw_section.get("grounded_facts_used")
                ),
            }
        )
        traces.append(
            {
                "section_id": section_id,
                "heading": heading,
                "target_word_count": targets.section_target_word_count,
                "min_word_count": targets.section_min_word_count,
                "max_word_count": targets.section_max_word_count,
                "first_pass_word_count": word_count,
                "retry_attempted": False,
                "retry_word_count": None,
                "selected_word_count": word_count,
                "selected_reason": "group_first_pass_selected",
                "group_generation_used": True,
                "first_pass_token_usage": _dict_value(response.get("token_usage"))
                if offset == 0
                else {},
                "retry_token_usage": {},
            }
        )
    return {"sections": normalized_sections, "traces": traces}


def _group_needs_retry(
    traces: list[dict[str, object]],
    targets: DynamicLengthTargets,
) -> bool:
    return any(
        _coerce_int(trace.get("selected_word_count")) < targets.section_min_word_count
        for trace in traces
    )


def _select_group_retry(
    original: dict[str, list[dict[str, object]]],
    retry: dict[str, list[dict[str, object]]],
) -> dict[str, list[dict[str, object]]]:
    selected_sections: list[dict[str, object]] = []
    selected_traces: list[dict[str, object]] = []
    for index, (original_section, retry_section, original_trace, retry_trace) in (
        enumerate(
            zip(
                original["sections"],
                retry["sections"],
                original["traces"],
                retry["traces"],
                strict=True,
            )
        )
    ):
        original_words = _coerce_int(original_trace.get("selected_word_count"))
        retry_words = _coerce_int(retry_trace.get("selected_word_count"))
        trace = dict(original_trace)
        trace["retry_attempted"] = True
        trace["retry_word_count"] = retry_words
        trace["retry_token_usage"] = (
            _dict_value(retry_trace.get("first_pass_token_usage"))
            if index == 0
            else {}
        )
        if retry_words >= original_words:
            selected_section = retry_section
            trace["selected_word_count"] = retry_words
            trace["selected_reason"] = "group_retry_selected_longer_or_equal"
        else:
            selected_section = original_section
            trace["selected_reason"] = "group_first_pass_selected_retry_shorter"
        selected_sections.append(selected_section)
        selected_traces.append(trace)
    return {"sections": selected_sections, "traces": selected_traces}


def _dynamic_section_payload(
    stable_payload: dict[str, object],
    section: dict[str, object],
    section_index: int,
    targets: DynamicLengthTargets,
) -> dict[str, object]:
    heading = str(section.get("heading") or section.get("section_name") or "")
    purpose = str(section.get("purpose") or section.get("section_purpose") or "")
    return {
        **stable_payload,
        "section_heading": heading,
        "section_purpose": purpose,
        "planned_section": section,
        "section_index": section_index,
        "section_instruction": (
            "Generate only this section body. Output strict JSON. The section "
            f"target is {targets.section_target_word_count} Tamil words, minimum "
            f"{targets.section_min_word_count}, maximum "
            f"{targets.section_max_word_count}. Use only grounded facts from the "
            "brief. Neutral connective context is allowed only if it does not "
            "introduce new factual claims. Do not summarize too aggressively."
        ),
    }


def _dynamic_section_stable_payload(
    base_payload: dict[str, object],
    targets: DynamicLengthTargets,
    *,
    use_compact_context: bool,
) -> dict[str, object]:
    grounded_article_plan = base_payload.get("grounded_article_plan")
    plan_payload = (
        grounded_article_plan if isinstance(grounded_article_plan, dict) else {}
    )
    if use_compact_context:
        context_key = "generation_context_pack"
        context_value: object = _generation_context_pack(base_payload, plan_payload)
    else:
        context_key = "full_generation_context"
        context_value = {
            "style_profile_for_voice_only": base_payload.get(
                "style_profile_for_voice_only"
            ),
            "grounded_brief_for_facts_only": base_payload.get(
                "grounded_brief_for_facts_only"
            ),
            "grounded_article_plan": plan_payload,
        }
    return {
        "target_language": base_payload.get("target_language"),
        "article_type": base_payload.get("article_type"),
        "tone_override": base_payload.get("tone_override"),
        "author_instruction": base_payload.get("author_instruction"),
        context_key: context_value,
        "grounded_article_plan_summary": plan_payload.get("plan_summary"),
        "claims_to_avoid": plan_payload.get("claims_to_avoid"),
        "dynamic_length_targets": _targets_payload(targets),
        "section_generation_rule": (
            "Write each requested section from the stable style profile, grounded "
            "context pack, grounded facts, and article plan above. Section-specific "
            "fields follow after this stable context and should control only the "
            "current section or section group."
        ),
    }


def _generation_context_pack(
    base_payload: dict[str, object],
    plan_payload: dict[str, object],
) -> dict[str, object]:
    style_context = _style_context_summary(
        _dict_value(base_payload.get("style_profile_for_voice_only"))
    )
    brief_wrapper = _dict_value(base_payload.get("grounded_brief_for_facts_only"))
    brief = _dict_value(brief_wrapper.get("brief"))
    planned_sections = _list_value(plan_payload.get("planned_sections"))
    return {
        "style_constraints_summary": style_context,
        "grounded_facts_pack": {
            "topic": brief.get("topic"),
            "one_line_summary": brief.get("one_line_summary"),
            "confirmed_facts": _limited_list(brief.get("confirmed_facts"), 12),
            "key_entities": _limited_list(brief.get("key_entities"), 10),
            "places": _limited_list(brief.get("places"), 8),
            "dates_or_timeline": _limited_list(brief.get("dates_or_timeline"), 8),
            "numbers_and_statistics": _limited_list(
                brief.get("numbers_and_statistics"),
                10,
            ),
            "quotes": _limited_list(brief.get("quotes"), 8),
            "background_from_source": _limited_list(
                brief.get("background_from_source"),
                8,
            ),
            "policy_or_legal_context": _limited_list(
                brief.get("policy_or_legal_context"),
                8,
            ),
            "affected_groups": _limited_list(brief.get("affected_groups"), 8),
            "claims_to_avoid": _limited_list(
                brief.get("claims_to_avoid") or plan_payload.get("claims_to_avoid"),
                12,
            ),
            "suggested_tamil_angle": brief.get("suggested_tamil_angle"),
            "editorial_risk_notes": _limited_list(
                brief.get("editorial_risk_notes"),
                8,
            ),
        },
        "article_plan_context": {
            "plan_summary": plan_payload.get("plan_summary"),
            "claims_to_avoid": _limited_list(plan_payload.get("claims_to_avoid"), 12),
            "sections": [
                _compact_section_plan(section, index)
                for index, section in enumerate(planned_sections, start=1)
                if isinstance(section, dict)
            ],
        },
    }


def _style_context_summary(style_wrapper: dict[str, object]) -> dict[str, object]:
    profile = _dict_value(style_wrapper.get("profile"))
    return {
        "language": style_wrapper.get("language"),
        "overall_tone": profile.get("overall_tone"),
        "intro_style": profile.get("intro_style"),
        "paragraph_style": profile.get("paragraph_style"),
        "sentence_style": profile.get("sentence_style"),
        "vocabulary_style": profile.get("vocabulary_style"),
        "narrative_flow": profile.get("narrative_flow"),
        "tamil_register": profile.get("tamil_register"),
        "dos": _limited_list(profile.get("dos"), 8),
        "donts": _limited_list(profile.get("donts"), 8),
        "generation_guidance": profile.get("generation_guidance"),
    }


def _compact_section_plan(section: dict[str, object], index: int) -> dict[str, object]:
    return {
        "section_index": index,
        "section_name": section.get("section_name"),
        "purpose": section.get("purpose") or section.get("section_purpose"),
        "target_words": section.get("target_words"),
        "grounded_facts_to_use": _limited_list(
            section.get("grounded_facts_to_use"),
            6,
        ),
        "quotes_or_attributions_to_use": _limited_list(
            section.get("quotes_or_attributions_to_use"),
            4,
        ),
        "claims_to_avoid": _limited_list(section.get("claims_to_avoid"), 6),
        "must_not_add": _limited_list(section.get("must_not_add"), 6),
    }


def _generation_context_metrics(
    original_context: dict[str, object],
    compressed_context: dict[str, object],
) -> dict[str, object]:
    original_json = json.dumps(original_context, ensure_ascii=False)
    compressed_json = json.dumps(compressed_context, ensure_ascii=False)
    original_chars = len(original_json)
    compressed_chars = len(compressed_json)
    return {
        "generation_context_pack_chars": compressed_chars,
        "generation_context_pack_tokens": _approx_token_count(compressed_json),
        "original_generation_context_chars": original_chars,
        "compressed_generation_context_chars": compressed_chars,
        "generation_context_compression_ratio": round(
            compressed_chars / original_chars,
            4,
        )
        if original_chars
        else None,
    }


def _section_groups(
    planned_sections: list[dict[str, object]],
    group_size: int,
) -> list[list[tuple[int, dict[str, object]]]]:
    size = max(group_size, 1)
    indexed = list(enumerate(planned_sections, start=1))
    return [indexed[index : index + size] for index in range(0, len(indexed), size)]


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


def _generate_structured_json(
    client: StructuredJsonClient,
    system_prompt: str,
    user_payload: str,
    *,
    prompt_cache_key: str | None = None,
) -> dict[str, object]:
    try:
        return client.generate_structured_json(
            system_prompt,
            user_payload,
            prompt_cache_key=prompt_cache_key,
        )
    except TypeError:
        if prompt_cache_key is None:
            raise
        return client.generate_structured_json(system_prompt, user_payload)


def _prompt_cache_key(namespace: str, stable_payload: str) -> str:
    digest = sha256(stable_payload.encode("utf-8")).hexdigest()[:32]
    return f"stylescribe-{namespace}-{digest}"


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _sum_token_usage(usages: list[dict[str, object]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cached_prompt_tokens": 0,
    }
    for usage in usages:
        for key in totals:
            value = usage.get(key)
            if isinstance(value, int):
                totals[key] += value
    return totals


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _limited_list(value: object, limit: int) -> list[object]:
    return _list_value(value)[:limit]


def _approx_token_count(text: str) -> int:
    return max(1, round(len(text) / 4)) if text else 0


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


def _build_newsroom_warnings(
    brief_record: GroundedBriefRecord,
    target_language: str,
    desired_word_count: int,
) -> list[str]:
    warnings: list[str] = []
    brief_warnings = StyleScribeRepository.decode_json_list(brief_record.warnings_json)
    warnings.extend(f"Grounded brief warning: {warning}" for warning in brief_warnings)
    if target_language != "ta":
        warnings.append(
            "Newsroom v1 article draft generation is Tamil-focused; "
            "target_language is not ta."
        )
    brief = StyleScribeRepository.decode_json_object(brief_record.brief_json)
    confirmed_facts = brief.get("confirmed_facts")
    fact_count = len(confirmed_facts) if isinstance(confirmed_facts, list) else 0
    if desired_word_count >= 800 and fact_count < 5:
        warnings.append(
            "Grounded brief may be too thin to support a long article without padding."
        )
    return warnings


def _newsroom_prompt_metadata(
    newsroom_prompt_version: str = "oneindia_newsroom_v1.0",
) -> dict[str, object]:
    _prompt_path, metadata_path = _newsroom_prompt_paths(newsroom_prompt_version)
    return StyleScribeRepository.decode_json_object(
        metadata_path.read_text(encoding="utf-8")
    )


def _newsroom_prompt_paths(newsroom_prompt_version: str) -> tuple[Path, Path]:
    paths = NEWSROOM_PROMPT_VERSION_PATHS.get(newsroom_prompt_version)
    if paths is None:
        raise ArticleGenerationError(
            f"Unsupported newsroom prompt version: {newsroom_prompt_version}"
        )
    return paths


def _annotate_newsroom_generation_metadata(
    draft: dict[str, object],
    *,
    prompt_metadata: dict[str, object],
    provider: str,
    model: str,
    input_identifier: str | None,
    git_commit: str | None,
) -> dict[str, object]:
    annotated = dict(draft)
    annotated["generation_mode"] = "newsroom_v1"
    annotated["generation_prompt_version"] = prompt_metadata["prompt_version"]
    annotated["newsroom_profile_version"] = prompt_metadata[
        "newsroom_profile_version"
    ]
    annotated["generation_metadata"] = {
        "generation_mode": "newsroom_v1",
        "prompt_version": prompt_metadata["prompt_version"],
        "provider": provider,
        "model": model,
        "git_commit": git_commit,
        "newsroom_profile_version": prompt_metadata["newsroom_profile_version"],
        "input_identifier": input_identifier,
    }
    notes = annotated.get("style_usage_notes")
    if isinstance(notes, list):
        annotated["style_usage_notes"] = [
            *notes,
            "Used generic Oneindia Tamil newsroom v1 guidance; no author profile.",
        ]
    else:
        annotated["style_usage_notes"] = [
            "Used generic Oneindia Tamil newsroom v1 guidance; no author profile."
        ]
    return annotated


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
