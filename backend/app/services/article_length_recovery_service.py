"""Grounded length recovery for short revised Tamil articles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticleRevisionRecord,
    AuthorStyleProfileRecord,
    DraftEvaluationRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_revision_service import cleanup_revised_article_tamil
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count

PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "article_length_recovery_prompt.txt"
)
SUFFICIENT_EXPANSION_ITEM_COUNT = 5


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class ArticleLengthRecoveryError(RuntimeError):
    """Raised when article length recovery cannot be completed."""


@dataclass(frozen=True)
class LengthRecoveryDecision:
    length_recovery_required: bool
    short_output_invalid: bool
    expansion_items_available: int
    target_min_word_count: int | None
    target_max_word_count: int | None
    current_word_count: int
    brief_has_sufficient_expansion_material: bool


@dataclass(frozen=True)
class ArticleLengthRecoveryResponse:
    revision_id: str
    expanded_draft: dict[str, object]
    expansion_summary: str
    expansion_items_used: list[object]
    warnings: list[str]
    final_article_word_count: int
    model_provider: str
    model_name: str
    token_usage: dict[str, object]


def assess_length_recovery_need(
    article: dict[str, object],
    brief: dict[str, object],
    desired_word_count: int | None,
) -> LengthRecoveryDecision:
    """Decide whether short output is invalid and needs grounded expansion."""

    current_word_count = approximate_tamil_word_count(
        str(article.get("article_body") or "")
    )
    if not desired_word_count:
        return LengthRecoveryDecision(
            length_recovery_required=False,
            short_output_invalid=False,
            expansion_items_available=count_expansion_items(brief),
            target_min_word_count=None,
            target_max_word_count=None,
            current_word_count=current_word_count,
            brief_has_sufficient_expansion_material=False,
        )
    target_min = int(desired_word_count * 0.75)
    target_max = int(desired_word_count * 1.15)
    expansion_items_available = count_expansion_items(brief)
    sufficient = expansion_items_available >= SUFFICIENT_EXPANSION_ITEM_COUNT
    short_invalid = current_word_count < target_min and sufficient
    return LengthRecoveryDecision(
        length_recovery_required=short_invalid,
        short_output_invalid=short_invalid,
        expansion_items_available=expansion_items_available,
        target_min_word_count=target_min,
        target_max_word_count=target_max,
        current_word_count=current_word_count,
        brief_has_sufficient_expansion_material=sufficient,
    )


def expand_article_to_target_length(
    current_revision: ArticleRevisionRecord,
    brief: GroundedBriefRecord,
    evaluation: DraftEvaluationRecord,
    profile: AuthorStyleProfileRecord,
    desired_word_count: int,
    article_type: str,
    target_language: str,
    tone_override: str | None,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> ArticleLengthRecoveryResponse:
    """Expand a short revised article using only grounded material."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    recovery_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article length recovery."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_article_length_recovery_input(
        current_revision=current_revision,
        brief=brief,
        evaluation=evaluation,
        profile=profile,
        desired_word_count=desired_word_count,
        article_type=article_type,
        target_language=target_language,
        tone_override=tone_override,
    )
    try:
        expanded = recovery_client.generate_structured_json(prompt, user_payload)
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise ArticleLengthRecoveryError(
            "OpenAI article length recovery failed."
        ) from exc

    expanded = cleanup_revised_article_tamil(expanded)
    expanded_draft: dict[str, object] = {
        "headline": str(expanded.get("headline") or current_revision.revised_headline),
        "subheadline": str(
            expanded.get("subheadline") or current_revision.revised_subheadline
        ),
        "article_body": str(expanded.get("article_body") or ""),
        "seo_title": str(
            expanded.get("seo_title") or current_revision.revised_seo_title
        ),
        "meta_description": str(
            expanded.get("meta_description")
            or current_revision.revised_meta_description
        ),
        "suggested_tags": _list_value(expanded.get("suggested_tags")),
    }
    created_at = datetime.now(UTC).isoformat()
    record = ArticleRevisionRecord(
        revision_id=str(uuid4()),
        draft_id=current_revision.draft_id,
        evaluation_id=current_revision.evaluation_id,
        author_id=current_revision.author_id,
        revised_headline=str(expanded_draft["headline"]),
        revised_subheadline=str(expanded_draft["subheadline"]),
        revised_article_body=str(expanded_draft["article_body"]),
        revised_seo_title=str(expanded_draft["seo_title"]),
        revised_meta_description=str(expanded_draft["meta_description"]),
        revised_tags_json=StyleScribeRepository.encode_json(
            _list_value(expanded_draft.get("suggested_tags"))
        ),
        revision_summary=str(expanded.get("expansion_summary") or ""),
        removed_or_softened_claims_json=StyleScribeRepository.encode_json([]),
        model_provider=recovery_client.provider,
        model_name=recovery_client.model_name,
        token_usage_json=StyleScribeRepository.encode_json(
            _dict_value(expanded.get("token_usage"))
        ),
        created_at=created_at,
    )
    repo.save_article_revision(record)
    return ArticleLengthRecoveryResponse(
        revision_id=record.revision_id,
        expanded_draft=expanded_draft,
        expansion_summary=record.revision_summary,
        expansion_items_used=_list_value(expanded.get("expansion_items_used")),
        warnings=[str(warning) for warning in _list_value(expanded.get("warnings"))],
        final_article_word_count=approximate_tamil_word_count(
            record.revised_article_body
        ),
        model_provider=record.model_provider,
        model_name=record.model_name,
        token_usage=_dict_value(expanded.get("token_usage")),
    )


def build_article_length_recovery_input(
    current_revision: ArticleRevisionRecord,
    brief: GroundedBriefRecord,
    evaluation: DraftEvaluationRecord,
    profile: AuthorStyleProfileRecord,
    desired_word_count: int,
    article_type: str,
    target_language: str,
    tone_override: str | None,
) -> str:
    """Build a structured expansion payload."""

    brief_json = StyleScribeRepository.decode_json_object(brief.brief_json)
    evaluation_json = StyleScribeRepository.decode_json_object(
        evaluation.evaluation_json
    )
    current_draft = {
        "headline": current_revision.revised_headline,
        "subheadline": current_revision.revised_subheadline,
        "article_body": current_revision.revised_article_body,
        "seo_title": current_revision.revised_seo_title,
        "meta_description": current_revision.revised_meta_description,
        "suggested_tags": json.loads(current_revision.revised_tags_json),
    }
    payload = {
        "task": "Expand short revised Tamil article using only grounded material.",
        "target_language": target_language,
        "article_type": article_type,
        "tone_override": tone_override,
        "desired_word_count": desired_word_count,
        "target_word_count_range": {
            "minimum_75_percent": int(desired_word_count * 0.75),
            "target": desired_word_count,
            "maximum_115_percent": int(desired_word_count * 1.15),
        },
        "current_revised_article": current_draft,
        "current_article_body_word_count": approximate_tamil_word_count(
            current_revision.revised_article_body
        ),
        "grounded_brief_for_facts_only": {
            "brief_id": brief.brief_id,
            "brief": brief_json,
            "source_excerpt": brief.source_text_excerpt,
            "expansion_material": expansion_material(brief_json),
            "expansion_items_available": count_expansion_items(brief_json),
        },
        "grounding_evaluation_feedback": {
            "evaluation_id": evaluation.evaluation_id,
            "unsupported_claims": evaluation_json.get("unsupported_claims", []),
            "overclaim_phrases": evaluation_json.get("overclaim_phrases", []),
            "rewrite_guidance": evaluation_json.get("rewrite_guidance", []),
        },
        "style_profile_for_voice_only": {
            "profile_id": profile.profile_id,
            "language": profile.language,
            "profile": StyleScribeRepository.decode_json_object(profile.profile_json),
        },
        "structured_article_guidance": [
            "opening paragraph with core news",
            "court decision or official action",
            "affected groups",
            "numbers/statistics",
            "attributed quote or source-supported statement",
            "legal/policy context",
            "grounded closing paragraph",
        ],
        "expansion_rules": (
            "Expand using only grounded material. Do not add filler, new facts, "
            "unsupported impact claims, unsupported future benefits, or repeated "
            "sentences. If a section is unsupported, skip it."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def expansion_material(brief: dict[str, object]) -> dict[str, list[object]]:
    return {
        "confirmed_facts": _list_value(brief.get("confirmed_facts")),
        "numbers_and_statistics": _list_value(brief.get("numbers_and_statistics")),
        "affected_groups": _list_value(brief.get("affected_groups")),
        "dates_or_timeline": _list_value(brief.get("dates_or_timeline")),
        "quotes": _list_value(brief.get("quotes")),
        "policy_or_legal_context": _list_value(brief.get("policy_or_legal_context")),
        "background_from_source": _list_value(brief.get("background_from_source")),
    }


def count_expansion_items(brief: dict[str, object]) -> int:
    material = expansion_material(brief)
    return sum(len(items) for items in material.values())


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}
