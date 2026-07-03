"""Generate and persist grounded article plans."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticlePlanRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_length_recovery_service import (
    count_expansion_items,
    expansion_material,
)
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "article_plan_prompt.txt"


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class ArticlePlanError(RuntimeError):
    """Raised when article planning cannot be completed."""


@dataclass(frozen=True)
class ArticlePlanResponse:
    plan_id: str
    brief_id: str
    author_id: str
    article_type: str
    desired_word_count: int | None
    target_word_count: int | None
    target_min_word_count: int | None
    target_max_word_count: int | None
    planned_sections: list[object]
    expansion_items_used: list[object]
    claims_to_avoid: list[object]
    plan_summary: str
    model_provider: str
    model_name: str
    token_usage: dict[str, object]
    warnings: list[str]
    created_at: str


def generate_article_plan(
    brief_id: str,
    author_id: str,
    article_type: str,
    desired_word_count: int | None,
    target_language: str,
    tone_override: str | None = None,
    author_instruction: str | None = None,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> ArticlePlanResponse:
    """Generate and persist a grounded article plan."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    brief = repo.fetch_grounded_brief(brief_id)
    if brief is None:
        raise ArticlePlanError(f"No grounded brief found for brief_id: {brief_id}")

    plan_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article planning."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_article_plan_input(
        brief=brief,
        author_id=author_id,
        article_type=article_type,
        desired_word_count=desired_word_count,
        target_language=target_language,
        tone_override=tone_override,
        author_instruction=author_instruction,
    )
    try:
        plan = plan_client.generate_structured_json(prompt, user_payload)
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise ArticlePlanError("OpenAI article plan generation failed.") from exc

    target_min = _optional_int(plan.get("target_min_word_count"))
    target_max = _optional_int(plan.get("target_max_word_count"))
    target = _optional_int(plan.get("target_word_count")) or desired_word_count
    if desired_word_count:
        target_min = target_min or int(desired_word_count * 0.75)
        target_max = target_max or int(desired_word_count * 1.15)
        target = target or desired_word_count
    created_at = datetime.now(UTC).isoformat()
    record = ArticlePlanRecord(
        plan_id=str(uuid4()),
        brief_id=brief.brief_id,
        author_id=author_id,
        article_type=article_type,
        desired_word_count=desired_word_count,
        target_min_word_count=target_min,
        target_max_word_count=target_max,
        planned_sections_json=StyleScribeRepository.encode_json(
            _list_value(plan.get("planned_sections"))
        ),
        expansion_items_used_json=StyleScribeRepository.encode_json(
            _list_value(plan.get("expansion_items_used"))
        ),
        claims_to_avoid_json=StyleScribeRepository.encode_json(
            _list_value(plan.get("claims_to_avoid"))
        ),
        plan_summary=str(plan.get("plan_summary") or ""),
        model_provider=plan_client.provider,
        model_name=plan_client.model_name,
        token_usage_json=StyleScribeRepository.encode_json(
            _dict_value(plan.get("token_usage"))
        ),
        created_at=created_at,
    )
    repo.save_article_plan(record)
    return _plan_response(
        record,
        warnings=[str(warning) for warning in _list_value(plan.get("warnings"))],
    )


def build_article_plan_input(
    brief: GroundedBriefRecord,
    author_id: str,
    article_type: str,
    desired_word_count: int | None,
    target_language: str,
    tone_override: str | None,
    author_instruction: str | None,
) -> str:
    """Build grounded plan input."""

    brief_json = StyleScribeRepository.decode_json_object(brief.brief_json)
    target_range = (
        {
            "minimum_75_percent": int(desired_word_count * 0.75),
            "target": desired_word_count,
            "maximum_115_percent": int(desired_word_count * 1.15),
        }
        if desired_word_count
        else None
    )
    payload = {
        "task": "Create grounded Tamil article plan before generation.",
        "author_id": author_id,
        "article_type": article_type,
        "target_language": target_language,
        "tone_override": tone_override,
        "author_instruction": author_instruction,
        "desired_word_count": desired_word_count,
        "target_word_count_range": target_range,
        "grounded_brief_for_facts_only": {
            "brief_id": brief.brief_id,
            "brief": brief_json,
            "source_excerpt": brief.source_text_excerpt,
            "expansion_material": expansion_material(brief_json),
            "expansion_items_available": count_expansion_items(brief_json),
            "claims_to_avoid": brief_json.get("claims_to_avoid", []),
        },
        "planning_rule": (
            "Use only grounded facts. Assign section target_words. For a 600-word "
            "article, prefer 6-8 sections if material supports it. Forbid filler "
            "and unsupported benefit or future-impact claims."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _plan_response(
    record: ArticlePlanRecord,
    warnings: list[str],
) -> ArticlePlanResponse:
    return ArticlePlanResponse(
        plan_id=record.plan_id,
        brief_id=record.brief_id,
        author_id=record.author_id,
        article_type=record.article_type,
        desired_word_count=record.desired_word_count,
        target_word_count=record.desired_word_count,
        target_min_word_count=record.target_min_word_count,
        target_max_word_count=record.target_max_word_count,
        planned_sections=list(
            StyleScribeRepository.decode_json_list(record.planned_sections_json)
        ),
        expansion_items_used=list(
            StyleScribeRepository.decode_json_list(record.expansion_items_used_json)
        ),
        claims_to_avoid=list(
            StyleScribeRepository.decode_json_list(record.claims_to_avoid_json)
        ),
        plan_summary=record.plan_summary,
        model_provider=record.model_provider,
        model_name=record.model_name,
        token_usage=StyleScribeRepository.decode_json_object(
            record.token_usage_json
        ),
        warnings=warnings,
        created_at=record.created_at,
    )


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None
