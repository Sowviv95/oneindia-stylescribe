"""Evaluate generated article drafts against grounded briefs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticleDraftRecord,
    ArticleRevisionRecord,
    DraftEvaluationRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.models.draft_evaluation_models import DraftEvaluationResponse
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)

PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "draft_grounding_evaluation_prompt.txt"
)


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class DraftEvaluationError(RuntimeError):
    """Raised when draft grounding evaluation cannot be completed."""


def evaluate_draft_grounding(
    draft_id: str,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> DraftEvaluationResponse:
    """Evaluate a saved draft against its grounded brief."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    draft = repo.fetch_article_draft(draft_id)
    if draft is None:
        raise DraftEvaluationError(f"No article draft found for draft_id: {draft_id}")

    brief = repo.fetch_grounded_brief(draft.brief_id)
    if brief is None:
        raise DraftEvaluationError(
            f"No grounded brief found for brief_id: {draft.brief_id}"
        )

    evaluation_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for draft evaluation."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_draft_evaluation_input(draft, brief)

    try:
        evaluation = evaluation_client.generate_structured_json(prompt, user_payload)
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise DraftEvaluationError("OpenAI draft evaluation failed.") from exc

    created_at = datetime.now(UTC).isoformat()
    record = DraftEvaluationRecord(
        evaluation_id=str(uuid4()),
        draft_id=draft.draft_id,
        brief_id=draft.brief_id,
        author_id=draft.author_id,
        model_provider=evaluation_client.provider,
        model_name=evaluation_client.model_name,
        status="completed",
        evaluation_json=StyleScribeRepository.encode_json(evaluation),
        warnings_json=StyleScribeRepository.encode_warnings([]),
        created_at=created_at,
    )
    repo.save_draft_evaluation(record)
    return _evaluation_response(record, evaluation, [])


def evaluate_revision_grounding(
    revision_id: str,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> DraftEvaluationResponse:
    """Evaluate a saved revision against the original draft's grounded brief."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    revision = repo.fetch_article_revision(revision_id)
    if revision is None:
        raise DraftEvaluationError(
            f"No article revision found for revision_id: {revision_id}"
        )
    draft = repo.fetch_article_draft(revision.draft_id)
    if draft is None:
        raise DraftEvaluationError(
            f"No article draft found for draft_id: {revision.draft_id}"
        )
    brief = repo.fetch_grounded_brief(draft.brief_id)
    if brief is None:
        raise DraftEvaluationError(
            f"No grounded brief found for brief_id: {draft.brief_id}"
        )

    evaluation_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for draft evaluation."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_revision_evaluation_input(revision, draft, brief)

    try:
        evaluation = evaluation_client.generate_structured_json(prompt, user_payload)
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise DraftEvaluationError("OpenAI revision evaluation failed.") from exc

    evaluation["evaluated_revision_id"] = revision.revision_id
    evaluation["initial_evaluation_id"] = revision.evaluation_id
    created_at = datetime.now(UTC).isoformat()
    record = DraftEvaluationRecord(
        evaluation_id=str(uuid4()),
        draft_id=draft.draft_id,
        brief_id=draft.brief_id,
        author_id=draft.author_id,
        model_provider=evaluation_client.provider,
        model_name=evaluation_client.model_name,
        status="completed",
        evaluation_json=StyleScribeRepository.encode_json(evaluation),
        warnings_json=StyleScribeRepository.encode_warnings([]),
        created_at=created_at,
    )
    repo.save_draft_evaluation(record)
    return _evaluation_response(record, evaluation, [])


def get_draft_evaluation(
    evaluation_id: str,
    repository: StyleScribeRepository | None = None,
) -> DraftEvaluationResponse:
    """Fetch a saved draft evaluation by ID."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_draft_evaluation(evaluation_id)
    if record is None:
        raise DraftEvaluationError(
            f"No draft evaluation found for evaluation_id: {evaluation_id}"
        )
    return _evaluation_response(
        record,
        StyleScribeRepository.decode_json_object(record.evaluation_json),
        StyleScribeRepository.decode_json_list(record.warnings_json),
    )


def get_latest_draft_evaluation(
    draft_id: str,
    repository: StyleScribeRepository | None = None,
) -> DraftEvaluationResponse:
    """Fetch the latest saved draft evaluation."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_latest_draft_evaluation(draft_id)
    if record is None:
        raise DraftEvaluationError(
            f"No draft evaluation found for draft_id: {draft_id}"
        )
    return _evaluation_response(
        record,
        StyleScribeRepository.decode_json_object(record.evaluation_json),
        StyleScribeRepository.decode_json_list(record.warnings_json),
    )


def build_draft_evaluation_input(
    draft: ArticleDraftRecord,
    brief: GroundedBriefRecord,
) -> str:
    """Build bounded evaluation input with draft and grounded brief only."""

    brief_json = StyleScribeRepository.decode_json_object(brief.brief_json)
    draft_json = StyleScribeRepository.decode_json_object(draft.draft_json)
    payload = {
        "task": "Evaluate generated draft grounding against grounded brief.",
        "grounded_brief_only_factual_source": {
            "brief_id": brief.brief_id,
            "brief": brief_json,
            "claims_to_avoid": brief_json.get("claims_to_avoid", []),
        },
        "generated_article_draft_to_check": {
            "draft_id": draft.draft_id,
            "draft": draft_json,
        },
        "evaluation_rule": (
            "Use only grounded_brief_only_factual_source. Do not use outside "
            "knowledge, author samples, source article full text, or style "
            "profile facts."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_revision_evaluation_input(
    revision: ArticleRevisionRecord,
    original_draft: ArticleDraftRecord,
    brief: GroundedBriefRecord,
) -> str:
    """Build bounded evaluation input for a revised draft."""

    brief_json = StyleScribeRepository.decode_json_object(brief.brief_json)
    revised_draft = _revision_draft_json(revision)
    payload = {
        "task": "Evaluate revised generated draft grounding against grounded brief.",
        "grounded_brief_only_factual_source": {
            "brief_id": brief.brief_id,
            "brief": brief_json,
            "claims_to_avoid": brief_json.get("claims_to_avoid", []),
        },
        "generated_article_draft_to_check": {
            "draft_id": original_draft.draft_id,
            "revision_id": revision.revision_id,
            "draft": revised_draft,
        },
        "evaluation_rule": (
            "Use only grounded_brief_only_factual_source. Do not use outside "
            "knowledge, author samples, source article full text, or style "
            "profile facts."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _revision_draft_json(revision: ArticleRevisionRecord) -> dict[str, object]:
    return {
        "headline": revision.revised_headline,
        "subheadline": revision.revised_subheadline,
        "article_body": revision.revised_article_body,
        "seo_title": revision.revised_seo_title,
        "meta_description": revision.revised_meta_description,
        "suggested_tags": json.loads(revision.revised_tags_json),
    }


def _evaluation_response(
    record: DraftEvaluationRecord,
    evaluation: dict[str, object],
    warnings: list[str],
) -> DraftEvaluationResponse:
    return DraftEvaluationResponse(
        evaluation_id=record.evaluation_id,
        draft_id=record.draft_id,
        brief_id=record.brief_id,
        author_id=record.author_id,
        model_provider=record.model_provider,
        model_name=record.model_name,
        status=record.status,
        evaluation=evaluation,
        warnings=warnings,
        created_at=record.created_at,
    )
