"""Generate and persist controlled Tamil article drafts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    ArticleDraftRecord,
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.models.article_draft_models import ArticleDraftResponse
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "article_generation_prompt.txt"


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

    warnings = _build_warnings(profile_record, brief_record, target_language)
    draft_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for article draft generation."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    user_payload = build_article_generation_input(
        profile_record=profile_record,
        brief_record=brief_record,
        author_instruction=author_instruction,
        target_language=target_language,
    )

    try:
        draft = draft_client.generate_structured_json(prompt, user_payload)
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
) -> str:
    """Build explicitly separated style/fact input for the model."""

    payload = {
        "target_language": target_language,
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
    }
    return json.dumps(payload, ensure_ascii=False)


def _build_warnings(
    profile_record: AuthorStyleProfileRecord,
    brief_record: GroundedBriefRecord,
    target_language: str,
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
        draft=draft,
        warnings=warnings,
        created_at=record.created_at,
    )
