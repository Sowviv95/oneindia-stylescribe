"""Generate and persist OpenAI grounded factual briefs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.models.grounded_brief_models import GroundedBriefResponse
from backend.app.services.language_detection_service import detect_language
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)
from backend.app.services.source_processor import SourceProcessingError, process_source

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "grounded_brief_prompt.txt"
MAX_SOURCE_CHARS_FOR_MODEL = 16000


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class GroundedBriefError(RuntimeError):
    """Raised when a grounded brief cannot be generated or fetched."""


def generate_grounded_brief(
    source_type: str,
    source_input: str,
    target_language: str = "ta",
    source_input_mode: str = "plain_text",
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> GroundedBriefResponse:
    """Generate, save, and return a grounded brief."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    try:
        processed = process_source(source_type, source_input, source_input_mode)
    except SourceProcessingError:
        raise

    source_language = detect_language(processed.cleaned_text)
    warnings = list(processed.warnings)
    if source_language == "unknown":
        warnings.append("Source language detection returned unknown.")

    bounded_text = processed.cleaned_text
    if len(bounded_text) > MAX_SOURCE_CHARS_FOR_MODEL:
        bounded_text = bounded_text[:MAX_SOURCE_CHARS_FOR_MODEL]
        warnings.append(
            f"Source text was truncated to {MAX_SOURCE_CHARS_FOR_MODEL} characters."
        )

    brief_client = model_client or OpenAIJsonClient(
        missing_key_message="OPENAI_API_KEY is required for grounded briefs."
    )
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    llm_input = build_grounded_brief_llm_input(
        source_type=processed.source_type,
        source_url=processed.source_url,
        source_language=source_language,
        target_language=target_language,
        source_text=bounded_text,
    )

    try:
        brief = brief_client.generate_structured_json(
            system_prompt=prompt,
            user_payload=llm_input,
        )
    except OpenAIClientError:
        raise
    except Exception as exc:
        raise GroundedBriefError("OpenAI grounded brief generation failed.") from exc
    brief = cleanup_grounded_brief_tamil(brief)

    created_at = datetime.now(UTC).isoformat()
    record = GroundedBriefRecord(
        brief_id=str(uuid4()),
        source_type=processed.source_type,
        source_input_hash=processed.source_input_hash,
        source_url=processed.source_url,
        source_text_excerpt=processed.source_text_excerpt,
        source_language=source_language,
        target_language=target_language,
        model_provider=brief_client.provider,
        model_name=brief_client.model_name,
        status="completed",
        brief_json=StyleScribeRepository.encode_json(brief),
        warnings_json=StyleScribeRepository.encode_warnings(warnings),
        created_at=created_at,
    )
    repo.save_grounded_brief(record)
    return _brief_response(record, brief, warnings)


def get_grounded_brief(
    brief_id: str,
    repository: StyleScribeRepository | None = None,
) -> GroundedBriefResponse:
    """Fetch a saved grounded brief."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_grounded_brief(brief_id)
    if record is None:
        raise GroundedBriefError(f"No grounded brief found for brief_id: {brief_id}")
    return _brief_response(
        record,
        StyleScribeRepository.decode_json_object(record.brief_json),
        StyleScribeRepository.decode_json_list(record.warnings_json),
    )


def build_grounded_brief_llm_input(
    source_type: str,
    source_url: str | None,
    source_language: str,
    target_language: str,
    source_text: str,
) -> str:
    payload = {
        "task": "Extract a grounded factual brief for Tamil news generation.",
        "source_type": source_type,
        "source_url": source_url,
        "source_language": source_language,
        "target_language": target_language,
        "source_text": source_text,
        "instruction": (
            "Use only this source_text. Preserve facts accurately and list "
            "claims_to_avoid for anything unsupported."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def cleanup_grounded_brief_tamil(brief: dict[str, object]) -> dict[str, object]:
    """Clean known mixed-language artifacts before the brief feeds later stages."""

    cleaned = _clean_object_strings(brief)
    return cleaned if isinstance(cleaned, dict) else brief


def _clean_object_strings(value: object) -> object:
    if isinstance(value, str):
        return (
            value.replace("pertenc செய்கிறார்கள்", "சேர்ந்துள்ளனர்")
            .replace("pertencிக்கிறார்கள்", "சேர்ந்துள்ளனர்")
            .replace("pertenc", "சேர்ந்துள்ளனர்")
        )
    if isinstance(value, list):
        return [_clean_object_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_object_strings(item) for key, item in value.items()}
    return value


def _brief_response(
    record: GroundedBriefRecord,
    brief: dict[str, object],
    warnings: list[str],
) -> GroundedBriefResponse:
    return GroundedBriefResponse(
        brief_id=record.brief_id,
        source_type=record.source_type,
        source_url=record.source_url,
        source_language=record.source_language,
        target_language=record.target_language,
        model_provider=record.model_provider,
        model_name=record.model_name,
        status=record.status,
        brief=brief,
        warnings=warnings,
        created_at=record.created_at,
        source_text_excerpt=record.source_text_excerpt,
    )
