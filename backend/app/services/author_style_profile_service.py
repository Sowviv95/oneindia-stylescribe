"""Generate and persist LLM-based author style profiles."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from backend.app.db.repository import (
    AuthorStyleProfileRecord,
    StyleScribeRepository,
    StyleSnapshotRecord,
)
from backend.app.models.style_profile_models import AuthorStyleProfileResponse
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIStyleClient,
)

PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "author_style_profile_prompt.txt"
MAX_EXCERPTS_FOR_MODEL = 24
LOW_ARTICLE_COUNT_THRESHOLD = 5
SMALL_EXCERPT_COUNT_THRESHOLD = 3


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


class AuthorStyleProfileError(RuntimeError):
    """Raised when a style profile cannot be generated or fetched."""


def generate_author_style_profile(
    author_id: str,
    repository: StyleScribeRepository | None = None,
    model_client: StructuredJsonClient | None = None,
) -> AuthorStyleProfileResponse:
    """Generate, save, and return an OpenAI style profile."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    snapshot = repo.fetch_latest_style_snapshot(author_id)
    if snapshot is None:
        raise AuthorStyleProfileError(
            "No deterministic style snapshot found. Build a style snapshot first."
        )

    profile_client = model_client or OpenAIStyleClient()
    stats = StyleScribeRepository.decode_json_object(snapshot.stats_json)
    excerpt_pack = StyleScribeRepository.decode_json_object(
        snapshot.excerpt_pack_json
    )
    source_excerpt_refs = build_source_excerpt_refs(excerpt_pack)
    warnings = _build_warnings(snapshot.article_count, source_excerpt_refs)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    llm_input = build_profile_llm_input(snapshot, stats, source_excerpt_refs)

    try:
        profile = profile_client.generate_structured_json(
            system_prompt=prompt,
            user_payload=llm_input,
        )
    except OpenAIClientError:
        raise
    except Exception as exc:
        message = "OpenAI style profile generation failed."
        raise AuthorStyleProfileError(message) from exc

    created_at = datetime.now(UTC).isoformat()
    profile_id = str(uuid4())
    record = AuthorStyleProfileRecord(
        profile_id=profile_id,
        author_id=author_id,
        snapshot_id=snapshot.snapshot_id,
        language=snapshot.language,
        model_provider=profile_client.provider,
        model_name=profile_client.model_name,
        status="completed",
        profile_json=StyleScribeRepository.encode_json(profile),
        source_excerpt_refs_json=StyleScribeRepository.encode_json(
            source_excerpt_refs
        ),
        warnings_json=StyleScribeRepository.encode_warnings(warnings),
        created_at=created_at,
    )
    repo.save_author_style_profile(record)

    return _profile_response(record, profile, source_excerpt_refs, warnings)


def get_latest_author_style_profile(
    author_id: str,
    repository: StyleScribeRepository | None = None,
) -> AuthorStyleProfileResponse:
    """Return the latest saved author style profile."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    record = repo.fetch_latest_author_style_profile(author_id)
    if record is None:
        raise AuthorStyleProfileError(
            f"No author style profile found for author_id: {author_id}"
        )
    return _profile_response(
        record,
        StyleScribeRepository.decode_json_object(record.profile_json),
        _decode_source_excerpt_refs(record.source_excerpt_refs_json),
        StyleScribeRepository.decode_json_list(record.warnings_json),
    )


def build_profile_llm_input(
    snapshot: StyleSnapshotRecord,
    stats: dict[str, object],
    source_excerpt_refs: list[dict[str, object]],
) -> str:
    """Build bounded JSON input for style profile generation."""

    payload = {
        "task": "Generate reusable Tamil author writing-style guidance.",
        "author_id": snapshot.author_id,
        "snapshot_id": snapshot.snapshot_id,
        "language": snapshot.language,
        "article_count": snapshot.article_count,
        "deterministic_stats": stats,
        "curated_style_excerpts": source_excerpt_refs[:MAX_EXCERPTS_FOR_MODEL],
        "instruction": (
            "Use excerpts only as style evidence. Do not convert article facts "
            "into style rules."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_source_excerpt_refs(
    excerpt_pack: dict[str, object],
) -> list[dict[str, object]]:
    """Flatten the bounded excerpt pack into source refs used for generation."""

    refs: list[dict[str, object]] = []
    for section_name in (
        "headline_examples",
        "intro_examples",
        "body_examples",
        "closing_examples",
        "short_article_examples",
        "long_article_examples",
    ):
        section = excerpt_pack.get(section_name)
        if not isinstance(section, list):
            continue
        for item in section:
            if not isinstance(item, dict):
                continue
            refs.append(
                {
                    "pack_section": section_name,
                    "article_id": item.get("article_id"),
                    "filename": item.get("filename"),
                    "title_or_heading": item.get("title_or_heading"),
                    "category": item.get("category"),
                    "excerpt_type": item.get("excerpt_type"),
                    "char_count": item.get("char_count"),
                    "excerpt_text": item.get("excerpt_text"),
                }
            )
    return refs[:MAX_EXCERPTS_FOR_MODEL]


def _profile_response(
    record: AuthorStyleProfileRecord,
    profile: dict[str, object],
    source_excerpt_refs: list[dict[str, object]],
    warnings: list[str],
) -> AuthorStyleProfileResponse:
    return AuthorStyleProfileResponse(
        profile_id=record.profile_id,
        author_id=record.author_id,
        snapshot_id=record.snapshot_id,
        language=record.language,
        model_provider=record.model_provider,
        model_name=record.model_name,
        status=record.status,
        profile=profile,
        source_excerpt_refs=source_excerpt_refs,
        warnings=warnings,
        created_at=record.created_at,
    )


def _build_warnings(
    article_count: int,
    source_excerpt_refs: list[dict[str, object]],
) -> list[str]:
    warnings: list[str] = []
    if article_count < LOW_ARTICLE_COUNT_THRESHOLD:
        warnings.append(
            f"Low article count for style profiling: {article_count} articles."
        )
    if len(source_excerpt_refs) < SMALL_EXCERPT_COUNT_THRESHOLD:
        excerpt_count = len(source_excerpt_refs)
        warnings.append(
            f"Small excerpt pack for style profiling: {excerpt_count} excerpts."
        )
    return warnings


def _decode_source_excerpt_refs(value: str) -> list[dict[str, object]]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        return []
    return [item for item in decoded if isinstance(item, dict)]
