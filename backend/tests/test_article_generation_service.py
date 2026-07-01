from pathlib import Path

import pytest

from backend.app.db.repository import (
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_generation_service import (
    ArticleGenerationError,
    build_article_generation_input,
    generate_article_draft,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_generate_article_draft_saves_to_sqlite(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    client = MockDraftClient()

    response = generate_article_draft(
        author_id="v_vasanthi",
        brief_id="brief-1",
        author_instruction="Write a concise Tamil article.",
        repository=repository,
        model_client=client,
    )
    saved = repository.fetch_article_draft(response.draft_id)

    assert response.draft_id
    assert response.profile_id == "profile-1"
    assert response.brief_id == "brief-1"
    assert response.draft["headline"] == "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி"
    assert saved is not None


def test_missing_style_profile_error(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()

    with pytest.raises(ArticleGenerationError, match="Generate style profile first"):
        generate_article_draft("missing", "brief-1", repository=repository)


def test_missing_grounded_brief_error(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    with pytest.raises(ArticleGenerationError, match="No grounded brief found"):
        generate_article_draft(
            "v_vasanthi",
            "missing",
            repository=repository,
            model_client=MockDraftClient(),
        )


def test_invalid_json_from_openai_handled(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        generate_article_draft(
            "v_vasanthi",
            "brief-1",
            repository=repository,
            model_client=InvalidDraftClient(),
        )


def test_target_language_warning_when_not_ta(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    response = generate_article_draft(
        "v_vasanthi",
        "brief-1",
        target_language="en",
        repository=repository,
        model_client=MockDraftClient(),
    )

    assert any("Tamil-focused" in warning for warning in response.warnings)


def test_prompt_input_separates_style_and_facts(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    profile = repository.fetch_latest_author_style_profile("v_vasanthi")
    brief = repository.fetch_grounded_brief("brief-1")
    assert profile is not None
    assert brief is not None

    payload = build_article_generation_input(profile, brief, None, "ta")

    assert "style_profile_for_voice_only" in payload
    assert "grounded_brief_for_facts_only" in payload
    assert "FULL_AUTHOR_SAMPLE_CORPUS" not in payload
    assert "FULL_SOURCE_ARTICLE_TEXT" not in payload


class MockDraftClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "Use only the grounded brief for facts" in system_prompt
        assert "style_profile_for_voice_only" in user_payload
        assert "grounded_brief_for_facts_only" in user_payload
        return {
            "headline": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "மூன்று பகுதிகளில் அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி தொடங்க உள்ளது.",
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை திட்டம்",
            "meta_description": "சென்னையில் புதிய வெள்ள எச்சரிக்கை முயற்சி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "fact_usage_notes": ["18 sensors preserved."],
            "style_usage_notes": ["Used emotional but grounded tone."],
        }


class InvalidDraftClient(MockDraftClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")


def _repository_with_profile_and_brief(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
    repository.save_author_style_profile(
        AuthorStyleProfileRecord(
            profile_id="profile-1",
            author_id="v_vasanthi",
            snapshot_id="snapshot-1",
            language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            profile_json=StyleScribeRepository.encode_json(
                {
                    "overall_tone": "Measured",
                    "headline_style": "Direct",
                    "intro_style": "Context first",
                    "paragraph_style": "Compact",
                    "tamil_register": "Conversational Tamil",
                    "dos": ["Stay grounded"],
                    "donts": ["Do not invent facts"],
                }
            ),
            source_excerpt_refs_json="[]",
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_grounded_brief(
        GroundedBriefRecord(
            brief_id="brief-1",
            source_type="text",
            source_input_hash="hash",
            source_url=None,
            source_text_excerpt="Short excerpt only",
            source_language="en",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            brief_json=StyleScribeRepository.encode_json(
                {
                    "topic": "Flood warning pilot",
                    "one_line_summary": "Chennai starts a flood warning pilot.",
                    "confirmed_facts": ["18 sensors will be installed."],
                    "claims_to_avoid": ["Do not claim it has succeeded."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository
