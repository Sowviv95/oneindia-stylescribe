from pathlib import Path

import pytest

from backend.app.db.repository import StyleScribeRepository, StyleSnapshotRecord
from backend.app.services.author_style_profile_service import (
    AuthorStyleProfileError,
    build_profile_llm_input,
    generate_author_style_profile,
    get_latest_author_style_profile,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_generate_author_style_profile_saves_profile(tmp_path: Path) -> None:
    repository = _repository_with_snapshot(tmp_path)
    client = MockProfileClient()

    response = generate_author_style_profile("v_vasanthi", repository, client)
    latest = get_latest_author_style_profile("v_vasanthi", repository)

    assert response.profile_id
    assert response.author_id == "v_vasanthi"
    assert response.snapshot_id == "snapshot-1"
    assert response.model_provider == "openai"
    assert response.model_name == "gpt-4o-mini"
    assert response.profile["overall_tone"] == "Measured and news-focused."
    assert response.source_excerpt_refs[0]["filename"] == "article-one.docx"
    assert latest.profile_id == response.profile_id


def test_generate_author_style_profile_errors_without_snapshot(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()

    with pytest.raises(AuthorStyleProfileError, match="Build a style snapshot first"):
        generate_author_style_profile("missing", repository, MockProfileClient())


def test_generate_author_style_profile_handles_invalid_model_json(
    tmp_path: Path,
) -> None:
    repository = _repository_with_snapshot(tmp_path)

    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        generate_author_style_profile("v_vasanthi", repository, InvalidJsonClient())


def test_profile_input_uses_bounded_snapshot_excerpts_only(tmp_path: Path) -> None:
    repository = _repository_with_snapshot(tmp_path)
    client = MockProfileClient()

    generate_author_style_profile("v_vasanthi", repository, client)

    assert client.last_payload is not None
    assert "extracted_text" not in client.last_payload
    assert "source_path" not in client.last_payload
    assert "Short bounded excerpt" in client.last_payload
    assert len(client.last_payload) < 5000


def test_build_profile_llm_input_limits_excerpt_count() -> None:
    snapshot = StyleSnapshotRecord(
        snapshot_id="snapshot-1",
        author_id="author",
        article_count=10,
        language="ta",
        status="completed",
        stats_json="{}",
        excerpt_pack_json="{}",
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
    refs = [{"excerpt_text": f"excerpt {index}"} for index in range(40)]

    payload = build_profile_llm_input(snapshot, {}, refs)

    assert "excerpt 23" in payload
    assert "excerpt 24" not in payload


class MockProfileClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.last_payload: str | None = None

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "examples are for style only" in system_prompt
        self.last_payload = user_payload
        return {
            "overall_tone": "Measured and news-focused.",
            "headline_style": "Direct.",
            "intro_style": "Context first.",
            "paragraph_style": "Compact.",
            "sentence_style": "Clear.",
            "vocabulary_style": "Tamil-forward.",
            "narrative_flow": "Fact to implication.",
            "closing_style": "Summative.",
            "reader_engagement_style": "Low hype.",
            "tamil_register": "Standard Tamil.",
            "english_or_transliterated_word_usage": "Minimal.",
            "category_specific_observations": ["Politics uses context."],
            "dos": ["Keep the tone grounded."],
            "donts": ["Do not copy facts."],
            "few_shot_usage_guidance": "Use excerpts for rhythm only.",
            "generation_guidance": "Write concise Tamil news prose.",
            "style_risks": ["Overfitting to sample topics."],
        }


class InvalidJsonClient(MockProfileClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")


def _repository_with_snapshot(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
    excerpt_pack = {
        "intro_examples": [
            {
                "article_id": "a1",
                "filename": "article-one.docx",
                "title_or_heading": "Article One",
                "category": "Politics",
                "excerpt_type": "intro",
                "char_count": 21,
                "excerpt_text": "Short bounded excerpt",
            }
        ],
        "body_examples": [
            {
                "article_id": "a1",
                "filename": "article-one.docx",
                "title_or_heading": "Article One",
                "category": "Politics",
                "excerpt_type": "body",
                "char_count": 20,
                "excerpt_text": "Another bounded bit",
            }
        ],
    }
    repository.save_style_snapshot(
        StyleSnapshotRecord(
            snapshot_id="snapshot-1",
            author_id="v_vasanthi",
            article_count=1,
            language="ta",
            status="completed",
            stats_json=StyleScribeRepository.encode_json(
                {"article_count": 1, "average_char_count": 21}
            ),
            excerpt_pack_json=StyleScribeRepository.encode_json(excerpt_pack),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository
