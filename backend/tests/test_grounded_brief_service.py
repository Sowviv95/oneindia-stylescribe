from pathlib import Path

import pytest

from backend.app.db.repository import StyleScribeRepository
from backend.app.services.grounded_brief_service import (
    MAX_SOURCE_CHARS_FOR_MODEL,
    cleanup_grounded_brief_tamil,
    generate_grounded_brief,
    get_grounded_brief,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError

SOURCE_TEXT = "Chennai officials said the project will start on Monday with 25 workers."


def test_generate_grounded_brief_saves_to_sqlite(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    client = MockBriefClient()

    response = generate_grounded_brief(
        source_type="text",
        source_input=SOURCE_TEXT,
        repository=repository,
        model_client=client,
    )
    saved = get_grounded_brief(response.brief_id, repository)

    assert response.brief_id
    assert response.source_language == "en"
    assert response.target_language == "ta"
    assert response.model_provider == "openai"
    assert response.brief["topic"] == "Project launch"
    assert saved.brief_id == response.brief_id


def test_generate_grounded_brief_invalid_json_handled(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")

    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        generate_grounded_brief(
            source_type="text",
            source_input=SOURCE_TEXT,
            repository=repository,
            model_client=InvalidJsonClient(),
        )


def test_generate_grounded_brief_truncates_source_text(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    client = MockBriefClient()
    source_text = "A" * (MAX_SOURCE_CHARS_FOR_MODEL + 100)

    response = generate_grounded_brief(
        source_type="text",
        source_input=source_text,
        repository=repository,
        model_client=client,
    )

    assert "truncated" in " ".join(response.warnings)
    assert client.last_payload is not None
    assert len(client.last_payload) < MAX_SOURCE_CHARS_FOR_MODEL + 2000


def test_grounded_brief_cleanup_removes_corrupted_mixed_token() -> None:
    cleaned = cleanup_grounded_brief_tamil(
        {
            "quotes": [
                {
                    "quote_tamil_or_summary": (
                        "அமெரிக்காவில் யார் pertencிக்கிறார்கள் என்பதற்கான கருத்து."
                    )
                }
            ]
        }
    )

    assert "pertenc" not in str(cleaned)
    assert "சேர்ந்துள்ளனர்" in str(cleaned)


class MockBriefClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.last_payload: str | None = None

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "Use only the supplied source text" in system_prompt
        assert "affected groups" in system_prompt
        assert "policy/legal context" in system_prompt
        self.last_payload = user_payload
        return {
            "topic": "Project launch",
            "one_line_summary": "Officials announced a project launch.",
            "source_language": "en",
            "target_language": "ta",
            "confirmed_facts": ["Project starts on Monday."],
            "key_entities": [],
            "places": [{"name_original": "Chennai", "name_tamil": "சென்னை"}],
            "dates_or_timeline": ["Monday"],
            "numbers_and_statistics": ["25 workers"],
            "affected_groups": ["workers"],
            "quotes": [],
            "policy_or_legal_context": [],
            "background_from_source": [],
            "missing_or_unclear_information": [],
            "claims_to_avoid": ["Do not state the project is completed."],
            "suggested_tamil_angle": "சென்னையில் திட்ட தொடக்கம்",
            "editorial_risk_notes": [],
        }


class InvalidJsonClient(MockBriefClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")
