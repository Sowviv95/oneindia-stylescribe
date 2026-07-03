import json
from pathlib import Path

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.services.article_plan_service import (
    build_article_plan_input,
    generate_article_plan,
)


def test_article_plan_prompt_includes_length_and_claim_rules(tmp_path: Path) -> None:
    repository = _repository_with_brief(tmp_path)
    brief = repository.fetch_grounded_brief("brief-1")
    assert brief is not None

    prompt = Path("backend/app/prompts/article_plan_prompt.txt").read_text(
        encoding="utf-8"
    )
    payload = build_article_plan_input(
        brief=brief,
        author_id="v_vasanthi",
        article_type="news",
        desired_word_count=600,
        target_language="ta",
        tone_override="clear",
        author_instruction="Write for Oneindia readers.",
    )

    assert "desired_word_count" in payload
    assert "target_word_count_range" in payload
    assert "target_words" in prompt
    assert "unsupported benefit" in prompt
    assert "generic filler" in prompt
    assert "6-8 sections" in prompt


def test_generate_article_plan_persists_multiple_sections(tmp_path: Path) -> None:
    repository = _repository_with_brief(tmp_path)

    response = generate_article_plan(
        brief_id="brief-1",
        author_id="v_vasanthi",
        article_type="news",
        desired_word_count=600,
        target_language="ta",
        repository=repository,
        model_client=MockPlanClient(),
    )
    saved = repository.fetch_article_plan(response.plan_id)

    assert saved is not None
    assert response.target_min_word_count == 450
    assert response.target_max_word_count == 690
    assert len(response.planned_sections) == 6
    assert response.expansion_items_used


class MockPlanClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        decoded = json.loads(user_payload)
        assert decoded["desired_word_count"] == 600
        return {
            "target_word_count": 600,
            "target_min_word_count": 450,
            "target_max_word_count": 690,
            "planned_sections": [
                {"section_name": f"section-{index}", "target_words": 90}
                for index in range(6)
            ],
            "expansion_items_used": ["18 sensors", "SMS alerts"],
            "claims_to_avoid": ["Do not claim effectiveness."],
            "plan_summary": "Six-section grounded plan.",
            "warnings": [],
        }


def _repository_with_brief(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
    repository.save_grounded_brief(
        GroundedBriefRecord(
            brief_id="brief-1",
            source_type="text",
            source_input_hash="hash",
            source_url=None,
            source_text_excerpt="Officials said a flood pilot begins next month.",
            source_language="en",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            brief_json=StyleScribeRepository.encode_json(
                {
                    "confirmed_facts": ["Pilot begins next month", "18 sensors"],
                    "numbers_and_statistics": ["18 sensors"],
                    "affected_groups": ["residents"],
                    "quotes": ["Officials said"],
                    "claims_to_avoid": ["Do not claim success."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository
