from pathlib import Path

import pytest

from backend.app.db.repository import (
    ArticleDraftRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.draft_grounding_evaluation_service import (
    DraftEvaluationError,
    build_draft_evaluation_input,
    evaluate_draft_grounding,
    get_latest_draft_evaluation,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_evaluate_draft_grounding_saves_to_sqlite(tmp_path: Path) -> None:
    repository = _repository_with_draft_and_brief(tmp_path)
    response = evaluate_draft_grounding(
        "draft-1",
        repository=repository,
        model_client=MockEvaluationClient(),
    )
    latest = get_latest_draft_evaluation("draft-1", repository)

    assert response.evaluation_id
    assert response.draft_id == "draft-1"
    assert response.evaluation["overall_risk"] == "high"
    assert response.evaluation["unsupported_claims"]
    assert latest.evaluation_id == response.evaluation_id


def test_evaluate_draft_grounding_missing_draft(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()

    with pytest.raises(DraftEvaluationError, match="No article draft found"):
        evaluate_draft_grounding("missing", repository, MockEvaluationClient())


def test_evaluate_draft_grounding_invalid_json(tmp_path: Path) -> None:
    repository = _repository_with_draft_and_brief(tmp_path)

    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        evaluate_draft_grounding("draft-1", repository, InvalidEvaluationClient())


def test_evaluation_prompt_input_is_bounded_and_grounded_only(tmp_path: Path) -> None:
    repository = _repository_with_draft_and_brief(tmp_path)
    draft = repository.fetch_article_draft("draft-1")
    brief = repository.fetch_grounded_brief("brief-1")
    assert draft is not None
    assert brief is not None

    payload = build_draft_evaluation_input(draft, brief)

    assert "grounded_brief_only_factual_source" in payload
    assert "generated_article_draft_to_check" in payload
    assert "claims_to_avoid" in payload
    assert "FULL_AUTHOR_SAMPLE_CORPUS" not in payload
    assert "FULL_SOURCE_ARTICLE_TEXT" not in payload
    assert "outside knowledge" in payload


class MockEvaluationClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounded brief as the only factual source" in system_prompt
        assert "generated_article_draft_to_check" in user_payload
        return _evaluation_json()


class InvalidEvaluationClient(MockEvaluationClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")


def _repository_with_draft_and_brief(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
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
                    "one_line_summary": "Pilot starts next month.",
                    "confirmed_facts": ["18 sensors will be installed."],
                    "claims_to_avoid": ["Do not claim effectiveness."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_article_draft(
        ArticleDraftRecord(
            draft_id="draft-1",
            author_id="v_vasanthi",
            profile_id="profile-1",
            brief_id="brief-1",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            author_instruction=None,
            article_type="news",
            desired_word_count=500,
            tone_override=None,
            include_seo=True,
            draft_json=StyleScribeRepository.encode_json(
                {
                    "headline": "Headline",
                    "article_body": "The system will ensure safety and reduce impact.",
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository


def _evaluation_json() -> dict[str, object]:
    return {
        "grounding_score": 45,
        "claim_safety_score": 40,
        "fact_preservation_score": 80,
        "overall_risk": "high",
        "editorial_readiness": "revision_required",
        "unsupported_claims": [
            {
                "claim": "system will ensure safety",
                "reason": "Benefit is not in brief.",
                "suggested_fix": "State only that SMS alerts will be sent.",
            }
        ],
        "overclaim_phrases": [
            {
                "phrase": "reduce impact",
                "reason": "Effectiveness is not established.",
                "suggested_fix": "Remove or qualify.",
            }
        ],
        "invented_facts": [],
        "contradictions": [],
        "claims_to_avoid_violations": ["Do not claim effectiveness."],
        "missing_key_facts": [],
        "preserved_facts": ["18 sensors will be installed."],
        "number_date_name_checks": ["18 preserved."],
        "rewrite_guidance": ["Avoid benefit language."],
        "summary": "Revision required due to unsupported benefits.",
    }
