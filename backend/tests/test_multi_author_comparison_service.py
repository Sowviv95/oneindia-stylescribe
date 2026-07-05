import json
from pathlib import Path

from backend.app.db.repository import AuthorStyleProfileRecord, StyleScribeRepository
from backend.app.services.multi_author_comparison_service import (
    run_multi_author_comparison_workflow,
)


def test_multi_author_comparison_reuses_brief_and_separates_author_branches(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profiles(tmp_path)
    brief_client = CountingBriefClient()
    plan_client = TrackingPlanClient()

    response = run_multi_author_comparison_workflow(
        source_text=_source_text(),
        author_id_a="v_vasanthi",
        author_id_b="hema_vandhana",
        desired_word_count=600,
        workflow_mode="standard",
        repository=repository,
        brief_model_client=brief_client,
        plan_model_client=plan_client,
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    assert response.workflow_completed is True
    assert response.author_a.author_id == "v_vasanthi"
    assert response.author_b.author_id == "hema_vandhana"
    assert response.shared_grounded_brief.brief_id == response.author_a.telemetry[
        "plan_token_usage"
    ].get("brief_marker")
    assert response.author_a.draft_id != response.author_b.draft_id
    assert response.author_a.plan_id != response.author_b.plan_id
    assert brief_client.calls == 1
    assert plan_client.author_ids == ["v_vasanthi", "hema_vandhana"]
    assert response.author_a.grounding_score == 86
    assert response.author_b.grounding_score == 86
    assert response.author_a.editor_attention_items
    unsupported_item = response.author_a.editor_attention_items[0]
    assert unsupported_item.category == "unsupported_claim"
    assert unsupported_item.severity == "blocker"
    assert unsupported_item.claim_text == "This pilot will prevent flooding."
    assert unsupported_item.matched_article_text is None
    assert any(
        item.category == "claims_to_avoid_violation"
        and item.avoid_rule == "Do not claim effectiveness."
        for item in response.author_a.editor_attention_items
    )
    assert any(
        item.category == "overclaim_phrase"
        and item.severity == "warning"
        for item in response.author_a.editor_attention_items
    )
    assert response.comparison_summary.recommended_draft in {
        "author_a",
        "author_b",
        "no_clear_recommendation",
    }
    assert response.aggregate_token_usage["total_tokens"] > 0
    assert response.telemetry["llm_call_count_total"] >= 5


class CountingBriefClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.calls = 0

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        self.calls += 1
        assert "source_text" in user_payload
        return {
            "topic": "Flood warning pilot",
            "one_line_summary": "Chennai officials announced a pilot.",
            "source_language": "en",
            "target_language": "ta",
            "confirmed_facts": [
                "A flood-warning pilot will begin next month.",
                "18 sensors will be installed.",
            ],
            "key_entities": [],
            "places": ["Chennai"],
            "dates_or_timeline": ["next month"],
            "numbers_and_statistics": ["18 sensors"],
            "quotes": [],
            "background_from_source": [],
            "missing_or_unclear_information": [],
            "claims_to_avoid": ["Do not claim effectiveness."],
            "suggested_tamil_angle": "Flood warning pilot",
            "editorial_risk_notes": [],
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 40,
                "total_tokens": 140,
                "cached_prompt_tokens": 0,
            },
        }


class TrackingPlanClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def __init__(self) -> None:
        self.author_ids: list[str] = []

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        payload = json.loads(user_payload)
        self.author_ids.append(str(payload["author_id"]))
        return {
            "target_word_count": 600,
            "target_min_word_count": 450,
            "target_max_word_count": 690,
            "planned_sections": [
                {
                    "section_name": "lead",
                    "purpose": "core news",
                    "target_words": 600,
                    "grounded_facts_to_use": ["18 sensors"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": ["Do not claim effectiveness."],
                    "must_not_add": ["No filler"],
                }
            ],
            "expansion_items_used": ["18 sensors"],
            "claims_to_avoid": ["Do not claim effectiveness."],
            "plan_summary": "One-section grounded test plan.",
            "warnings": [],
            "token_usage": {
                "prompt_tokens": 80,
                "completion_tokens": 30,
                "total_tokens": 110,
                "cached_prompt_tokens": 0,
                "brief_marker": payload["grounded_brief_for_facts_only"][
                    "brief_id"
                ],
            },
        }


class MockDraftClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        payload = json.loads(user_payload)
        if "section_group" in payload:
            return {
                "article_sections": [
                    {
                        "section_text": (
                            "சென்னை நகரில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் "
                            "தொடங்கும் என்று அதிகாரிகள் தெரிவித்துள்ளனர். குறைந்த "
                            "உயரப்பகுதி சாலைகளின் அருகே 18 சென்சார்கள் "
                            "அமைக்கப்படும்."
                        ),
                        "grounded_facts_used": ["18 sensors"],
                    }
                ],
                "token_usage": {
                    "prompt_tokens": 60,
                    "completion_tokens": 45,
                    "total_tokens": 105,
                    "cached_prompt_tokens": 0,
                },
            }
        return {
            "headline": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி தொடங்குகிறது.",
            "article_body": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி தொடங்க உள்ளது.",
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை திட்டம்",
            "meta_description": "சென்னை வெள்ள எச்சரிக்கை முயற்சி குறித்த செய்தி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 50,
                "total_tokens": 170,
                "cached_prompt_tokens": 0,
            },
        }


class MockEvaluationClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounded_brief_only_factual_source" in user_payload
        return {
            "grounding_score": 86,
            "claim_safety_score": 84,
            "fact_preservation_score": 90,
            "overall_risk": "medium",
            "editorial_readiness": "review_required",
            "unsupported_claims": [
                {
                    "claim": "This pilot will prevent flooding.",
                    "reason": "The brief does not support effectiveness.",
                    "suggested_fix": "Say only that officials announced a pilot.",
                }
            ],
            "overclaim_phrases": [
                {
                    "phrase": "protect residents",
                    "reason": "The brief does not support a protection benefit.",
                    "suggested_fix": "Use neutral wording.",
                }
            ],
            "invented_facts": [],
            "contradictions": [],
            "claims_to_avoid_violations": [
                {
                    "claim": "This pilot will prevent flooding.",
                    "avoid_rule": "Do not claim effectiveness.",
                    "reason": "Effectiveness is explicitly unavailable.",
                }
            ],
            "missing_key_facts": [],
            "preserved_facts": ["18 sensors"],
            "number_date_name_checks": [],
            "rewrite_guidance": [],
            "summary": "Mostly grounded.",
            "token_usage": {
                "prompt_tokens": 90,
                "completion_tokens": 35,
                "total_tokens": 125,
                "cached_prompt_tokens": 0,
            },
        }


def _repository_with_profiles(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
    for author_id in ("v_vasanthi", "hema_vandhana"):
        repository.save_author_style_profile(
            AuthorStyleProfileRecord(
                profile_id=f"profile-{author_id}",
                author_id=author_id,
                snapshot_id=f"snapshot-{author_id}",
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
    return repository


def _source_text() -> str:
    return """
    Advertisement
    Chennai city officials said a new flood-warning pilot will begin next month.
    The civic body said 18 sensors will be installed near low-lying streets.
    Subscribe
    """
