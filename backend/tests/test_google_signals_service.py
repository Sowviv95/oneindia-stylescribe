from backend.app.models.google_signals_models import (
    GoogleSignalsComponentConfig,
    GoogleSignalsScore,
    GoogleSignalsScoringConfig,
)
from backend.app.services.google_signals_service import (
    GOOGLE_SIGNALS_V1_CONFIG,
    build_google_signals_input,
    build_google_signals_score,
    evaluate_google_signals,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_google_signals_weighted_average() -> None:
    result = build_google_signals_score(
        {
            "components": [
                {
                    "name": "search_intent_clarity",
                    "score": 80,
                    "rationale": "Clear topic.",
                    "risk_level": "low",
                },
                {
                    "name": "headline_search_clarity",
                    "score": 60,
                    "rationale": "Headline could be more specific.",
                    "risk_level": "medium",
                },
                {"name": "freshness_timeliness", "score": 70},
                {"name": "originality_angle", "score": 90},
                {"name": "eeat_trust", "score": 100},
                {"name": "snippet_meta_readiness", "score": 50},
                {"name": "structured_data_readiness", "score": 40},
            ],
            "risk_flags": ["Headline is broad."],
            "recommendations": ["Make the headline more specific."],
            "metadata": {"primary_search_intent": "flood warning pilot"},
        }
    )

    assert result.score == 73
    assert result.version == "google_signals_v1"
    assert result.risk_flags == ["Headline is broad."]
    assert result.metadata["schema_type"] == "NewsArticle"


def test_google_signals_missing_and_disabled_components() -> None:
    config = GoogleSignalsScoringConfig(
        version="google_signals_test",
        components=[
            GoogleSignalsComponentConfig(name="enabled_present", weight=75),
            GoogleSignalsComponentConfig(name="enabled_missing", weight=25),
            GoogleSignalsComponentConfig(
                name="disabled_component",
                weight=100,
                enabled=False,
            ),
        ],
    )

    result = build_google_signals_score(
        {
            "components": [
                {"name": "enabled_present", "score": 80},
                {"name": "disabled_component", "score": 100},
            ],
        },
        config,
    )

    assert result.score == 60
    assert result.components[1].name == "enabled_missing"
    assert result.components[1].score == 0
    assert result.components[1].risk_level == "high"
    assert result.components[2].enabled is False


def test_google_signals_vague_headline_cap_recalculates_overall() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            risk_flags=["vague_headline"],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert _component_score(result, "headline_search_clarity") == 55
    assert result.score == 85
    assert result.metadata["applied_score_caps"][0]["reason"] == "vague_headline"


def test_google_signals_weak_first_paragraph_caps_intent_and_snippet() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            risk_flags=["weak first paragraph"],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert _component_score(result, "search_intent_clarity") == 65
    assert _component_score(result, "snippet_meta_readiness") == 65


def test_google_signals_thin_content_caps_snippet_and_structured_data() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            risk_flags=["thin_content"],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert _component_score(result, "snippet_meta_readiness") == 60
    assert _component_score(result, "structured_data_readiness") == 55


def test_google_signals_unsupported_claims_cap_eeat() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            risk_flags=["unsupported claims"],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert _component_score(result, "eeat_trust") == 65


def test_google_signals_generic_content_caps_originality() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            risk_flags=["generic_rewritten_content"],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert _component_score(result, "originality_angle") == 65


def test_google_signals_missing_primary_intent_and_entity_caps_scores() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            metadata={
                "primary_search_intent": "unclear",
                "primary_entities": [],
            },
        )
    )

    assert _component_score(result, "search_intent_clarity") == 60
    assert _component_score(result, "structured_data_readiness") == 55


def test_google_signals_unsafe_recommendations_are_sanitized() -> None:
    result = build_google_signals_score(
        _raw_google_signals_evaluation(
            recommendations=[
                "Add expert quotes to improve authority.",
                "Add recent developments for freshness.",
                "Add statistics if present in the source brief.",
                "Make the headline more specific.",
                "Consider adding expert quotes or citations to improve trust signals.",
                "Add more specific details or expert quotes.",
                "Consider adding recent developments or updates.",
                "Incorporate more direct quotes or expert opinions.",
                "Include specific dates or references.",
                "Enhance originality by adding unique insights or context.",
                (
                    "Include more specific details or quotes from stakeholders to "
                    "improve trust signals."
                ),
                (
                    "Consider adding more specific details in the article body to "
                    "improve structured data readiness."
                ),
                (
                    "Consider adding more specific details about the survey "
                    "methodology to improve trust signals."
                ),
                (
                    "Clarify the structured data elements, including publication "
                    "date and author information."
                ),
            ],
            metadata={
                "primary_search_intent": "Chennai Metro service change",
                "primary_entities": ["Chennai Metro"],
            },
        )
    )

    assert "Add expert quotes to improve authority." not in result.recommendations
    assert "Add recent developments for freshness." not in result.recommendations
    assert (
        "Consider adding expert quotes or citations to improve trust signals."
        not in result.recommendations
    )
    assert "Add more specific details or expert quotes." not in result.recommendations
    assert (
        "Consider adding recent developments or updates."
        not in result.recommendations
    )
    assert (
        "Incorporate more direct quotes or expert opinions."
        not in result.recommendations
    )
    assert "Include specific dates or references." not in result.recommendations
    assert (
        "Enhance originality by adding unique insights or context."
        not in result.recommendations
    )
    assert (
        "Include more specific details or quotes from stakeholders to improve trust "
        "signals."
        not in result.recommendations
    )
    assert (
        "Consider adding more specific details in the article body to improve "
        "structured data readiness."
        not in result.recommendations
    )
    assert (
        "Consider adding more specific details about the survey methodology to "
        "improve trust signals."
        not in result.recommendations
    )
    assert (
        "Clarify the structured data elements, including publication date and "
        "author information."
        not in result.recommendations
    )
    assert "Add statistics if present in the source brief." in result.recommendations
    assert (
        "Use source-provided details more clearly if present; do not add new "
        "details unless they are in the source brief."
    ) in result.recommendations
    assert (
        "Surface stakeholder views only if they are present in the source brief; "
        "do not add new quotes."
    ) in result.recommendations
    assert (
        "Clarify methodology only using details already present in the source."
    ) in result.recommendations
    assert (
        "Populate structured data only from available workflow metadata or "
        "source-grounded fields."
    ) in result.recommendations
    assert (
        "Surface source-provided expert attribution more clearly if present; "
        "do not add new quotes or citations unless present in the source brief."
    ) in result.recommendations
    assert (
        "Clarify the available freshness cues from the source; do not add "
        "recent developments unless present in the source brief."
    ) in result.recommendations
    assert (
        "Strengthen originality using the source-grounded angle; do not add "
        "unsupported context."
    ) in result.recommendations
    assert (
        "Do not add new facts, quotes, citations, statistics, dates, methodology, "
        "metadata, or developments unless present in the source brief or workflow "
        "metadata."
    ) in result.recommendations
    assert (
        "Strengthen this section using available grounded details, not external "
        "additions."
    ) in result.recommendations
    assert "Make the headline more specific." in result.recommendations


def test_google_signals_evaluator_fallback_on_invalid_response() -> None:
    result = evaluate_google_signals(
        final_article=_article(),
        grounded_brief=_brief(),
        author_id="v_vasanthi",
        article_type="news",
        target_language="ta",
        model_client=InvalidGoogleSignalsClient(),
    )

    assert result.available is False
    assert result.google_signals is None
    assert result.error is not None
    assert "component list" in result.error


def test_google_signals_prompt_input_is_grounded_and_versioned() -> None:
    payload = build_google_signals_input(
        final_article=_article(),
        grounded_brief=_brief(),
        author_id="v_vasanthi",
        article_type="news",
        target_language="ta",
        desired_word_count=600,
        workflow_metadata={"final_article_source_stage": "draft"},
        config=GOOGLE_SIGNALS_V1_CONFIG,
    )

    assert "google_signals_v1" in payload
    assert "grounded_brief_only_factual_source" in payload
    assert "generated_article_to_evaluate" in payload
    assert "Do not use outside knowledge" in payload
    assert "FULL_SOURCE_ARTICLE_TEXT" not in payload


class InvalidGoogleSignalsClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        return {"components": "not-a-list"}


class FailingGoogleSignalsClient(InvalidGoogleSignalsClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")


def _raw_google_signals_evaluation(
    *,
    risk_flags: list[str] | None = None,
    recommendations: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "components": [
            {"name": "search_intent_clarity", "score": 90},
            {"name": "headline_search_clarity", "score": 90},
            {"name": "freshness_timeliness", "score": 90},
            {"name": "originality_angle", "score": 90},
            {"name": "eeat_trust", "score": 90},
            {"name": "snippet_meta_readiness", "score": 90},
            {"name": "structured_data_readiness", "score": 90},
        ],
        "risk_flags": risk_flags or [],
        "recommendations": recommendations or [],
        "metadata": metadata or {},
    }


def _component_score(result: GoogleSignalsScore, name: str) -> int:
    for component in result.components:
        if component.name == name:
            return component.score
    raise AssertionError(f"Missing component: {name}")


def _article() -> dict[str, object]:
    return {
        "headline": "Chennai flood warning pilot",
        "subheadline": "18 sensors will be installed next month.",
        "article_body": "Chennai officials announced a flood-warning pilot.",
        "seo_title": "Chennai flood warning pilot",
        "meta_description": "Chennai flood warning pilot details.",
        "suggested_tags": ["Chennai", "Flood warning"],
    }


def _brief() -> dict[str, object]:
    return {
        "topic": "Flood warning pilot",
        "one_line_summary": "Chennai officials announced a pilot.",
        "confirmed_facts": ["A flood-warning pilot will begin next month."],
        "claims_to_avoid": ["Do not claim effectiveness."],
    }
