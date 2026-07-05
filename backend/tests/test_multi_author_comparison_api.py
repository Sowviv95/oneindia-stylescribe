from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.multi_author_comparison_models import (
    AuthorComparisonOutput,
    MultiAuthorComparisonResponse,
    MultiAuthorComparisonSummary,
    SharedGroundedBriefMetadata,
)
from backend.app.models.pasted_text_workflow_models import (
    SourceCleanupSummary,
    WorkflowBriefSummary,
)


def test_multi_author_comparison_endpoint(monkeypatch: object) -> None:
    def fake_workflow(**kwargs: object) -> MultiAuthorComparisonResponse:
        assert kwargs["author_id_a"] == "v_vasanthi"
        assert kwargs["author_id_b"] == "hema_vandhana"
        assert kwargs["workflow_mode"] == "fast_review"
        return MultiAuthorComparisonResponse(
            workflow_id="workflow-compare-1",
            workflow_completed=True,
            status="completed",
            desired_word_count=600,
            target_min_word_count=450,
            target_max_word_count=690,
            workflow_mode="fast_review",
            source_cleanup=SourceCleanupSummary(
                original_char_count=100,
                cleaned_char_count=90,
                removed_line_count=1,
                warnings=[],
            ),
            brief_summary=WorkflowBriefSummary(
                topic="Flood warning",
                one_line_summary="Summary",
                confirmed_facts=["Fact"],
                claims_to_avoid=[],
            ),
            shared_grounded_brief=SharedGroundedBriefMetadata(
                brief_id="brief-1",
                source_language="en",
                target_language="ta",
                model_provider="openai",
                model_name="gpt-4o-mini",
                status="completed",
                source_text_excerpt="Source",
            ),
            author_a=_author_output("v_vasanthi", "author_a"),
            author_b=_author_output("hema_vandhana", "author_b"),
            comparison_summary=MultiAuthorComparisonSummary(
                factual_faithfulness_comparison="Similar.",
                author_style_difference="Separate author profiles used.",
                readability_difference="Similar length.",
                recommended_draft="no_clear_recommendation",
                recommendation_rationale="Scores are close.",
            ),
        )

    monkeypatch.setattr(
        "backend.app.main.run_multi_author_comparison_workflow",
        fake_workflow,
    )
    client = TestClient(app)

    response = client.post(
        "/workflows/multi-author-comparison",
        json={
            "source_text": "Chennai officials said enough facts here.",
            "author_id_a": "v_vasanthi",
            "author_id_b": "hema_vandhana",
            "desired_word_count": 600,
            "workflow_mode": "fast_review",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_id"] == "workflow-compare-1"
    assert payload["shared_grounded_brief"]["brief_id"] == "brief-1"
    assert payload["author_a"]["author_id"] == "v_vasanthi"
    assert payload["author_b"]["author_id"] == "hema_vandhana"
    assert "OPENAI_API_KEY" not in response.text


def test_multi_author_comparison_endpoint_requires_two_authors() -> None:
    client = TestClient(app)

    response = client.post(
        "/workflows/multi-author-comparison",
        json={
            "source_text": "Chennai officials said enough facts here.",
            "author_id_a": "v_vasanthi",
        },
    )

    assert response.status_code == 422


def _author_output(author_id: str, role: str) -> AuthorComparisonOutput:
    return AuthorComparisonOutput(
        author_id=author_id,
        role=role,  # type: ignore[arg-type]
        profile_id=f"profile-{author_id}",
        draft_id=f"draft-{author_id}",
        evaluation_id=f"evaluation-{author_id}",
        generated_headline="Headline",
        generated_subheadline="Subheadline",
        article_body="Article body",
        word_count=500,
        grounding_score=86,
        final_readiness="review_required",
        blockers=[],
        warnings=[],
    )
