from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.models.pasted_text_workflow_models import (
    PastedTextWorkflowResponse,
    SourceCleanupSummary,
    WorkflowBriefSummary,
    WorkflowDraftSummary,
    WorkflowEvaluationSummary,
)


def test_pasted_text_workflow_endpoint(monkeypatch: object) -> None:
    def fake_workflow(**kwargs: object) -> PastedTextWorkflowResponse:
        assert kwargs["author_id"] == "v_vasanthi"
        assert kwargs["workflow_mode"] == "fast_review"
        return PastedTextWorkflowResponse(
            workflow_id="workflow-1",
            status="completed",
            author_id="v_vasanthi",
            brief_id="brief-1",
            draft_id="draft-1",
            evaluation_id="evaluation-1",
            source_cleanup=SourceCleanupSummary(
                original_char_count=100,
                cleaned_char_count=80,
                removed_line_count=2,
                warnings=[],
            ),
            brief_summary=WorkflowBriefSummary(
                topic="Flood warning",
                one_line_summary="Summary",
                confirmed_facts=["Fact"],
                claims_to_avoid=["Avoid"],
            ),
            draft_summary=WorkflowDraftSummary(
                headline="Headline",
                subheadline="Subheadline",
                seo_title="SEO",
                tags=["tag"],
            ),
            evaluation_summary=WorkflowEvaluationSummary(
                grounding_score=82,
                claim_safety_score=80,
                overall_risk="medium",
                editorial_readiness="review_required",
            ),
            export_paths=[],
            warnings=[],
        )

    monkeypatch.setattr(
        "backend.app.main.run_pasted_text_to_draft_workflow",
        fake_workflow,
    )
    client = TestClient(app)

    response = client.post(
        "/workflows/pasted-text-to-draft",
        json={
            "author_id": "v_vasanthi",
            "source_text": "Advertisement\nChennai officials said enough facts here.",
            "author_instruction": "Write this as Tamil news.",
            "desired_word_count": 500,
            "workflow_mode": "fast_review",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workflow_id"] == "workflow-1"
    assert payload["brief_id"] == "brief-1"
    assert payload["draft_id"] == "draft-1"
    assert payload["evaluation_id"] == "evaluation-1"
    assert "OPENAI_API_KEY" not in response.text


def test_pasted_text_workflow_endpoint_requires_source_text() -> None:
    client = TestClient(app)

    response = client.post(
        "/workflows/pasted-text-to-draft",
        json={"author_id": "v_vasanthi"},
    )

    assert response.status_code == 422
