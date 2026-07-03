import pytest
from fastapi.testclient import TestClient

from backend.app.db.repository import DraftEvaluationRecord, StyleScribeRepository
from backend.app.main import app
from backend.app.models.article_revision_models import (
    ArticleRevisionResponse,
)

client = TestClient(app)


def test_create_article_grounding_revision_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = ArticleRevisionResponse(
        revision_id="revision-1",
        draft_id="draft-1",
        evaluation_id="evaluation-1",
        author_id="v_vasanthi",
        model_provider="openai",
        model_name="gpt-4o-mini",
        revised_draft={"headline": "Revised"},
        revision_summary="Removed unsupported claims.",
        removed_or_softened_claims=["பாதுகாப்பு உறுதி"],
        token_usage={},
        created_at="2026-01-01T00:00:00+00:00",
        warnings=[],
    )

    monkeypatch.setattr(
        "backend.app.main.revise_article_grounding",
        lambda draft_id, evaluation_id, repository: revision,
    )
    monkeypatch.setattr("backend.app.main.StyleScribeRepository", FakeRepository)

    response = client.post(
        "/drafts/draft-1/revise-grounding",
        json={"evaluation_id": "evaluation-1"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["revision"]["revision_id"] == "revision-1"
    assert payload["revision"]["revised_draft"]["headline"] == "Revised"


def test_latest_article_grounding_revision_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = ArticleRevisionResponse(
        revision_id="revision-1",
        draft_id="draft-1",
        evaluation_id="evaluation-1",
        author_id="v_vasanthi",
        model_provider="openai",
        model_name="gpt-4o-mini",
        revised_draft={"headline": "Revised"},
        revision_summary="Summary",
        removed_or_softened_claims=[],
        token_usage={},
        created_at="2026-01-01T00:00:00+00:00",
        warnings=[],
    )
    monkeypatch.setattr(
        "backend.app.main.get_latest_article_revision",
        lambda draft_id: expected,
    )

    response = client.get("/drafts/draft-1/revision/latest")

    assert response.status_code == 200
    assert response.json()["revision_id"] == "revision-1"


class FakeRepository:
    decode_json_object = staticmethod(StyleScribeRepository.decode_json_object)

    def initialize_schema(self) -> None:
        pass

    def fetch_draft_evaluation(self, evaluation_id: str) -> DraftEvaluationRecord:
        return DraftEvaluationRecord(
            evaluation_id=evaluation_id,
            draft_id="draft-1",
            brief_id="brief-1",
            author_id="v_vasanthi",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            evaluation_json=StyleScribeRepository.encode_json(
                {"overall_risk": "high", "editorial_readiness": "revision_required"}
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
