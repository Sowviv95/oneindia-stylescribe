from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db.repository import DraftEvaluationRecord, StyleScribeRepository
from backend.app.main import app
from backend.app.models.draft_evaluation_models import DraftEvaluationResponse

client = TestClient(app)


def test_create_draft_grounding_evaluation_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = DraftEvaluationResponse(
        evaluation_id="evaluation-api",
        draft_id="draft-1",
        brief_id="brief-1",
        author_id="v_vasanthi",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        evaluation={"overall_risk": "high"},
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "backend.app.main.evaluate_draft_grounding",
        lambda draft_id: expected,
    )

    response = client.post("/drafts/draft-1/evaluate-grounding")

    assert response.status_code == 200
    assert response.json()["evaluation_id"] == "evaluation-api"


def test_latest_draft_grounding_evaluation_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    monkeypatch.setenv("STYLESCRIBE_DB_PATH", str(db_path))
    get_settings.cache_clear()
    repository = StyleScribeRepository(db_path)
    repository.initialize_schema()
    repository.save_draft_evaluation(
        DraftEvaluationRecord(
            evaluation_id="evaluation-1",
            draft_id="draft-1",
            brief_id="brief-1",
            author_id="v_vasanthi",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            evaluation_json=StyleScribeRepository.encode_json(
                {"overall_risk": "medium", "editorial_readiness": "review_required"}
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    response = client.get("/drafts/draft-1/evaluation/latest")

    assert response.status_code == 200
    assert response.json()["evaluation"]["overall_risk"] == "medium"
