from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.main import app
from backend.app.models.grounded_brief_models import GroundedBriefResponse

client = TestClient(app)


def test_create_grounded_brief_endpoint_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = GroundedBriefResponse(
        brief_id="brief-api",
        source_type="text",
        source_url=None,
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief={"topic": "Project launch"},
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
        source_text_excerpt="Short source excerpt",
    )
    monkeypatch.setattr(
        "backend.app.main.generate_grounded_brief",
        lambda source_type, source_input, target_language, source_input_mode: expected,
    )

    response = client.post(
        "/briefs/grounded",
        json={
            "source_type": "text",
            "source_input": "This is a source text long enough for testing.",
        },
    )

    assert response.status_code == 200
    assert response.json()["brief_id"] == "brief-api"


def test_get_grounded_brief_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    _set_db_path(monkeypatch, db_path)
    repository = StyleScribeRepository(db_path)
    repository.initialize_schema()
    repository.save_grounded_brief(_brief_record())

    response = client.get("/briefs/brief-1")

    assert response.status_code == 200
    body = response.json()
    assert body["brief_id"] == "brief-1"
    assert body["brief"]["topic"] == "Project launch"


def _set_db_path(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setenv("STYLESCRIBE_DB_PATH", str(db_path))
    get_settings.cache_clear()


def _brief_record() -> GroundedBriefRecord:
    return GroundedBriefRecord(
        brief_id="brief-1",
        source_type="text",
        source_input_hash="hash",
        source_url=None,
        source_text_excerpt="Short source excerpt",
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief_json=StyleScribeRepository.encode_json({"topic": "Project launch"}),
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
