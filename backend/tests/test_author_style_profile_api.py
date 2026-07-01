from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db.repository import StyleScribeRepository, StyleSnapshotRecord
from backend.app.main import app
from backend.app.models.style_profile_models import AuthorStyleProfileResponse

client = TestClient(app)


def test_create_style_profile_endpoint_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = AuthorStyleProfileResponse(
        profile_id="profile-api",
        author_id="v_vasanthi",
        snapshot_id="snapshot-api",
        language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        profile={"overall_tone": "Measured."},
        source_excerpt_refs=[],
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "backend.app.main.generate_author_style_profile",
        lambda author_id: expected,
    )

    response = client.post("/authors/v_vasanthi/style-profile")

    assert response.status_code == 200
    body = response.json()
    assert body["profile_id"] == "profile-api"
    assert body["model_provider"] == "openai"


def test_latest_style_profile_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    _set_db_path(monkeypatch, db_path)
    repository = StyleScribeRepository(db_path)
    repository.initialize_schema()
    repository.save_style_snapshot(_snapshot())

    from backend.app.services.author_style_profile_service import (
        generate_author_style_profile,
    )
    from backend.tests.test_author_style_profile_service import MockProfileClient

    generate_author_style_profile("v_vasanthi", repository, MockProfileClient())

    response = client.get("/authors/v_vasanthi/style-profile/latest")

    assert response.status_code == 200
    body = response.json()
    assert body["author_id"] == "v_vasanthi"
    assert body["profile"]["overall_tone"] == "Measured and news-focused."


def test_style_profile_endpoint_returns_404_without_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_db_path(monkeypatch, tmp_path / "stylescribe.db")

    response = client.post("/authors/missing/style-profile")

    assert response.status_code == 404
    assert "Build a style snapshot first" in response.json()["detail"]


def _set_db_path(monkeypatch: pytest.MonkeyPatch, db_path: Path) -> None:
    monkeypatch.setenv("STYLESCRIBE_DB_PATH", str(db_path))
    get_settings.cache_clear()


def _snapshot() -> StyleSnapshotRecord:
    return StyleSnapshotRecord(
        snapshot_id="snapshot-api",
        author_id="v_vasanthi",
        article_count=1,
        language="ta",
        status="completed",
        stats_json=StyleScribeRepository.encode_json({"article_count": 1}),
        excerpt_pack_json=StyleScribeRepository.encode_json(
            {
                "intro_examples": [
                    {
                        "article_id": "a1",
                        "filename": "article-one.docx",
                        "title_or_heading": "Article One",
                        "category": "Politics",
                        "excerpt_type": "intro",
                        "char_count": 21,
                        "excerpt_text": "Short bounded excerpt",
                    }
                ]
            }
        ),
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
