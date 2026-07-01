from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db.repository import ArticleDraftRecord, StyleScribeRepository
from backend.app.main import app
from backend.app.models.article_draft_models import ArticleDraftResponse

client = TestClient(app)


def test_create_article_draft_endpoint_success(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = ArticleDraftResponse(
        draft_id="draft-api",
        author_id="v_vasanthi",
        profile_id="profile-1",
        brief_id="brief-1",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        article_type="news",
        desired_word_count=600,
        tone_override=None,
        include_seo=True,
        draft={"headline": "தலைப்பு"},
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(
        "backend.app.main.generate_article_draft",
        lambda author_id,
        brief_id,
        author_instruction,
        target_language,
        article_type,
        desired_word_count,
        tone_override,
        include_seo: expected,
    )

    response = client.post(
        "/drafts/article",
        json={"author_id": "v_vasanthi", "brief_id": "brief-1"},
    )

    assert response.status_code == 200
    assert response.json()["draft_id"] == "draft-api"


def test_get_article_draft_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    monkeypatch.setenv("STYLESCRIBE_DB_PATH", str(db_path))
    get_settings.cache_clear()
    repository = StyleScribeRepository(db_path)
    repository.initialize_schema()
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
            desired_word_count=600,
            tone_override=None,
            include_seo=True,
            draft_json=StyleScribeRepository.encode_json({"headline": "தலைப்பு"}),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )

    response = client.get("/drafts/draft-1")

    assert response.status_code == 200
    assert response.json()["draft"]["headline"] == "தலைப்பு"
