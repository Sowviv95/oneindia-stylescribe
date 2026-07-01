from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.config import get_settings
from backend.app.db.repository import ArticleRecord, AuthorRecord, StyleScribeRepository
from backend.app.main import app

client = TestClient(app)


def test_create_and_get_latest_style_snapshot_endpoint(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    _set_db_path(monkeypatch, db_path)
    _seed_article(db_path)

    create_response = client.post("/authors/v_vasanthi/style-snapshot")

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["author_id"] == "v_vasanthi"
    assert created["article_count"] == 1
    assert created["language"] == "ta"
    assert created["status"] == "completed"
    assert created["stats"]["article_count"] == 1
    assert created["excerpt_pack"]["intro_examples"]

    latest_response = client.get("/authors/v_vasanthi/style-snapshot/latest")

    assert latest_response.status_code == 200
    latest = latest_response.json()
    assert latest["snapshot_id"] == created["snapshot_id"]
    assert latest["excerpt_pack"]["intro_examples"]


def test_create_style_snapshot_endpoint_returns_404_without_articles(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    _set_db_path(monkeypatch, db_path)

    response = client.post("/authors/missing/style-snapshot")

    assert response.status_code == 404
    assert "No ingested articles found" in response.json()["detail"]


def test_latest_style_snapshot_endpoint_returns_404_without_snapshot(
    tmp_path: Path,
    monkeypatch: object,
) -> None:
    db_path = tmp_path / "stylescribe.db"
    _set_db_path(monkeypatch, db_path)

    response = client.get("/authors/missing/style-snapshot/latest")

    assert response.status_code == 404
    assert "No style snapshot found" in response.json()["detail"]


def _set_db_path(monkeypatch: object, db_path: Path) -> None:
    monkeypatch.setenv("STYLESCRIBE_DB_PATH", str(db_path))
    get_settings.cache_clear()


def _seed_article(db_path: Path) -> None:
    repository = StyleScribeRepository(db_path)
    repository.initialize_schema()
    repository.upsert_author(
        AuthorRecord(
            author_id="v_vasanthi",
            display_name="V Vasanthi",
            language="ta",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
    text = "தமிழ் intro?\nBody paragraph.\nClosing paragraph!"
    repository.upsert_article(
        ArticleRecord(
            article_id="v_vasanthi:article-one",
            author_id="v_vasanthi",
            filename="article-one.docx",
            title="Article One",
            heading="Heading One",
            url="https://example.com/article-one",
            category="Politics",
            tags=None,
            keywords=None,
            meta_description=None,
            added_date=None,
            content_from_metadata=None,
            extracted_text=text,
            text_char_count=len(text),
            source_path="synthetic/article-one.docx",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )
