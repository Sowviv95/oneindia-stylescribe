from pathlib import Path

import pytest

from backend.app.db.repository import (
    ArticleRecord,
    AuthorRecord,
    StyleScribeRepository,
)
from backend.app.services.style_snapshot_service import (
    AuthorStyleSnapshotError,
    build_author_style_snapshot,
    get_latest_author_style_snapshot,
)


def test_build_author_style_snapshot_saves_snapshot(tmp_path: Path) -> None:
    repository = _repository_with_article(tmp_path)

    response = build_author_style_snapshot("v_vasanthi", repository)
    latest = get_latest_author_style_snapshot("v_vasanthi", repository)

    assert response.author_id == "v_vasanthi"
    assert response.article_count == 1
    assert response.language == "ta"
    assert response.status == "completed"
    assert response.stats["article_count"] == 1
    assert response.excerpt_pack["intro_examples"]
    assert latest.snapshot_id == response.snapshot_id


def test_build_author_style_snapshot_errors_without_articles(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()

    with pytest.raises(AuthorStyleSnapshotError):
        build_author_style_snapshot("missing", repository)


def _repository_with_article(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
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
    return repository
