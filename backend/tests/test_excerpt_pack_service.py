from backend.app.db.repository import ArticleForAnalysis
from backend.app.services.excerpt_pack_service import (
    MAX_EXCERPT_CHARS,
    build_excerpt_pack,
)


def test_build_excerpt_pack_selects_bounded_examples() -> None:
    articles = [
        _article("a1", "a1.docx", "Title One", "Politics", "P1\nP2\nP3"),
        _article("a2", "a2.docx", "Title Two", "Sports", "A" * 1000),
    ]

    pack = build_excerpt_pack(articles)

    assert len(pack["headline_examples"]) == 2
    assert len(pack["intro_examples"]) == 2
    assert len(pack["body_examples"]) == 2
    assert len(pack["closing_examples"]) == 2
    assert len(pack["short_article_examples"]) == 2
    assert len(pack["long_article_examples"]) == 2
    long_excerpt = pack["long_article_examples"][0]["excerpt_text"]
    assert len(long_excerpt) <= MAX_EXCERPT_CHARS
    assert long_excerpt.endswith("...")


def _article(
    article_id: str,
    filename: str,
    title: str,
    category: str,
    text: str,
) -> ArticleForAnalysis:
    return ArticleForAnalysis(
        article_id=article_id,
        author_id="author",
        filename=filename,
        title=title,
        heading=None,
        url=None,
        category=category,
        tags=None,
        keywords=None,
        meta_description=None,
        added_date=None,
        content_from_metadata=None,
        extracted_text=text,
        text_char_count=len(text),
    )
