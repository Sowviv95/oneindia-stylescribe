"""Curated excerpt pack generation for style profile preparation."""

from backend.app.db.repository import ArticleForAnalysis
from backend.app.services.style_statistics_service import split_paragraphs

MAX_EXCERPT_CHARS = 700


def build_excerpt_pack(articles: list[ArticleForAnalysis]) -> dict[str, object]:
    """Build bounded, deterministic excerpts from ingested articles."""

    sorted_articles = sorted(articles, key=lambda article: article.filename)
    by_length = sorted(sorted_articles, key=lambda article: article.text_char_count)
    return {
        "headline_examples": _headline_examples(sorted_articles[:10]),
        "intro_examples": _paragraph_examples(
            sorted_articles,
            excerpt_type="intro",
            limit=5,
            position="intro",
        ),
        "body_examples": _paragraph_examples(
            sorted_articles,
            excerpt_type="body",
            limit=5,
            position="body",
        ),
        "closing_examples": _paragraph_examples(
            sorted_articles,
            excerpt_type="closing",
            limit=5,
            position="closing",
        ),
        "short_article_examples": _article_excerpt_examples(
            by_length[:3],
            excerpt_type="short_article",
        ),
        "long_article_examples": _article_excerpt_examples(
            list(reversed(by_length[-3:])),
            excerpt_type="long_article",
        ),
    }


def _headline_examples(
    articles: list[ArticleForAnalysis],
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for article in articles:
        headline = article.title or article.heading
        if not headline:
            continue
        examples.append(_example(article, headline, "headline"))
    return examples


def _paragraph_examples(
    articles: list[ArticleForAnalysis],
    excerpt_type: str,
    limit: int,
    position: str,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    for article in articles:
        paragraphs = split_paragraphs(article.extracted_text)
        selected = _select_paragraphs(paragraphs, position)
        if not selected:
            continue
        examples.append(_example(article, "\n".join(selected), excerpt_type))
        if len(examples) >= limit:
            break
    return examples


def _article_excerpt_examples(
    articles: list[ArticleForAnalysis],
    excerpt_type: str,
) -> list[dict[str, object]]:
    return [
        _example(article, article.extracted_text, excerpt_type)
        for article in articles
        if article.extracted_text.strip()
    ]


def _select_paragraphs(paragraphs: list[str], position: str) -> list[str]:
    if not paragraphs:
        return []
    if position == "intro":
        return paragraphs[:2]
    if position == "closing":
        return paragraphs[-2:]
    middle_index = len(paragraphs) // 2
    return paragraphs[middle_index : middle_index + 1]


def _example(
    article: ArticleForAnalysis,
    text: str,
    excerpt_type: str,
) -> dict[str, object]:
    excerpt_text = _truncate(text.strip(), MAX_EXCERPT_CHARS)
    return {
        "article_id": article.article_id,
        "filename": article.filename,
        "title_or_heading": article.title or article.heading,
        "category": article.category,
        "excerpt_text": excerpt_text,
        "excerpt_type": excerpt_type,
        "char_count": len(excerpt_text),
    }


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
