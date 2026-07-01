"""Deterministic writing-style statistics for ingested articles."""

from __future__ import annotations

import re
from collections import Counter

from backend.app.db.repository import ArticleForAnalysis

SENTENCE_SPLIT_PATTERN = re.compile(r"[.?!।॥\u0964\u0965؟]+")
TAMIL_CHAR_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")


def split_paragraphs(text: str) -> list[str]:
    """Split text into non-empty paragraphs."""

    return [paragraph.strip() for paragraph in text.splitlines() if paragraph.strip()]


def split_sentences(text: str) -> list[str]:
    """Approximate sentence splitting using Tamil and common punctuation."""

    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT_PATTERN.split(text)
        if sentence.strip()
    ]


def calculate_char_ratios(text: str) -> dict[str, float]:
    """Calculate approximate Tamil and Latin character ratios."""

    non_space_chars = [character for character in text if not character.isspace()]
    total = len(non_space_chars)
    if total == 0:
        return {
            "approximate_tamil_char_ratio": 0.0,
            "approximate_latin_char_ratio": 0.0,
        }

    tamil_count = len(TAMIL_CHAR_PATTERN.findall(text))
    latin_count = len(LATIN_CHAR_PATTERN.findall(text))
    return {
        "approximate_tamil_char_ratio": round(tamil_count / total, 4),
        "approximate_latin_char_ratio": round(latin_count / total, 4),
    }


def calculate_style_statistics(
    articles: list[ArticleForAnalysis],
) -> dict[str, object]:
    """Calculate deterministic style statistics from ingested article text."""

    article_count = len(articles)
    char_counts = [article.text_char_count for article in articles]
    all_text = "\n".join(article.extracted_text for article in articles)
    paragraph_lists = [split_paragraphs(article.extracted_text) for article in articles]
    all_paragraphs = [
        paragraph for paragraphs in paragraph_lists for paragraph in paragraphs
    ]
    sentence_lists = [split_sentences(article.extracted_text) for article in articles]
    all_sentences = [sentence for sentences in sentence_lists for sentence in sentences]
    title_lengths = [len(article.title) for article in articles if article.title]
    heading_lengths = [len(article.heading) for article in articles if article.heading]
    category_distribution = Counter(
        article.category for article in articles if article.category
    )
    intro_lengths = [
        len(paragraphs[0]) for paragraphs in paragraph_lists if paragraphs
    ]
    closing_lengths = [
        len(paragraphs[-1]) for paragraphs in paragraph_lists if paragraphs
    ]
    char_ratios = calculate_char_ratios(all_text)

    return {
        "article_count": article_count,
        "total_char_count": sum(char_counts),
        "average_char_count": _average(char_counts),
        "min_char_count": min(char_counts, default=0),
        "max_char_count": max(char_counts, default=0),
        "average_paragraph_count": _average(
            [len(paragraphs) for paragraphs in paragraph_lists]
        ),
        "average_paragraph_char_count": _average(
            [len(paragraph) for paragraph in all_paragraphs]
        ),
        "short_paragraph_ratio": _ratio(
            len([paragraph for paragraph in all_paragraphs if len(paragraph) < 120]),
            len(all_paragraphs),
        ),
        "long_paragraph_ratio": _ratio(
            len([paragraph for paragraph in all_paragraphs if len(paragraph) > 500]),
            len(all_paragraphs),
        ),
        "average_sentence_count": _average(
            [len(sentences) for sentences in sentence_lists]
        ),
        "average_sentence_char_count": _average(
            [len(sentence) for sentence in all_sentences]
        ),
        "title_available_count": len(title_lengths),
        "heading_available_count": len(heading_lengths),
        "average_title_char_count": _average(title_lengths),
        "average_heading_char_count": _average(heading_lengths),
        "category_distribution": dict(sorted(category_distribution.items())),
        "average_intro_paragraph_char_count": _average(intro_lengths),
        "average_closing_paragraph_char_count": _average(closing_lengths),
        "question_mark_count": all_text.count("?") + all_text.count("؟"),
        "exclamation_mark_count": all_text.count("!"),
        "colon_count": all_text.count(":"),
        "semicolon_count": all_text.count(";"),
        "quote_count": _quote_count(all_text),
        "ellipsis_count": all_text.count("...") + all_text.count("…"),
        **char_ratios,
    }


def _average(values: list[int]) -> float:
    if not values:
        return 0.0
    return round(sum(values) / len(values), 2)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _quote_count(text: str) -> int:
    quote_characters = ['"', "'", "“", "”", "‘", "’", "«", "»"]
    return sum(text.count(character) for character in quote_characters)
