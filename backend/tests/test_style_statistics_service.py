from backend.app.db.repository import ArticleForAnalysis
from backend.app.services.style_statistics_service import (
    calculate_char_ratios,
    calculate_style_statistics,
    split_paragraphs,
    split_sentences,
)


def test_split_paragraphs_removes_empty_lines() -> None:
    assert split_paragraphs("Intro\n\n  \nBody\nClosing") == [
        "Intro",
        "Body",
        "Closing",
    ]


def test_split_sentences_uses_common_and_indic_punctuation() -> None:
    assert split_sentences("One. Two? மூன்று! நான்கு। ஐந்து॥") == [
        "One",
        "Two",
        "மூன்று",
        "நான்கு",
        "ஐந்து",
    ]


def test_calculate_char_ratios_counts_tamil_and_latin() -> None:
    ratios = calculate_char_ratios("தமிழ் ABC 123")

    assert ratios["approximate_tamil_char_ratio"] == 0.4545
    assert ratios["approximate_latin_char_ratio"] == 0.2727


def test_calculate_style_statistics() -> None:
    articles = [
        _article(
            article_id="a1",
            filename="a1.docx",
            title="Title One",
            heading="Heading One",
            category="Politics",
            text="தமிழ் intro?\nBody paragraph: quote \"yes\".\nClosing!",
        ),
        _article(
            article_id="a2",
            filename="a2.docx",
            title=None,
            heading="Heading Two",
            category="Sports",
            text="English intro.\nSecond paragraph...",
        ),
    ]

    stats = calculate_style_statistics(articles)
    char_counts = [article.text_char_count for article in articles]

    assert stats["article_count"] == 2
    assert stats["total_char_count"] == sum(char_counts)
    assert stats["min_char_count"] == min(char_counts)
    assert stats["max_char_count"] == max(char_counts)
    assert stats["title_available_count"] == 1
    assert stats["heading_available_count"] == 2
    assert stats["category_distribution"] == {"Politics": 1, "Sports": 1}
    assert stats["question_mark_count"] == 1
    assert stats["exclamation_mark_count"] == 1
    assert stats["colon_count"] == 1
    assert stats["quote_count"] == 2
    assert stats["ellipsis_count"] == 1


def _article(
    article_id: str,
    filename: str,
    title: str | None,
    heading: str | None,
    category: str,
    text: str,
) -> ArticleForAnalysis:
    return ArticleForAnalysis(
        article_id=article_id,
        author_id="author",
        filename=filename,
        title=title,
        heading=heading,
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
