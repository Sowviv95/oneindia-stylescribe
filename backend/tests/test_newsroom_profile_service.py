import json
from pathlib import Path

from backend.app.services.newsroom_profile_service import (
    NewsroomProfilePathConfig,
    build_pattern_conclusions,
    extract_phrase_records,
    run_newsroom_profile_analysis,
)


def test_pattern_statistics_are_deterministic() -> None:
    articles = [
        _article("a1", "author_a", "Chennai: alpha beta 123.\nContext said more."),
        _article("a2", "author_b", "Delhi: alpha beta 456.\nEarlier context."),
        _article("a3", "author_c", "Madurai: alpha beta.\nMore context."),
    ]

    first = build_pattern_conclusions(articles)
    second = build_pattern_conclusions(articles)

    assert first == second
    opening = next(item for item in first if item.pattern_id == "opening_place_prefix")
    assert opening.frequency == 3
    assert opening.confidence == "high"


def test_phrase_extraction_prefers_cross_author_phrases() -> None:
    articles = [
        _article("a1", "author_a", "shared phrase appears here. shared phrase again."),
        _article("a2", "author_b", "shared phrase appears here too."),
        _article("a3", "author_c", "shared phrase appears here also."),
    ]

    preferred, review = extract_phrase_records(articles, min_articles=2)

    assert any(record.phrase == "shared phrase" for record in preferred)
    assert all(record.phrase != "shared phrase" for record in review)


def test_author_skew_detection_routes_phrase_to_review() -> None:
    articles = [
        _article("a1", "author_a", "skewed phrase appears here."),
        _article("a2", "author_a", "skewed phrase appears again."),
        _article("a3", "author_a", "skewed phrase appears repeatedly."),
        _article("a4", "author_b", "different wording appears."),
    ]

    preferred, review = extract_phrase_records(articles, min_articles=2)

    assert not any(record.phrase == "skewed phrase" for record in preferred)
    assert any(record.phrase == "skewed phrase" for record in review)


def test_newsroom_profile_outputs_include_traceable_evidence(tmp_path: Path) -> None:
    cleaned_path = tmp_path / "cleaned.jsonl"
    classified_path = tmp_path / "classified.jsonl"
    articles = [
        _article("a1", "author_a", "Chennai: shared phrase 123.\nEarlier context."),
        _article("a2", "author_b", "Delhi: shared phrase 456.\nEarlier context."),
        _article("a3", "author_c", "Madurai: shared phrase 789.\nEarlier context."),
    ]
    cleaned_path.write_text(
        "".join(json.dumps(article, ensure_ascii=False) + "\n" for article in articles),
        encoding="utf-8",
    )
    classified_path.write_text(
        "".join(
            json.dumps(
                {
                    **article,
                    "topic": "politics",
                    "topic_confidence": 0.8,
                    "topic_low_confidence": False,
                    "topic_multi_category_conflict": False,
                    "topic_evidence": ["politics"],
                },
                ensure_ascii=False,
            )
            + "\n"
            for article in articles
        ),
        encoding="utf-8",
    )

    result = run_newsroom_profile_analysis(
        NewsroomProfilePathConfig(
            cleaned_articles_jsonl=cleaned_path,
            classified_articles_jsonl=classified_path,
            reports_dir=tmp_path / "reports",
        )
    )

    evidence_lines = result.output_paths["evidence_jsonl"].read_text(
        encoding="utf-8"
    ).splitlines()
    assert result.summary["accepted_articles_analyzed"] == 3
    assert evidence_lines
    first_evidence = json.loads(evidence_lines[0])
    assert first_evidence["article_id"]
    assert first_evidence["relative_source_path"]


def _article(article_id: str, author_id: str, body_text: str) -> dict[str, object]:
    paragraphs = body_text.splitlines()
    return {
        "article_id": article_id,
        "author_id": author_id,
        "relative_source_path": f"{author_id}/{article_id}.docx",
        "body_text": body_text,
        "paragraph_sequence": paragraphs,
        "topic": "other",
        "topic_confidence": 0.0,
        "topic_low_confidence": True,
        "topic_multi_category_conflict": False,
    }
