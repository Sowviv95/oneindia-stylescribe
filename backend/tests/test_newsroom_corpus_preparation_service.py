from pathlib import Path

from backend.app.services.newsroom_corpus_preparation_service import (
    PreparationPathConfig,
    classify_topic,
    detect_near_duplicate_clusters,
    normalize_for_comparison,
    prepare_articles,
    profile_article,
    recommend_canonical_article,
    run_newsroom_corpus_preparation,
)


def test_normalize_for_comparison_removes_zero_width_and_folds_latin_case() -> None:
    text = "Hello\u200b WORLD\r\n  தமிழ்\tசெய்தி  "

    assert normalize_for_comparison(text) == "hello world\nதமிழ் செய்தி"


def test_profile_article_flags_suspicious_structure() -> None:
    article = _article(
        article_id="a1",
        headline=" ".join(f"h{i}" for i in range(40)),
        subheadline="Sub",
        body_text="short body",
        paragraphs=[" ".join(f"h{i}" for i in range(40)), "Sub", "short body"],
    )

    profile = profile_article(article)

    assert "long_headline_candidate" in profile.structure_warning_flags
    assert "uncertain_body_boundary" in profile.content_review_flags
    assert profile.headline_word_count == 40
    assert profile.paragraph_count == 3


def test_duplicate_clustering_and_canonical_recommendation() -> None:
    first = _article(
        article_id="a1",
        headline="சென்னை அரசியல் செய்தி",
        body_text="தமிழக அரசு தேர்தல் அறிவிப்பு மக்கள் கூட்டம் விவாதம்",
        paragraphs=["சென்னை அரசியல் செய்தி", "தமிழக அரசு தேர்தல் அறிவிப்பு"],
    )
    second = _article(
        article_id="a2",
        headline="சென்னை அரசியல் செய்தி",
        body_text=(
            "தமிழக அரசு தேர்தல் அறிவிப்பு மக்கள் கூட்டம் விவாதம் கூடுதல் தகவல்"
        ),
        paragraphs=[
            "சென்னை அரசியல் செய்தி",
            "தமிழக அரசு தேர்தல் அறிவிப்பு",
            "கூடுதல் தகவல்",
        ],
    )

    clusters = detect_near_duplicate_clusters([first, second])

    assert len(clusters) == 1
    assert clusters[0].canonical_article_id == "a2"
    assert recommend_canonical_article([first, second]) == "a2"


def test_prepare_articles_rejects_non_canonical_duplicate() -> None:
    shared = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu "
        "nu xi omicron pi rho sigma tau upsilon phi chi psi omega article "
        "news politics district people public update"
    )
    first = _article(article_id="a1", body_text=shared)
    second = _article(
        article_id="a2",
        body_text=f"{shared} கூடுதல் தகவல்",
        paragraphs=["தலைப்பு", shared, "கூடுதல் தகவல்"],
    )
    articles = [first, second]
    profiles = [profile_article(article) for article in articles]
    clusters = detect_near_duplicate_clusters(articles)
    topics = {article["article_id"]: classify_topic(article) for article in articles}

    prepared = prepare_articles(articles, profiles, clusters, topics)

    statuses = {article.article_id: article.preparation_status for article in prepared}
    assert statuses["a1"] == "rejected"
    assert statuses["a2"] in {"accepted", "review_required"}
    rejected = next(article for article in prepared if article.article_id == "a1")
    assert "near_duplicate_non_canonical" in rejected.decision_reasons


def test_classify_topic_returns_evidence_and_review_flag() -> None:
    article = _article(
        headline="தமிழக தேர்தல் அரசியல்",
        body_text="முதல்வர் அமைச்சர் திமுக பாஜக கூட்டணி விவாதம்",
    )

    result = classify_topic(article)

    assert result.topic == "politics"
    assert result.confidence > 0
    assert "திமுக" in result.matched_evidence
    assert result.review_flag is False
    assert result.low_confidence is False


def test_valid_body_with_no_explicit_headline_is_accepted() -> None:
    article = _article(
        headline="",
        body_text=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
            "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
            "public article body with enough usable text"
        ),
    )
    article["headline_status"] = "uncertain"
    article["headline_candidate"] = "alpha beta gamma delta epsilon"
    article["paragraph_sequence"] = [str(article["body_text"])]

    profile = profile_article(article)
    prepared = prepare_articles(
        [article],
        [profile],
        [],
        {str(article["article_id"]): classify_topic(article)},
    )[0]

    assert prepared.preparation_status == "accepted"
    assert prepared.article_usable is True
    assert "no_headline" in prepared.informational_flags
    assert "uncertain_headline_body_boundary" in prepared.structure_warning_flags
    assert prepared.body_text == article["body_text"]


def test_informational_anomalies_do_not_trigger_review() -> None:
    article = _article(
        headline="",
        subheadline="",
        body_text=(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
            "mu nu xi omicron pi rho sigma tau upsilon phi chi psi omega "
            "one two three four five"
        ),
    )
    article["headline_status"] = "not_present"
    article["subheadline_status"] = "not_present"

    profile = profile_article(article)
    prepared = prepare_articles(
        [article],
        [profile],
        [],
        {str(article["article_id"]): classify_topic(article)},
    )[0]

    assert prepared.preparation_status == "accepted"
    assert prepared.informational_flags
    assert prepared.content_review_flags == []


def test_content_review_is_distinct_from_structure_warning() -> None:
    article = _article(
        body_text=(
            "valid article text with enough words for usability "
            "alpha beta gamma delta epsilon zeta eta theta iota kappa "
            "lambda mu nu xi omicron pi rho sigma tau upsilon "
            "<p class=\"gmail-yj6qo\"> <p class=\"gmail-adL\">"
        ),
        paragraphs=[
            "valid article text with enough words for usability",
            "<p class=\"gmail-yj6qo\">",
            "<p class=\"gmail-adL\">",
        ],
    )

    profile = profile_article(article)
    prepared = prepare_articles(
        [article],
        [profile],
        [],
        {str(article["article_id"]): classify_topic(article)},
    )[0]

    assert "malformed_or_mixed_content" in prepared.content_review_flags
    assert prepared.preparation_status == "review_required"


def test_low_evidence_topic_classification_may_return_other() -> None:
    result = classify_topic(
        _article(headline="brief update", body_text="single weak weather mention")
    )

    assert result.topic == "other"
    assert result.low_confidence is True


def test_run_preparation_is_deterministic(tmp_path: Path) -> None:
    articles_path = tmp_path / "articles.jsonl"
    article = _article(
        article_id="a1",
        body_text="முதல்வர் அமைச்சர் திமுக பாஜக கூட்டணி விவாதம்",
    )
    articles_path.write_text(_json_line(article), encoding="utf-8")
    paths = PreparationPathConfig(
        articles_jsonl=articles_path,
        cleaned_dir=tmp_path / "cleaned",
        rejected_dir=tmp_path / "rejected",
        classified_dir=tmp_path / "classified",
        reports_dir=tmp_path / "reports",
    )

    first = run_newsroom_corpus_preparation(paths=paths)
    second = run_newsroom_corpus_preparation(paths=paths)

    assert first.summary == second.summary
    assert first.prepared_articles == second.prepared_articles


def _article(
    *,
    article_id: str = "a1",
    author_id: str = "v_vasanthi",
    headline: str = "தலைப்பு",
    subheadline: str = "துணைத் தலைப்பு",
    body_text: str = "இது அரசியல் செய்தி மக்கள் விவாதம் கூடுதல் தகவல்",
    paragraphs: list[str] | None = None,
) -> dict[str, object]:
    resolved_paragraphs = paragraphs or [headline, subheadline, body_text]
    text = "\n".join(resolved_paragraphs)
    return {
        "article_id": article_id,
        "author_id": author_id,
        "source_filename": f"{article_id}.docx",
        "relative_source_path": f"{author_id}/{article_id}.docx",
        "file_size_bytes": 100,
        "file_sha256": "f" * 64,
        "text_sha256": "t" * 64,
        "extraction_status": "extracted",
        "extraction_warnings": [],
        "headline": headline,
        "headline_status": "inferred" if headline else "not_present",
        "headline_candidate": headline or None,
        "subheadline": subheadline,
        "subheadline_status": "inferred" if subheadline else "not_present",
        "structure_confidence": "medium" if headline else "low",
        "body_text": body_text,
        "paragraph_sequence": resolved_paragraphs,
        "paragraph_metadata": [],
        "total_word_count": len(text.split()),
        "char_count": len(text),
    }


def _json_line(article: dict[str, object]) -> str:
    import json

    return json.dumps(article, ensure_ascii=False) + "\n"
