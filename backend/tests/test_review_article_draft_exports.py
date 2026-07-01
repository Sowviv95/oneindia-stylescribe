from backend.app.models.article_draft_models import ArticleDraftResponse
from backend.app.scripts import review_article_draft


def test_review_article_draft_markdown_and_html_exports(
    tmp_path,
    monkeypatch,
) -> None:
    markdown_path = tmp_path / "review_outputs" / "draft_review.md"
    html_path = tmp_path / "review_outputs" / "draft_review.html"
    monkeypatch.setattr(
        review_article_draft,
        "load_review_context",
        lambda draft_id: _review_context(),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_article_draft",
            "--draft-id",
            "draft-1",
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
        ],
    )
    review_article_draft.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_article_draft",
            "--draft-id",
            "draft-1",
            "--format",
            "html",
            "--output",
            str(html_path),
        ],
    )
    review_article_draft.main()

    markdown = markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "# Article Draft Review" in markdown
    assert "Source Brief Highlights" in markdown
    assert "Style Profile Highlights" in markdown
    assert "Headline" in markdown
    assert "Article body" in markdown
    assert "<!doctype html>" in html
    assert '<meta charset="utf-8">' in html
    assert "Nirmala UI" in html
    assert "Generated Draft" in html


def _review_context() -> dict[str, object]:
    return {
        "draft": ArticleDraftResponse(
            draft_id="draft-1",
            author_id="v_vasanthi",
            profile_id="profile-1",
            brief_id="brief-1",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            article_type="news",
            desired_word_count=500,
            tone_override="clear",
            include_seo=True,
            draft={
                "headline": "Headline",
                "subheadline": "Subheadline",
                "article_body": "Article body",
                "seo_title": "SEO",
                "meta_description": "Meta",
                "suggested_tags": ["tag"],
            },
            warnings=[],
            created_at="2026-01-01T00:00:00+00:00",
        ),
        "brief": {
            "topic": "Flood warning",
            "one_line_summary": "Summary",
            "confirmed_facts": ["Fact"],
            "claims_to_avoid": ["Avoid"],
        },
        "profile": {
            "overall_tone": "Measured",
            "headline_style": "Direct",
            "intro_style": "Context",
            "paragraph_style": "Compact",
            "tamil_register": "Conversational",
            "dos": ["Do"],
            "donts": ["Dont"],
        },
    }
