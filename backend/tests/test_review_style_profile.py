from backend.app.models.style_profile_models import AuthorStyleProfileResponse
from backend.app.scripts import review_style_profile


def test_review_helper_formats_bounded_output(
    capsys: object,
    monkeypatch: object,
) -> None:
    long_excerpt = "A" * 900
    profile = AuthorStyleProfileResponse(
        profile_id="profile-1",
        author_id="v_vasanthi",
        snapshot_id="snapshot-1",
        language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        profile={"overall_tone": "Measured.", "dos": ["Be concise."]},
        source_excerpt_refs=[
            {
                "filename": "article-one.docx",
                "title_or_heading": "Article One",
                "category": "Politics",
                "excerpt_type": "intro",
                "excerpt_text": long_excerpt,
            }
        ],
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
    )

    monkeypatch.setattr(
        review_style_profile,
        "get_latest_author_style_profile",
        lambda author_id: profile,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["review_style_profile", "--author-id", "v_vasanthi", "--limit", "1"],
    )

    review_style_profile.main()

    output = capsys.readouterr().out
    assert "Profile ID: profile-1" in output
    assert "article-one.docx" in output
    assert "Generated style profile" in output
    assert "A" * 800 not in output
