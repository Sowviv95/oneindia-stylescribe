from scripts.run_live_workflow import _render_html_output


def test_manual_html_output_includes_generated_headline() -> None:
    html, missing = _render_html_output(
        {
            "workflow_completed": True,
            "generated_headline": "விசா விதிகளில் புதிய மாற்றம்",
            "generated_subheadline": "H-1B தொடர்பான விவரங்கள் விளக்கம்.",
            "final_article": "தமிழ் கட்டுரை உடல்.",
            "final_publication_blockers": ["review_needed"],
            "final_publication_warnings": ["editor_check"],
        },
        {"source_text": "English source headline\nSource body"},
    )

    assert missing == []
    assert "விசா விதிகளில் புதிய மாற்றம்" in html
    assert "H-1B தொடர்பான விவரங்கள் விளக்கம்." in html
    assert "தமிழ் கட்டுரை உடல்." in html


def test_manual_html_output_falls_back_to_source_headline() -> None:
    html, missing = _render_html_output(
        {
            "workflow_completed": True,
            "final_article": "தமிழ் கட்டுரை உடல்.",
            "final_publication_blockers": [],
            "final_publication_warnings": [],
        },
        {"source_text": "Source headline fallback\nSource body"},
    )

    assert missing == []
    assert "Source headline fallback" in html
    assert "Headline source: source_text.first_line" in html
