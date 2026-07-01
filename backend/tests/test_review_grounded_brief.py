from backend.app.models.grounded_brief_models import GroundedBriefResponse
from backend.app.scripts import review_grounded_brief


def test_review_grounded_brief_helper_output(
    capsys: object,
    monkeypatch: object,
) -> None:
    response = GroundedBriefResponse(
        brief_id="brief-1",
        source_type="text",
        source_url=None,
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief={"topic": "Project launch", "claims_to_avoid": ["Do not invent."]},
        warnings=["Source language detection returned unknown."],
        created_at="2026-01-01T00:00:00+00:00",
        source_text_excerpt="Short source excerpt",
    )
    monkeypatch.setattr(
        review_grounded_brief,
        "get_grounded_brief",
        lambda brief_id: response,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["review_grounded_brief", "--brief-id", "brief-1"],
    )

    review_grounded_brief.main()

    output = capsys.readouterr().out
    assert "Brief ID: brief-1" in output
    assert "Source excerpt" in output
    assert "Generated grounded brief" in output
    assert "Do not invent." in output
