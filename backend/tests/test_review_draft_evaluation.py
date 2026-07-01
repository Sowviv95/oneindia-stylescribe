from backend.app.models.draft_evaluation_models import DraftEvaluationResponse
from backend.app.scripts import review_draft_evaluation


def test_review_draft_evaluation_markdown_and_html_exports(
    tmp_path,
    monkeypatch,
) -> None:
    markdown_path = tmp_path / "review_outputs" / "evaluation.md"
    html_path = tmp_path / "review_outputs" / "evaluation.html"
    monkeypatch.setattr(
        review_draft_evaluation,
        "get_latest_draft_evaluation",
        lambda draft_id: _response(),
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_draft_evaluation",
            "--draft-id",
            "draft-1",
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
        ],
    )
    review_draft_evaluation.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "review_draft_evaluation",
            "--draft-id",
            "draft-1",
            "--format",
            "html",
            "--output",
            str(html_path),
        ],
    )
    review_draft_evaluation.main()

    markdown = markdown_path.read_text(encoding="utf-8")
    html = html_path.read_text(encoding="utf-8")
    assert "Draft Grounding Evaluation" in markdown
    assert "Unsupported Claims" in markdown
    assert "ensure safety" in markdown
    assert "<!doctype html>" in html
    assert '<meta charset="utf-8">' in html
    assert "Nirmala UI" in html


def _response() -> DraftEvaluationResponse:
    return DraftEvaluationResponse(
        evaluation_id="evaluation-1",
        draft_id="draft-1",
        brief_id="brief-1",
        author_id="v_vasanthi",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        evaluation={
            "grounding_score": 45,
            "claim_safety_score": 40,
            "fact_preservation_score": 80,
            "overall_risk": "high",
            "editorial_readiness": "revision_required",
            "unsupported_claims": [{"claim": "ensure safety"}],
            "overclaim_phrases": [{"phrase": "reduce impact"}],
            "claims_to_avoid_violations": ["Do not claim effectiveness."],
            "rewrite_guidance": ["Remove benefit language."],
            "summary": "Revision required.",
        },
        warnings=[],
        created_at="2026-01-01T00:00:00+00:00",
    )
