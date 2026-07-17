from backend.app.services.customer_challenge_mapping_service import (
    build_customer_challenge_mapping,
    customer_challenge_mapping_markdown_lines,
)


def test_customer_challenge_mapping_status_values() -> None:
    report = build_customer_challenge_mapping(
        initial_evaluation={"grounding_score": 92, "unsupported_claims": []},
        google_signals=_google_signals(),
        length_status="pass",
        section_coverage_status="pass",
        final_article_word_count=620,
    )

    allowed = {
        "already_handled",
        "partially_handled",
        "needs_validation",
        "not_currently_addressed",
        "not_applicable_to_stylescribe",
        "platform_or_cms_dependency",
        "future_scaling_risk",
    }
    assert {challenge.status for challenge in report.challenges} <= allowed


def test_unsupported_claims_mostly_handled_with_headline_validation() -> None:
    report = build_customer_challenge_mapping(
        initial_evaluation={"grounding_score": 92, "unsupported_claims": []},
        google_signals=_google_signals(),
        unsupported_claim_findings_count=1,
        unsupported_claims_unresolved_count=0,
    )

    challenge = _challenge(report, "Unsupported claims")
    assert challenge.status == "already_handled"
    assert challenge.risk_level == "low"
    assert "headline-specific" in " ".join(challenge.evidence_missing)
    assert "headline claim integrity" in challenge.notes


def test_thin_content_partially_handled() -> None:
    report = build_customer_challenge_mapping(
        length_status="warning",
        length_recovery_attempted=True,
        length_recovery_succeeded=True,
        section_coverage_status="warning",
    )

    challenge = _challenge(report, "Thin content")
    assert challenge.status == "partially_handled"
    assert challenge.risk_level == "medium"


def test_repetition_at_scale_is_future_scaling_risk() -> None:
    report = build_customer_challenge_mapping()

    challenge = _challenge(report, "Repetition at scale")
    assert challenge.status == "future_scaling_risk"
    assert challenge.owner_type == "future"


def test_platform_checklist_items_are_not_stylescribe_defects() -> None:
    report = build_customer_challenge_mapping()
    platform_items = {
        item.checklist_area: item
        for item in report.google_checklist
        if item.scope == "platform_cms_seo_dependency"
    }

    assert "Images" in platform_items
    assert "Manual actions" in platform_items
    assert all(
        not item.treated_as_stylescribe_defect for item in platform_items.values()
    )


def test_mapping_markdown_includes_section_and_framing() -> None:
    report = build_customer_challenge_mapping(google_signals=_google_signals())
    markdown = "\n".join(customer_challenge_mapping_markdown_lines(report))

    assert "Customer Challenge Mapping" in markdown
    assert "These are customer-observed content generation risk areas" in markdown
    assert "Google Checklist Mapping" in markdown


def test_recommendations_do_not_ask_for_unsupported_facts() -> None:
    report = build_customer_challenge_mapping(google_signals=_google_signals())
    recommendations = " ".join(
        challenge.recommended_next_action.lower()
        for challenge in report.challenges
    )

    banned_phrases = [
        "add unsupported facts",
        "add unverified facts",
        "include unsupported facts",
        "invent",
        "external context",
    ]
    assert not any(phrase in recommendations for phrase in banned_phrases)


def _challenge(report: object, name: str):
    for challenge in report.challenges:
        if challenge.challenge_name == name:
            return challenge
    raise AssertionError(f"Missing challenge: {name}")


def _google_signals() -> dict[str, object]:
    return {
        "score": 82,
        "version": "google_signals_v1",
        "components": [
            {"name": "search_intent_clarity", "score": 82},
            {"name": "headline_search_clarity", "score": 80},
            {"name": "freshness_timeliness", "score": 78},
            {"name": "originality_angle", "score": 76},
            {"name": "eeat_trust", "score": 84},
            {"name": "snippet_meta_readiness", "score": 81},
            {"name": "structured_data_readiness", "score": 79},
        ],
        "risk_flags": [],
    }
