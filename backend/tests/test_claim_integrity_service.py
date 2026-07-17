from backend.app.services.claim_integrity_service import (
    assess_claim_integrity,
    claim_integrity_markdown_lines,
)


def test_supported_headline_number() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "City installs 18 flood sensors",
            "subheadline": "",
            "article_body": (
                "City officials said the pilot will install 18 flood sensors.\n\n"
                "The pilot will install 18 flood sensors in low-lying streets.\n\n"
                "The sensors will be placed near low-lying streets."
            ),
        },
        grounded_brief={
            "confirmed_facts": ["The city will install 18 flood sensors."],
        },
        source_text="The city will install 18 flood sensors.",
    )

    finding = report.findings[0]
    assert report.claim_integrity_status == "pass"
    assert finding.claim_location == "headline"
    assert finding.claim_type == "number"
    assert finding.integrity_status == "supported"


def test_unsupported_headline_discount_blocks() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Phone gets 50% discount today",
            "subheadline": "",
            "article_body": "The phone launched in India today.",
        },
        grounded_brief={"confirmed_facts": ["The phone launched in India today."]},
        source_text="The phone launched in India today.",
    )

    finding = report.findings[0]
    assert report.claim_integrity_status == "block"
    assert finding.claim_type == "discount"
    assert finding.integrity_status in {"unsupported", "body_mismatch"}
    assert report.number_price_discount_offer_review_required is True


def test_intro_claim_not_present_in_body_is_body_mismatch() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Flood pilot begins",
            "subheadline": "",
            "article_body": (
                "The pilot will install 18 flood sensors.\n\n"
                "Officials said alerts will be sent to residents."
            ),
        },
        grounded_brief={
            "confirmed_facts": ["The pilot will install 18 flood sensors."],
        },
        source_text="The pilot will install 18 flood sensors.",
    )

    intro_finding = _finding(report, "intro")
    assert report.claim_integrity_status == "block"
    assert intro_finding.integrity_status == "body_mismatch"


def test_price_claim_supported_by_source_and_body() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Plan starts at Rs 499",
            "subheadline": "",
            "article_body": (
                "The plan starts at Rs 499 for the monthly pack.\n\n"
                "The company said the plan starts at Rs 499."
            ),
        },
        grounded_brief={
            "confirmed_facts": ["The company said the plan starts at Rs 499."],
        },
        source_text="The company said the plan starts at Rs 499.",
    )

    finding = _finding(report, "headline")
    assert report.claim_integrity_status == "pass"
    assert finding.claim_type == "price"
    assert finding.source_support_status == "supported"
    assert finding.body_support_status == "supported"


def test_comparison_claim_without_source_support_requires_review() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Metro update announced",
            "subheadline": "",
            "article_body": (
                "Officials announced the route update.\n\n"
                "Officials announced the route update."
                " The new route is faster than the old route."
            ),
        },
        grounded_brief={"confirmed_facts": ["Officials announced the route update."]},
        source_text="Officials announced the route update.",
    )

    finding = _finding(report, "body")
    assert report.claim_integrity_status == "review"
    assert finding.claim_type == "comparison"
    assert finding.integrity_status == "source_missing"


def test_superlative_claim_requires_review() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Best budget phone launched",
            "subheadline": "",
            "article_body": "The budget phone launched in India today.",
        },
        grounded_brief={"confirmed_facts": ["The budget phone launched in India."]},
        source_text="The budget phone launched in India.",
    )

    finding = _finding(report, "headline")
    assert report.claim_integrity_status == "review"
    assert finding.claim_type == "superlative"
    assert finding.integrity_status == "not_verifiable"


def test_ordinary_descriptive_headline_does_not_false_block() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "City flood warning pilot announced",
            "subheadline": "",
            "article_body": "City officials announced a flood warning pilot.",
        },
        grounded_brief={
            "confirmed_facts": ["City officials announced a flood warning pilot."],
        },
        source_text="City officials announced a flood warning pilot.",
    )

    assert report.claim_integrity_status == "pass"
    assert report.findings == []


def test_claim_integrity_markdown_renders_report() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "City installs 18 flood sensors",
            "subheadline": "",
            "article_body": "City officials said the pilot installs 18 sensors.",
        },
        grounded_brief={"confirmed_facts": ["The pilot installs 18 sensors."]},
        source_text="The pilot installs 18 sensors.",
    )

    markdown = "\n".join(claim_integrity_markdown_lines(report))
    assert "Claim Integrity Check" in markdown
    assert "final claim_integrity_status" in markdown
    assert "Headline claims supported by body and source" in markdown


def test_claim_integrity_recommendations_do_not_add_unsupported_facts() -> None:
    report = assess_claim_integrity(
        final_article={
            "headline": "Phone gets 50% discount today",
            "subheadline": "",
            "article_body": "The phone launched in India today.",
        },
        grounded_brief={"confirmed_facts": ["The phone launched in India today."]},
        source_text="The phone launched in India today.",
    )
    recommendations = " ".join(
        finding.recommended_action.lower() for finding in report.findings
    )

    banned = [
        "add unsupported facts",
        "add unverified facts",
        "invent",
        "external context",
    ]
    assert not any(phrase in recommendations for phrase in banned)


def _finding(report: object, location: str):
    for finding in report.findings:
        if finding.claim_location == location:
            return finding
    raise AssertionError(f"Missing finding at {location}")
