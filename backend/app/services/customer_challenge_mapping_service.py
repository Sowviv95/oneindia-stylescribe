"""Map customer-observed content risks to current StyleScribe evidence."""

from __future__ import annotations

from typing import Any

from backend.app.models.customer_challenge_mapping_models import (
    CustomerChallengeAssessment,
    CustomerChallengeMappingReport,
    GoogleChecklistMapping,
)


def build_customer_challenge_mapping(
    *,
    initial_evaluation: dict[str, object] | None = None,
    final_evaluation: dict[str, object] | None = None,
    google_signals: dict[str, object] | None = None,
    length_status: str | None = None,
    length_recovery_attempted: bool = False,
    length_recovery_succeeded: bool = False,
    section_coverage_status: str | None = None,
    tamil_quality_status: str | None = None,
    unsupported_claim_findings_count: int = 0,
    unsupported_claims_unresolved_count: int = 0,
    final_article_word_count: int | None = None,
) -> CustomerChallengeMappingReport:
    """Build a non-generative assessment against customer risk areas."""

    grounding_evaluation = final_evaluation or initial_evaluation
    grounding_available = grounding_evaluation is not None
    google_available = google_signals is not None
    google_components = _google_component_names(google_signals)

    challenges = [
        _thin_content_assessment(
            length_status=length_status,
            length_recovery_attempted=length_recovery_attempted,
            length_recovery_succeeded=length_recovery_succeeded,
            section_coverage_status=section_coverage_status,
            final_article_word_count=final_article_word_count,
        ),
        _unsupported_claims_assessment(
            grounding_available=grounding_available,
            unsupported_claim_findings_count=unsupported_claim_findings_count,
            unsupported_claims_unresolved_count=unsupported_claims_unresolved_count,
        ),
        CustomerChallengeAssessment(
            challenge_name=(
                "Numbers/offers in headlines or intros not substantiated in body"
            ),
            customer_concern=(
                "Numbers, offers, or prominent lead claims may appear in high-"
                "visibility fields without matching support in the body."
            ),
            current_stylescribe_coverage=(
                "Grounding evaluation checks number/date/name consistency and "
                "Google Signals checks headline search clarity, but there is no "
                "separate headline/body claim-integrity hard gate."
            ),
            status="needs_validation" if grounding_available else "partially_handled",
            evidence_available=_present(
                "grounding evaluation available" if grounding_available else None,
                "google headline_search_clarity component available"
                if "headline_search_clarity" in google_components
                else None,
            ),
            evidence_missing=[
                "headline/body-specific numeric and offer substantiation gate",
            ],
            risk_level="medium",
            recommended_next_action=(
                "Add a validation gate that compares headline, subheadline, and "
                "intro claims against body text and grounded source facts."
            ),
            owner_type="stylescribe",
            notes="This is a validation gap, not evidence that this output failed.",
        ),
        _freshness_assessment(google_available, google_components),
        CustomerChallengeAssessment(
            challenge_name="Unfair or weak comparisons",
            customer_concern=(
                "Generated articles can make comparisons that are one-sided, "
                "under-sourced, or not comparable on the same basis."
            ),
            current_stylescribe_coverage=(
                "Grounding can flag unsupported comparison claims, but current "
                "workflow metadata does not show comparison-specific fairness logic."
            ),
            status="partially_handled" if grounding_available else "needs_validation",
            evidence_available=_present(
                "grounding evaluation available" if grounding_available else None,
            ),
            evidence_missing=[
                "comparison-specific fairness and like-for-like validation",
            ],
            risk_level="medium",
            recommended_next_action=(
                "Introduce a comparison-specific review checklist when comparison "
                "language is detected."
            ),
            owner_type="editorial",
            notes=(
                "Grounding helps with factual support but not full comparison "
                "fairness."
            ),
        ),
        CustomerChallengeAssessment(
            challenge_name="Flattened author voice",
            customer_concern=(
                "Generated output may preserve facts while losing the author's "
                "recognizable style and judgment."
            ),
            current_stylescribe_coverage=(
                "Author style profiles are used during generation and the multi-"
                "author comparison foundation can support style review."
            ),
            status="partially_handled",
            evidence_available=[
                "author style profile used during generation",
                "multi-author comparison foundation exists",
            ],
            evidence_missing=[
                "editorial validation of voice fidelity for this generated output",
            ],
            risk_level="medium",
            recommended_next_action=(
                "Use editorial review or author-pair comparison to validate voice "
                "fidelity on sampled outputs."
            ),
            owner_type="editorial",
            notes="Style coverage exists, but voice quality remains judgment-based.",
        ),
        CustomerChallengeAssessment(
            challenge_name="Repetition at scale",
            customer_concern=(
                "Many generated outputs may become repetitive even when each single "
                "article looks acceptable."
            ),
            current_stylescribe_coverage=(
                "This run assesses a single output. No cross-output similarity or "
                "portfolio-level repetition evidence is available in this report."
            ),
            status="future_scaling_risk",
            evidence_available=[],
            evidence_missing=[
                "cross-output similarity detection",
                "topic/angle repetition tracking across generated articles",
            ],
            risk_level="unknown",
            recommended_next_action=(
                "Add batch-level similarity and angle-diversity reporting before "
                "large-scale rollout."
            ),
            owner_type="future",
            notes="This is a scale risk, not a single-output defect finding.",
        ),
        _google_readiness_assessment(google_available, google_components),
        _originality_assessment(google_available, google_components),
    ]

    return CustomerChallengeMappingReport(
        challenges=challenges,
        google_checklist=_google_checklist_mapping(),
    )


def customer_challenge_mapping_markdown_lines(
    report: CustomerChallengeMappingReport | dict[str, object] | None,
) -> list[str]:
    """Render the mapping report as markdown lines for existing exporters."""

    if report is None:
        return []
    report_dict = (
        report.model_dump()
        if isinstance(report, CustomerChallengeMappingReport)
        else report
    )
    lines = [
        "",
        "## Customer Challenge Mapping",
        "",
        str(report_dict.get("framing_note") or ""),
        "",
        "### Customer Risk Areas",
        "",
    ]
    for challenge in _list_value(report_dict.get("challenges")):
        if not isinstance(challenge, dict):
            continue
        lines.extend(
            [
                f"- Challenge: {challenge.get('challenge_name')}",
                f"  - Status: {challenge.get('status')}",
                f"  - Risk level: {challenge.get('risk_level')}",
                f"  - Owner: {challenge.get('owner_type')}",
                "  - Current coverage: "
                f"{challenge.get('current_stylescribe_coverage')}",
                "  - Evidence available: "
                f"{challenge.get('evidence_available')}",
                "  - Evidence missing: "
                f"{challenge.get('evidence_missing')}",
                "  - Recommended next action: "
                f"{challenge.get('recommended_next_action')}",
                f"  - Notes: {challenge.get('notes')}",
            ]
        )
    lines.extend(["", "### Google Checklist Mapping", ""])
    for checklist_item in _list_value(report_dict.get("google_checklist")):
        if not isinstance(checklist_item, dict):
            continue
        lines.extend(
            [
                f"- Checklist area: {checklist_item.get('checklist_area')}",
                f"  - Scope: {checklist_item.get('scope')}",
                "  - Treated as StyleScribe defect: "
                f"{checklist_item.get('treated_as_stylescribe_defect')}",
                "  - Relevance: "
                f"{checklist_item.get('current_stylescribe_relevance')}",
            ]
        )
    return lines


def _thin_content_assessment(
    *,
    length_status: str | None,
    length_recovery_attempted: bool,
    length_recovery_succeeded: bool,
    section_coverage_status: str | None,
    final_article_word_count: int | None,
) -> CustomerChallengeAssessment:
    evidence = _present(
        f"length status: {length_status}" if length_status else None,
        "length recovery attempted" if length_recovery_attempted else None,
        "length recovery succeeded" if length_recovery_succeeded else None,
        f"section coverage status: {section_coverage_status}"
        if section_coverage_status
        else None,
        f"final article word count: {final_article_word_count}"
        if final_article_word_count is not None
        else None,
    )
    return CustomerChallengeAssessment(
        challenge_name="Thin content",
        customer_concern=(
            "Generated output may be too short or lack enough useful coverage for "
            "the reader."
        ),
        current_stylescribe_coverage=(
            "Length targets, section planning, section coverage checks, and length "
            "recovery address short output, but completeness is broader than length."
        ),
        status="partially_handled",
        evidence_available=evidence,
        evidence_missing=[
            "independent assessment of content completeness beyond word count",
        ],
        risk_level="medium" if length_status == "warning" else "low",
        recommended_next_action=(
            "Keep length recovery as a guardrail and add a completeness review "
            "that checks whether grounded brief essentials are covered."
        ),
        owner_type="stylescribe",
        notes="Length is measurable here; usefulness and completeness need validation.",
    )


def _unsupported_claims_assessment(
    *,
    grounding_available: bool,
    unsupported_claim_findings_count: int,
    unsupported_claims_unresolved_count: int,
) -> CustomerChallengeAssessment:
    status = "already_handled" if grounding_available else "needs_validation"
    risk = (
        "low"
        if grounding_available and unsupported_claims_unresolved_count == 0
        else "medium"
    )
    return CustomerChallengeAssessment(
        challenge_name="Unsupported claims",
        customer_concern=(
            "Generated output may include claims that are not supported by the "
            "grounded factual brief or source facts."
        ),
        current_stylescribe_coverage=(
            "Grounding evaluation, unsupported-claim findings, and revision "
            "guardrail metadata are available for claim safety review."
        ),
        status=status,
        evidence_available=_present(
            "grounding evaluation available" if grounding_available else None,
            f"unsupported claim findings: {unsupported_claim_findings_count}",
            f"unsupported claims unresolved: {unsupported_claims_unresolved_count}",
        ),
        evidence_missing=[
            "headline-specific hard gate for substantiating prominent claims",
        ],
        risk_level=risk,
        recommended_next_action=(
            "Use the existing grounding scanner as the primary gate and add "
            "headline-specific validation for prominent claims."
        ),
        owner_type="stylescribe",
        notes=(
            "Mostly handled by grounding; headline claim integrity still needs "
            "validation."
        ),
    )


def _freshness_assessment(
    google_available: bool,
    google_components: set[str],
) -> CustomerChallengeAssessment:
    freshness_available = "freshness_timeliness" in google_components
    return CustomerChallengeAssessment(
        challenge_name="Missing news hook / freshness context",
        customer_concern=(
            "The article may fail to make clear why the story matters now."
        ),
        current_stylescribe_coverage=(
            "Google Signals v1 includes freshness/timeliness when available; "
            "grounded brief dates and timeline fields can support review."
        ),
        status=(
            "partially_handled"
            if google_available and freshness_available
            else "needs_validation"
        ),
        evidence_available=_present(
            "google freshness_timeliness component available"
            if freshness_available
            else None,
        ),
        evidence_missing=[
            "hard news-hook validation tied to grounded dates or current trigger",
        ],
        risk_level="medium",
        recommended_next_action=(
            "Validate that the lead and headline express a grounded current hook "
            "when the brief contains timing evidence."
        ),
        owner_type="editorial",
        notes="Freshness can be scored only from grounded article and brief evidence.",
    )


def _google_readiness_assessment(
    google_available: bool,
    google_components: set[str],
) -> CustomerChallengeAssessment:
    return CustomerChallengeAssessment(
        challenge_name="Weak Google/search readiness",
        customer_concern=(
            "Generated output may be unclear for search intent, snippets, "
            "structured data, or headline relevance."
        ),
        current_stylescribe_coverage=(
            "Google Signals v1 evaluates search intent clarity, headline clarity, "
            "freshness, E-E-A-T/trust, snippet readiness, and structured data."
        ),
        status="already_handled" if google_available else "needs_validation",
        evidence_available=sorted(google_components),
        evidence_missing=[] if google_available else ["Google Signals v1 output"],
        risk_level="low" if google_available else "unknown",
        recommended_next_action=(
            "Continue reporting Google Signals v1 alongside grounding and use low "
            "component scores as editorial review prompts."
        ),
        owner_type="seo",
        notes="This is an assessment signal, not an automated revision instruction.",
    )


def _originality_assessment(
    google_available: bool,
    google_components: set[str],
) -> CustomerChallengeAssessment:
    originality_available = "originality_angle" in google_components
    return CustomerChallengeAssessment(
        challenge_name="Weak originality/usefulness/engagement signals",
        customer_concern=(
            "The article may be generic, lightly rewritten, or not useful enough "
            "for readers."
        ),
        current_stylescribe_coverage=(
            "Google Signals v1 includes originality angle and helpfulness-related "
            "risk flags; grounding constrains factual claims."
        ),
        status=(
            "partially_handled"
            if google_available and originality_available
            else "needs_validation"
        ),
        evidence_available=_present(
            "google originality_angle component available"
            if originality_available
            else None,
            "google risk flags available" if google_available else None,
        ),
        evidence_missing=[
            "reader engagement validation",
            "cross-output originality comparison",
        ],
        risk_level="medium",
        recommended_next_action=(
            "Track originality_angle and risk flags, then sample outputs for "
            "editorial usefulness review."
        ),
        owner_type="editorial",
        notes="Originality is assessed only from provided article and brief evidence.",
    )


def _google_checklist_mapping() -> list[GoogleChecklistMapping]:
    direct = [
        "Scaled content abuse",
        "Scraping / lightly rewritten content",
        "Doorway abuse",
        "Transparency",
        "Misleading content",
        "Helpful people-first content",
        "No clickbait / sensationalism",
        "Titles & headlines",
    ]
    platform = [
        "Images",
        "Page experience",
        "Discover monitoring",
        "Ads and sponsored content",
        "Manual actions",
    ]
    not_scope = [
        "Cloaking",
        "Sneaky redirects",
        "Site reputation abuse",
        "Hacked content / UGC spam",
        "Feature-content policies",
    ]
    return [
        GoogleChecklistMapping(
            checklist_area=item,
            scope="directly_relevant_to_stylescribe_article_quality",
            current_stylescribe_relevance=(
                "Relevant to generated article quality assessment and editorial "
                "review signals."
            ),
        )
        for item in direct
    ] + [
        GoogleChecklistMapping(
            checklist_area=item,
            scope="platform_cms_seo_dependency",
            current_stylescribe_relevance=(
                "Depends on publishing platform, CMS, page setup, monitoring, or "
                "business policy outside this generated article assessment."
            ),
        )
        for item in platform
    ] + [
        GoogleChecklistMapping(
            checklist_area=item,
            scope="not_applicable_to_current_sprint",
            current_stylescribe_relevance=(
                "Not part of the current generated article quality mapping sprint."
            ),
        )
        for item in not_scope
    ]


def _google_component_names(google_signals: dict[str, object] | None) -> set[str]:
    if not google_signals:
        return set()
    names: set[str] = set()
    for component in _list_value(google_signals.get("components")):
        if isinstance(component, dict) and isinstance(component.get("name"), str):
            names.add(str(component["name"]))
    return names


def _present(*items: str | None) -> list[str]:
    return [item for item in items if item]


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
