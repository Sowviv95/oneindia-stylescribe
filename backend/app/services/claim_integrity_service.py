"""Deterministic claim integrity checks for generated article top matter."""

from __future__ import annotations

import re
from typing import Any

from backend.app.models.claim_integrity_models import (
    ClaimIntegrityFinding,
    ClaimIntegrityReport,
    ClaimIntegrityStatus,
    ClaimLocation,
    ClaimRiskLevel,
    ClaimSupportStatus,
    ClaimType,
)

NUMBER_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?%?")
PRICE_RE = re.compile(
    r"(?:₹|rs\.?|inr|\$)\s*\d[\d,.]*|\d[\d,.]*\s*(?:rupees|lakh|crore)",
    re.I,
)
PERCENT_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?\s*%|\bpercent\b|\bpercentage\b", re.I)
DATE_RE = re.compile(
    r"\b(?:today|tomorrow|yesterday|next month|this month|last month|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december|20\d{2})\b",
    re.I,
)
DISCOUNT_RE = re.compile(r"\b(?:discount|off|cashback|sale|deal)\b", re.I)
OFFER_RE = re.compile(r"\b(?:offer|free|bonus|coupon|promo|limited-time)\b", re.I)
RANKING_RE = re.compile(r"\b(?:top\s+\d+|ranked|ranking|no\.\s*1|number\s+one)\b", re.I)
COMPARISON_RE = re.compile(
    r"\b(?:more than|less than|higher than|lower than|cheaper than|costlier than|"
    r"better than|worse than|faster than|slower than|compared with|compared to|"
    r"versus|vs\.?)\b",
    re.I,
)
SUPERLATIVE_RE = re.compile(
    r"\b(?:best|cheapest|costliest|first|only|top|largest|smallest|highest|lowest|"
    r"biggest|fastest|most|least)\b",
    re.I,
)
CAUSAL_RE = re.compile(
    r"\b(?:causes|caused|will cause|leads to|will lead to|results in|prevents|"
    r"guarantees|boosts|cuts|saves|slashes)\b",
    re.I,
)
TOKEN_RE = re.compile(r"[a-z0-9₹$%]+", re.I)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।])\s+|\n+")

STOPWORDS = {
    "the",
    "and",
    "for",
    "from",
    "with",
    "this",
    "that",
    "will",
    "has",
    "have",
    "into",
    "over",
    "under",
    "than",
    "more",
    "less",
    "best",
    "top",
    "only",
    "first",
    "offer",
    "offers",
    "discount",
    "off",
    "free",
}

BLOCKING_TOP_TYPES = {
    "number",
    "percentage",
    "price",
    "discount",
    "offer",
    "comparison",
}


def assess_claim_integrity(
    *,
    final_article: dict[str, object],
    grounded_brief: dict[str, object],
    source_text: str = "",
) -> ClaimIntegrityReport:
    """Assess risk-bearing generated claims against body and grounded evidence."""

    article_body = str(final_article.get("article_body") or "")
    intro, body_after_intro = _intro_and_remaining_body(article_body)
    source_evidence = _normalize_text(
        " ".join([source_text, _flatten_for_evidence(grounded_brief)])
    )
    body_evidence = _normalize_text(article_body)
    body_after_intro_evidence = _normalize_text(body_after_intro)
    top_matter: list[tuple[ClaimLocation, str, str]] = [
        ("headline", str(final_article.get("headline") or ""), body_evidence),
        ("subheadline", str(final_article.get("subheadline") or ""), body_evidence),
        ("intro", intro, body_after_intro_evidence),
    ]

    findings: list[ClaimIntegrityFinding] = []
    for location, text, body_support_text in top_matter:
        findings.extend(
            _findings_for_text(
                text=text,
                location=location,
                source_evidence=source_evidence,
                body_evidence=body_support_text,
            )
        )

    for sentence in _risk_sentences(body_after_intro):
        findings.extend(
            _findings_for_text(
                text=sentence,
                location="body",
                source_evidence=source_evidence,
                body_evidence=body_evidence,
            )
        )

    return _build_report(_dedupe_findings(findings))


def claim_integrity_markdown_lines(
    report: ClaimIntegrityReport | dict[str, object] | None,
) -> list[str]:
    """Render claim integrity output as markdown for existing report exporters."""

    if report is None:
        return []
    report_dict = (
        report.model_dump() if isinstance(report, ClaimIntegrityReport) else report
    )
    lines = [
        "",
        "## Claim Integrity Check",
        "",
        f"- final claim_integrity_status: {report_dict.get('claim_integrity_status')}",
        "- Headline claims supported by body and source: "
        f"{report_dict.get('headline_claims_supported')}",
        "- Intro claims supported by body and source: "
        f"{report_dict.get('intro_claims_supported')}",
        "- Number/price/discount/offer requires editorial review: "
        f"{report_dict.get('number_price_discount_offer_review_required')}",
        f"- Summary: {report_dict.get('summary')}",
        "- Findings:",
    ]
    for finding in _list_value(report_dict.get("findings")):
        if not isinstance(finding, dict):
            continue
        lines.extend(
            [
                f"  - Claim: {finding.get('claim_text')}",
                f"    - Location: {finding.get('claim_location')}",
                f"    - Type: {finding.get('claim_type')}",
                f"    - Source support: {finding.get('source_support_status')}",
                f"    - Body support: {finding.get('body_support_status')}",
                f"    - Integrity: {finding.get('integrity_status')}",
                f"    - Risk level: {finding.get('risk_level')}",
                f"    - Evidence: {finding.get('evidence_summary')}",
                f"    - Recommended action: {finding.get('recommended_action')}",
            ]
        )
    return lines


def _findings_for_text(
    *,
    text: str,
    location: ClaimLocation,
    source_evidence: str,
    body_evidence: str,
) -> list[ClaimIntegrityFinding]:
    findings: list[ClaimIntegrityFinding] = []
    for claim_text in _risk_sentences(text):
        claim_type = _claim_type(claim_text)
        if claim_type is None:
            continue
        source_support = _support_status(claim_text, source_evidence, claim_type)
        body_support = (
            "supported"
            if location == "body"
            else _support_status(claim_text, body_evidence, claim_type)
        )
        integrity = _integrity_status(
            location=location,
            claim_type=claim_type,
            source_support=source_support,
            body_support=body_support,
        )
        findings.append(
            ClaimIntegrityFinding(
                claim_text=claim_text,
                claim_location=location,
                claim_type=claim_type,
                source_support_status=source_support,
                body_support_status=body_support,
                integrity_status=integrity,
                evidence_summary=_evidence_summary(source_support, body_support),
                risk_level=_risk_level(location, claim_type, integrity),
                recommended_action=_recommended_action(integrity),
            )
        )
    return findings


def _build_report(findings: list[ClaimIntegrityFinding]) -> ClaimIntegrityReport:
    status = "pass"
    if any(_is_blocking_finding(finding) for finding in findings):
        status = "block"
    elif any(
        finding.integrity_status
        in {"partially_supported", "not_verifiable", "source_missing"}
        for finding in findings
    ):
        status = "review"

    headline_claims_supported = all(
        finding.integrity_status == "supported"
        for finding in findings
        if finding.claim_location in {"headline", "subheadline"}
    )
    intro_claims_supported = all(
        finding.integrity_status == "supported"
        for finding in findings
        if finding.claim_location == "intro"
    )
    number_review = any(
        finding.claim_type in {"number", "percentage", "price", "discount", "offer"}
        and finding.integrity_status != "supported"
        for finding in findings
    )
    return ClaimIntegrityReport(
        claim_integrity_status=status,
        findings=findings,
        headline_claims_supported=headline_claims_supported,
        intro_claims_supported=intro_claims_supported,
        number_price_discount_offer_review_required=number_review,
        summary=_summary(status, findings),
    )


def _is_blocking_finding(finding: ClaimIntegrityFinding) -> bool:
    return (
        finding.claim_location in {"headline", "subheadline", "intro"}
        and finding.claim_type in BLOCKING_TOP_TYPES
        and finding.integrity_status
        in {"unsupported", "body_mismatch", "source_missing"}
    )


def _integrity_status(
    *,
    location: ClaimLocation,
    claim_type: ClaimType,
    source_support: ClaimSupportStatus,
    body_support: ClaimSupportStatus,
) -> ClaimIntegrityStatus:
    if body_support == "missing" and location in {"headline", "subheadline", "intro"}:
        return "body_mismatch"
    if source_support == "supported" and body_support == "supported":
        return "supported"
    if source_support == "partially_supported" or body_support == "partially_supported":
        return "partially_supported"
    if source_support == "not_verifiable" or claim_type == "superlative":
        return "not_verifiable"
    if source_support == "missing":
        if (
            location in {"headline", "subheadline", "intro"}
            and claim_type in BLOCKING_TOP_TYPES
        ):
            return "unsupported"
        return "source_missing"
    return "not_verifiable"


def _support_status(
    claim_text: str,
    evidence_text: str,
    claim_type: ClaimType,
) -> ClaimSupportStatus:
    if not evidence_text.strip():
        return "missing"
    normalized_claim = _normalize_text(claim_text)
    if normalized_claim and normalized_claim in evidence_text:
        return "supported"

    claim_numbers = _number_tokens(normalized_claim)
    evidence_numbers = _number_tokens(evidence_text)
    content_tokens = _content_tokens(normalized_claim)
    evidence_tokens = set(_content_tokens(evidence_text))
    token_overlap = [token for token in content_tokens if token in evidence_tokens]

    if claim_type == "superlative" and normalized_claim not in evidence_text:
        return "not_verifiable"
    if claim_numbers:
        if all(number in evidence_numbers for number in claim_numbers):
            if len(token_overlap) >= min(2, len(content_tokens)):
                return "supported"
            return "partially_supported"
        return "missing"
    if claim_type in {"comparison", "causal_outcome", "ranking"}:
        if len(token_overlap) >= 3:
            return "partially_supported"
        return "missing"
    if claim_type in {"discount", "offer", "price"}:
        if len(token_overlap) >= 2:
            return "partially_supported"
        return "missing"
    if len(token_overlap) >= 3:
        return "partially_supported"
    return "missing"


def _claim_type(text: str) -> ClaimType | None:
    if DISCOUNT_RE.search(text):
        return "discount"
    if OFFER_RE.search(text):
        return "offer"
    if PRICE_RE.search(text):
        return "price"
    if PERCENT_RE.search(text):
        return "percentage"
    if RANKING_RE.search(text):
        return "ranking"
    if COMPARISON_RE.search(text):
        return "comparison"
    if SUPERLATIVE_RE.search(text):
        return "superlative"
    if CAUSAL_RE.search(text):
        return "causal_outcome"
    if DATE_RE.search(text):
        return "date"
    if NUMBER_RE.search(text):
        return "number"
    return None


def _risk_sentences(text: str) -> list[str]:
    candidates = [
        sentence.strip(" \t\r\n-")
        for sentence in SENTENCE_SPLIT_RE.split(text)
        if sentence.strip(" \t\r\n-")
    ]
    if not candidates and text.strip():
        candidates = [text.strip()]
    return [candidate for candidate in candidates if _claim_type(candidate) is not None]


def _intro_and_remaining_body(article_body: str) -> tuple[str, str]:
    paragraphs = [
        part.strip()
        for part in re.split(r"\n\s*\n|\r?\n", article_body)
        if part.strip()
    ]
    if not paragraphs:
        return "", ""
    return paragraphs[0], "\n\n".join(paragraphs[1:])


def _flatten_for_evidence(value: object) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten_for_evidence(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_for_evidence(item) for item in value)
    if value is None:
        return ""
    return str(value)


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def _number_tokens(text: str) -> list[str]:
    return [
        token.replace(",", "").replace(" ", "")
        for token in NUMBER_RE.findall(text)
    ]


def _content_tokens(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.casefold()):
        if token in STOPWORDS or len(token) < 3 or token.isdigit():
            continue
        tokens.append(token)
    return tokens


def _evidence_summary(
    source_support: ClaimSupportStatus,
    body_support: ClaimSupportStatus,
) -> str:
    return (
        f"Source support is {source_support}; body support is {body_support}. "
        "Assessment used only the grounded brief, source text, and generated article."
    )


def _risk_level(
    location: ClaimLocation,
    claim_type: ClaimType,
    integrity: ClaimIntegrityStatus,
) -> ClaimRiskLevel:
    if (
        location in {"headline", "subheadline", "intro"}
        and claim_type in BLOCKING_TOP_TYPES
        and integrity != "supported"
    ):
        return "high"
    if integrity in {"not_verifiable", "partially_supported", "source_missing"}:
        return "medium"
    return "low"


def _recommended_action(integrity: ClaimIntegrityStatus) -> str:
    if integrity == "supported":
        return "No claim-integrity action required."
    if integrity == "body_mismatch":
        return (
            "Review the claim against the grounded evidence and remove or soften it "
            "if the article body does not support it."
        )
    if integrity == "unsupported":
        return (
            "Hold for editorial review; remove or soften the claim unless existing "
            "grounded evidence supports it."
        )
    if integrity == "source_missing":
        return (
            "Review against the available source facts and remove or soften the "
            "claim if support is not found."
        )
    if integrity == "partially_supported":
        return (
            "Review the wording and keep only the portion supported by the existing "
            "grounded evidence."
        )
    return (
        "Route to editorial review and avoid presenting the claim as verified "
        "unless existing grounded evidence supports it."
    )


def _summary(status: str, findings: list[ClaimIntegrityFinding]) -> str:
    if not findings:
        return "No risk-bearing headline, intro, or body claims were detected."
    return (
        f"{len(findings)} risk-bearing claim(s) assessed; "
        f"overall status is {status}."
    )


def _dedupe_findings(
    findings: list[ClaimIntegrityFinding],
) -> list[ClaimIntegrityFinding]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[ClaimIntegrityFinding] = []
    for finding in findings:
        key = (
            _normalize_text(finding.claim_text),
            finding.claim_location,
            finding.claim_type,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    return deduped


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []
