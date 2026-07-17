"""Models for headline, intro, and body claim integrity checks."""

from typing import Literal

from pydantic import BaseModel, Field

ClaimLocation = Literal["headline", "subheadline", "intro", "body"]
ClaimType = Literal[
    "number",
    "percentage",
    "price",
    "discount",
    "offer",
    "date",
    "ranking",
    "comparison",
    "superlative",
    "causal_outcome",
]
ClaimSupportStatus = Literal[
    "supported",
    "partially_supported",
    "missing",
    "not_verifiable",
    "not_applicable",
]
ClaimIntegrityStatus = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "body_mismatch",
    "source_missing",
    "not_verifiable",
]
ClaimRiskLevel = Literal["low", "medium", "high"]
ClaimIntegrityOverallStatus = Literal["pass", "review", "block"]


class ClaimIntegrityFinding(BaseModel):
    claim_text: str
    claim_location: ClaimLocation
    claim_type: ClaimType
    source_support_status: ClaimSupportStatus
    body_support_status: ClaimSupportStatus
    integrity_status: ClaimIntegrityStatus
    evidence_summary: str
    risk_level: ClaimRiskLevel
    recommended_action: str


class ClaimIntegrityReport(BaseModel):
    claim_integrity_status: ClaimIntegrityOverallStatus
    findings: list[ClaimIntegrityFinding] = Field(default_factory=list)
    headline_claims_supported: bool = True
    intro_claims_supported: bool = True
    number_price_discount_offer_review_required: bool = False
    summary: str = ""

