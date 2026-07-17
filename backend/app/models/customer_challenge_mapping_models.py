"""Models for mapping customer-observed content risks to StyleScribe evidence."""

from typing import Literal

from pydantic import BaseModel, Field

CustomerChallengeStatus = Literal[
    "already_handled",
    "partially_handled",
    "needs_validation",
    "not_currently_addressed",
    "not_applicable_to_stylescribe",
    "platform_or_cms_dependency",
    "future_scaling_risk",
]

CustomerChallengeRiskLevel = Literal["low", "medium", "high", "unknown"]

CustomerChallengeOwnerType = Literal[
    "stylescribe",
    "editorial",
    "platform",
    "seo",
    "future",
]

GoogleChecklistScope = Literal[
    "directly_relevant_to_stylescribe_article_quality",
    "platform_cms_seo_dependency",
    "not_applicable_to_current_sprint",
]


class CustomerChallengeAssessment(BaseModel):
    challenge_name: str
    customer_concern: str
    current_stylescribe_coverage: str
    status: CustomerChallengeStatus
    evidence_available: list[str] = Field(default_factory=list)
    evidence_missing: list[str] = Field(default_factory=list)
    risk_level: CustomerChallengeRiskLevel
    recommended_next_action: str
    owner_type: CustomerChallengeOwnerType
    notes: str = ""


class GoogleChecklistMapping(BaseModel):
    checklist_area: str
    scope: GoogleChecklistScope
    current_stylescribe_relevance: str
    treated_as_stylescribe_defect: bool = False
    notes: str = ""


class CustomerChallengeMappingReport(BaseModel):
    framing_note: str = (
        "These are customer-observed content generation risk areas. They are not "
        "assumed defects in this generated output."
    )
    challenges: list[CustomerChallengeAssessment]
    google_checklist: list[GoogleChecklistMapping]

