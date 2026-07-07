"""Models for modular Google Signals scoring."""

from typing import Any, Literal

from pydantic import BaseModel, Field

GoogleSignalsRiskLevel = Literal["low", "medium", "high", "unknown"]


class GoogleSignalsComponentConfig(BaseModel):
    name: str
    weight: int = Field(ge=0)
    enabled: bool = True


class GoogleSignalsScoringConfig(BaseModel):
    version: str
    components: list[GoogleSignalsComponentConfig]

    @property
    def enabled_components(self) -> list[GoogleSignalsComponentConfig]:
        return [component for component in self.components if component.enabled]


class GoogleSignalsComponentScore(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)
    weight: int = Field(ge=0)
    rationale: str = ""
    risk_level: GoogleSignalsRiskLevel = "unknown"
    enabled: bool = True


class GoogleSignalsScore(BaseModel):
    score: int
    version: str
    components: list[GoogleSignalsComponentScore]
    risk_flags: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    overall_rationale: str | None = None


class GoogleSignalsEvaluationResult(BaseModel):
    available: bool
    google_signals: GoogleSignalsScore | None = None
    error: str | None = None

