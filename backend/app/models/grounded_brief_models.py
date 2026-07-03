"""Models for grounded factual briefs."""

from typing import Any, Literal

from pydantic import BaseModel, Field

SourceType = Literal["text", "url"]
SourceInputMode = Literal["plain_text", "pasted_web_text"]


class GroundedBriefRequest(BaseModel):
    source_type: SourceType
    source_input: str = Field(..., min_length=1)
    target_language: str = Field(default="ta", min_length=1)
    source_input_mode: SourceInputMode = "plain_text"


class GroundedBriefSummary(BaseModel):
    brief_id: str
    source_type: str
    source_url: str | None
    source_language: str
    target_language: str
    status: str
    created_at: str


class GroundedBriefResponse(GroundedBriefSummary):
    model_provider: str
    model_name: str
    brief: dict[str, Any]
    warnings: list[str]
    source_text_excerpt: str
