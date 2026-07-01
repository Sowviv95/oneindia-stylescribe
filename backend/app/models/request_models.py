"""Request models for StyleScribe endpoints."""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["url", "text"]
ModelName = Literal["openai", "qwen", "gemma"]


def default_models() -> list[ModelName]:
    """Return the default generation model list."""

    return ["openai"]


class ArticleGenerationRequest(BaseModel):
    """Request body for article generation."""

    author_id: str = Field(..., min_length=1)
    target_language: str = Field(default="ta", min_length=1)
    source_type: SourceType
    source_input: str = Field(..., min_length=1)
    author_instruction: str | None = Field(default=None)
    category: str | None = Field(default=None)
    models: list[ModelName] = Field(default_factory=default_models)
