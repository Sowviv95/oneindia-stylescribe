"""Application configuration loaded from environment variables."""

import os
from functools import lru_cache

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr

load_dotenv()


class Settings(BaseModel):
    """Runtime settings for the StyleScribe API.

    Secret values are represented with ``SecretStr`` so accidental model dumps or
    repr calls do not expose API keys.
    """

    openai_api_key: SecretStr | None = Field(default=None)
    openai_model: str | None = Field(default=None)
    openai_model_default: str | None = Field(default=None)
    openai_model_planning: str | None = Field(default=None)
    openai_model_generation: str | None = Field(default=None)
    openai_model_revision: str | None = Field(default=None)
    openai_model_evaluation: str | None = Field(default=None)
    openai_model_length_recovery: str | None = Field(default=None)
    qwen_provider: str | None = Field(default=None)
    qwen_base_url: str | None = Field(default=None)
    qwen_model: str | None = Field(default=None)
    gemma_provider: str | None = Field(default=None)
    gemma_base_url: str | None = Field(default=None)
    gemma_model: str | None = Field(default=None)
    default_target_language: str = Field(default="ta")
    default_source_language: str | None = Field(default=None)
    stylescribe_db_path: str = Field(default="data/stylescribe.db")
    openai_timeout_seconds: float = Field(default=90.0)
    openai_max_retries: int = Field(default=2)
    max_concurrent_section_calls: int = Field(default=3, ge=1, le=8)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""

    openai_api_key = os.getenv("OPENAI_API_KEY")
    return Settings(
        openai_api_key=SecretStr(openai_api_key) if openai_api_key else None,
        openai_model=os.getenv("OPENAI_MODEL"),
        openai_model_default=os.getenv("OPENAI_MODEL_DEFAULT"),
        openai_model_planning=os.getenv("OPENAI_MODEL_PLANNING"),
        openai_model_generation=os.getenv("OPENAI_MODEL_GENERATION"),
        openai_model_revision=os.getenv("OPENAI_MODEL_REVISION"),
        openai_model_evaluation=os.getenv("OPENAI_MODEL_EVALUATION"),
        openai_model_length_recovery=os.getenv("OPENAI_MODEL_LENGTH_RECOVERY"),
        qwen_provider=os.getenv("QWEN_PROVIDER"),
        qwen_base_url=os.getenv("QWEN_BASE_URL"),
        qwen_model=os.getenv("QWEN_MODEL"),
        gemma_provider=os.getenv("GEMMA_PROVIDER"),
        gemma_base_url=os.getenv("GEMMA_BASE_URL"),
        gemma_model=os.getenv("GEMMA_MODEL"),
        default_target_language=os.getenv("DEFAULT_TARGET_LANGUAGE", "ta"),
        default_source_language=os.getenv("DEFAULT_SOURCE_LANGUAGE"),
        stylescribe_db_path=os.getenv("STYLESCRIBE_DB_PATH", "data/stylescribe.db"),
        openai_timeout_seconds=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
        openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
        max_concurrent_section_calls=int(
            os.getenv("MAX_CONCURRENT_SECTION_CALLS", "3")
        ),
    )
