"""Model provider registry placeholders."""

from pydantic import BaseModel

from backend.app.config import Settings, get_settings


class ModelConfiguration(BaseModel):
    """Safe model provider configuration exposed to the application."""

    name: str
    provider: str
    model: str | None
    base_url: str | None = None
    enabled: bool


def get_model_configurations(
    settings: Settings | None = None,
) -> dict[str, ModelConfiguration]:
    """Return logical model configurations without exposing secret values."""

    resolved_settings = settings or get_settings()
    return {
        "openai": ModelConfiguration(
            name="openai",
            provider="openai",
            model=resolved_settings.openai_model,
            enabled=resolved_settings.openai_api_key is not None,
        ),
        "qwen": ModelConfiguration(
            name="qwen",
            provider=resolved_settings.qwen_provider or "ollama",
            model=resolved_settings.qwen_model,
            base_url=resolved_settings.qwen_base_url,
            enabled=bool(resolved_settings.qwen_model),
        ),
        "gemma": ModelConfiguration(
            name="gemma",
            provider=resolved_settings.gemma_provider or "ollama",
            model=resolved_settings.gemma_model,
            base_url=resolved_settings.gemma_base_url,
            enabled=bool(resolved_settings.gemma_model),
        ),
    }


def get_enabled_model_configurations(
    settings: Settings | None = None,
) -> dict[str, ModelConfiguration]:
    """Return only enabled model configurations."""

    configurations = get_model_configurations(settings)
    return {
        name: configuration
        for name, configuration in configurations.items()
        if configuration.enabled
    }
