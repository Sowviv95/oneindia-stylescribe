import pytest

from backend.app.config import get_settings
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIStyleClient,
    _parse_json_object,
)


def test_openai_client_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()

    with pytest.raises(OpenAIClientError, match="OPENAI_API_KEY"):
        OpenAIStyleClient()


def test_parse_json_object_rejects_invalid_json() -> None:
    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        _parse_json_object("not-json")
