import httpx
import pytest
from openai import APITimeoutError

from backend.app.config import get_settings
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
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


def test_openai_json_client_uses_configured_timeout_and_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "1")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()

    OpenAIJsonClient()

    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 1


def test_openai_json_client_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            request = httpx.Request("POST", "https://api.openai.com/v1/chat")
            raise APITimeoutError(request=request)

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "3")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient()

    with pytest.raises(OpenAIClientError, match="timed out after 3 seconds"):
        client.generate_structured_json("system", "payload")
