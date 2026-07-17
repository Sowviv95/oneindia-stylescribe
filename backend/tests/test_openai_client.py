import httpx
import pytest
from openai import APITimeoutError, BadRequestError

from backend.app.config import get_settings
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
    OpenAIStyleClient,
    _parse_json_object,
    _token_usage,
    request_runtime_metadata,
    temperature_metadata,
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


def test_gpt_5_5_uses_extended_timeout_and_disables_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()

    client = OpenAIJsonClient(model_name="gpt-5.5")

    assert client.model_name == "gpt-5.5"
    assert client.timeout_seconds == 240
    assert captured["timeout"] == 240
    assert captured["max_retries"] == 0


def test_non_generation_models_keep_existing_timeout_and_sdk_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("OPENAI_MAX_RETRIES", "2")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()

    client = OpenAIJsonClient(model_name="gpt-4o-mini")

    assert client.model_name == "gpt-4o-mini"
    assert client.timeout_seconds == 12.5
    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 2


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


def test_gpt_5_5_timeout_retries_once_then_preserves_final_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    class FakeCompletions:
        call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            request = httpx.Request("POST", "https://api.openai.com/v1/chat")
            raise APITimeoutError(request=request)

    completions = FakeCompletions()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = completions

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.time.sleep",
        sleeps.append,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient(model_name="gpt-5.5")

    with pytest.raises(OpenAIClientError, match="timed out after 240 seconds"):
        client.generate_structured_json("system", "payload")

    assert completions.call_count == 2
    assert client.last_attempt_count == 2
    assert client.last_retry_count == 1
    assert sleeps == [2.0]


def test_gpt_5_5_permanent_4xx_errors_are_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []

    class FakeCompletions:
        call_count = 0

        def create(self, **kwargs: object) -> object:
            self.call_count += 1
            request = httpx.Request("POST", "https://api.openai.com/v1/chat")
            response = httpx.Response(400, request=request)
            raise BadRequestError("bad request", response=response, body=None)

    completions = FakeCompletions()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = completions

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.time.sleep",
        sleeps.append,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient(model_name="gpt-5.5")

    with pytest.raises(BadRequestError, match="bad request"):
        client.generate_structured_json("system", "payload")

    assert completions.call_count == 1
    assert client.last_attempt_count == 1
    assert client.last_retry_count == 0
    assert sleeps == []


def test_openai_json_client_passes_prompt_cache_key_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMessage:
        content = '{"ok": true}'

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15
        prompt_tokens_details = {"cached_tokens": 4}

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient()

    result = client.generate_structured_json(
        "system",
        "payload",
        prompt_cache_key="stylescribe-test",
    )

    assert captured["prompt_cache_key"] == "stylescribe-test"
    assert result["token_usage"]["cached_prompt_tokens"] == 4


def test_openai_json_client_omits_temperature_for_gpt_5_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMessage:
        content = '{"ok": true}'

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient(model_name="gpt-5.5")

    result = client.generate_structured_json(
        "system",
        "payload",
        prompt_cache_key="stylescribe-test",
    )

    assert captured["model"] == "gpt-5.5"
    assert captured["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "payload"},
    ]
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["prompt_cache_key"] == "stylescribe-test"
    assert "temperature" not in captured
    assert result["ok"] is True
    assert result["token_usage"]["prompt_tokens"] == 10


def test_openai_json_client_keeps_temperature_for_supported_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeMessage:
        content = '{"ok": true}'

    class FakeChoice:
        message = FakeMessage()

    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()

    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            captured.update(kwargs)
            return FakeResponse()

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            self.chat = FakeChat()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        "backend.app.services.model_clients.openai_client.OpenAI",
        FakeOpenAI,
    )
    get_settings.cache_clear()
    client = OpenAIJsonClient(model_name="gpt-4o-mini")

    result = client.generate_structured_json("system", "payload")

    assert captured["model"] == "gpt-4o-mini"
    assert captured["messages"] == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "payload"},
    ]
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.1
    assert result["ok"] is True


def test_temperature_metadata_matches_request_behavior() -> None:
    assert temperature_metadata("gpt-5.5", 0.1) == {
        "temperature_mode": "model_default",
        "temperature_requested": None,
    }
    assert temperature_metadata("gpt-4o-mini", 0.1) == {
        "temperature_mode": "explicit",
        "temperature_requested": 0.1,
    }


def test_request_runtime_metadata_includes_timeout_and_temperature() -> None:
    assert request_runtime_metadata("gpt-5.5", 0.1, 90.0) == {
        "temperature_mode": "model_default",
        "temperature_requested": None,
        "timeout_seconds": 240,
    }
    assert request_runtime_metadata("gpt-4o-mini", 0.1, 90.0) == {
        "temperature_mode": "explicit",
        "temperature_requested": 0.1,
        "timeout_seconds": 90.0,
    }


def test_token_usage_handles_missing_cached_details() -> None:
    class FakeUsage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    assert _token_usage(FakeUsage()) == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "cached_prompt_tokens": None,
    }
