import json
from pathlib import Path

import httpx
import pytest
from openai import APIConnectionError, APIStatusError

from backend.app.config import get_settings
from backend.app.services.model_clients import grok_client
from backend.app.services.model_clients.grok_client import (
    GROK_BASE_URL,
    GROK_GENERATION_MODEL,
    GrokJsonClient,
    _cost_usd_from_ticks,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_grok_client_uses_xai_base_url_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_openai(**kwargs: object) -> FakeOpenAI:
        captured.update(kwargs)
        return FakeOpenAI([FakeResponse('{"ok": true}')])

    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", fake_openai)

    client = GrokJsonClient()

    assert client.provider == "grok"
    assert client.model_name == GROK_GENERATION_MODEL
    assert captured["api_key"] == "xai-secret"
    assert str(captured["base_url"]) == GROK_BASE_URL
    assert captured["timeout"] == 180.0
    assert captured["max_retries"] == 0


def test_grok_client_missing_key_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XAI_API_KEY", raising=False)

    with pytest.raises(OpenAIClientError, match="XAI_API_KEY is required"):
        GrokJsonClient()


def test_grok_client_sends_temperature_and_json_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOpenAI([FakeResponse('{"headline": "தலைப்பு"}')])
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)

    result = GrokJsonClient().generate_structured_json("system", "payload")
    request = fake.chat.completions.requests[0]

    assert result["headline"] == "தலைப்பு"
    assert request["model"] == GROK_GENERATION_MODEL
    assert request["temperature"] == 0.1
    assert request["response_format"] == {"type": "json_object"}
    assert request["messages"][0] == {"role": "system", "content": "system"}
    assert request["messages"][1] == {"role": "user", "content": "payload"}


def test_grok_client_maps_usage_and_provider_cost_ticks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeOpenAI([FakeResponse('{"ok": true}', usage=FakeUsage())])
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)

    result = GrokJsonClient().generate_structured_json("system", "payload")
    usage = result["token_usage"]

    assert usage["prompt_tokens"] == 100
    assert usage["cached_prompt_tokens"] == 25
    assert usage["completion_tokens"] == 40
    assert usage["total_tokens"] == 140
    assert usage["reasoning_tokens"] == 0
    assert usage["accepted_prediction_tokens"] == 3
    assert usage["rejected_prediction_tokens"] == 4
    assert usage["provider_cost_ticks"] == 123456789
    assert usage["provider_reported_cost_usd"] == 0.012346
    assert usage["provider_cost_conversion_status"] == "converted"
    assert _cost_usd_from_ticks(10_000_000_000) == 1.0


def test_grok_client_retries_transient_once(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps: list[float] = []
    fake = FakeOpenAI(
        [
            APIConnectionError(request=httpx.Request("POST", GROK_BASE_URL)),
            FakeResponse('{"ok": true}'),
        ]
    )
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr(grok_client.time, "sleep", sleeps.append)

    result = GrokJsonClient().generate_structured_json("system", "payload")

    assert result["ok"] is True
    assert fake.chat.completions.call_count == 2
    assert sleeps == [2.0]


def test_grok_client_does_not_retry_permanent_4xx(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    fake = FakeOpenAI([_status_error(400)])
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr(grok_client.time, "sleep", sleeps.append)

    with pytest.raises(OpenAIClientError, match="Grok request failed"):
        GrokJsonClient().generate_structured_json("system", "payload")

    assert fake.chat.completions.call_count == 1
    assert sleeps == []


def test_grok_client_retries_invalid_json_once_and_saves_raw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = FakeOpenAI([FakeResponse('{"bad": '), FakeResponse('{"ok": true}')])
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr(grok_client.time, "sleep", lambda _: None)
    client = GrokJsonClient()
    client.configure_diagnostics(output_dir=tmp_path, input_id="input_01")

    result = client.generate_structured_json("system", '{"section_group": []}')

    assert result["ok"] is True
    assert fake.chat.completions.call_count == 2
    assert client.last_retry_count == 1
    raw_files = list(
        (tmp_path / "raw").glob("section_group_generation_attempt_01*.txt")
    )
    assert len(raw_files) == 1
    metadata = json.loads(
        raw_files[0]
        .with_name(raw_files[0].stem + "_metadata.json")
        .read_text(encoding="utf-8")
    )
    assert metadata["provider"] == "grok"
    assert metadata["model"] == GROK_GENERATION_MODEL
    assert metadata["failure_type"] == "truncated_response"
    assert "xai-secret" not in metadata


def test_grok_client_preserves_final_invalid_json_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake = FakeOpenAI([FakeResponse('{"bad": '), FakeResponse('{"still_bad": ')])
    monkeypatch.setenv("XAI_API_KEY", "xai-secret")
    monkeypatch.setattr(grok_client, "OpenAI", lambda **kwargs: fake)
    monkeypatch.setattr(grok_client.time, "sleep", lambda _: None)
    client = GrokJsonClient()
    client.configure_diagnostics(output_dir=tmp_path, input_id="input_01")

    with pytest.raises(OpenAIClientError, match="attempt 2/2"):
        client.generate_structured_json("system", '{"section_group": []}')

    assert fake.chat.completions.call_count == 2
    raw_files = list((tmp_path / "raw").glob("section_group_generation_attempt_*.txt"))
    assert len(raw_files) == 2


class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 40
    total_tokens = 140
    cost_in_usd_ticks = 123456789
    prompt_tokens_details = {"cached_tokens": 25}
    completion_tokens_details = {
        "reasoning_tokens": 0,
        "accepted_prediction_tokens": 3,
        "rejected_prediction_tokens": 4,
    }


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChoice:
    finish_reason = "stop"

    def __init__(self, content: str) -> None:
        self.message = FakeMessage(content)


class FakeResponse:
    def __init__(self, content: str, usage: object | None = None) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = usage or FakeUsage()


class FakeCompletions:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.requests: list[dict[str, object]] = []
        self.call_count = 0

    def create(self, **kwargs: object) -> object:
        self.call_count += 1
        self.requests.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeChat:
    def __init__(self, outcomes: list[object]) -> None:
        self.completions = FakeCompletions(outcomes)


class FakeOpenAI:
    def __init__(self, outcomes: list[object]) -> None:
        self.chat = FakeChat(outcomes)


def _status_error(status_code: int) -> APIStatusError:
    request = httpx.Request("POST", GROK_BASE_URL)
    response = httpx.Response(status_code, request=request)
    return APIStatusError("status error", response=response, body=None)
