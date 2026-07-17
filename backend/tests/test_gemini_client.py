import pytest

from backend.app.services.model_clients.gemini_client import (
    GEMINI_GENERATION_MODEL,
    GeminiJsonClient,
    GeminiParseFailure,
    _generation_config,
    _parse_raw_json_object,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_parse_raw_json_object_accepts_strict_json() -> None:
    parsed = _parse_raw_json_object('{"ok": true}')

    assert parsed == {"ok": True}


def test_parse_raw_json_object_accepts_single_outer_code_fence() -> None:
    parsed = _parse_raw_json_object('```json\n{"ok": true}\n```')

    assert parsed == {"ok": True}


def test_parse_raw_json_object_extracts_single_object_with_prose() -> None:
    parsed = _parse_raw_json_object('Here is JSON:\n{"ok": true}\nDone.')

    assert parsed == {"ok": True}


def test_parse_raw_json_object_rejects_multiple_objects() -> None:
    with pytest.raises(GeminiParseFailure) as exc_info:
        _parse_raw_json_object('{"ok": true}\n{"extra": true}')

    assert exc_info.value.failure_type == "multiple_json_objects"


def test_parse_raw_json_object_rejects_incomplete_json() -> None:
    with pytest.raises(GeminiParseFailure) as exc_info:
        _parse_raw_json_object('{"ok": true')

    assert exc_info.value.failure_type == "truncated_response"


def test_parse_raw_json_object_rejects_invalid_json() -> None:
    with pytest.raises(GeminiParseFailure, match="Expecting value"):
        _parse_raw_json_object("not-json")


def test_parse_raw_json_object_rejects_bad_escaping_without_repair() -> None:
    with pytest.raises(GeminiParseFailure) as exc_info:
        _parse_raw_json_object('{"bad": "\\q"}')

    assert exc_info.value.failure_type == "invalid_json"


def test_parse_raw_json_object_rejects_non_object() -> None:
    with pytest.raises(OpenAIClientError, match="not an object"):
        _parse_raw_json_object('["not", "object"]')


def test_generation_config_requests_json_object_schema() -> None:
    config = _generation_config("system")

    assert config["system_instruction"] == "system"
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"] == {
        "type": "object",
        "additionalProperties": True,
    }
    assert config["temperature"] == 0.1


def test_gemini_client_retries_transient_503_with_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    models = FakeModels(
        [
            RuntimeError("503 UNAVAILABLE: overloaded"),
            RuntimeError("503 UNAVAILABLE: overloaded"),
            FakeResponse('{"ok": true}'),
        ]
    )
    client = _fake_gemini_client(models)
    monkeypatch.setattr(
        "backend.app.services.model_clients.gemini_client.time.sleep",
        sleeps.append,
    )

    result = client.generate_structured_json("system", "payload")

    assert result["ok"] is True
    assert models.call_count == 3
    assert sleeps == [2.0, 4.0]


def test_gemini_client_does_not_retry_permanent_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    models = FakeModels([RuntimeError("404 NOT_FOUND: no such model")])
    client = _fake_gemini_client(models)
    monkeypatch.setattr(
        "backend.app.services.model_clients.gemini_client.time.sleep",
        sleeps.append,
    )

    with pytest.raises(OpenAIClientError, match="404_NOT_FOUND|404 NOT_FOUND"):
        client.generate_structured_json("system", "payload")

    assert models.call_count == 1
    assert sleeps == []


def test_gemini_client_preserves_final_503_after_bounded_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps: list[float] = []
    models = FakeModels(
        [
            RuntimeError("503 UNAVAILABLE: overloaded 1"),
            RuntimeError("503 UNAVAILABLE: overloaded 2"),
            RuntimeError("503 UNAVAILABLE: overloaded 3"),
            RuntimeError("503 UNAVAILABLE: overloaded final"),
        ]
    )
    client = _fake_gemini_client(models)
    monkeypatch.setattr(
        "backend.app.services.model_clients.gemini_client.time.sleep",
        sleeps.append,
    )

    with pytest.raises(OpenAIClientError, match="overloaded final"):
        client.generate_structured_json("system", "payload")

    assert models.call_count == 4
    assert sleeps == [2.0, 4.0, 8.0]


def test_gemini_client_retries_invalid_json_once_and_saves_raw(tmp_path) -> None:
    models = FakeModels(
        [
            FakeResponse('{"bad": '),
            FakeResponse('{"ok": true}'),
        ]
    )
    client = _fake_gemini_client(models)
    client.configure_diagnostics(output_dir=tmp_path, input_id="input_04")

    result = client.generate_structured_json("system", '{"section_group": []}')

    assert result["ok"] is True
    assert models.call_count == 2
    assert client.last_retry_count == 1
    raw_files = list(
        (tmp_path / "raw").glob("section_group_generation_attempt_01*.txt")
    )
    assert len(raw_files) == 1
    assert client.call_records[0]["failure_type"] == "truncated_response"
    assert client.call_records[0]["usage"]["total_tokens"] == 5


def test_gemini_client_preserves_both_failed_raw_responses(tmp_path) -> None:
    models = FakeModels(
        [
            FakeResponse('{"bad": '),
            FakeResponse('{"still_bad": '),
        ]
    )
    client = _fake_gemini_client(models)
    client.configure_diagnostics(output_dir=tmp_path, input_id="input_04")

    with pytest.raises(OpenAIClientError, match="attempt 2/2"):
        client.generate_structured_json("system", '{"section_group": []}')

    raw_files = list((tmp_path / "raw").glob("section_group_generation_attempt_*.txt"))
    assert len(raw_files) == 2
    assert models.call_count == 2


class FakeUsage:
    prompt_token_count = 3
    candidates_token_count = 2
    total_token_count = 5
    cached_content_token_count = 1


class FakeResponse:
    usage_metadata = FakeUsage()

    def __init__(self, text: str) -> None:
        self.text = text


class FakeModels:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = outcomes
        self.call_count = 0

    def generate_content(self, **kwargs: object) -> object:
        self.call_count += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, models: FakeModels) -> None:
        self.models = models


def _fake_gemini_client(models: FakeModels) -> GeminiJsonClient:
    client = GeminiJsonClient.__new__(GeminiJsonClient)
    client.model_name = GEMINI_GENERATION_MODEL
    client.timeout_seconds = 90.0
    client.last_attempt_count = 0
    client.last_retry_count = 0
    client.call_records = []
    client.diagnostic_dir = None
    client.diagnostic_context = {}
    client._client = FakeClient(models)
    return client
