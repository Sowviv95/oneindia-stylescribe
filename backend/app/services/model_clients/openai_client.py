"""OpenAI client wrapper for structured JSON generation."""

import inspect
import json
import time
from collections.abc import Sequence
from typing import Any, cast

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
)

from backend.app.config import get_settings

OPENAI_MODELS_WITH_DEFAULT_ONLY_TEMPERATURE = {"gpt-5.5"}
OPENAI_MODELS_WITH_EXTENDED_TIMEOUT = {"gpt-5.5"}
OPENAI_EXTENDED_TIMEOUT_SECONDS = 240.0
OPENAI_EXTENDED_TIMEOUT_MAX_ATTEMPTS = 2
OPENAI_EXTENDED_TIMEOUT_BACKOFF_SECONDS = 2.0


class OpenAIClientError(RuntimeError):
    """Raised when OpenAI generation cannot be completed."""


class OpenAIStyleClient:
    """Small reusable OpenAI client for JSON generation."""

    provider = "openai"

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        if settings.openai_api_key is None:
            raise OpenAIClientError("OPENAI_API_KEY is required for style profiles.")

        self.model_name = model_name or settings.openai_model or "gpt-4o-mini"
        self.timeout_seconds = _timeout_seconds_for_model(
            self.model_name,
            settings.openai_timeout_seconds,
        )
        self.max_retries = _sdk_max_retries_for_model(
            self.model_name,
            settings.openai_max_retries,
        )
        self.last_attempt_count = 0
        self.last_retry_count = 0
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        self._supports_prompt_cache_key = _supports_prompt_cache_key(self._client)

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        """Generate and parse a JSON object from OpenAI."""

        try:
            response = _create_completion(
                self._client,
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                prompt_cache_key=prompt_cache_key,
                supports_prompt_cache_key=self._supports_prompt_cache_key,
            )
        except APITimeoutError as exc:
            raise OpenAIClientError(
                f"OpenAI request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        content = response.choices[0].message.content
        if content is None:
            raise OpenAIClientError("OpenAI returned an empty response.")
        parsed = _parse_json_object(content)
        parsed["token_usage"] = _token_usage(response.usage)
        return parsed


class OpenAIJsonClient(OpenAIStyleClient):
    """Reusable OpenAI client that returns raw JSON objects."""

    def __init__(
        self,
        model_name: str | None = None,
        missing_key_message: str = "OPENAI_API_KEY is required.",
    ) -> None:
        settings = get_settings()
        if settings.openai_api_key is None:
            raise OpenAIClientError(missing_key_message)

        self.model_name = model_name or settings.openai_model or "gpt-4o-mini"
        self.timeout_seconds = _timeout_seconds_for_model(
            self.model_name,
            settings.openai_timeout_seconds,
        )
        self.max_retries = _sdk_max_retries_for_model(
            self.model_name,
            settings.openai_max_retries,
        )
        self.last_attempt_count = 0
        self.last_retry_count = 0
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=self.timeout_seconds,
            max_retries=self.max_retries,
        )
        self._supports_prompt_cache_key = _supports_prompt_cache_key(self._client)

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        response = self._create_completion_with_retries(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            prompt_cache_key=prompt_cache_key,
        )
        content = response.choices[0].message.content
        if content is None:
            raise OpenAIClientError("OpenAI returned an empty response.")
        parsed = _parse_raw_json_object(content)
        parsed["token_usage"] = _token_usage(response.usage)
        return parsed

    def _create_completion_with_retries(
        self,
        *,
        messages: list[dict[str, str]],
        response_format: dict[str, str],
        temperature: float,
        prompt_cache_key: str | None,
    ) -> Any:
        max_attempts = _max_attempts_for_model(self.model_name)
        self.last_attempt_count = 0
        self.last_retry_count = 0
        while True:
            self.last_attempt_count += 1
            try:
                return _create_completion(
                    self._client,
                    model=self.model_name,
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                    prompt_cache_key=prompt_cache_key,
                    supports_prompt_cache_key=self._supports_prompt_cache_key,
                )
            except Exception as exc:
                if (
                    self.last_attempt_count >= max_attempts
                    or not _is_transient_openai_error(exc)
                ):
                    if isinstance(exc, APITimeoutError):
                        raise OpenAIClientError(
                            "OpenAI request timed out after "
                            f"{self.timeout_seconds:g} seconds."
                        ) from exc
                    raise
                self.last_retry_count += 1
                time.sleep(OPENAI_EXTENDED_TIMEOUT_BACKOFF_SECONDS)


def _token_usage(usage: object) -> dict[str, int | None]:
    prompt_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "cached_prompt_tokens": _cached_prompt_tokens(prompt_details),
    }


def _supports_prompt_cache_key(client: OpenAI) -> bool:
    try:
        signature = inspect.signature(client.chat.completions.create)
    except (AttributeError, TypeError, ValueError):
        return False
    return "prompt_cache_key" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _create_completion(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, str],
    temperature: float,
    prompt_cache_key: str | None,
    supports_prompt_cache_key: bool,
) -> Any:
    create = cast(Any, client.chat.completions.create)
    request: dict[str, object] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
    }
    if _should_send_temperature(model):
        request["temperature"] = temperature
    if prompt_cache_key and supports_prompt_cache_key:
        request["prompt_cache_key"] = prompt_cache_key
    return create(**request)


def _should_send_temperature(model: str) -> bool:
    return model not in OPENAI_MODELS_WITH_DEFAULT_ONLY_TEMPERATURE


def _timeout_seconds_for_model(model: str, default_timeout: float) -> float:
    if model in OPENAI_MODELS_WITH_EXTENDED_TIMEOUT:
        return OPENAI_EXTENDED_TIMEOUT_SECONDS
    return default_timeout


def _sdk_max_retries_for_model(model: str, default_max_retries: int) -> int:
    if model in OPENAI_MODELS_WITH_EXTENDED_TIMEOUT:
        return 0
    return default_max_retries


def _max_attempts_for_model(model: str) -> int:
    if model in OPENAI_MODELS_WITH_EXTENDED_TIMEOUT:
        return OPENAI_EXTENDED_TIMEOUT_MAX_ATTEMPTS
    return 1


def _is_transient_openai_error(exc: Exception) -> bool:
    if isinstance(exc, APITimeoutError | APIConnectionError | InternalServerError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code >= 500
    return False


def temperature_metadata(
    model: str,
    configured_temperature: float,
) -> dict[str, object]:
    if _should_send_temperature(model):
        return {
            "temperature_mode": "explicit",
            "temperature_requested": configured_temperature,
        }
    return {
        "temperature_mode": "model_default",
        "temperature_requested": None,
    }


def request_runtime_metadata(
    model: str,
    configured_temperature: float,
    default_timeout: float,
) -> dict[str, object]:
    return {
        **temperature_metadata(model, configured_temperature),
        "timeout_seconds": _timeout_seconds_for_model(model, default_timeout),
    }


def _cached_prompt_tokens(prompt_details: object) -> int | None:
    if isinstance(prompt_details, dict):
        value = prompt_details.get("cached_tokens")
    else:
        value = getattr(prompt_details, "cached_tokens", None)
    return value if isinstance(value, int) else None


def _parse_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAIClientError("OpenAI returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise OpenAIClientError("OpenAI returned JSON that is not an object.")
    return _coerce_profile_object(parsed)


def _parse_raw_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise OpenAIClientError("OpenAI returned invalid JSON.") from exc
    if not isinstance(parsed, dict):
        raise OpenAIClientError("OpenAI returned JSON that is not an object.")
    return parsed


def _coerce_profile_object(parsed: dict[str, object]) -> dict[str, object]:
    return {
        "overall_tone": _string_value(parsed.get("overall_tone")),
        "headline_style": _string_value(parsed.get("headline_style")),
        "intro_style": _string_value(parsed.get("intro_style")),
        "paragraph_style": _string_value(parsed.get("paragraph_style")),
        "sentence_style": _string_value(parsed.get("sentence_style")),
        "vocabulary_style": _string_value(parsed.get("vocabulary_style")),
        "narrative_flow": _string_value(parsed.get("narrative_flow")),
        "closing_style": _string_value(parsed.get("closing_style")),
        "reader_engagement_style": _string_value(
            parsed.get("reader_engagement_style")
        ),
        "tamil_register": _string_value(parsed.get("tamil_register")),
        "english_or_transliterated_word_usage": _string_value(
            parsed.get("english_or_transliterated_word_usage")
        ),
        "category_specific_observations": _string_list(
            parsed.get("category_specific_observations")
        ),
        "dos": _string_list(parsed.get("dos")),
        "donts": _string_list(parsed.get("donts")),
        "few_shot_usage_guidance": _string_value(
            parsed.get("few_shot_usage_guidance")
        ),
        "generation_guidance": _string_value(parsed.get("generation_guidance")),
        "style_risks": _string_list(parsed.get("style_risks")),
    }


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]
