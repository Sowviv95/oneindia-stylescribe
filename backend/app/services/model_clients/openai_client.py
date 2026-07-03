"""OpenAI client wrapper for structured JSON generation."""

import json
from collections.abc import Sequence

from openai import APITimeoutError, OpenAI

from backend.app.config import get_settings


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
        self.timeout_seconds = settings.openai_timeout_seconds
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        """Generate and parse a JSON object from OpenAI."""

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
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
        self.timeout_seconds = settings.openai_timeout_seconds
        self._client = OpenAI(
            api_key=settings.openai_api_key.get_secret_value(),
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_payload},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except APITimeoutError as exc:
            raise OpenAIClientError(
                f"OpenAI request timed out after {self.timeout_seconds:g} seconds."
            ) from exc
        content = response.choices[0].message.content
        if content is None:
            raise OpenAIClientError("OpenAI returned an empty response.")
        parsed = _parse_raw_json_object(content)
        parsed["token_usage"] = _token_usage(response.usage)
        return parsed


def _token_usage(usage: object) -> dict[str, int | None]:
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


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
