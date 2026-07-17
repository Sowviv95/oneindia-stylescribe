"""Gemini client wrapper for structured JSON generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from backend.app.config import get_settings
from backend.app.services.model_clients.openai_client import OpenAIClientError

GEMINI_GENERATION_MODEL = "gemini-3.5-flash"
GEMINI_MAX_TRANSIENT_RETRIES = 3
GEMINI_INITIAL_BACKOFF_SECONDS = 2.0
GEMINI_MAX_INVALID_JSON_ATTEMPTS = 2


@dataclass(frozen=True)
class GeminiParseFailure(Exception):
    failure_type: str
    parser_error: str
    raw_response_path: str | None = None
    attempt: int = 1
    max_attempts: int = GEMINI_MAX_INVALID_JSON_ATTEMPTS

    def __str__(self) -> str:
        path = (
            f" Raw response saved to {self.raw_response_path}."
            if self.raw_response_path
            else ""
        )
        return (
            f"Gemini returned invalid JSON during structured_json, attempt "
            f"{self.attempt}/{self.max_attempts}. {self.parser_error}.{path}"
        )


class GeminiJsonClient:
    """Reusable Gemini client that returns raw JSON objects."""

    provider = "gemini"

    def __init__(
        self,
        model_name: str = GEMINI_GENERATION_MODEL,
        missing_key_message: str = "GEMINI_API_KEY is required.",
    ) -> None:
        settings = get_settings()
        if settings.gemini_api_key is None:
            raise OpenAIClientError(missing_key_message)

        try:
            from google import genai
        except ImportError as exc:
            raise OpenAIClientError(
                "google-genai is required for Gemini article generation."
            ) from exc

        self.model_name = model_name
        self.timeout_seconds = settings.openai_timeout_seconds
        self.last_attempt_count = 0
        self.last_retry_count = 0
        self.call_records: list[dict[str, object]] = []
        self.diagnostic_dir: Path | None = None
        self.diagnostic_context: dict[str, object] = {}
        self._client = genai.Client(
            api_key=settings.gemini_api_key.get_secret_value()
        )

    def configure_diagnostics(
        self,
        *,
        output_dir: Path | None,
        input_id: str | None = None,
    ) -> None:
        self.diagnostic_dir = output_dir
        self.diagnostic_context = {"input_id": input_id}

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        """Generate and parse a JSON object from Gemini."""

        _ = prompt_cache_key
        operation = _operation_from_payload(user_payload)
        final_failure: GeminiParseFailure | None = None
        for attempt in range(1, GEMINI_MAX_INVALID_JSON_ATTEMPTS + 1):
            self.last_attempt_count += 1
            response = self._generate_content_with_retries(
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            content = _response_text(response)
            usage = _token_usage(response)
            record = _call_record(
                provider=self.provider,
                model=self.model_name,
                operation=operation,
                attempt=attempt,
                response=response,
                raw_text=content,
                usage=usage,
                status="received",
                context=self.diagnostic_context,
            )
            if content is None:
                record.update(
                    {
                        "status": "failed",
                        "failure_type": "empty_response",
                        "parser_error": "Gemini returned an empty response",
                    }
                )
                path = self._save_raw_response(record, "")
                record["raw_response_path"] = path
                self.call_records.append(record)
                final_failure = GeminiParseFailure(
                    "empty_response",
                    "Gemini returned an empty response",
                    raw_response_path=path,
                    attempt=attempt,
                )
            else:
                try:
                    parsed = _parse_raw_json_object(content)
                except GeminiParseFailure as exc:
                    record.update(
                        {
                            "status": "failed",
                            "failure_type": exc.failure_type,
                            "parser_error": exc.parser_error,
                        }
                    )
                    path = self._save_raw_response(record, content)
                    record["raw_response_path"] = path
                    self.call_records.append(record)
                    final_failure = GeminiParseFailure(
                        exc.failure_type,
                        exc.parser_error,
                        raw_response_path=path,
                        attempt=attempt,
                    )
                else:
                    record["status"] = "parsed"
                    self.call_records.append(record)
                    parsed["token_usage"] = usage
                    return parsed
            if attempt < GEMINI_MAX_INVALID_JSON_ATTEMPTS:
                self.last_retry_count += 1
        if final_failure is not None:
            raise OpenAIClientError(str(final_failure)) from final_failure
        raise OpenAIClientError("Gemini returned invalid JSON.")

    def _generate_content_with_retries(
        self,
        *,
        system_prompt: str,
        user_payload: str,
    ) -> object:
        retry_attempts = 0
        while True:
            try:
                return self._client.models.generate_content(
                    model=self.model_name,
                    contents=user_payload,
                    config=_generation_config(system_prompt),
                )
            except Exception as exc:
                if (
                    not _is_transient_503_unavailable(exc)
                    or retry_attempts >= GEMINI_MAX_TRANSIENT_RETRIES
                ):
                    raise OpenAIClientError(
                        f"Gemini request failed: {exc}"
                    ) from exc
                time.sleep(_backoff_seconds(retry_attempts))
                retry_attempts += 1

    def _save_raw_response(
        self,
        record: dict[str, object],
        raw_text: str,
    ) -> str | None:
        if self.diagnostic_dir is None:
            return None
        raw_dir = self.diagnostic_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        operation = str(record.get("operation") or "generation")
        raw_attempt = record.get("attempt")
        attempt = raw_attempt if isinstance(raw_attempt, int) else 1
        existing = len(list(raw_dir.glob(f"{operation}_attempt_{attempt:02d}*.txt")))
        suffix = f"_{existing + 1:02d}" if existing else ""
        raw_path = raw_dir / f"{operation}_attempt_{attempt:02d}{suffix}.txt"
        metadata_path = (
            raw_dir / f"{operation}_attempt_{attempt:02d}{suffix}_metadata.json"
        )
        raw_path.write_text(raw_text, encoding="utf-8")
        metadata = {**record, "raw_response_path": str(raw_path)}
        metadata.pop("raw_text", None)
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(raw_path)


def _generation_config(system_prompt: str) -> Any:
    return {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_json_schema": {
            "type": "object",
            "additionalProperties": True,
        },
        "temperature": 0.1,
    }


def _is_transient_503_unavailable(exc: Exception) -> bool:
    message = str(exc)
    return "503" in message and "UNAVAILABLE" in message


def _backoff_seconds(retry_attempt: int) -> float:
    return float(GEMINI_INITIAL_BACKOFF_SECONDS * (2**retry_attempt))


def _response_text(response: object) -> str | None:
    text = getattr(response, "text", None)
    return text if isinstance(text, str) else None


def _finish_reason(response: object) -> str | None:
    candidates = getattr(response, "candidates", None)
    if isinstance(candidates, list) and candidates:
        return str(getattr(candidates[0], "finish_reason", "")) or None
    return None


def _token_usage(response: object) -> dict[str, int | None]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "prompt_tokens": _int_attr(usage, "prompt_token_count"),
        "completion_tokens": _int_attr(usage, "candidates_token_count"),
        "total_tokens": _int_attr(usage, "total_token_count"),
        "cached_prompt_tokens": _int_attr(usage, "cached_content_token_count"),
    }


def _int_attr(value: object, name: str) -> int | None:
    if isinstance(value, dict):
        raw = value.get(name)
    else:
        raw = getattr(value, name, None)
    return raw if isinstance(raw, int) else None


def _parse_raw_json_object(content: str) -> dict[str, object]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        fenced = _strip_outer_code_fence(content)
        if fenced != content:
            try:
                parsed = json.loads(fenced)
            except json.JSONDecodeError as fence_exc:
                parsed = _extract_single_json_object(content, fence_exc)
        else:
            parsed = _extract_single_json_object(content, exc)
    if not isinstance(parsed, dict):
        raise OpenAIClientError("Gemini returned JSON that is not an object.")
    return cast(dict[str, object], parsed)


def _strip_outer_code_fence(content: str) -> str:
    stripped = content.strip()
    if not stripped.startswith("```"):
        return content
    lines = stripped.splitlines()
    if len(lines) < 3 or not lines[-1].strip() == "```":
        return content
    first = lines[0].strip()
    if first not in {"```", "```json", "```JSON"}:
        return content
    return "\n".join(lines[1:-1]).strip()


def _extract_single_json_object(
    content: str,
    original_error: json.JSONDecodeError,
) -> object:
    spans = _json_object_spans(content)
    if not spans:
        failure = (
            "truncated_response"
            if content.count("{") > content.count("}")
            else "invalid_json"
        )
        raise GeminiParseFailure(failure, str(original_error)) from original_error
    if len(spans) > 1:
        raise GeminiParseFailure(
            "multiple_json_objects",
            "Response contained multiple top-level JSON objects",
        ) from original_error
    start, end = spans[0]
    try:
        return json.loads(content[start:end])
    except json.JSONDecodeError as exc:
        raise GeminiParseFailure("invalid_json", str(exc)) from original_error


def _json_object_spans(content: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    in_string = False
    escaped = False
    depth = 0
    start: int | None = None
    for index, char in enumerate(content):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    spans.append((start, index + 1))
                    start = None
    return spans


def _operation_from_payload(user_payload: str) -> str:
    try:
        payload = json.loads(user_payload)
    except json.JSONDecodeError:
        return "generation"
    if not isinstance(payload, dict):
        return "generation"
    has_group = "section_group" in payload
    has_retry = "section_retry_request" in payload
    if has_retry and has_group:
        return "section_group_retry"
    if has_group:
        return "section_group_generation"
    if has_retry:
        return "section_retry"
    if "planned_section" in payload:
        return "section_generation"
    return "initial_article_generation"


def _call_record(
    *,
    provider: str,
    model: str,
    operation: str,
    attempt: int,
    response: object,
    raw_text: str | None,
    usage: dict[str, int | None],
    status: str,
    context: dict[str, object],
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "stage": "generation",
        "operation": operation,
        "input_id": context.get("input_id"),
        "attempt": attempt,
        "status": status,
        "raw_response_length": len(raw_text) if raw_text is not None else None,
        "finish_reason": _finish_reason(response),
        "usage": usage,
    }
