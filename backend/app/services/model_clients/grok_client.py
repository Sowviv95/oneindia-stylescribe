"""xAI Grok client wrapper for structured JSON generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI

from backend.app.config import get_settings
from backend.app.services.model_clients.gemini_client import (
    GeminiParseFailure,
    _operation_from_payload,
    _parse_raw_json_object,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError

GROK_GENERATION_MODEL = "grok-4.20-0309-non-reasoning"
GROK_BASE_URL = "https://api.x.ai/v1"
GROK_TIMEOUT_SECONDS = 180.0
GROK_MAX_ATTEMPTS = 2
GROK_BACKOFF_SECONDS = 2.0
GROK_TICKS_PER_USD = Decimal("10000000000")
USD_QUANT = Decimal("0.000001")


@dataclass(frozen=True)
class GrokParseFailure(Exception):
    failure_type: str
    parser_error: str
    raw_response_path: str | None = None
    attempt: int = 1
    max_attempts: int = GROK_MAX_ATTEMPTS

    def __str__(self) -> str:
        path = (
            f" Raw response saved to {self.raw_response_path}."
            if self.raw_response_path
            else ""
        )
        return (
            f"Grok returned invalid JSON during structured_json, attempt "
            f"{self.attempt}/{self.max_attempts}. {self.parser_error}.{path}"
        )


class GrokJsonClient:
    """Reusable xAI Grok client that returns raw JSON objects."""

    provider = "grok"

    def __init__(
        self,
        model_name: str = GROK_GENERATION_MODEL,
        missing_key_message: str = "XAI_API_KEY is required.",
    ) -> None:
        settings = get_settings()
        if settings.xai_api_key is None:
            raise OpenAIClientError(missing_key_message)

        self.model_name = model_name
        self.timeout_seconds = GROK_TIMEOUT_SECONDS
        self.last_attempt_count = 0
        self.last_retry_count = 0
        self.call_records: list[dict[str, object]] = []
        self.diagnostic_dir: Path | None = None
        self.diagnostic_context: dict[str, object] = {}
        self._client = OpenAI(
            api_key=settings.xai_api_key.get_secret_value(),
            base_url=GROK_BASE_URL,
            timeout=self.timeout_seconds,
            max_retries=0,
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
        """Generate and parse a JSON object from Grok."""

        _ = prompt_cache_key
        operation = _operation_from_payload(user_payload)
        final_failure: Exception | None = None
        for attempt in range(1, GROK_MAX_ATTEMPTS + 1):
            self.last_attempt_count += 1
            try:
                response = self._create_completion(
                    system_prompt=system_prompt,
                    user_payload=user_payload,
                )
            except Exception as exc:
                final_failure = exc
                if attempt >= GROK_MAX_ATTEMPTS or not _is_transient_grok_error(exc):
                    raise _client_error(exc) from exc
                self.last_retry_count += 1
                time.sleep(_backoff_seconds(attempt - 1))
                continue

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
                final_failure = GrokParseFailure(
                    "empty_response",
                    "Grok returned an empty response",
                    attempt=attempt,
                )
                record["failure_type"] = "empty_response"
                record["parser_error"] = "Grok returned an empty response"
                path = self._save_failed_response(record, "")
                record["raw_response_path"] = path
                self.call_records.append(record)
            else:
                try:
                    parsed = _parse_raw_json_object(content)
                except GeminiParseFailure as exc:
                    final_failure = GrokParseFailure(
                        exc.failure_type,
                        exc.parser_error,
                        attempt=attempt,
                    )
                    record.update(
                        {
                            "failure_type": exc.failure_type,
                            "parser_error": exc.parser_error,
                        }
                    )
                    path = self._save_failed_response(record, content)
                    record["raw_response_path"] = path
                    self.call_records.append(record)
                except OpenAIClientError as exc:
                    final_failure = GrokParseFailure(
                        "schema_validation_failed",
                        str(exc),
                        attempt=attempt,
                    )
                    record.update(
                        {
                            "failure_type": "schema_validation_failed",
                            "parser_error": str(exc),
                        }
                    )
                    path = self._save_failed_response(record, content)
                    record["raw_response_path"] = path
                    self.call_records.append(record)
                else:
                    record["status"] = "parsed"
                    self.call_records.append(record)
                    parsed["token_usage"] = usage
                    return parsed
            if attempt < GROK_MAX_ATTEMPTS:
                self.last_retry_count += 1
                time.sleep(_backoff_seconds(attempt - 1))
        if isinstance(final_failure, GrokParseFailure):
            raise OpenAIClientError(str(final_failure)) from final_failure
        raise OpenAIClientError("Grok returned invalid JSON.")

    def _create_completion(
        self,
        *,
        system_prompt: str,
        user_payload: str,
    ) -> object:
        return self._client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

    def _save_failed_response(
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


def _response_text(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else None
    return None


def _finish_reason(response: object) -> str | None:
    choices = getattr(response, "choices", None)
    if isinstance(choices, list) and choices:
        reason = getattr(choices[0], "finish_reason", None)
        return str(reason) if reason is not None else None
    return None


def _token_usage(response: object) -> dict[str, object]:
    usage = getattr(response, "usage", None)
    prompt_details = _detail_object(usage, "prompt_tokens_details")
    completion_details = _detail_object(usage, "completion_tokens_details")
    ticks = _int_attr(usage, "cost_in_usd_ticks")
    converted = _cost_usd_from_ticks(ticks)
    return {
        "prompt_tokens": _int_attr(usage, "prompt_tokens"),
        "completion_tokens": _int_attr(usage, "completion_tokens"),
        "total_tokens": _int_attr(usage, "total_tokens"),
        "cached_prompt_tokens": _int_attr(prompt_details, "cached_tokens"),
        "reasoning_tokens": _int_attr(completion_details, "reasoning_tokens"),
        "accepted_prediction_tokens": _int_attr(
            completion_details,
            "accepted_prediction_tokens",
        ),
        "rejected_prediction_tokens": _int_attr(
            completion_details,
            "rejected_prediction_tokens",
        ),
        "provider_cost_ticks": ticks,
        "provider_reported_cost_usd": converted,
        "provider_cost_conversion_status": (
            "converted" if ticks is not None else "unavailable"
        ),
    }


def _detail_object(parent: object, name: str) -> object:
    if isinstance(parent, dict):
        return parent.get(name)
    return getattr(parent, name, None)


def _int_attr(value: object, name: str) -> int | None:
    if isinstance(value, dict):
        raw = value.get(name)
    else:
        raw = getattr(value, name, None)
    return raw if isinstance(raw, int) else None


def _cost_usd_from_ticks(ticks: int | None) -> float | None:
    if ticks is None:
        return None
    value = Decimal(ticks) / GROK_TICKS_PER_USD
    return float(value.quantize(USD_QUANT, rounding=ROUND_HALF_UP))


def _call_record(
    *,
    provider: str,
    model: str,
    operation: str,
    attempt: int,
    response: object,
    raw_text: str | None,
    usage: dict[str, object],
    status: str,
    context: dict[str, object],
) -> dict[str, object]:
    return {
        "provider": provider,
        "model": model,
        "timestamp": datetime.now(UTC).isoformat(),
        "operation": operation,
        "attempt": attempt,
        "status": status,
        "raw_response_length": len(raw_text) if raw_text is not None else None,
        "finish_reason": _finish_reason(response),
        "usage": usage,
        **context,
    }


def _is_transient_grok_error(exc: Exception) -> bool:
    if isinstance(exc, APITimeoutError | APIConnectionError):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _client_error(exc: Exception) -> OpenAIClientError:
    if isinstance(exc, APITimeoutError):
        return OpenAIClientError(
            f"Grok request timed out after {GROK_TIMEOUT_SECONDS:g} seconds."
        )
    return OpenAIClientError(f"Grok request failed: {exc}")


def _backoff_seconds(retry_attempt: int) -> float:
    return float(GROK_BACKOFF_SECONDS * (2**retry_attempt))
