"""Workflow telemetry and cost helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from backend.app.config import get_settings

# Static fallback prices are estimates per 1M tokens and must be reviewed
# periodically. Environment variables override these values.
STATIC_OPENAI_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (5.0, 15.0),
    "gpt-4o-mini": (0.15, 0.60),
}


@dataclass
class WorkflowTelemetry:
    started_at: float
    runtime_by_stage: dict[str, float] = field(default_factory=dict)
    llm_call_count_by_stage: dict[str, int] = field(default_factory=dict)
    token_usage_by_stage: dict[str, dict[str, int | float | None]] = field(
        default_factory=dict
    )
    model_used_by_stage: dict[str, str] = field(default_factory=dict)

    def record_runtime(self, stage: str, runtime_seconds: float) -> None:
        self.runtime_by_stage[stage] = round(runtime_seconds, 3)

    def record_calls(self, stage: str, count: int) -> None:
        if count > 0:
            self.llm_call_count_by_stage[stage] = count

    def record_model(self, stage: str, model_name: str | None) -> None:
        if model_name:
            self.model_used_by_stage[stage] = model_name

    def record_tokens(self, stage: str, usage: dict[str, object] | None) -> None:
        if not usage:
            self.token_usage_by_stage[stage] = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "cached_prompt_tokens": None,
                "uncached_prompt_tokens": None,
                "prompt_cache_hit_ratio": None,
            }
            return
        prompt_tokens = _optional_int(usage.get("prompt_tokens"))
        cached_prompt_tokens = _optional_int(usage.get("cached_prompt_tokens"))
        self.token_usage_by_stage[stage] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": _optional_int(usage.get("completion_tokens")),
            "total_tokens": _optional_int(usage.get("total_tokens")),
            "cached_prompt_tokens": cached_prompt_tokens,
            "uncached_prompt_tokens": _uncached_prompt_tokens(
                prompt_tokens,
                cached_prompt_tokens,
            ),
            "prompt_cache_hit_ratio": _prompt_cache_hit_ratio(
                prompt_tokens,
                cached_prompt_tokens,
            ),
        }

    def summary(self, total_runtime_seconds: float) -> dict[str, Any]:
        cost = estimate_workflow_cost(
            self.token_usage_by_stage,
            self.model_used_by_stage,
        )
        prompt_by_stage = {
            stage: usage.get("prompt_tokens")
            for stage, usage in self.token_usage_by_stage.items()
        }
        completion_by_stage = {
            stage: usage.get("completion_tokens")
            for stage, usage in self.token_usage_by_stage.items()
        }
        cached_by_stage = {
            stage: usage.get("cached_prompt_tokens")
            for stage, usage in self.token_usage_by_stage.items()
        }
        uncached_by_stage = {
            stage: usage.get("uncached_prompt_tokens")
            for stage, usage in self.token_usage_by_stage.items()
        }
        cache_ratio_by_stage = {
            stage: usage.get("prompt_cache_hit_ratio")
            for stage, usage in self.token_usage_by_stage.items()
        }
        total_prompt = _sum_int_values(prompt_by_stage)
        total_completion = _sum_int_values(completion_by_stage)
        total_cached = _sum_int_values(cached_by_stage)
        total_uncached = _sum_int_values(uncached_by_stage)
        return {
            "total_runtime_seconds": round(total_runtime_seconds, 3),
            "llm_call_count_total": sum(self.llm_call_count_by_stage.values()),
            "llm_call_count_by_stage": self.llm_call_count_by_stage,
            "runtime_by_stage": self.runtime_by_stage,
            "slowest_stage": _max_key(self.runtime_by_stage),
            "model_used_by_stage": self.model_used_by_stage,
            "token_usage_by_stage": self.token_usage_by_stage,
            "prompt_tokens_by_stage": prompt_by_stage,
            "completion_tokens_by_stage": completion_by_stage,
            "cached_prompt_tokens_by_stage": cached_by_stage,
            "uncached_prompt_tokens_by_stage": uncached_by_stage,
            "prompt_cache_hit_ratio_by_stage": cache_ratio_by_stage,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "cached_prompt_tokens_total": total_cached,
            "uncached_prompt_tokens_total": total_uncached,
            "prompt_cache_hit_ratio": _prompt_cache_hit_ratio(
                total_prompt,
                total_cached,
            ),
            **cost,
        }


def resolve_stage_model(stage: str) -> str | None:
    settings = get_settings()
    stage_models = {
        "planning": settings.openai_model_planning,
        "generation": settings.openai_model_generation,
        "revision": settings.openai_model_revision,
        "evaluation": settings.openai_model_evaluation,
        "length_recovery": settings.openai_model_length_recovery,
    }
    return (
        stage_models.get(stage)
        or settings.openai_model
        or settings.openai_model_default
        or "gpt-4o-mini"
    )


def estimate_workflow_cost(
    token_usage_by_stage: dict[str, dict[str, int | float | None]],
    model_used_by_stage: dict[str, str],
) -> dict[str, Any]:
    cost_by_stage: dict[str, float | None] = {}
    model_breakdown: dict[str, float] = {}
    notes: list[str] = [
        "Costs are estimates based on configured or static per-token pricing."
    ]
    available = True
    for stage, usage in token_usage_by_stage.items():
        model = model_used_by_stage.get(stage)
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if model is None or prompt_tokens is None or completion_tokens is None:
            cost_by_stage[stage] = None
            available = False
            notes.append(f"{stage}: token usage unavailable")
            continue
        prices = _prices_for_model(model)
        if prices is None:
            cost_by_stage[stage] = None
            available = False
            notes.append(f"{stage}: price unknown for {model}")
            continue
        input_price, output_price = prices
        stage_cost = (
            (prompt_tokens / 1_000_000) * input_price
            + (completion_tokens / 1_000_000) * output_price
        )
        rounded = round(stage_cost, 6)
        cost_by_stage[stage] = rounded
        model_breakdown[model] = round(model_breakdown.get(model, 0.0) + rounded, 6)
    known_costs = [value for value in cost_by_stage.values() if value is not None]
    return {
        "estimated_cost_total_usd": (
            round(sum(known_costs), 6) if known_costs else None
        ),
        "estimated_cost_by_stage_usd": cost_by_stage,
        "estimated_cost_model_breakdown": model_breakdown,
        "cost_estimation_available": available,
        "cost_estimation_notes": notes,
        "highest_cost_stage": _max_key(
            {key: value for key, value in cost_by_stage.items() if value is not None}
        ),
    }


def _prices_for_model(model: str) -> tuple[float, float] | None:
    alias = _model_alias(model)
    input_env = os.getenv(f"OPENAI_COST_INPUT_PER_1M_{alias}")
    output_env = os.getenv(f"OPENAI_COST_OUTPUT_PER_1M_{alias}")
    if input_env and output_env:
        return float(input_env), float(output_env)
    return STATIC_OPENAI_PRICING_USD_PER_1M.get(model)


def _model_alias(model: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model.upper())


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _sum_int_values(values: Mapping[str, object]) -> int:
    return sum(value for value in values.values() if isinstance(value, int))


def _uncached_prompt_tokens(
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
) -> int | None:
    if prompt_tokens is None:
        return None
    if cached_prompt_tokens is None:
        return prompt_tokens
    return max(prompt_tokens - cached_prompt_tokens, 0)


def _prompt_cache_hit_ratio(
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
) -> float | None:
    if prompt_tokens is None or prompt_tokens <= 0 or cached_prompt_tokens is None:
        return None
    return round(cached_prompt_tokens / prompt_tokens, 4)


def _max_key(values: dict[str, float]) -> str | None:
    if not values:
        return None
    return max(values, key=lambda key: values[key])
