from backend.app.config import get_settings
from backend.app.services.workflow_telemetry import (
    WorkflowTelemetry,
    estimate_workflow_cost,
    resolve_stage_model,
)


def test_stage_model_config_fallback(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_MODEL", "legacy-model")
    monkeypatch.setenv("OPENAI_MODEL_GENERATION", "generation-model")
    monkeypatch.setenv("OPENAI_MODEL_DEFAULT", "default-model")
    get_settings.cache_clear()

    assert resolve_stage_model("generation") == "generation-model"
    assert resolve_stage_model("revision") == "legacy-model"


def test_stage_model_config_uses_default_without_legacy(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL_DEFAULT", "default-model")
    get_settings.cache_clear()

    assert resolve_stage_model("evaluation") == "default-model"


def test_token_usage_and_cost_aggregation() -> None:
    telemetry = WorkflowTelemetry(started_at=0.0)
    telemetry.record_runtime("generation", 2.5)
    telemetry.record_calls("generation", 3)
    telemetry.record_model("generation", "gpt-4o")
    telemetry.record_tokens(
        "generation",
        {
            "prompt_tokens": 1_000,
            "completion_tokens": 500,
            "total_tokens": 1_500,
        },
    )

    summary = telemetry.summary(total_runtime_seconds=3.0)

    assert summary["total_prompt_tokens"] == 1_000
    assert summary["total_completion_tokens"] == 500
    assert summary["total_tokens"] == 1_500
    assert summary["estimated_cost_by_stage_usd"]["generation"] == 0.0125
    assert summary["estimated_cost_total_usd"] == 0.0125
    assert summary["highest_cost_stage"] == "generation"
    assert summary["cost_estimation_available"] is True


def test_missing_token_usage_marks_cost_unavailable() -> None:
    telemetry = WorkflowTelemetry(started_at=0.0)
    telemetry.record_model("revision", "gpt-4o-mini")
    telemetry.record_tokens("revision", None)

    summary = telemetry.summary(total_runtime_seconds=1.0)

    assert summary["token_usage_by_stage"]["revision"]["prompt_tokens"] is None
    assert summary["estimated_cost_by_stage_usd"]["revision"] is None
    assert summary["cost_estimation_available"] is False


def test_unknown_model_pricing_is_safe() -> None:
    result = estimate_workflow_cost(
        {"generation": {"prompt_tokens": 100, "completion_tokens": 100}},
        {"generation": "unknown-model"},
    )

    assert result["estimated_cost_by_stage_usd"]["generation"] is None
    assert result["cost_estimation_available"] is False
    assert "price unknown" in " ".join(result["cost_estimation_notes"])
