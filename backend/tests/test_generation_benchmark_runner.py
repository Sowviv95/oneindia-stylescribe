# ruff: noqa: E501

import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from docx import Document

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "run_generation_benchmark.py"
spec = importlib.util.spec_from_file_location("generation_benchmark_runner", SCRIPT_PATH)
runner = importlib.util.module_from_spec(spec)
sys.modules["generation_benchmark_runner"] = runner
assert spec.loader is not None
spec.loader.exec_module(runner)


def test_manifest_validation_requires_ten_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "inputs.json"
    manifest.write_text('{"inputs": []}', encoding="utf-8")

    with pytest.raises(runner.BenchmarkError, match="exactly 10"):
        runner.load_input_manifest(manifest)


def test_manifest_supports_ten_non_generated_source_inputs(tmp_path: Path) -> None:
    manifest = _write_text_manifest(tmp_path)

    loaded = runner.load_input_manifest(manifest)

    assert len(loaded["inputs"]) == 10
    assert loaded["inputs"][0]["input_id"] == "input_01"
    assert loaded["inputs"][0]["source_text"].startswith("Source body")


def test_word_input_discovery_sorts_and_assigns_ids(tmp_path: Path) -> None:
    _write_docx(tmp_path / "B Story.docx", ["B source text long enough for extraction."])
    _write_docx(tmp_path / "a Story.docx", ["A source text long enough for extraction."])
    _write_docx(tmp_path / "legacy.doc", ["unsupported"])
    for index in range(8):
        _write_docx(tmp_path / f"z{index}.docx", [f"Z source {index} long enough for extraction."])

    manifest, discovery = runner.load_word_input_manifest(
        tmp_path,
        author_id="v_vasanthi",
        output_dir=tmp_path / "out",
    )

    assert discovery["word_file_count"] == 11
    assert discovery["unsupported_files"][0]["filename"] == "legacy.doc"
    assert manifest["inputs"][0]["input_id"] == "input_01"
    assert manifest["inputs"][0]["original_filename"] == "a Story.docx"
    assert manifest["inputs"][1]["original_filename"] == "B Story.docx"


def test_word_discovery_reports_empty_docx(tmp_path: Path) -> None:
    _write_docx(tmp_path / "empty.docx", ["   "])

    discovery = runner.discover_word_inputs(tmp_path, tmp_path / "out")

    assert discovery["empty_or_unreadable_files"][0]["filename"] == "empty.docx"
    assert discovery["files"][0]["expected_source_json_path"].endswith("source.json")


def test_prepare_dry_run_makes_no_api_calls(monkeypatch, tmp_path: Path) -> None:
    manifest = _write_text_manifest(tmp_path)

    def fail_api(*args, **kwargs):
        raise AssertionError("API call should not happen")

    monkeypatch.setattr(runner, "generate_grounded_brief", fail_api)
    monkeypatch.setattr(runner, "generate_article_plan", fail_api)

    result = runner.prepare_command(
        SimpleNamespace(
            input_manifest=str(manifest),
            input_dir=None,
            output_dir=str(tmp_path / "bench"),
            input_id=None,
            start_from=None,
            max_inputs=2,
            resume=False,
            overwrite=False,
            dry_run=True,
            author_id="v_vasanthi",
        )
    )

    assert result["selected_input_ids"] == ["input_01", "input_02"]


def test_prepare_creates_one_brief_and_plan_per_input(monkeypatch, tmp_path: Path) -> None:
    manifest = _write_text_manifest(tmp_path)
    calls = {"brief": 0, "plan": 0}

    def fake_brief(**kwargs):
        calls["brief"] += 1
        return _brief_response(f"brief_{calls['brief']:02d}")

    def fake_plan(**kwargs):
        calls["plan"] += 1
        return _plan_response(f"plan_{calls['plan']:02d}", kwargs["brief_id"])

    monkeypatch.setattr(runner, "generate_grounded_brief", fake_brief)
    monkeypatch.setattr(runner, "generate_article_plan", fake_plan)

    runner.prepare_command(
        SimpleNamespace(
            input_manifest=str(manifest),
            input_dir=None,
            output_dir=str(tmp_path / "bench"),
            input_id=None,
            start_from=None,
            max_inputs=2,
            resume=False,
            overwrite=False,
            dry_run=False,
            author_id="v_vasanthi",
        )
    )

    assert calls == {"brief": 2, "plan": 2}
    prepared = runner._read_json(tmp_path / "bench" / "shared" / "manifest.json")
    assert prepared["inputs"][0]["brief_id"] == "brief_01"
    assert prepared["inputs"][1]["plan_id"] == "plan_02"


def test_generation_reuses_brief_and_plan_without_regeneration(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    calls = {"generation": 0, "evaluation": 0}

    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("openai", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(runner, "generate_grounded_brief", lambda **kwargs: pytest.fail("brief regenerated"))
    monkeypatch.setattr(runner, "generate_article_plan", lambda **kwargs: pytest.fail("plan regenerated"))

    def fake_generation(**kwargs):
        calls["generation"] += 1
        assert kwargs["brief_id"] == "brief_01"
        assert kwargs["plan_id"] == "plan_01"
        return _draft_response("draft_01")

    def fake_evaluation(*args, **kwargs):
        calls["evaluation"] += 1
        return _evaluation_response()

    monkeypatch.setattr(runner, "generate_article_draft", fake_generation)
    monkeypatch.setattr(runner, "evaluate_draft_grounding", fake_evaluation)

    runner.generate_command(
        SimpleNamespace(
            provider="openai",
            model="gpt-5.5",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id="input_01",
            start_from=None,
            max_inputs=None,
            resume=False,
            overwrite=False,
            dry_run=False,
        )
    )

    assert calls == {"generation": 1, "evaluation": 1}


def test_provider_model_validation_accepts_integrated_models() -> None:
    runner.validate_provider_model("openai", "gpt-5.5")
    runner.validate_provider_model("gemini", "gemini-3.5-flash")
    runner.validate_provider_model("grok", "grok-4.20-0309-non-reasoning")

    with pytest.raises(runner.BenchmarkError, match="Unsupported model"):
        runner.validate_provider_model("openai", "gpt-4o-mini")
    with pytest.raises(runner.BenchmarkError, match="Unsupported model"):
        runner.validate_provider_model("grok", "grok-4")


def test_input_selection_modes(tmp_path: Path) -> None:
    manifest = runner.load_input_manifest(_write_text_manifest(tmp_path))
    entries = manifest["inputs"]

    assert runner.select_inputs(entries, input_id="input_03", start_from=None, max_inputs=None).input_ids == ["input_03"]
    assert runner.select_inputs(entries, input_id=None, start_from="input_08", max_inputs=None).input_ids == ["input_08", "input_09", "input_10"]
    assert runner.select_inputs(entries, input_id=None, start_from="input_04", max_inputs=2).input_ids == ["input_04", "input_05"]


def test_generate_dry_run_makes_no_api_calls(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)

    monkeypatch.setattr(runner, "make_generation_client", lambda *args: pytest.fail("API client should not be created"))

    result = runner.generate_command(
        SimpleNamespace(
            provider="openai",
            model="gpt-5.5",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id="input_01",
            start_from=None,
            max_inputs=None,
            resume=False,
            overwrite=False,
            dry_run=True,
        )
    )

    assert result["selected_input_ids"] == ["input_01"]


def test_resume_skips_completed_output_but_not_stale_model(tmp_path: Path) -> None:
    manifest = runner._read_json(_prepared_manifest(tmp_path))
    entry = manifest["inputs"][0]
    output = tmp_path / "bench" / "gpt_5_5" / "input_01" / "response.json"
    output.parent.mkdir(parents=True)
    runner._write_json(output, _response_payload(entry, "openai", "gpt-5.5", "completed"))

    assert runner.is_valid_completed_output(output, entry, "openai", "gpt-5.5")

    runner._write_json(output, _response_payload(entry, "openai", "gpt-4o-mini", "completed"))
    assert not runner.is_valid_completed_output(output, entry, "openai", "gpt-5.5")


def test_resume_does_not_skip_empty_completed_output(tmp_path: Path) -> None:
    manifest = runner._read_json(_prepared_manifest(tmp_path))
    entry = manifest["inputs"][0]
    output = tmp_path / "bench" / "gemini_3_5_flash" / "input_01" / "response.json"
    output.parent.mkdir(parents=True)
    payload = _response_payload(entry, "gemini", "gemini-3.5-flash", "completed")
    payload["generated_tamil_article"] = ""
    payload["word_count"] = None
    runner._write_json(output, payload)

    assert not runner.is_valid_completed_output(
        output,
        entry,
        "gemini",
        "gemini-3.5-flash",
    )


def test_overwrite_reruns_selected_input(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    calls = {"generation": 0}
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("openai", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(runner, "generate_article_draft", lambda **kwargs: calls.__setitem__("generation", calls["generation"] + 1) or _draft_response("draft"))
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda *args, **kwargs: _evaluation_response())

    runner.generate_command(_generate_args(tmp_path, manifest_path, resume=True, overwrite=False))
    runner.generate_command(_generate_args(tmp_path, manifest_path, resume=True, overwrite=False))
    runner.generate_command(_generate_args(tmp_path, manifest_path, resume=True, overwrite=True))

    assert calls["generation"] == 2


def test_failed_input_persists_and_later_inputs_continue(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("openai", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))

    def fake_generation(**kwargs):
        if kwargs["brief_id"] == "brief_01":
            raise RuntimeError("sample failed")
        return _draft_response("draft_ok")

    monkeypatch.setattr(runner, "generate_article_draft", fake_generation)
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda *args, **kwargs: _evaluation_response())

    runner.generate_command(
        SimpleNamespace(
            provider="openai",
            model="gpt-5.5",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id=None,
            start_from=None,
            max_inputs=2,
            resume=False,
            overwrite=False,
            dry_run=False,
        )
    )

    first = runner._read_json(tmp_path / "bench" / "gpt_5_5" / "input_01" / "response.json")
    second = runner._read_json(tmp_path / "bench" / "gpt_5_5" / "input_02" / "response.json")
    assert first["completion_status"] == "failed"
    assert second["completion_status"] == "completed"


def test_run_summary_json_and_csv_are_written(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("openai", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(runner, "generate_article_draft", lambda **kwargs: _draft_response("draft"))
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda *args, **kwargs: _evaluation_response())

    runner.generate_command(_generate_args(tmp_path, manifest_path))

    assert (tmp_path / "bench" / "gpt_5_5" / "run_summary.json").exists()
    assert (tmp_path / "bench" / "gpt_5_5" / "run_summary.csv").read_text(encoding="utf-8").startswith("input_id,provider")


def test_gemini_article_body_maps_to_canonical_article() -> None:
    draft = _draft_response("draft", article_field="article_body")

    article = runner.extract_canonical_article(draft)

    assert article.headline == "Headline"
    assert article.subheadline == "Subheadline"
    assert article.source_field == "draft.article_body"
    assert article.article_body.startswith("தமிழ் கட்டுரை")
    assert article.word_count > 0


def test_openai_article_body_maps_to_canonical_article() -> None:
    draft = _draft_response("draft", article_field="article_body")

    article = runner.extract_canonical_article(draft)

    assert article.headline == "Headline"
    assert article.subheadline == "Subheadline"
    assert article.article_body.startswith("தமிழ் கட்டுரை")


def test_legacy_article_field_maps_to_canonical_article() -> None:
    draft = _draft_response("draft", article_field="article")

    article = runner.extract_canonical_article(draft)

    assert article.source_field == "draft.article"
    assert article.article_body.startswith("தமிழ் கட்டுரை")


def test_same_canonical_article_is_persisted_to_response_and_html(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    captured = {}
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("gemini", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))

    draft = _draft_response("draft", article_field="article_body")
    canonical = runner.extract_canonical_article(draft)

    def fake_generation(**kwargs):
        return draft

    def fake_evaluation(draft_id, **kwargs):
        captured["draft_id"] = draft_id
        captured["canonical_article"] = canonical.article_body
        return _evaluation_response()

    monkeypatch.setattr(runner, "generate_article_draft", fake_generation)
    monkeypatch.setattr(runner, "evaluate_draft_grounding", fake_evaluation)

    runner.generate_command(
        SimpleNamespace(
            provider="gemini",
            model="gemini-3.5-flash",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id="input_01",
            start_from=None,
            max_inputs=None,
            resume=False,
            overwrite=False,
            dry_run=False,
        )
    )

    response = runner._read_json(
        tmp_path / "bench" / "gemini_3_5_flash" / "input_01" / "response.json"
    )
    html = (
        tmp_path / "bench" / "gemini_3_5_flash" / "input_01" / "article.html"
    ).read_text(encoding="utf-8")
    assert captured["draft_id"] == "draft"
    assert response["generated_tamil_article"] == captured["canonical_article"]
    assert response["readiness"] == "safe_to_review"
    assert response["generated_tamil_article"] in html


def test_word_count_is_calculated_from_article_text() -> None:
    draft = _draft_response("draft", article_field="article_body")

    article = runner.extract_canonical_article(draft)

    assert article.word_count == runner.approximate_tamil_word_count(article.article_body)
    assert article.model_reported_word_count == 999


def test_empty_article_is_failed_and_not_evaluated(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    calls = {"evaluation": 0}
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("gemini", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(
        runner,
        "generate_article_draft",
        lambda **kwargs: _draft_response("draft", article_text="", article_field="article_body"),
    )

    def fake_evaluation(*args, **kwargs):
        calls["evaluation"] += 1
        return _evaluation_response()

    monkeypatch.setattr(runner, "evaluate_draft_grounding", fake_evaluation)

    runner.generate_command(
        SimpleNamespace(
            provider="gemini",
            model="gemini-3.5-flash",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id="input_01",
            start_from=None,
            max_inputs=None,
            resume=False,
            overwrite=False,
            dry_run=False,
        )
    )

    response = runner._read_json(
        tmp_path / "bench" / "gemini_3_5_flash" / "input_01" / "response.json"
    )
    assert response["completion_status"] == "failed"
    assert response["error_message"] == "empty_generated_article"
    assert response["draft"]["draft"]["article_body"] == ""
    assert calls["evaluation"] == 0


def test_html_contains_article_and_does_not_render_none() -> None:
    response = {
        "input_id": "input_01",
        "source_title": "Source",
        "provider": "gemini",
        "generation_model": "gemini-3.5-flash",
        "author_id": "v_vasanthi",
        "brief_id": "brief",
        "plan_id": "plan",
        "generated_headline": "Headline",
        "generated_subheadline": "Subheadline",
        "generated_tamil_article": "தமிழ் கட்டுரை உரை",
        "word_count": 3,
        "grounding_score": None,
        "readiness": None,
        "blockers": [],
        "warnings": [],
        "unsupported_claims": [],
        "completion_status": "completed",
    }

    html = runner._article_html(response, {"total_elapsed_runtime_seconds": None})

    assert "தமிழ் கட்டுரை உரை" in html
    assert "None" not in html
    assert "Not evaluated" in html
    assert "Not available" in html


def test_article_html_displays_separate_cost_and_token_groups() -> None:
    response = {
        "input_id": "input_01",
        "source_title": "Source",
        "provider": "gemini",
        "generation_model": "gemini-3.5-flash",
        "author_id": "v_vasanthi",
        "brief_id": "brief",
        "plan_id": "plan",
        "generated_headline": "Headline",
        "generated_subheadline": "Subheadline",
        "generated_tamil_article": "தமிழ் கட்டுரை உரை",
        "word_count": 3,
        "grounding_score": 90,
        "readiness": "safe_to_review",
        "blockers": [],
        "warnings": [],
        "unsupported_claims": [],
        "completion_status": "completed",
    }
    telemetry = {
        "total_elapsed_runtime_seconds": 12.3,
        "cost_currency": "USD",
        "pricing_configuration_id": "benchmark_pricing_2026_07_16",
        "pricing_effective_date": "2026-07-16",
        "cost_accuracy": "per_call_calculated",
        "generation_call_ledger": [{"thinking_tokens": 7, "reasoning_tokens": 0}],
        "cost_breakdown": {
            "generation": {
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "prompt_tokens": 100,
                "cached_prompt_tokens": 10,
                "completion_tokens": 50,
                "provider_total_tokens": 200,
                "reasoning_tokens": 0,
                "provider_cost_ticks": 123456789,
                "provider_reported_cost_usd": 0.012346,
                "provider_cost_conversion_status": "converted",
                "token_reconciliation_status": "provider_includes_thinking_tokens",
                "total_cost_usd": 0.000601,
                "cost_status": "calculated",
            },
            "grounding_evaluation": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_tokens": 80,
                "cached_prompt_tokens": 5,
                "completion_tokens": 20,
                "provider_total_tokens": 100,
                "total_cost_usd": 0.000023,
                "cost_status": "calculated",
            },
            "combined": {
                "total_cost_usd": 0.000624,
                "cost_status": "calculated",
            },
        },
    }

    html = runner._article_html(response, telemetry)

    assert "Generation Usage" in html
    assert "Grounding Evaluation Usage" in html
    assert "Combined Cost" in html
    assert "Generation cost:</strong> USD 0.000601" in html
    assert "Grounding evaluation cost:</strong> USD 0.000023" in html
    assert "Combined total cost:</strong> USD 0.000624" in html
    assert "Provider-reported generation total tokens" in html
    assert "Generation reasoning tokens:</strong> 0" in html
    assert "Provider cost ticks:</strong> 123456789" in html
    assert "Provider-reported cost:</strong> USD 0.012346" in html
    assert "Provider cost conversion status:</strong> converted" in html
    assert "<strong>Tokens:</strong>" not in html
    assert "None" not in html


def test_article_html_renders_unavailable_cost_reason() -> None:
    response = {
        "input_id": "input_01",
        "source_title": "Source",
        "provider": "gemini",
        "generation_model": "gemini-3.5-flash-lite",
        "author_id": "v_vasanthi",
        "brief_id": "brief",
        "plan_id": "plan",
        "generated_headline": "Headline",
        "generated_subheadline": "Subheadline",
        "generated_tamil_article": "தமிழ் கட்டுரை உரை",
        "word_count": 3,
        "grounding_score": 90,
        "readiness": "safe_to_review",
        "blockers": [],
        "warnings": [],
        "unsupported_claims": [],
        "completion_status": "completed",
    }

    html = runner._article_html(
        response,
        {
            "cost_breakdown": {
                "generation": {
                    "provider": "gemini",
                    "model": "gemini-3.5-flash-lite",
                    "total_cost_usd": None,
                    "cost_status": "pricing_unavailable",
                }
            }
        },
    )

    assert "Generation cost:</strong> Not available" in html
    assert "Reason:</strong> pricing unavailable" in html


def test_pricing_lookup_has_exact_configured_models() -> None:
    assert runner.pricing_lookup("gemini", "gemini-3.5-flash")[
        "input_usd_per_million"
    ] == 1.5
    assert runner.pricing_lookup("openai", "gpt-5.5")[
        "output_usd_per_million"
    ] == 30.0
    assert runner.pricing_lookup("openai", "gpt-4o-mini")[
        "cached_input_usd_per_million"
    ] == 0.075
    assert runner.pricing_lookup("grok", "grok-4.20-0309-non-reasoning")[
        "input_usd_per_million"
    ] == 1.25
    assert runner.pricing_lookup("gemini", "gemini-3.1-flash-lite")[
        "output_usd_per_million"
    ] == 1.5
    assert runner.pricing_lookup("gemini", "gemini-3.5-flash-lite") is None


def test_decimal_cost_calculation_and_missing_pricing(tmp_path: Path) -> None:
    pricing = tmp_path / "pricing.json"
    pricing.write_text(
        '{"version":"test","models":[{"provider":"openai","model":"gpt-5.5","input_usd_per_million":2,"cached_input_usd_per_million":1,"output_usd_per_million":8,"currency":"USD","effective_date":"2026-01-01","pricing_mode":"standard"}]}',
        encoding="utf-8",
    )

    estimated = runner.calculate_cost(
        provider="openai",
        model="gpt-5.5",
        prompt_tokens=1000,
        cached_prompt_tokens=100,
        completion_tokens=500,
        pricing_path=pricing,
    )
    missing = runner.calculate_cost(
        provider="gemini",
        model="unknown",
        prompt_tokens=1000,
        cached_prompt_tokens=None,
        completion_tokens=500,
        pricing_path=pricing,
    )

    assert estimated["cost_status"] == "calculated"
    assert estimated["input_cost_usd"] == 0.0018
    assert estimated["cached_input_cost_usd"] == 0.0001
    assert estimated["output_cost_usd"] == 0.004
    assert estimated["total_cost_usd"] == 0.0059
    assert missing["total_cost_usd"] is None
    assert missing["cost_status"] == "pricing_unavailable"


def test_generation_and_evaluation_costs_are_separate_and_combined() -> None:
    payload = runner.build_cost_payload(
        generation_provider="gemini",
        generation_model="gemini-3.5-flash",
        generation_usage={
            "prompt_tokens": 1000,
            "cached_prompt_tokens": 100,
            "completion_tokens": 500,
            "total_tokens": 1600,
        },
        generation_ledger=[],
        evaluation_provider="openai",
        evaluation_model="gpt-4o-mini",
        evaluation_usage={
            "prompt_tokens": 1000,
            "cached_prompt_tokens": 100,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    )

    generation = payload["cost_breakdown"]["generation"]
    evaluation = payload["cost_breakdown"]["grounding_evaluation"]
    combined = payload["cost_breakdown"]["combined"]
    assert generation["total_cost_usd"] == 0.005865
    assert evaluation["total_cost_usd"] == 0.000443
    assert combined["total_cost_usd"] == 0.006308
    assert generation["model"] == "gemini-3.5-flash"
    assert evaluation["model"] == "gpt-4o-mini"


def test_openai_generation_uses_gpt_5_5_rates() -> None:
    cost = runner.calculate_cost(
        provider="openai",
        model="gpt-5.5",
        prompt_tokens=1000,
        cached_prompt_tokens=100,
        completion_tokens=500,
    )

    assert cost["total_cost_usd"] == 0.01955


def test_grok_generation_uses_exact_grok_pricing() -> None:
    cost = runner.calculate_cost(
        provider="grok",
        model="grok-4.20-0309-non-reasoning",
        prompt_tokens=1000,
        cached_prompt_tokens=100,
        completion_tokens=200,
    )

    assert cost["input_cost_usd"] == 0.001125
    assert cost["cached_input_cost_usd"] == 0.00002
    assert cost["output_cost_usd"] == 0.0005
    assert cost["total_cost_usd"] == 0.001645


def test_generation_ledger_includes_retries_and_unselected_retry_cost() -> None:
    draft = {
        "section_generation_trace": [
            {
                "section_id": "section_1",
                "first_pass_token_usage": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "retry_token_usage": {
                    "prompt_tokens": 120,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 60,
                    "total_tokens": 180,
                },
            }
        ],
        "token_usage": {
            "prompt_tokens": 220,
            "cached_prompt_tokens": 0,
            "completion_tokens": 110,
            "total_tokens": 330,
        },
    }

    ledger = runner.build_generation_call_ledger(
        provider="gemini",
        model="gemini-3.5-flash",
        draft=draft,
    )
    total = runner.aggregate_generation_ledger(
        "gemini",
        "gemini-3.5-flash",
        draft["token_usage"],
        ledger,
    )

    assert len(ledger) == 2
    assert ledger[1]["operation"] == "section_retry"
    assert total["prompt_tokens"] == 220
    assert total["completion_tokens"] == 110
    assert total["total_cost_usd"] == 0.00132


def test_failed_client_call_usage_is_billable() -> None:
    client = SimpleNamespace(
        call_records=[
            {
                "provider": "gemini",
                "model": "gemini-3.5-flash",
                "operation": "section_group_generation",
                "attempt": 1,
                "status": "failed",
                "failure_type": "invalid_json",
                "raw_response_path": "raw/attempt.txt",
                "usage": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 0,
                    "completion_tokens": 50,
                    "total_tokens": 500,
                },
            }
        ]
    )

    ledger = runner.build_client_call_ledger(client)
    total = runner.aggregate_generation_ledger(
        "gemini",
        "gemini-3.5-flash",
        {},
        ledger,
    )

    assert ledger[0]["call_status"] == "failed"
    assert ledger[0]["failure_type"] == "invalid_json"
    assert ledger[0]["total_cost_usd"] == 0.0006
    assert total["total_cost_usd"] == 0.0006


def test_grok_client_ledger_preserves_reasoning_and_provider_cost() -> None:
    client = SimpleNamespace(
        call_records=[
            {
                "provider": "grok",
                "model": "grok-4.20-0309-non-reasoning",
                "operation": "section_group_generation",
                "attempt": 1,
                "status": "parsed",
                "usage": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "reasoning_tokens": 0,
                    "accepted_prediction_tokens": 2,
                    "rejected_prediction_tokens": 1,
                    "provider_cost_ticks": 123456789,
                    "provider_reported_cost_usd": 0.012346,
                    "provider_cost_conversion_status": "converted",
                },
            }
        ]
    )

    ledger = runner.build_client_call_ledger(client)
    total = runner.aggregate_generation_ledger(
        "grok",
        "grok-4.20-0309-non-reasoning",
        {},
        ledger,
    )

    assert ledger[0]["reasoning_tokens"] == 0
    assert ledger[0]["provider_cost_ticks"] == 123456789
    assert total["provider_cost_ticks"] == 123456789
    assert total["provider_reported_cost_usd"] == 0.012346
    assert total["provider_cost_conversion_status"] == "converted"


def test_token_reconciliation_surfaces_thinking_difference() -> None:
    result = runner.aggregate_generation_ledger(
        "gemini",
        "gemini-3.5-flash",
        {
            "prompt_tokens": 100,
            "cached_prompt_tokens": 0,
            "completion_tokens": 50,
            "total_tokens": 1000,
        },
        [],
    )

    assert result["token_reconciliation_status"] == "provider_includes_thinking_tokens"
    assert result["token_reconciliation_difference"] == 850


def test_global_configuration_error_stops_invocation(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(runner, "make_generation_client", lambda *args: (_ for _ in ()).throw(runner.BenchmarkError("missing key")))

    with pytest.raises(runner.BenchmarkError):
        runner.generate_command(_generate_args(tmp_path, manifest_path))


def test_benchmark_runner_accepts_grok_provider_and_uses_openai_for_evaluation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    generation_clients: list[tuple[str, str]] = []
    evaluation_models: list[str] = []

    def make_client(provider: str, model: str):
        generation_clients.append((provider, model))
        return _Client(provider, model)

    def eval_client(*, model_name: str, missing_key_message: str):
        evaluation_models.append(model_name)
        return _Client("openai", model_name)

    monkeypatch.setattr(runner, "make_generation_client", make_client)
    monkeypatch.setattr(runner, "OpenAIJsonClient", eval_client)
    monkeypatch.setattr(runner, "generate_article_draft", lambda **kwargs: _draft_response("draft"))
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda **kwargs: _evaluation_response())

    args = _generate_args(tmp_path, manifest_path)
    args.provider = "grok"
    args.model = "grok-4.20-0309-non-reasoning"

    runner.generate_command(args)

    assert generation_clients == [("grok", "grok-4.20-0309-non-reasoning")]
    assert evaluation_models
    output_dir = tmp_path / "bench" / "grok_4_20_0309_non_reasoning" / "input_01"
    response = runner._read_json(output_dir / "response.json")
    assert response["provider"] == "grok"
    assert response["generation_model"] == "grok-4.20-0309-non-reasoning"
    assert (output_dir / "article.html").exists()


def test_grok_resume_skips_only_valid_grok_outputs(monkeypatch, tmp_path: Path, capsys) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    manifest = runner._read_json(manifest_path)
    entry = runner.normalize_input_entry(manifest["inputs"][0], 1)
    output = tmp_path / "bench" / "grok_4_20_0309_non_reasoning" / "input_01" / "response.json"
    output.parent.mkdir(parents=True)
    stale = _response_payload(entry, "openai", "gpt-5.5", "completed")
    runner._write_json(output, stale)
    runner._write_json(output.parent / "telemetry.json", {})
    (output.parent / "article.html").write_text("<html></html>", encoding="utf-8")

    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client(provider, model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(runner, "generate_article_draft", lambda **kwargs: _draft_response("draft"))
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda **kwargs: _evaluation_response())

    args = _generate_args(tmp_path, manifest_path, resume=True)
    args.provider = "grok"
    args.model = "grok-4.20-0309-non-reasoning"

    runner.generate_command(args)

    assert "SKIPPED" not in capsys.readouterr().out
    response = runner._read_json(output)
    assert response["provider"] == "grok"
    assert response["generation_model"] == "grok-4.20-0309-non-reasoning"


def test_consolidation_reads_saved_summaries_without_api_calls(monkeypatch, tmp_path: Path) -> None:
    model_dir = tmp_path / "bench" / "gpt_5_5"
    runner._write_json(model_dir / "run_summary.json", {"rows": [{"input_id": "input_01", "status": "completed"}]})
    monkeypatch.setattr(runner, "make_generation_client", lambda *args: pytest.fail("API should not be called"))

    result = runner.consolidate_command(SimpleNamespace(output_dir=str(tmp_path / "bench")))

    assert result["rows"][0]["input_id"] == "input_01"
    assert (tmp_path / "bench" / "consolidated" / "model_results.csv").exists()


def test_run_summary_includes_separate_cost_fields(monkeypatch, tmp_path: Path) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("gemini", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))
    monkeypatch.setattr(runner, "generate_article_draft", lambda **kwargs: _draft_response("draft"))
    monkeypatch.setattr(runner, "evaluate_draft_grounding", lambda *args, **kwargs: _evaluation_response())

    runner.generate_command(
        SimpleNamespace(
            provider="gemini",
            model="gemini-3.5-flash",
            manifest=str(manifest_path),
            output_dir=str(tmp_path / "bench"),
            input_id="input_01",
            start_from=None,
            max_inputs=None,
            resume=False,
            overwrite=False,
            dry_run=False,
        )
    )

    summary = runner._read_json(
        tmp_path / "bench" / "gemini_3_5_flash" / "run_summary.json"
    )
    row = summary["rows"][0]
    csv_text = (
        tmp_path / "bench" / "gemini_3_5_flash" / "run_summary.csv"
    ).read_text(encoding="utf-8")
    assert row["generation_total_cost_usd"] is not None
    assert row["evaluation_total_cost_usd"] is not None
    assert row["combined_total_cost_usd"] is not None
    assert "generation_total_cost_usd" in csv_text
    assert "evaluation_total_cost_usd" in csv_text


def test_cost_recalculation_makes_no_api_calls_and_labels_historical(monkeypatch, tmp_path: Path) -> None:
    input_dir = tmp_path / "bench" / "gemini_3_5_flash" / "input_01"
    input_dir.mkdir(parents=True)
    entry = runner._read_json(_prepared_manifest(tmp_path))["inputs"][0]
    response = _response_payload(entry, "gemini", "gemini-3.5-flash", "completed")
    response["draft"] = _draft_response("draft").model_dump()
    response["evaluation"] = _evaluation_response().model_dump()
    runner._write_json(input_dir / "response.json", response)
    runner._write_json(input_dir / "telemetry.json", {"input_id": "input_01"})
    (input_dir / "article.html").write_text("<html></html>", encoding="utf-8")
    original_response = runner._read_json(input_dir / "response.json")
    monkeypatch.setattr(runner, "make_generation_client", lambda *args: pytest.fail("API should not be called"))

    runner.recalculate_costs_command(SimpleNamespace(output_dir=str(tmp_path / "bench")))

    telemetry = runner._read_json(input_dir / "telemetry.json")
    response_after = runner._read_json(input_dir / "response.json")
    html = (input_dir / "article.html").read_text(encoding="utf-8")
    assert telemetry["cost_breakdown"]["generation"]["total_cost_usd"] is not None
    assert telemetry["cost_accuracy"] == "aggregate_estimate"
    assert (input_dir / "telemetry.before_cost_recalculation.json").exists()
    assert response_after == original_response
    assert "தமிழ் கட்டுரை உரை" in response_after["generated_tamil_article"]
    assert "Generation Usage" in html


def test_elapsed_messages_are_produced_while_input_runs(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    row, _ = runner._run_input_with_console_progress(
        index=1,
        total=10,
        input_id="input_01",
        provider="gemini",
        model="gemini-3.5-flash",
        stage={"value": "GENERATION"},
        operation=lambda: time.sleep(0.035) or {"status": "completed"},
    )

    output = capsys.readouterr().out
    assert row == {"status": "completed"}
    assert "[1/10] input_01 | gemini | gemini-3.5-flash | STARTED" in output
    assert "input_01 | GENERATION | elapsed=" in output


def test_timer_stops_after_success(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    runner._run_input_with_console_progress(
        index=1,
        total=1,
        input_id="input_01",
        provider="openai",
        model="gpt-5.5",
        stage={"value": "RUNNING"},
        operation=lambda: time.sleep(0.02) or {"status": "completed"},
    )
    first_output = capsys.readouterr().out
    time.sleep(0.03)
    later_output = capsys.readouterr().out

    assert "COMPLETED | runtime=" in first_output
    assert later_output == ""


def test_timer_stops_after_failure_and_preserves_exception(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_SECONDS", 0.01)
    expected = RuntimeError("original failure")

    def fail():
        time.sleep(0.02)
        raise expected

    with pytest.raises(RuntimeError) as exc_info:
        runner._run_input_with_console_progress(
            index=1,
            total=1,
            input_id="input_01",
            provider="openai",
            model="gpt-5.5",
            stage={"value": "GROUNDING"},
            operation=fail,
        )
    first_output = capsys.readouterr().out
    time.sleep(0.03)
    later_output = capsys.readouterr().out

    assert exc_info.value is expected
    assert "FAILED | runtime=" in first_output
    assert "error=original failure" in first_output
    assert later_output == ""


def test_failed_row_prints_runtime_without_raising(monkeypatch, capsys) -> None:
    monkeypatch.setattr(runner, "HEARTBEAT_INTERVAL_SECONDS", 0.01)

    row, _ = runner._run_input_with_console_progress(
        index=2,
        total=10,
        input_id="input_02",
        provider="gemini",
        model="gemini-3.5-flash",
        stage={"value": "GENERATION"},
        operation=lambda: {"status": "failed", "error_message": "sample failed"},
    )

    output = capsys.readouterr().out
    assert row["status"] == "failed"
    assert "[2/10] input_02 | FAILED | runtime=" in output
    assert "error=sample failed" in output


def test_skipped_inputs_do_not_start_timer(monkeypatch, tmp_path: Path, capsys) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    manifest = runner._read_json(manifest_path)
    entry = runner.normalize_input_entry(manifest["inputs"][0], 1)
    output = tmp_path / "bench" / "gpt_5_5" / "input_01" / "response.json"
    output.parent.mkdir(parents=True)
    runner._write_json(output, _response_payload(entry, "openai", "gpt-5.5", "completed"))
    runner._write_json(output.parent / "telemetry.json", {})
    (output.parent / "article.html").write_text("<html></html>", encoding="utf-8")

    def fail_start(self):
        raise AssertionError("timer should not start for skipped input")

    monkeypatch.setattr(runner.InputHeartbeat, "start", fail_start)
    monkeypatch.setattr(runner, "make_generation_client", lambda provider, model: _Client("openai", model))
    monkeypatch.setattr(runner, "OpenAIJsonClient", lambda *args, **kwargs: _Client("openai", "eval"))

    runner.generate_command(_generate_args(tmp_path, manifest_path, resume=True))

    output_text = capsys.readouterr().out
    assert "[1/1] input_01 | SKIPPED | existing valid output" in output_text


def test_benchmark_summary_calculates_runtime_average_fastest_and_slowest(capsys) -> None:
    runner._print_benchmark_summary(
        [
            runner.ConsoleRunRecord("input_01", "completed", 10.0),
            runner.ConsoleRunRecord("input_02", "completed", 20.0),
            runner.ConsoleRunRecord("input_03", "failed", 30.0),
            runner.ConsoleRunRecord("input_04", "skipped", 0.0),
        ],
        total_runtime_seconds=65.0,
    )

    output = capsys.readouterr().out
    assert "Completed: 2" in output
    assert "Failed: 1" in output
    assert "Skipped: 1" in output
    assert "Total runtime: 00:01:05" in output
    assert "Average runtime per completed input: 00:00:15" in output
    assert "Fastest input: input_01 - 00:00:10" in output
    assert "Slowest input: input_02 - 00:00:20" in output


def test_build_comparison_generates_saved_output_only_pages(monkeypatch, tmp_path: Path) -> None:
    bench = _comparison_benchmark(tmp_path)

    def api_call(*args, **kwargs):
        raise AssertionError("comparison builder must not make API calls")

    monkeypatch.setattr(runner, "make_generation_client", api_call)
    monkeypatch.setattr(runner, "generate_article_draft", api_call)
    monkeypatch.setattr(runner, "evaluate_draft_grounding", api_call)

    runner.build_comparison_command(_comparison_args(bench))

    index = bench / "comparisons" / "index.html"
    detail = bench / "comparisons" / "input_01_comparison.html"
    assert index.exists()
    assert detail.exists()
    assert (bench / "comparisons" / "input_02_comparison.html").exists()

    html = detail.read_text(encoding="utf-8")
    assert html.find("Gemini") < html.find("OpenAI")
    assert "StyleScribe Multi-Model Comparison" in html
    assert html.count("Author ID") == 1
    assert "Generation cost" in html
    assert "Grounding evaluation cost" in html
    assert "Combined cost" in html
    assert "<strong>Tokens:</strong>" not in html
    assert "Provider-reported generation total tokens" in html
    assert "Gemini headline" in html
    assert "OpenAI subheadline" in html
    assert "தமிழ் கட்டுரை" in html
    assert "இரண்டாம் பத்தி" in html
    assert "Grounding summary text" in html
    assert "hl-blocker" in html
    assert "hl-warning" in html
    assert "Runtime:" in html
    assert "Generation cost:" in html
    assert "../gemini_3_5_flash/input_01/response.json" in html
    assert "comparison unavailable" not in html

    index_html = index.read_text(encoding="utf-8")
    assert "input_01_comparison.html" in index_html
    assert "Average generation runtime" in index_html
    assert "Median generation runtime" in index_html
    assert "Total generation cost" in index_html
    assert "Completion rate" in index_html


def test_build_comparison_renders_failed_missing_and_integrity_warning(tmp_path: Path) -> None:
    bench = _comparison_benchmark(tmp_path, mismatch=True)

    runner.build_comparison_command(_comparison_args(bench))

    mismatched = (bench / "comparisons" / "input_01_comparison.html").read_text(encoding="utf-8")
    failed = (bench / "comparisons" / "input_02_comparison.html").read_text(encoding="utf-8")

    assert "Comparison integrity warning" in mismatched
    assert "plan_id" in mismatched
    assert "Failed" in failed
    assert "generation failed" in failed
    assert "Missing" in failed
    assert "No completed article available" in failed
    assert "None</p>" in failed
    assert "Not available" in failed


def test_build_comparison_does_not_modify_saved_outputs(tmp_path: Path) -> None:
    bench = _comparison_benchmark(tmp_path)
    response_path = bench / "gemini_3_5_flash" / "input_01" / "response.json"
    telemetry_path = bench / "gemini_3_5_flash" / "input_01" / "telemetry.json"
    before_response = response_path.read_text(encoding="utf-8")
    before_telemetry = telemetry_path.read_text(encoding="utf-8")

    runner.build_comparison_command(_comparison_args(bench))

    assert response_path.read_text(encoding="utf-8") == before_response
    assert telemetry_path.read_text(encoding="utf-8") == before_telemetry


def test_build_comparison_uses_run_summary_csv_fallback(tmp_path: Path) -> None:
    bench = _comparison_benchmark(tmp_path)
    (bench / "gemini_3_5_flash" / "run_summary.json").unlink()
    (bench / "gpt_5_5" / "run_summary.json").unlink()

    runner.build_comparison_command(_comparison_args(bench))

    html = (bench / "comparisons" / "index.html").read_text(encoding="utf-8")
    assert "Gemini: gemini-3.5-flash" in html
    assert "OpenAI: gpt-5.5" in html
    assert "input_01_comparison.html" in html


def _write_text_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "inputs.json"
    inputs = [
        {
            "input_id": f"input_{index:02d}",
            "source_text": f"Source body {index} with enough text for benchmark preparation and validation.",
            "source_title": f"Source {index}",
        }
        for index in range(1, 11)
    ]
    runner._write_json(manifest, {"inputs": inputs})
    return manifest


def _prepared_manifest(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    inputs = []
    for index in range(1, 11):
        input_id = f"input_{index:02d}"
        shared = bench / "shared" / input_id
        shared.mkdir(parents=True, exist_ok=True)
        brief_path = shared / "brief.json"
        plan_path = shared / "plan.json"
        source_path = shared / "source.json"
        runner._write_json(source_path, {"source_text": "source"})
        runner._write_json(brief_path, {"brief_id": f"brief_{index:02d}"})
        runner._write_json(
            plan_path,
            {
                "plan_id": f"plan_{index:02d}",
                "target_min_word_count": 450,
                "target_max_word_count": 690,
            },
        )
        inputs.append(
            {
                "input_id": input_id,
                "source_title": f"Source {index}",
                "source_language": "en",
                "source_text": "source text long enough",
                "source_path": None,
                "author_id": "v_vasanthi",
                "desired_word_count": 600,
                "tone": "clear",
                "article_type": "news",
                "brief_id": f"brief_{index:02d}",
                "brief_path": str(brief_path),
                "plan_id": f"plan_{index:02d}",
                "plan_path": str(plan_path),
                "shared_artifacts_status": "completed",
            }
        )
    manifest_path = bench / "shared" / "manifest.json"
    runner._write_json(manifest_path, {"inputs": inputs})
    return manifest_path


def _generate_args(tmp_path: Path, manifest_path: Path, *, resume: bool = False, overwrite: bool = False):
    return SimpleNamespace(
        provider="openai",
        model="gpt-5.5",
        manifest=str(manifest_path),
        output_dir=str(tmp_path / "bench"),
        input_id="input_01",
        start_from=None,
        max_inputs=None,
        resume=resume,
        overwrite=overwrite,
        dry_run=False,
    )


def _comparison_args(bench: Path):
    return SimpleNamespace(
        output_dir=str(bench),
        left_provider="gemini",
        left_model="gemini-3.5-flash",
        right_provider="openai",
        right_model="gpt-5.5",
    )


def _comparison_benchmark(tmp_path: Path, *, mismatch: bool = False) -> Path:
    manifest_path = _prepared_manifest(tmp_path)
    bench = manifest_path.parents[1]
    manifest = runner._read_json(manifest_path)
    input_01 = manifest["inputs"][0]
    input_02 = manifest["inputs"][1]
    _write_saved_model_output(
        bench,
        input_01,
        provider="gemini",
        model="gemini-3.5-flash",
        headline="Gemini headline",
        subheadline="Gemini subheadline",
        article="தமிழ் கட்டுரை blocker claim warning phrase\n\nஇரண்டாம் பத்தி.",
        runtime=10,
        generation_cost=0.000111,
        grounding_score=85,
        readiness="safe_to_review",
    )
    right_entry = dict(input_01)
    if mismatch:
        right_entry["plan_id"] = "different_plan"
    _write_saved_model_output(
        bench,
        right_entry,
        provider="openai",
        model="gpt-5.5",
        headline="OpenAI headline",
        subheadline="OpenAI subheadline",
        article="தமிழ் கட்டுரை blocker claim warning phrase\n\nமூன்றாம் பத்தி.",
        runtime=20,
        generation_cost=0.000222,
        grounding_score=90,
        readiness="revision_required",
    )
    _write_saved_model_output(
        bench,
        input_02,
        provider="gemini",
        model="gemini-3.5-flash",
        headline="",
        subheadline="",
        article="",
        runtime=5,
        generation_cost=0.000033,
        grounding_score=None,
        readiness=None,
        status="failed",
        error_message="generation failed",
    )
    return bench


def _write_saved_model_output(
    bench: Path,
    entry: dict,
    *,
    provider: str,
    model: str,
    headline: str,
    subheadline: str,
    article: str,
    runtime: float,
    generation_cost: float,
    grounding_score: int | None,
    readiness: str | None,
    status: str = "completed",
    error_message: str | None = None,
) -> None:
    model_dir = bench / runner.safe_model_dir(model)
    input_dir = model_dir / entry["input_id"]
    input_dir.mkdir(parents=True, exist_ok=True)
    response = {
        "input_id": entry["input_id"],
        "source_title": entry["source_title"],
        "source_language": entry["source_language"],
        "author_id": entry["author_id"],
        "brief_id": entry["brief_id"],
        "plan_id": entry["plan_id"],
        "provider": provider,
        "generation_model": model,
        "generated_headline": headline,
        "generated_subheadline": subheadline,
        "generated_tamil_article": article,
        "word_count": 120 if status == "completed" else None,
        "desired_word_count": entry["desired_word_count"],
        "target_minimum": 90,
        "target_maximum": 150,
        "within_target_range": status == "completed",
        "grounding_score": grounding_score,
        "readiness": readiness,
        "unsupported_claims": [{"claim": "blocker claim", "reason": "unsupported", "editor_action": "remove"}] if status == "completed" else [],
        "claims_to_avoid_violations": [],
        "overclaims": [],
        "repetition_indicators": [],
        "blockers": [],
        "warnings": [{"text": "warning phrase", "reason": "check source"}] if status == "completed" else [],
        "grounding_evaluation_result": {"summary": "Grounding summary text"},
        "workflow_settings": {
            "grounding_evaluation": True,
            "auto_revision": False,
            "final_evaluation": False,
        },
        "completion_status": status,
        "error_type": "benchmark_error" if error_message else None,
        "error_message": error_message,
    }
    telemetry = {
        "input_id": entry["input_id"],
        "provider": provider,
        "model": model,
        "completion_status": status,
        "generation_runtime_seconds": runtime,
        "grounding_evaluation_runtime_seconds": 2,
        "total_elapsed_runtime_seconds": runtime + 2,
        "attempt_count": 2,
        "retry_count": 1,
        "timeout_count": 0,
        "cost_currency": "USD",
        "pricing_configuration_id": "benchmark_pricing_2026_07_16",
        "cost_accuracy": "per_call_calculated",
        "error_message": error_message,
        "generation_call_ledger": [
            {
                "failure_type": "invalid_json" if provider == "gemini" else None,
                "attempt": 2 if provider == "gemini" else 1,
            }
        ],
        "cost_breakdown": {
            "generation": {
                "provider": provider,
                "model": model,
                "prompt_tokens": 100,
                "cached_prompt_tokens": 10,
                "completion_tokens": 50,
                "provider_total_tokens": 175,
                "total_cost_usd": generation_cost,
                "token_reconciliation_status": "reconciled",
            },
            "grounding_evaluation": {
                "provider": "openai",
                "model": "gpt-4o-mini",
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "provider_total_tokens": 100,
                "total_cost_usd": 0.00001,
            },
            "combined": {
                "total_cost_usd": generation_cost + 0.00001,
            },
        },
    }
    runner._write_json(input_dir / "response.json", response)
    runner._write_json(input_dir / "telemetry.json", telemetry)
    (input_dir / "article.html").write_text("<html>saved article</html>", encoding="utf-8")
    rows = runner._load_existing_summary_rows(model_dir)
    existing = [row for row in rows if row.get("input_id") != entry["input_id"]]
    existing.append(
        runner._summary_row(
            response,
            telemetry,
            {
                "response": str(input_dir / "response.json"),
                "telemetry": str(input_dir / "telemetry.json"),
                "html": str(input_dir / "article.html"),
            },
        )
    )
    runner._write_run_summary(model_dir, existing)


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)


class _Client:
    provider = "openai"

    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name
        self.timeout_seconds = 240.0
        self.last_attempt_count = 1
        self.last_retry_count = 0


def _brief_response(brief_id: str):
    return SimpleNamespace(
        brief_id=brief_id,
        model_name="gpt-4o-mini",
        brief={"token_usage": {"prompt_tokens": 1, "completion_tokens": 1}},
        model_dump=lambda mode="json": {"brief_id": brief_id, "brief": {}},
    )


def _plan_response(plan_id: str, brief_id: str):
    return SimpleNamespace(
        plan_id=plan_id,
        brief_id=brief_id,
        model_name="gpt-4o-mini",
        token_usage={"prompt_tokens": 1, "completion_tokens": 1},
        target_min_word_count=450,
        target_max_word_count=690,
        __dict__={
            "plan_id": plan_id,
            "brief_id": brief_id,
            "target_min_word_count": 450,
            "target_max_word_count": 690,
        },
    )


def _draft_response(
    draft_id: str,
    *,
    article_text: str | None = None,
    article_field: str = "article",
):
    article = article_text if article_text is not None else "தமிழ் கட்டுரை உரை " * 50
    draft_payload = {
        "headline": "Headline",
        "subheadline": "Subheadline",
        article_field: article,
        "section_assembled_article_word_count": 999,
        "token_usage": {
            "prompt_tokens": 1000,
            "cached_prompt_tokens": 100,
            "completion_tokens": 500,
            "total_tokens": 1500,
        },
    }
    return SimpleNamespace(
        draft_id=draft_id,
        model_dump=lambda mode="json": {
            "draft_id": draft_id,
            "draft": draft_payload,
        },
    )


def _evaluation_response():
    return SimpleNamespace(
        model_dump=lambda mode="json": {
            "evaluation": {
                "grounding_score": 75,
                "editorial_readiness": "safe_to_review",
                "unsupported_claims": [],
                "blockers": [],
                "warnings": [],
                "token_usage": {
                    "prompt_tokens": 1000,
                    "cached_prompt_tokens": 100,
                    "completion_tokens": 500,
                    "total_tokens": 1500,
                },
            }
        }
    )


def _response_payload(entry: dict, provider: str, model: str, status: str) -> dict:
    return {
        "completion_status": status,
        "provider": provider,
        "generation_model": model,
        "generated_tamil_article": "தமிழ் கட்டுரை உரை " * 20,
        "word_count": 60,
        "author_id": entry["author_id"],
        "input_id": entry["input_id"],
        "brief_id": entry["brief_id"],
        "plan_id": entry["plan_id"],
        "workflow_settings": {
            "grounding_evaluation": True,
            "auto_revision": False,
            "final_evaluation": False,
        },
    }
