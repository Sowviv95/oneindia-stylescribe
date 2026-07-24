from pathlib import Path
from types import SimpleNamespace

import scripts.run_generation_benchmark as runner


def test_retrieval_mode_is_explicit_and_uses_separate_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(
        runner,
        "_retrieval_dry_run_payload",
        lambda entries, options: {"previews": [{"input_id": "input_01"}]},
    )

    dry_run = runner.generate_command(
        _args(
            tmp_path,
            manifest_path,
            dry_run=True,
            generation_mode="newsroom_v1_retrieval",
        )
    )

    response_path = dry_run["output_paths"]["input_01"]["response"]
    assert dry_run["generation_mode"] == "newsroom_v1_retrieval"
    assert dry_run["prompt_version"] == "oneindia_newsroom_v1.0_retrieval_v1"
    assert dry_run["retrieval_prompt_version"] == (
        "oneindia_newsroom_v1.0_retrieval_v1"
    )
    assert "newsroom_v1_retrieval_gemini_gemini_3_5_flash" in response_path


def test_non_default_retrieval_prompt_version_uses_versioned_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(
        runner,
        "_retrieval_dry_run_payload",
        lambda entries, options: {"previews": [{"input_id": "input_01"}]},
    )
    args = _args(
        tmp_path,
        manifest_path,
        dry_run=True,
        generation_mode="newsroom_v1_retrieval",
    )
    args.retrieval_prompt_version = (
        "oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard"
    )

    dry_run = runner.generate_command(args)

    response_path = dry_run["output_paths"]["input_01"]["response"]
    assert dry_run["prompt_version"] == (
        "oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard"
    )
    assert "v1_1_impact_guard" in response_path


def test_retrieval_trace_and_leakage_are_persisted(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(
        runner,
        "make_generation_client",
        lambda provider, model: _Client(provider, model),
    )
    monkeypatch.setattr(
        runner,
        "OpenAIJsonClient",
        lambda *args, **kwargs: _Client("openai", "eval"),
    )
    monkeypatch.setattr(
        runner,
        "_retrieve_for_entry",
        lambda entry, options: {
            "records": ["record"],
            "scores": ["score"],
            "trace": _trace(),
        },
    )
    monkeypatch.setattr(runner, "_prepare_retrieval_runtime", lambda options: None)

    def retrieval_generation(**kwargs):
        assert kwargs["retrieved_records"] == ["record"]
        assert kwargs["retrieval_scores"] == ["score"]
        return _draft_response(kwargs["retrieval_trace"])

    monkeypatch.setattr(
        runner,
        "generate_newsroom_retrieval_article_draft",
        retrieval_generation,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_draft_grounding",
        lambda draft_id, **kwargs: _evaluation_response(),
    )
    monkeypatch.setattr(
        runner,
        "_run_retrieval_leakage_for_entry",
        lambda *args, **kwargs: {"finding_count": 0, "status": "clear"},
    )
    monkeypatch.setattr(runner, "_current_git_commit", lambda: "abc123")

    runner.generate_command(
        _args(
            tmp_path,
            manifest_path,
            generation_mode="newsroom_v1_retrieval",
        )
    )

    response = runner._read_json(
        tmp_path
        / "bench"
        / "newsroom_v1_retrieval_gemini_gemini_3_5_flash"
        / "input_01"
        / "response.json"
    )
    telemetry = runner._read_json(
        tmp_path
        / "bench"
        / "newsroom_v1_retrieval_gemini_gemini_3_5_flash"
        / "input_01"
        / "telemetry.json"
    )

    assert response["generation_mode"] == "newsroom_v1_retrieval"
    assert response["retrieval_trace"]["retrieved_article_ids"] == ["r1"]
    assert response["retrieval_leakage_diagnostic"]["status"] == "clear"
    assert response["topic_metadata"]["original_brief_topic"] == "AI support"
    assert telemetry["retrieval_latency_seconds"] == 0.123
    summary = runner._read_json(
        tmp_path
        / "bench"
        / "newsroom_v1_retrieval_gemini_gemini_3_5_flash"
        / "run_summary.json"
    )
    assert summary["rows"][0]["retrieved_article_ids"] == "r1"
    assert summary["rows"][0]["retrieval_leakage_status"] == "clear"


def test_retrieval_runtime_is_prepared_once_for_multiple_inputs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    prepare_calls = 0
    generated_inputs = []

    monkeypatch.setattr(
        runner,
        "make_generation_client",
        lambda provider, model: _Client(provider, model),
    )
    monkeypatch.setattr(
        runner,
        "OpenAIJsonClient",
        lambda *args, **kwargs: _Client("openai", "eval"),
    )

    def prepare(options):
        nonlocal prepare_calls
        prepare_calls += 1
        options["_index"] = object()
        options["_embedding_provider"] = object()
        options["_index_load_seconds"] = 0.01
        options["_model_load_seconds"] = 0.02

    def retrieve(entry, options):
        assert options["_index"] is not None
        assert options["_embedding_provider"] is not None
        return {"records": ["record"], "scores": ["score"], "trace": _trace()}

    def retrieval_generation(**kwargs):
        generated_inputs.append(kwargs["input_identifier"])
        return _draft_response(kwargs["retrieval_trace"])

    monkeypatch.setattr(runner, "_prepare_retrieval_runtime", prepare)
    monkeypatch.setattr(runner, "_retrieve_for_entry", retrieve)
    monkeypatch.setattr(
        runner,
        "generate_newsroom_retrieval_article_draft",
        retrieval_generation,
    )
    monkeypatch.setattr(
        runner,
        "evaluate_draft_grounding",
        lambda draft_id, **kwargs: _evaluation_response(),
    )
    monkeypatch.setattr(
        runner,
        "_run_retrieval_leakage_for_entry",
        lambda *args, **kwargs: {"finding_count": 0, "status": "clear"},
    )
    monkeypatch.setattr(runner, "_current_git_commit", lambda: "abc123")

    args = _args(tmp_path, manifest_path, generation_mode="newsroom_v1_retrieval")
    args.input_id = None
    args.max_inputs = 2
    runner.generate_command(args)

    assert prepare_calls == 1
    assert generated_inputs == ["input_01", "input_02"]


def test_non_retrieval_mode_does_not_prepare_retrieval_runtime(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = _prepared_manifest(tmp_path)
    monkeypatch.setattr(
        runner,
        "_prepare_retrieval_runtime",
        lambda options: (_ for _ in ()).throw(AssertionError("unexpected")),
    )

    dry_run = runner.generate_command(
        _args(tmp_path, manifest_path, dry_run=True, generation_mode="newsroom_v1")
    )

    assert dry_run["generation_mode"] == "newsroom_v1"


def test_fallback_policy_never_uses_hashing_retrieval() -> None:
    assert runner.RETRIEVAL_OPERATIONAL_FALLBACK_POLICY[
        "fallback_generation_mode"
    ] == "newsroom_v1"
    assert runner.RETRIEVAL_OPERATIONAL_FALLBACK_POLICY[
        "fallback_newsroom_prompt_version"
    ] == "oneindia_newsroom_v1.0"
    assert runner.RETRIEVAL_OPERATIONAL_FALLBACK_POLICY[
        "never_fallback_embedding_provider"
    ] == "local_hashing"


def test_evaluation_anomaly_diagnostic_flags_self_contradictory_finding() -> None:
    diagnostic = runner._evaluation_anomaly_diagnostics(
        {
            "claims_to_avoid_violations": [
                {
                    "claim": "Supported claim",
                    "reason": "This is directly supported by the grounded brief.",
                }
            ]
        }
    )

    assert diagnostic["status"] == "review_required"
    assert diagnostic["anomaly_count"] == 1
    assert diagnostic["anomalies"][0]["anomaly_type"] == (
        "claims_to_avoid_self_contradiction"
    )


def test_evaluation_anomaly_diagnostic_does_not_weaken_genuine_violation() -> None:
    diagnostic = runner._evaluation_anomaly_diagnostics(
        {
            "claims_to_avoid_violations": [
                {
                    "claim": "Immediate rule changes start tomorrow.",
                    "reason": "The brief explicitly says no timeline was provided.",
                }
            ]
        }
    )

    assert diagnostic["status"] == "clear"
    assert diagnostic["anomaly_count"] == 0


def _prepared_manifest(tmp_path: Path) -> Path:
    bench = tmp_path / "bench"
    inputs = []
    for index in range(1, 11):
        input_id = f"input_{index:02d}"
        shared = bench / "shared" / input_id
        shared.mkdir(parents=True, exist_ok=True)
        brief_path = shared / "brief.json"
        plan_path = shared / "plan.json"
        runner._write_json(
            brief_path,
            {
                "brief_id": f"brief_{index:02d}",
                "brief": {
                    "topic": "AI support",
                    "one_line_summary": "AI support systems are being tested.",
                    "confirmed_facts": ["AI support systems are being tested."],
                },
            },
        )
        runner._write_json(
            plan_path,
            {
                "plan_id": f"plan_{index:02d}",
                "target_min_word_count": 450,
                "target_max_word_count": 690,
            },
        )
        runner._write_json(
            shared / "source.json",
            {
                "input_id": input_id,
                "source_title": f"Source {index}",
                "source_text": "source text",
            },
        )
        inputs.append(
            {
                "input_id": input_id,
                "source_title": f"Source {index}",
                "source_language": "en",
                "source_text": "source text",
                "source_path": None,
                "author_id": "v_vasanthi",
                "desired_word_count": 600,
                "tone": "clear",
                "article_type": "news",
                "author_instruction": "Write for Oneindia.",
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


def _args(
    tmp_path: Path,
    manifest_path: Path,
    *,
    generation_mode: str,
    dry_run: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        provider="gemini",
        model="gemini-3.5-flash",
        generation_mode=generation_mode,
        newsroom_prompt_version="oneindia_newsroom_v1.0",
        retrieval_prompt_version="oneindia_newsroom_v1.0_retrieval_v1",
        retrieval_index_path=str(tmp_path / "index.json"),
        retrieval_records_path=str(tmp_path / "records.jsonl"),
        embedding_provider="local_hashing",
        embedding_model="word_hashing_256_v1",
        retrieval_top_k=3,
        candidate_pool_size=12,
        topic_boost=False,
        topic_boost_weight=0.05,
        max_examples_per_author=1,
        max_retrieval_context_chars=9000,
        rebuild_index=False,
        reuse_index=True,
        manifest=str(manifest_path),
        output_dir=str(tmp_path / "bench"),
        input_id="input_01",
        start_from=None,
        max_inputs=None,
        resume=False,
        overwrite=False,
        dry_run=dry_run,
    )


def _trace() -> dict[str, object]:
    return {
        "retrieval_mode": "newsroom_v1_retrieval",
        "retrieval_prompt_version": "oneindia_newsroom_v1.0_retrieval_v1",
        "index_version": "idx",
        "corpus_version": "corpus",
        "profile_version": "profile",
        "retrieved_article_ids": ["r1"],
        "retrieved_authors": ["author_a"],
        "retrieved_topics": [{"article_id": "r1", "topic": "technology"}],
        "selected_scores": [{"article_id": "r1", "similarity_score": 0.8}],
        "candidate_scores": [{"article_id": "r1", "similarity_score": 0.8}],
        "exclusions": [{"article_id": "r2", "reason": "source_article"}],
        "retrieval_latency_seconds": 0.123,
        "ranking_configuration": {"top_k": 3},
    }


def _draft_response(trace: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        draft_id="draft-1",
        model_dump=lambda mode="json": {
            "draft_id": "draft-1",
            "draft": {
                "headline": "Headline",
                "subheadline": "Subheadline",
                "article_body": "தமிழ் கட்டுரை உரை " * 50,
                "retrieval_trace": trace,
                "token_usage": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            },
        },
    )


def _evaluation_response() -> SimpleNamespace:
    return SimpleNamespace(
        model_dump=lambda mode="json": {
            "evaluation": {
                "grounding_score": 80,
                "editorial_readiness": "safe_to_review",
                "unsupported_claims": [],
                "blockers": [],
                "warnings": [],
                "token_usage": {
                    "prompt_tokens": 100,
                    "cached_prompt_tokens": 10,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        }
    )


class _Client:
    def __init__(self, provider: str, model_name: str) -> None:
        self.provider = provider
        self.model_name = model_name
        self.timeout_seconds = 240.0
        self.last_attempt_count = 1
        self.last_retry_count = 0
        self.call_records = []
