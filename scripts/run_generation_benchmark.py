# ruff: noqa: E501,I001
"""Prepare and run resumable one-model article generation benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import traceback
from subprocess import DEVNULL, check_output
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository  # noqa: E402
from backend.app.services.docx_extractor import (  # noqa: E402
    DocxExtractionError,
    extract_docx_text,
)
from backend.app.services.article_generation_service import (  # noqa: E402
    NEWSROOM_PROMPT_VERSION_PATHS,
    NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS,
    build_newsroom_retrieval_generation_input,
    generate_article_draft,
    generate_newsroom_article_draft,
    generate_newsroom_retrieval_article_draft,
)
from backend.app.services.article_plan_service import generate_article_plan  # noqa: E402
from backend.app.services.draft_grounding_evaluation_service import evaluate_draft_grounding  # noqa: E402
from backend.app.services.grounded_brief_service import generate_grounded_brief  # noqa: E402
from backend.app.services.model_clients.gemini_client import GeminiJsonClient  # noqa: E402
from backend.app.services.model_clients.grok_client import GrokJsonClient  # noqa: E402
from backend.app.services.model_clients.openai_client import OpenAIJsonClient, request_runtime_metadata  # noqa: E402
from backend.app.services.newsroom_retrieval_service import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_INDEX_PATH,
    DEFAULT_RECORDS_PATH,
    RetrievalRankingConfig,
    build_or_load_index,
    build_retrieval_query,
    make_embedding_provider,
    retrieve_examples,
    topic_metadata_from_brief,
)
from backend.app.services.retrieval_leakage_diagnostic_service import (  # noqa: E402
    run_retrieval_leakage_diagnostic,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count  # noqa: E402
from backend.app.services.workflow_telemetry import resolve_stage_model  # noqa: E402

AUTHOR_ID = "v_vasanthi"
TARGET_LANGUAGE = "ta"
EXPERIMENT_TYPE = "raw_generation"
DEFAULT_AUTHOR_INSTRUCTION = "Write this as a Tamil news article for Oneindia readers."
DEFAULT_DESIRED_WORD_COUNT = 600
DEFAULT_TONE = "clear, engaging and factual"
DEFAULT_ARTICLE_TYPE = "news"
HEARTBEAT_INTERVAL_SECONDS = 10.0
PRICE_CONFIG_PATH = REPO_ROOT / "backend" / "app" / "config" / "model_pricing.json"
USD_QUANT = Decimal("0.000001")
COST_ASSUMPTIONS = [
    "Standard API pricing used",
    "No Batch, Flex, or Priority discount applied",
    "No regional processing uplift applied",
    "No tool-call charges included",
    "Gemini thinking tokens priced as output where included in provider usage",
]
GENERATION_MODES = {"legacy", "newsroom_v1", "newsroom_v1_retrieval"}
LEGACY_PROMPT_VERSION = "legacy_article_generation_prompt"
DEFAULT_NEWSROOM_PROMPT_VERSION = "oneindia_newsroom_v1.0"
DEFAULT_RETRIEVAL_PROMPT_VERSION = "oneindia_newsroom_v1.0_retrieval_v1"
IMPACT_GUARD_RETRIEVAL_PROMPT_VERSION = (
    "oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard"
)
RETRIEVAL_OPERATIONAL_FALLBACK_POLICY = {
    "fallback_generation_mode": "newsroom_v1",
    "fallback_newsroom_prompt_version": DEFAULT_NEWSROOM_PROMPT_VERSION,
    "never_fallback_embedding_provider": "local_hashing",
    "record_fallback_reason": True,
}

SUPPORTED_MODELS: dict[str, set[str]] = {
    "gemini": {"gemini-3.5-flash", "gemini-3.5-flash-lite"},
    "openai": {"gpt-5.5"},
    "grok": {"grok-4.20-0309-non-reasoning"},
}
FUTURE_MODELS: dict[str, set[str]] = {}

SUMMARY_FIELDS = [
    "input_id",
    "provider",
    "model",
    "generation_mode",
    "prompt_version",
    "newsroom_profile_version",
    "retrieval_prompt_version",
    "retrieval_index_version",
    "git_commit",
    "status",
    "source_title",
    "source_language",
    "brief_topic",
    "provisional_topic",
    "provisional_topic_confidence",
    "topic_low_confidence",
    "topic_multi_category_conflict",
    "topic_review_flag",
    "author_id",
    "brief_id",
    "plan_id",
    "headline",
    "word_count",
    "desired_word_count",
    "target_minimum",
    "target_maximum",
    "word_count_variance",
    "within_target_range",
    "grounding_score",
    "readiness",
    "unsupported_claim_count",
    "claims_to_avoid_violation_count",
    "evaluation_anomaly_count",
    "blocker_count",
    "warning_count",
    "generation_runtime_seconds",
    "grounding_evaluation_runtime_seconds",
    "total_elapsed_runtime_seconds",
    "prompt_tokens",
    "cached_prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "attempt_count",
    "retry_count",
    "retrieval_index_load_seconds",
    "retrieval_model_load_seconds",
    "retrieval_latency_seconds",
    "retrieved_article_ids",
    "retrieved_authors",
    "retrieval_leakage_finding_count",
    "retrieval_leakage_status",
    "estimated_total_cost",
    "cost_status",
    "generation_prompt_tokens",
    "generation_cached_prompt_tokens",
    "generation_uncached_prompt_tokens",
    "generation_completion_tokens",
    "generation_provider_total_tokens",
    "generation_reasoning_tokens",
    "generation_accepted_prediction_tokens",
    "generation_rejected_prediction_tokens",
    "generation_total_cost_usd",
    "generation_provider_cost_ticks",
    "generation_provider_reported_cost_usd",
    "generation_provider_cost_conversion_status",
    "generation_cost_status",
    "evaluation_prompt_tokens",
    "evaluation_cached_prompt_tokens",
    "evaluation_uncached_prompt_tokens",
    "evaluation_completion_tokens",
    "evaluation_total_cost_usd",
    "evaluation_cost_status",
    "combined_total_cost_usd",
    "combined_cost_status",
    "cost_incurred_before_failure_usd",
    "billable_failed_call_count",
    "pricing_configuration_id",
    "pricing_effective_date",
    "token_reconciliation_status",
    "response_path",
    "html_path",
    "telemetry_path",
    "error_type",
    "error_message",
]


class BenchmarkError(RuntimeError):
    """Raised for global benchmark configuration errors."""


@dataclass(frozen=True)
class InputSelection:
    input_ids: list[str]
    entries: list[dict[str, Any]]


@dataclass(frozen=True)
class ConsoleRunRecord:
    input_id: str
    status: str
    elapsed_seconds: float


@dataclass(frozen=True)
class CanonicalArticle:
    headline: str
    subheadline: str
    article_body: str
    word_count: int
    source_field: str | None
    model_reported_word_count: int | None


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_command(args)
        elif args.command == "generate":
            generate_command(args)
        elif args.command == "consolidate":
            consolidate_command(args)
        elif args.command == "recalculate-costs":
            recalculate_costs_command(args)
        elif args.command == "build-comparison":
            build_comparison_command(args)
        else:
            parser.error("Unknown command.")
    except BenchmarkError as exc:
        print(f"BENCHMARK_BLOCKED: {exc}", file=sys.stderr)
        return 2
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input-manifest")
    prepare.add_argument("--input-dir")
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--author-id", default=AUTHOR_ID)
    prepare.add_argument("--input-id")
    prepare.add_argument("--start-from")
    prepare.add_argument("--max-inputs", type=int)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")
    prepare.add_argument("--dry-run", action="store_true")

    generate = subparsers.add_parser("generate")
    generate.add_argument("--provider", required=True)
    generate.add_argument("--model", required=True)
    generate.add_argument(
        "--generation-mode",
        choices=sorted(GENERATION_MODES),
        default="legacy",
    )
    generate.add_argument(
        "--newsroom-prompt-version",
        choices=sorted(NEWSROOM_PROMPT_VERSION_PATHS),
        default=DEFAULT_NEWSROOM_PROMPT_VERSION,
    )
    generate.add_argument(
        "--retrieval-prompt-version",
        choices=sorted(NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS),
        default=DEFAULT_RETRIEVAL_PROMPT_VERSION,
    )
    generate.add_argument(
        "--retrieval-index-path",
        default=str(DEFAULT_INDEX_PATH),
    )
    generate.add_argument(
        "--retrieval-records-path",
        default=str(DEFAULT_RECORDS_PATH),
    )
    generate.add_argument("--embedding-provider", default=DEFAULT_EMBEDDING_PROVIDER)
    generate.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    generate.add_argument("--retrieval-top-k", type=int, default=3)
    generate.add_argument("--candidate-pool-size", type=int, default=12)
    generate.add_argument("--topic-boost", action="store_true")
    generate.add_argument("--topic-boost-weight", type=float, default=0.05)
    generate.add_argument("--max-examples-per-author", type=int, default=1)
    generate.add_argument("--max-retrieval-context-chars", type=int, default=9000)
    generate.add_argument("--rebuild-index", action="store_true")
    generate.add_argument("--reuse-index", action="store_true")
    generate.add_argument("--manifest", required=True)
    generate.add_argument("--output-dir", required=True)
    generate.add_argument("--input-id")
    generate.add_argument("--start-from")
    generate.add_argument("--max-inputs", type=int)
    generate.add_argument("--resume", action="store_true")
    generate.add_argument("--overwrite", action="store_true")
    generate.add_argument("--dry-run", action="store_true")

    consolidate = subparsers.add_parser("consolidate")
    consolidate.add_argument("--output-dir", required=True)
    recalculate = subparsers.add_parser("recalculate-costs")
    recalculate.add_argument("--output-dir", required=True)
    comparison = subparsers.add_parser("build-comparison")
    comparison.add_argument("--output-dir", required=True)
    comparison.add_argument("--comparisons-dir-name", default="comparisons")
    comparison.add_argument("--left-provider", required=True)
    comparison.add_argument("--left-model", required=True)
    comparison.add_argument("--left-generation-mode", default="legacy")
    comparison.add_argument(
        "--left-newsroom-prompt-version",
        default=DEFAULT_NEWSROOM_PROMPT_VERSION,
    )
    comparison.add_argument(
        "--left-retrieval-prompt-version",
        default=DEFAULT_RETRIEVAL_PROMPT_VERSION,
    )
    comparison.add_argument("--right-provider", required=True)
    comparison.add_argument("--right-model", required=True)
    comparison.add_argument("--right-generation-mode", default="legacy")
    comparison.add_argument(
        "--right-newsroom-prompt-version",
        default=DEFAULT_NEWSROOM_PROMPT_VERSION,
    )
    comparison.add_argument(
        "--right-retrieval-prompt-version",
        default=DEFAULT_RETRIEVAL_PROMPT_VERSION,
    )
    comparison.add_argument("--third-provider")
    comparison.add_argument("--third-model")
    comparison.add_argument("--third-generation-mode", default="legacy")
    comparison.add_argument(
        "--third-newsroom-prompt-version",
        default=DEFAULT_NEWSROOM_PROMPT_VERSION,
    )
    comparison.add_argument(
        "--third-retrieval-prompt-version",
        default=DEFAULT_RETRIEVAL_PROMPT_VERSION,
    )
    return parser


def prepare_command(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    manifest, discovery = _prepare_input_manifest(args, output_dir)
    selection = select_inputs(
        manifest["inputs"],
        input_id=args.input_id,
        start_from=args.start_from,
        max_inputs=args.max_inputs,
    )
    shared_dir = output_dir / "shared"
    prepared_manifest_path = shared_dir / "manifest.json"
    prepared_manifest = _load_existing_prepared_manifest(prepared_manifest_path, manifest)
    selected_ids = set(selection.input_ids)

    dry_run = {
        "mode": "prepare",
        "word_file_discovery": discovery,
        "selected_input_ids": selection.input_ids,
        "manifest_path": str(prepared_manifest_path),
        "output_paths": {
            entry["input_id"]: {
                "source": str(shared_dir / entry["input_id"] / "source.json"),
                "brief": str(shared_dir / entry["input_id"] / "brief.json"),
                "plan": str(shared_dir / entry["input_id"] / "plan.json"),
                "telemetry": str(shared_dir / entry["input_id"] / "preparation_telemetry.json"),
            }
            for entry in selection.entries
        },
    }
    if args.dry_run:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return dry_run

    repo = StyleScribeRepository()
    repo.initialize_schema()
    shared_dir.mkdir(parents=True, exist_ok=True)
    for entry in prepared_manifest["inputs"]:
        if entry["input_id"] not in selected_ids:
            continue
        if (
            args.resume
            and not args.overwrite
            and _shared_artifacts_complete(entry, output_dir)
        ):
            print(f"SKIP_PREPARED {entry['input_id']}", flush=True)
            continue
        _prepare_one(entry, output_dir, repo)
        _write_json(prepared_manifest_path, prepared_manifest)
        print(f"PREPARED {entry['input_id']}", flush=True)
    _write_json(prepared_manifest_path, prepared_manifest)
    return prepared_manifest


def _prepare_input_manifest(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if bool(args.input_manifest) == bool(args.input_dir):
        raise BenchmarkError("Prepare requires exactly one of --input-manifest or --input-dir.")
    prepared_manifest_path = output_dir / "shared" / "manifest.json"
    if args.input_dir:
        if prepared_manifest_path.exists() and args.resume and not args.overwrite:
            return load_prepared_or_pending_manifest(prepared_manifest_path), {
                **discover_word_inputs(Path(args.input_dir), output_dir),
                "mapping_source": "existing_manifest",
                "manifest_path": str(prepared_manifest_path),
            }
        return load_word_input_manifest(
            Path(args.input_dir),
            author_id=args.author_id,
            output_dir=output_dir,
        )
    return load_input_manifest(Path(args.input_manifest)), None


def generate_command(args: argparse.Namespace) -> dict[str, Any]:
    provider = args.provider.strip().lower()
    model = args.model.strip()
    generation_mode = _validate_generation_mode(
        getattr(args, "generation_mode", "legacy")
    )
    newsroom_prompt_version = _validate_newsroom_prompt_version(
        getattr(args, "newsroom_prompt_version", DEFAULT_NEWSROOM_PROMPT_VERSION)
    )
    retrieval_prompt_version = _validate_retrieval_prompt_version(
        getattr(args, "retrieval_prompt_version", DEFAULT_RETRIEVAL_PROMPT_VERSION)
    )
    validate_provider_model(provider, model)
    manifest = load_prepared_manifest(Path(args.manifest))
    output_dir = Path(args.output_dir)
    selection = select_inputs(
        manifest["inputs"],
        input_id=args.input_id,
        start_from=args.start_from,
        max_inputs=args.max_inputs,
    )
    model_dir = _generation_model_dir(
        output_dir,
        provider,
        model,
        generation_mode,
        newsroom_prompt_version=newsroom_prompt_version,
        retrieval_prompt_version=retrieval_prompt_version,
    )
    prompt_metadata = _generation_prompt_metadata(
        generation_mode,
        newsroom_prompt_version=newsroom_prompt_version,
        retrieval_prompt_version=getattr(
            args,
            "retrieval_prompt_version",
            DEFAULT_RETRIEVAL_PROMPT_VERSION,
        ),
    )
    retrieval_options = _retrieval_options_from_args(args)
    retrieval_dry_run = None
    if generation_mode == "newsroom_v1_retrieval" and args.dry_run:
        retrieval_dry_run = _retrieval_dry_run_payload(
            entries=selection.entries,
            options=retrieval_options,
        )
    dry_run = {
        "mode": "generate",
        "provider": provider,
        "model": model,
        "generation_mode": generation_mode,
        "prompt_version": prompt_metadata["prompt_version"],
        "retrieval_prompt_version": prompt_metadata.get("retrieval_prompt_version"),
        "newsroom_profile_version": prompt_metadata.get("newsroom_profile_version"),
        "selected_input_ids": selection.input_ids,
        "retrieval": retrieval_dry_run,
        "output_paths": {
            entry["input_id"]: _model_output_paths(model_dir, entry["input_id"])
            for entry in selection.entries
        },
    }
    for entry in selection.entries:
        _validate_shared_artifact_paths(entry)
    if args.dry_run:
        print(json.dumps(dry_run, ensure_ascii=False, indent=2))
        return dry_run

    client = make_generation_client(provider, model)
    eval_client = OpenAIJsonClient(
        model_name=resolve_stage_model("evaluation"),
        missing_key_message="OPENAI_API_KEY is required for draft evaluation.",
    )
    if generation_mode == "newsroom_v1_retrieval":
        _prepare_retrieval_runtime(retrieval_options)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = _load_existing_summary_rows(model_dir)
    rows_by_input = {str(row.get("input_id")): row for row in rows}
    console_records: list[ConsoleRunRecord] = []
    run_started = perf_counter()
    total_count = len(selection.entries)
    for index, entry in enumerate(selection.entries, start=1):
        paths = _model_output_paths(model_dir, entry["input_id"])
        if (
            args.resume
            and not args.overwrite
            and is_valid_completed_output(
                Path(paths["response"]),
                entry,
                provider,
                model,
                generation_mode,
            )
        ):
            print(
                f"[{index}/{total_count}] {entry['input_id']} | SKIPPED | existing valid output",
                flush=True,
            )
            console_records.append(
                ConsoleRunRecord(
                    input_id=entry["input_id"],
                    status="skipped",
                    elapsed_seconds=0.0,
                )
            )
            rows_by_input[entry["input_id"]] = _summary_row_from_response(
                Path(paths["response"]),
                Path(paths["telemetry"]),
                Path(paths["html"]),
            )
            continue
        stage = {"value": "LOADING"}
        if hasattr(client, "configure_diagnostics"):
            client.configure_diagnostics(
                output_dir=Path(paths["response"]).parent,
                input_id=entry["input_id"],
            )
        row, elapsed = _run_input_with_console_progress(
            index=index,
            total=total_count,
            input_id=entry["input_id"],
            provider=provider,
            model=model,
            stage=stage,
            operation=(
                lambda current_entry=entry, current_stage=stage: _generate_one(
                        current_entry,
                        provider,
                        model,
                        output_dir,
                        client,
                        eval_client,
                        generation_mode=generation_mode,
                        newsroom_prompt_version=newsroom_prompt_version,
                        retrieval_options=retrieval_options,
                        progress_stage=current_stage,
                    )
            ),
        )
        console_records.append(
            ConsoleRunRecord(
                input_id=entry["input_id"],
                status=str(row["status"]),
                elapsed_seconds=elapsed,
            )
        )
        rows_by_input[entry["input_id"]] = row
        _write_run_summary(model_dir, list(rows_by_input.values()))
    summary = _write_run_summary(model_dir, list(rows_by_input.values()))
    _print_benchmark_summary(console_records, perf_counter() - run_started)
    _print_cost_summary(list(rows_by_input.values()))
    return summary


def consolidate_command(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for summary_path in output_dir.glob("*/run_summary.json"):
        payload = _read_json(summary_path)
        for row in payload.get("rows", []):
            if isinstance(row, dict):
                rows.append(row)
    consolidated_dir = output_dir / "consolidated"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _now(),
        "source": "saved_run_summaries",
        "rows": rows,
    }
    _write_json(consolidated_dir / "model_results.json", payload)
    _write_csv(consolidated_dir / "model_results.csv", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def recalculate_costs_command(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    updated_rows_by_model: dict[Path, list[dict[str, Any]]] = {}
    for response_path in output_dir.glob("*/input_*/response.json"):
        telemetry_path = response_path.parent / "telemetry.json"
        html_path = response_path.parent / "article.html"
        if not telemetry_path.exists():
            continue
        response = _read_json(response_path)
        telemetry = _read_json(telemetry_path)
        backup_path = response_path.parent / "telemetry.before_cost_recalculation.json"
        if not backup_path.exists():
            _write_json(backup_path, telemetry)
        updated = recalculate_saved_costs(response, telemetry)
        _write_json_atomic(telemetry_path, updated)
        _write_text_atomic(html_path, _article_html(response, updated))
        row = _summary_row(
            response,
            updated,
            {
                "response": str(response_path),
                "telemetry": str(telemetry_path),
                "html": str(html_path),
            },
        )
        updated_rows_by_model.setdefault(response_path.parents[1], []).append(row)
    summaries = {}
    for model_dir, rows in updated_rows_by_model.items():
        summary_backup = model_dir / "run_summary.before_cost_recalculation.json"
        if (model_dir / "run_summary.json").exists() and not summary_backup.exists():
            _write_json(summary_backup, _read_json(model_dir / "run_summary.json"))
        summaries[str(model_dir)] = _write_run_summary(model_dir, rows)
    result = {
        "mode": "recalculate-costs",
        "updated_model_dirs": sorted(summaries),
        "pricing_configuration_id": _pricing_version(PRICE_CONFIG_PATH),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def recalculate_saved_costs(
    response: dict[str, Any],
    telemetry: dict[str, Any],
) -> dict[str, Any]:
    draft = _dict_value(_dict_value(response.get("draft")).get("draft"))
    evaluation = _dict_value(response.get("evaluation"))
    eval_payload = _dict_value(evaluation.get("evaluation"))
    provider = str(response.get("provider") or telemetry.get("provider") or "")
    model = str(response.get("generation_model") or telemetry.get("model") or "")
    ledger = _list_value(telemetry.get("generation_call_ledger"))
    if not ledger:
        ledger = build_generation_call_ledger(
            provider=provider,
            model=model,
            draft=draft,
        )
    generation_usage = _dict_value(draft.get("token_usage"))
    eval_usage = _dict_value(eval_payload.get("token_usage"))
    cost_payload = build_cost_payload(
        generation_provider=provider,
        generation_model=model,
        generation_usage=generation_usage,
        generation_ledger=[row for row in ledger if isinstance(row, dict)],
        evaluation_provider=str(evaluation.get("model_provider") or "openai"),
        evaluation_model=str(evaluation.get("model_name") or "gpt-4o-mini"),
        evaluation_usage=eval_usage,
    )
    if not _list_value(telemetry.get("generation_call_ledger")):
        cost_payload["cost_assumptions"] = [
            *COST_ASSUMPTIONS,
            "Historical recalculation used saved aggregate or partial trace token usage.",
        ]
    generation = cost_payload["cost_breakdown"]["generation"]
    return {
        **telemetry,
        "prompt_tokens": generation.get("prompt_tokens"),
        "completion_tokens": generation.get("completion_tokens"),
        "total_tokens": generation.get("provider_total_tokens"),
        "cached_prompt_tokens": generation.get("cached_prompt_tokens"),
        "uncached_prompt_tokens": generation.get("uncached_prompt_tokens"),
        "estimated_input_cost": generation.get("input_cost_usd"),
        "estimated_cached_input_cost": generation.get("cached_input_cost_usd"),
        "estimated_output_cost": generation.get("output_cost_usd"),
        "estimated_total_cost": generation.get("total_cost_usd"),
        "cost_status": generation.get("cost_status"),
        **cost_payload,
        "billable_failed_call_count": sum(
            1
            for row in cost_payload.get("generation_call_ledger", [])
            if isinstance(row, dict)
            and row.get("call_status") == "failed"
            and row.get("total_cost_usd") is not None
        ),
        "cost_incurred_before_failure_usd": (
            generation.get("total_cost_usd")
            if response.get("completion_status") == "failed"
            else None
        ),
    }


class InputHeartbeat:
    def __init__(
        self,
        *,
        index: int,
        total: int,
        input_id: str,
        stage: dict[str, str],
        started_at: float,
        interval_seconds: float | None = None,
    ) -> None:
        self._index = index
        self._total = total
        self._input_id = input_id
        self._stage = stage
        self._started_at = started_at
        self._interval_seconds = (
            HEARTBEAT_INTERVAL_SECONDS
            if interval_seconds is None
            else interval_seconds
        )
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(self._interval_seconds, 0.1) + 1.0)

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            elapsed = perf_counter() - self._started_at
            stage = self._stage.get("value") or "RUNNING"
            print(
                f"[{self._index}/{self._total}] {self._input_id} | {stage} | elapsed={_format_duration(elapsed)}",
                flush=True,
            )


def _run_input_with_console_progress(
    *,
    index: int,
    total: int,
    input_id: str,
    provider: str,
    model: str,
    stage: dict[str, str],
    operation: Any,
) -> tuple[dict[str, Any], float]:
    started = perf_counter()
    print(
        f"[{index}/{total}] {input_id} | {provider} | {model} | STARTED",
        flush=True,
    )
    heartbeat = InputHeartbeat(
        index=index,
        total=total,
        input_id=input_id,
        stage=stage,
        started_at=started,
    )
    heartbeat.start()
    try:
        row = operation()
        elapsed = perf_counter() - started
        if row.get("status") == "completed":
            print(
                f"[{index}/{total}] {input_id} | COMPLETED | runtime={_format_duration(elapsed)}",
                flush=True,
            )
        else:
            error = _concise_error(row.get("error_message") or row.get("error_type"))
            print(
                f"[{index}/{total}] {input_id} | FAILED | runtime={_format_duration(elapsed)} | error={error}",
                flush=True,
            )
        return row, elapsed
    except Exception as exc:
        elapsed = perf_counter() - started
        print(
            f"[{index}/{total}] {input_id} | FAILED | runtime={_format_duration(elapsed)} | error={_concise_error(exc)}",
            flush=True,
        )
        raise
    finally:
        heartbeat.stop()


def _print_benchmark_summary(
    records: list[ConsoleRunRecord],
    total_runtime_seconds: float,
) -> None:
    completed = [record for record in records if record.status == "completed"]
    failed = [record for record in records if record.status == "failed"]
    skipped = [record for record in records if record.status == "skipped"]
    average = (
        sum(record.elapsed_seconds for record in completed) / len(completed)
        if completed
        else 0.0
    )
    fastest = min(completed, key=lambda record: record.elapsed_seconds, default=None)
    slowest = max(completed, key=lambda record: record.elapsed_seconds, default=None)
    print("BENCHMARK COMPLETE", flush=True)
    print(f"Completed: {len(completed)}", flush=True)
    print(f"Failed: {len(failed)}", flush=True)
    print(f"Skipped: {len(skipped)}", flush=True)
    print(f"Total runtime: {_format_duration(total_runtime_seconds)}", flush=True)
    print(
        "Average runtime per completed input: "
        f"{_format_duration(average)}",
        flush=True,
    )
    print(
        "Fastest input: "
        f"{fastest.input_id} - {_format_duration(fastest.elapsed_seconds)}"
        if fastest
        else "Fastest input: None",
        flush=True,
    )
    print(
        "Slowest input: "
        f"{slowest.input_id} - {_format_duration(slowest.elapsed_seconds)}"
        if slowest
        else "Slowest input: None",
        flush=True,
    )


def _print_cost_summary(rows: list[dict[str, Any]]) -> None:
    completed = [row for row in rows if row.get("status") == "completed"]
    generation_total = _sum_money(row.get("generation_total_cost_usd") for row in rows)
    evaluation_total = _sum_money(row.get("evaluation_total_cost_usd") for row in rows)
    combined_total = _sum_money(row.get("combined_total_cost_usd") for row in rows)
    average_generation = (
        generation_total / Decimal(len(completed)) if completed else Decimal("0")
    )
    average_combined = (
        combined_total / Decimal(len(completed)) if completed else Decimal("0")
    )
    incomplete = [
        str(row.get("input_id"))
        for row in rows
        if row.get("combined_cost_status") != "calculated"
    ]
    print("COST SUMMARY", flush=True)
    print(f"Generation cost: USD {_money_text(generation_total)}", flush=True)
    print(f"Grounding evaluation cost: USD {_money_text(evaluation_total)}", flush=True)
    print(f"Combined cost: USD {_money_text(combined_total)}", flush=True)
    print(
        "Average generation cost per completed input: "
        f"USD {_money_text(average_generation)}",
        flush=True,
    )
    print(
        "Average combined cost per completed input: "
        f"USD {_money_text(average_combined)}",
        flush=True,
    )
    print(f"Pricing version: {_pricing_version(PRICE_CONFIG_PATH)}", flush=True)
    print(
        f"Cost coverage: {'partial' if incomplete else 'complete'}",
        flush=True,
    )
    if incomplete:
        print(
            "Inputs with incomplete cost data: " + ", ".join(incomplete),
            flush=True,
        )


def load_input_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json_any(path)
    inputs = payload if isinstance(payload, list) else payload.get("inputs")
    if not isinstance(inputs, list):
        raise BenchmarkError("Input manifest must contain an inputs list.")
    normalized = [normalize_input_entry(raw, index) for index, raw in enumerate(inputs, start=1)]
    if len(normalized) != 10:
        raise BenchmarkError(f"Input manifest must contain exactly 10 inputs; found {len(normalized)}.")
    expected = [f"input_{index:02d}" for index in range(1, 11)]
    actual = [entry["input_id"] for entry in normalized]
    if actual != expected:
        raise BenchmarkError("Input IDs must be input_01 through input_10 in order.")
    return {"inputs": normalized}


def load_word_input_manifest(
    input_dir: Path,
    *,
    author_id: str,
    output_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    discovery = discover_word_inputs(input_dir, output_dir)
    supported = [
        item for item in discovery["files"] if item["supported"] and not item["empty_or_unreadable"]
    ]
    if len(supported) != 10:
        raise BenchmarkError(
            "Word input directory must contain exactly 10 readable supported files; "
            f"found {len(supported)}."
        )
    inputs = []
    for index, item in enumerate(supported, start=1):
        input_id = f"input_{index:02d}"
        item["assigned_input_id"] = input_id
        inputs.append(
            {
                "input_id": input_id,
                "source_title": Path(str(item["filename"])).stem,
                "source_language": "unknown",
                "source_text": None,
                "source_path": item["path"],
                "source_type": "text",
                "source_input_mode": "plain_text",
                "author_id": author_id,
                "desired_word_count": DEFAULT_DESIRED_WORD_COUNT,
                "tone": DEFAULT_TONE,
                "article_type": DEFAULT_ARTICLE_TYPE,
                "author_instruction": DEFAULT_AUTHOR_INSTRUCTION,
                "original_filename": item["filename"],
                "file_type": item["file_type"],
                "brief_id": None,
                "brief_path": None,
                "plan_id": None,
                "plan_path": None,
                "shared_artifacts_status": "pending",
            }
        )
    discovery["supported_file_count"] = len(supported)
    return {"inputs": inputs}, discovery


def load_prepared_or_pending_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    inputs = payload.get("inputs")
    if not isinstance(inputs, list):
        raise BenchmarkError("Prepared manifest is invalid.")
    return {"inputs": [normalize_input_entry(entry, index) for index, entry in enumerate(inputs, start=1)]}


def discover_word_inputs(input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    if not input_dir.exists():
        raise BenchmarkError(f"Input directory does not exist: {input_dir}")
    word_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".docx", ".doc"}
            and not path.name.startswith("~$")
        ],
        key=lambda path: path.name.casefold(),
    )
    for path in word_paths:
        supported = path.suffix.lower() == ".docx"
        empty_or_unreadable = False
        message = None
        char_count = None
        if supported:
            try:
                extracted = extract_docx_text(path)
                char_count = extracted.char_count
                empty_or_unreadable = not bool(extracted.text.strip())
                if empty_or_unreadable:
                    message = "empty extracted text"
            except DocxExtractionError as exc:
                empty_or_unreadable = True
                message = str(exc)
        else:
            message = ".doc extraction is not supported by the current repository path"
        item: dict[str, Any] = {
            "filename": path.name,
            "path": str(path),
            "file_type": path.suffix.lower(),
            "supported": supported,
            "empty_or_unreadable": empty_or_unreadable,
            "message": message,
            "char_count": char_count,
        }
        if output_dir is not None:
            source_path = output_dir / "shared" / f"input_{len([f for f in files if f.get('supported') and not f.get('empty_or_unreadable')]) + 1:02d}" / "source.json"
            item["expected_source_json_path"] = str(source_path)
        files.append(item)
    return {
        "input_dir": str(input_dir),
        "word_file_count": len(word_paths),
        "supported_file_count": len([item for item in files if item["supported"]]),
        "unsupported_files": [item for item in files if not item["supported"]],
        "empty_or_unreadable_files": [item for item in files if item["empty_or_unreadable"]],
        "files": files,
    }


def load_prepared_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise BenchmarkError("Prepared manifest is invalid.")
    normalized = [
        normalize_input_entry(entry, index)
        for index, entry in enumerate(inputs, start=1)
    ]
    for entry in normalized:
        if not isinstance(entry, dict):
            raise BenchmarkError("Prepared manifest contains a non-object input.")
        if not _shared_artifacts_complete(entry, path.parents[1]):
            raise BenchmarkError(f"Shared artifacts are incomplete for {entry.get('input_id')}.")
    return {"inputs": normalized}


def normalize_input_entry(raw: object, index: int) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise BenchmarkError("Each input manifest entry must be an object.")
    input_id = str(raw.get("input_id") or f"input_{index:02d}")
    source_text = raw.get("source_text")
    source_path = raw.get("source_path")
    if source_text is None and source_path is None:
        raise BenchmarkError(f"{input_id} must include source_text or source_path.")
    if source_text is not None and not str(source_text).strip():
        raise BenchmarkError(f"{input_id} source_text is empty.")
    return {
        "input_id": input_id,
        "source_title": str(raw.get("source_title") or _title_from_source(source_text, source_path, input_id)),
        "source_language": str(raw.get("source_language") or "unknown"),
        "source_text": str(source_text) if source_text is not None else None,
        "source_path": str(source_path) if source_path is not None else None,
        "source_type": str(raw.get("source_type") or "text"),
        "source_input_mode": str(raw.get("source_input_mode") or "plain_text"),
        "author_id": str(raw.get("author_id") or AUTHOR_ID),
        "desired_word_count": int(raw.get("desired_word_count") or DEFAULT_DESIRED_WORD_COUNT),
        "tone": str(raw.get("tone") or DEFAULT_TONE),
        "article_type": str(raw.get("article_type") or DEFAULT_ARTICLE_TYPE),
        "author_instruction": str(raw.get("author_instruction") or DEFAULT_AUTHOR_INSTRUCTION),
        "original_filename": raw.get("original_filename"),
        "file_type": raw.get("file_type"),
        "brief_id": raw.get("brief_id"),
        "brief_path": raw.get("brief_path"),
        "plan_id": raw.get("plan_id"),
        "plan_path": raw.get("plan_path"),
        "topic_metadata": raw.get("topic_metadata"),
        "shared_artifacts_status": raw.get("shared_artifacts_status") or "pending",
    }


def select_inputs(
    entries: list[dict[str, Any]],
    *,
    input_id: str | None,
    start_from: str | None,
    max_inputs: int | None,
) -> InputSelection:
    selected = entries
    if input_id:
        selected = [entry for entry in entries if entry["input_id"] == input_id]
        if not selected:
            raise BenchmarkError(f"Unknown input_id: {input_id}")
    elif start_from:
        ids = [entry["input_id"] for entry in entries]
        if start_from not in ids:
            raise BenchmarkError(f"Unknown start-from input_id: {start_from}")
        selected = entries[ids.index(start_from) :]
    if max_inputs is not None:
        if max_inputs < 1:
            raise BenchmarkError("--max-inputs must be at least 1.")
        selected = selected[:max_inputs]
    return InputSelection(
        input_ids=[entry["input_id"] for entry in selected],
        entries=selected,
    )


def validate_provider_model(provider: str, model: str) -> None:
    if provider in FUTURE_MODELS and model in FUTURE_MODELS[provider]:
        raise BenchmarkError(f"{provider} provider is not yet integrated.")
    if provider not in SUPPORTED_MODELS:
        raise BenchmarkError(f"Unsupported provider: {provider}")
    if model not in SUPPORTED_MODELS[provider]:
        raise BenchmarkError(f"Unsupported model for {provider}: {model}")


def safe_model_dir(model: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in model).strip("_")


def _validate_generation_mode(mode: str) -> str:
    normalized = str(mode or "legacy").strip().lower()
    if normalized not in GENERATION_MODES:
        raise BenchmarkError(f"Unsupported generation mode: {mode}")
    return normalized


def _validate_newsroom_prompt_version(version: str) -> str:
    normalized = str(version or DEFAULT_NEWSROOM_PROMPT_VERSION).strip()
    if normalized not in NEWSROOM_PROMPT_VERSION_PATHS:
        raise BenchmarkError(f"Unsupported newsroom prompt version: {version}")
    return normalized


def _validate_retrieval_prompt_version(version: str) -> str:
    normalized = str(version or DEFAULT_RETRIEVAL_PROMPT_VERSION).strip()
    if normalized not in NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS:
        raise BenchmarkError(f"Unsupported retrieval prompt version: {version}")
    return normalized


def _generation_model_dir(
    output_dir: Path,
    provider: str,
    model: str,
    generation_mode: str,
    *,
    newsroom_prompt_version: str = DEFAULT_NEWSROOM_PROMPT_VERSION,
    retrieval_prompt_version: str = DEFAULT_RETRIEVAL_PROMPT_VERSION,
) -> Path:
    if generation_mode == "legacy":
        return output_dir / safe_model_dir(model)
    if generation_mode == "newsroom_v1_retrieval":
        if retrieval_prompt_version != DEFAULT_RETRIEVAL_PROMPT_VERSION:
            version_slug = safe_model_dir(retrieval_prompt_version)
            return output_dir / (
                f"{generation_mode}_{version_slug}_{provider}_{safe_model_dir(model)}"
            )
        return output_dir / f"{generation_mode}_{provider}_{safe_model_dir(model)}"
    if newsroom_prompt_version != DEFAULT_NEWSROOM_PROMPT_VERSION:
        version_slug = safe_model_dir(newsroom_prompt_version)
        return output_dir / f"{version_slug}_{provider}_{safe_model_dir(model)}"
    return output_dir / f"{generation_mode}_{provider}_{safe_model_dir(model)}"


def _generation_prompt_metadata(
    generation_mode: str,
    *,
    newsroom_prompt_version: str = DEFAULT_NEWSROOM_PROMPT_VERSION,
    retrieval_prompt_version: str = DEFAULT_RETRIEVAL_PROMPT_VERSION,
) -> dict[str, Any]:
    if generation_mode == "legacy":
        return {
            "generation_mode": "legacy",
            "prompt_version": LEGACY_PROMPT_VERSION,
            "newsroom_profile_version": None,
        }
    if generation_mode == "newsroom_v1":
        _prompt_path, metadata_path = NEWSROOM_PROMPT_VERSION_PATHS[
            newsroom_prompt_version
        ]
        payload = _read_json(metadata_path)
        return {
            "generation_mode": "newsroom_v1",
            "prompt_version": payload["prompt_version"],
            "newsroom_profile_version": payload["newsroom_profile_version"],
        }
    if generation_mode == "newsroom_v1_retrieval":
        _prompt_path, metadata_path = NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS[
            retrieval_prompt_version
        ]
        payload = _read_json(metadata_path)
        return {
            "generation_mode": "newsroom_v1_retrieval",
            "prompt_version": payload["prompt_version"],
            "base_prompt_version": payload["base_prompt_version"],
            "retrieval_prompt_version": payload["retrieval_prompt_version"],
            "newsroom_profile_version": payload["newsroom_profile_version"],
        }
    raise BenchmarkError(f"Unsupported generation mode: {generation_mode}")


def _current_git_commit() -> str | None:
    try:
        return check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            stderr=DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def is_valid_completed_output(
    response_path: Path,
    entry: dict[str, Any],
    provider: str,
    model: str,
    generation_mode: str = "legacy",
) -> bool:
    if not response_path.exists():
        return False
    try:
        payload = _read_json(response_path)
    except BenchmarkError:
        return False
    workflow = payload.get("workflow_settings")
    article = str(payload.get("generated_tamil_article") or "").strip()
    word_count = payload.get("word_count")
    return (
        payload.get("completion_status") == "completed"
        and bool(article)
        and len(article) >= 80
        and isinstance(word_count, int)
        and word_count > 0
        and bool(payload.get("provider"))
        and bool(payload.get("generation_model"))
        and payload.get("provider") == provider
        and payload.get("generation_model") == model
        and (
            payload.get("generation_mode") == generation_mode
            or (generation_mode == "legacy" and payload.get("generation_mode") is None)
        )
        and payload.get("author_id") == entry.get("author_id")
        and payload.get("input_id") == entry.get("input_id")
        and payload.get("brief_id") == entry.get("brief_id")
        and payload.get("plan_id") == entry.get("plan_id")
        and workflow
        == {
            "grounding_evaluation": True,
            "auto_revision": False,
            "final_evaluation": False,
        }
    )


def make_generation_client(provider: str, model: str) -> Any:
    if provider == "openai":
        return OpenAIJsonClient(
            model_name=model,
            missing_key_message="OPENAI_API_KEY is required for OpenAI generation.",
        )
    if provider == "gemini":
        return GeminiJsonClient(
            model_name=model,
            missing_key_message="GEMINI_API_KEY is required for Gemini generation.",
        )
    if provider == "grok":
        return GrokJsonClient(
            model_name=model,
            missing_key_message="XAI_API_KEY is required for Grok generation.",
        )
    raise BenchmarkError(f"{provider} provider is not yet integrated.")


def calculate_cost(
    *,
    provider: str,
    model: str,
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing_path: Path = PRICE_CONFIG_PATH,
) -> dict[str, Any]:
    return price_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        pricing_path=pricing_path,
    )


def price_token_usage(
    *,
    provider: str,
    model: str,
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
    completion_tokens: int | None,
    pricing_path: Path = PRICE_CONFIG_PATH,
) -> dict[str, Any]:
    pricing = pricing_lookup(provider, model, pricing_path)
    base = {
        "provider": provider,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_prompt_tokens,
        "uncached_prompt_tokens": _uncached_tokens(prompt_tokens, cached_prompt_tokens),
        "completion_tokens": completion_tokens,
        "currency": pricing.get("currency") if pricing else None,
        "pricing_configuration_id": _pricing_version(pricing_path),
        "pricing_effective_date": pricing.get("effective_date") if pricing else None,
        "pricing_mode": pricing.get("pricing_mode") if pricing else None,
    }
    if pricing is None:
        return _unavailable_cost(base, "pricing_unavailable")
    if prompt_tokens is None or completion_tokens is None:
        return _unavailable_cost(base, "token_usage_unavailable")
    cached_tokens = cached_prompt_tokens or 0
    uncached_tokens = max(prompt_tokens - cached_tokens, 0)
    input_cost = _token_cost(uncached_tokens, pricing["input_usd_per_million"])
    cached_input_cost = _token_cost(
        cached_tokens,
        pricing["cached_input_usd_per_million"],
    )
    output_cost = _token_cost(completion_tokens, pricing["output_usd_per_million"])
    total = input_cost + cached_input_cost + output_cost
    return {
        **base,
        "input_cost_usd": _money(input_cost),
        "cached_input_cost_usd": _money(cached_input_cost),
        "output_cost_usd": _money(output_cost),
        "total_cost_usd": _money(total),
        "cost_status": "calculated",
    }


def build_generation_call_ledger(
    *,
    provider: str,
    model: str,
    draft: dict[str, Any],
) -> list[dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    traces = _list_value(draft.get("section_generation_trace"))
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        section_ids = [str(trace.get("section_id") or "")]
        first_usage = _dict_value(trace.get("first_pass_token_usage"))
        if first_usage:
            ledger.append(
                _ledger_entry(
                    provider=provider,
                    model=model,
                    operation=(
                        "section_group_generation"
                        if trace.get("group_generation_used")
                        else "section_generation"
                    ),
                    section_ids=section_ids,
                    attempt=1,
                    usage=first_usage,
                )
            )
        retry_usage = _dict_value(trace.get("retry_token_usage"))
        if retry_usage:
            ledger.append(
                _ledger_entry(
                    provider=provider,
                    model=model,
                    operation=(
                        "section_group_retry"
                        if trace.get("group_generation_used")
                        else "section_retry"
                    ),
                    section_ids=section_ids,
                    attempt=2,
                    usage=retry_usage,
                )
            )
    aggregate_usage = _dict_value(draft.get("token_usage"))
    if aggregate_usage:
        aggregate_prompt = _optional_int(aggregate_usage.get("prompt_tokens")) or 0
        aggregate_cached = _optional_int(aggregate_usage.get("cached_prompt_tokens")) or 0
        aggregate_completion = _optional_int(aggregate_usage.get("completion_tokens")) or 0
        aggregate_total = _optional_int(aggregate_usage.get("total_tokens")) or 0
        ledger_prompt = sum(_optional_int(row.get("prompt_tokens")) or 0 for row in ledger)
        ledger_cached = sum(_optional_int(row.get("cached_prompt_tokens")) or 0 for row in ledger)
        ledger_completion = sum(_optional_int(row.get("completion_tokens")) or 0 for row in ledger)
        ledger_total = sum(_optional_int(row.get("provider_total_tokens")) or 0 for row in ledger)
        missing_usage = {
            "prompt_tokens": aggregate_prompt - ledger_prompt,
            "cached_prompt_tokens": aggregate_cached - ledger_cached,
            "completion_tokens": aggregate_completion - ledger_completion,
            "total_tokens": aggregate_total - ledger_total,
        }
        if any(value > 0 for value in missing_usage.values()):
            ledger.append(
                _ledger_entry(
                    provider=provider,
                    model=model,
                    operation="aggregate_untraced_generation",
                    section_ids=[],
                    attempt=1,
                    usage={key: max(value, 0) for key, value in missing_usage.items()},
                    cost_accuracy="aggregate_estimate",
                )
            )
    return ledger


def build_client_call_ledger(client: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in getattr(client, "call_records", []) or []:
        if not isinstance(record, dict):
            continue
        usage = _dict_value(record.get("usage"))
        if not usage:
            continue
        rows.append(
            _ledger_entry(
                provider=str(record.get("provider") or ""),
                model=str(record.get("model") or ""),
                operation=str(record.get("operation") or "generation"),
                section_ids=[],
                attempt=_optional_int(record.get("attempt")) or 1,
                usage={
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "cached_prompt_tokens": usage.get("cached_prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                    "reasoning_tokens": usage.get("reasoning_tokens"),
                    "accepted_prediction_tokens": usage.get("accepted_prediction_tokens"),
                    "rejected_prediction_tokens": usage.get("rejected_prediction_tokens"),
                    "provider_cost_ticks": usage.get("provider_cost_ticks"),
                    "provider_reported_cost_usd": usage.get("provider_reported_cost_usd"),
                    "provider_cost_conversion_status": usage.get("provider_cost_conversion_status"),
                },
                cost_accuracy="per_call_calculated",
                status=str(record.get("status") or ""),
                failure_type=record.get("failure_type"),
                raw_response_path=record.get("raw_response_path"),
            )
        )
    return rows


def build_cost_payload(
    *,
    generation_provider: str,
    generation_model: str,
    generation_usage: dict[str, Any],
    generation_ledger: list[dict[str, Any]],
    evaluation_provider: str,
    evaluation_model: str,
    evaluation_usage: dict[str, Any],
) -> dict[str, Any]:
    generation_totals = aggregate_generation_ledger(
        generation_provider,
        generation_model,
        generation_usage,
        generation_ledger,
    )
    evaluation_cost = _cost_breakdown_from_usage(
        provider=evaluation_provider,
        model=evaluation_model,
        usage=evaluation_usage,
    )
    combined_total, combined_status = _combined_cost(
        generation_totals.get("total_cost_usd"),
        generation_totals.get("cost_status"),
        evaluation_cost.get("total_cost_usd"),
        evaluation_cost.get("cost_status"),
    )
    pricing = pricing_lookup(generation_provider, generation_model) or pricing_lookup(
        evaluation_provider,
        evaluation_model,
    )
    return {
        "cost_breakdown": {
            "generation": generation_totals,
            "grounding_evaluation": evaluation_cost,
            "combined": {
                "total_cost_usd": combined_total,
                "cost_status": combined_status,
            },
        },
        "generation_call_ledger": generation_ledger,
        "pricing_configuration_id": _pricing_version(PRICE_CONFIG_PATH),
        "pricing_effective_date": pricing.get("effective_date") if pricing else None,
        "pricing_mode": pricing.get("pricing_mode") if pricing else None,
        "cost_currency": pricing.get("currency") if pricing else "USD",
        "cost_accuracy": _cost_accuracy(generation_ledger, generation_totals, evaluation_cost),
        "cost_assumptions": COST_ASSUMPTIONS,
        "cost_exclusions": [
            "Provider-side charges not represented in token usage metadata",
            "Non-generation preparation costs",
        ],
    }


def aggregate_generation_ledger(
    provider: str,
    model: str,
    aggregate_usage: dict[str, Any],
    ledger: list[dict[str, Any]],
) -> dict[str, Any]:
    if ledger:
        prompt_tokens = sum(_optional_int(row.get("prompt_tokens")) or 0 for row in ledger)
        cached_tokens = sum(_optional_int(row.get("cached_prompt_tokens")) or 0 for row in ledger)
        completion_tokens = sum(_optional_int(row.get("completion_tokens")) or 0 for row in ledger)
        provider_total = sum(_optional_int(row.get("provider_total_tokens")) or 0 for row in ledger)
        reasoning_tokens = _sum_optional_int(row.get("reasoning_tokens") for row in ledger)
        accepted_prediction_tokens = _sum_optional_int(
            row.get("accepted_prediction_tokens") for row in ledger
        )
        rejected_prediction_tokens = _sum_optional_int(
            row.get("rejected_prediction_tokens") for row in ledger
        )
        provider_cost_ticks = _sum_optional_int(row.get("provider_cost_ticks") for row in ledger)
        provider_reported_cost_usd = _sum_money_or_none(
            row.get("provider_reported_cost_usd") for row in ledger
        )
        provider_cost_conversion_status = _provider_cost_conversion_status(ledger)
    else:
        prompt_tokens = _optional_int(aggregate_usage.get("prompt_tokens"))
        cached_tokens = _optional_int(aggregate_usage.get("cached_prompt_tokens"))
        completion_tokens = _optional_int(aggregate_usage.get("completion_tokens"))
        provider_total = _optional_int(aggregate_usage.get("total_tokens"))
        reasoning_tokens = _optional_int(aggregate_usage.get("reasoning_tokens"))
        accepted_prediction_tokens = _optional_int(
            aggregate_usage.get("accepted_prediction_tokens")
        )
        rejected_prediction_tokens = _optional_int(
            aggregate_usage.get("rejected_prediction_tokens")
        )
        provider_cost_ticks = _optional_int(aggregate_usage.get("provider_cost_ticks"))
        provider_reported_cost_usd = aggregate_usage.get("provider_reported_cost_usd")
        provider_cost_conversion_status = aggregate_usage.get(
            "provider_cost_conversion_status"
        )
    priced = price_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )
    reconciliation = _token_reconciliation(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        provider_total_tokens=provider_total,
    )
    return {
        **priced,
        "provider_total_tokens": provider_total,
        "reasoning_tokens": reasoning_tokens,
        "accepted_prediction_tokens": accepted_prediction_tokens,
        "rejected_prediction_tokens": rejected_prediction_tokens,
        "provider_cost_ticks": provider_cost_ticks,
        "provider_reported_cost_usd": provider_reported_cost_usd,
        "provider_cost_conversion_status": provider_cost_conversion_status,
        **reconciliation,
    }


def _prepare_one(
    entry: dict[str, Any],
    output_dir: Path,
    repo: StyleScribeRepository,
) -> None:
    shared_input_dir = output_dir / "shared" / entry["input_id"]
    shared_input_dir.mkdir(parents=True, exist_ok=True)
    source_text = _source_text(entry)
    source_payload = {
        "input_id": entry["input_id"],
        "source_title": entry["source_title"],
        "source_language": entry["source_language"],
        "original_filename": entry.get("original_filename"),
        "file_type": entry.get("file_type"),
        "original_source_path": entry.get("source_path"),
        "source_text": source_text if entry["source_text"] is not None else None,
        "source_path": entry["source_path"],
        "desired_word_count": entry["desired_word_count"],
        "tone": entry["tone"],
        "article_type": entry["article_type"],
        "author_id": entry["author_id"],
    }
    _write_json(shared_input_dir / "source.json", source_payload)
    telemetry: dict[str, Any] = {
        "input_id": entry["input_id"],
        "preparation_status": "started",
        "started_at": _now(),
        "errors": None,
    }
    try:
        brief_started = perf_counter()
        brief = generate_grounded_brief(
            source_type=entry["source_type"],
            source_input=source_text,
            target_language=TARGET_LANGUAGE,
            source_input_mode=entry["source_input_mode"],
            repository=repo,
        )
        brief_runtime = round(perf_counter() - brief_started, 3)
        entry["brief_id"] = brief.brief_id
        entry["brief_path"] = str(shared_input_dir / "brief.json")
        _write_json(shared_input_dir / "brief.json", brief.model_dump(mode="json"))
        entry["topic_metadata"] = topic_metadata_from_brief(
            brief.brief,
            input_id=str(entry["input_id"]),
        )

        planning_started = perf_counter()
        plan = generate_article_plan(
            brief_id=brief.brief_id,
            author_id=entry["author_id"],
            article_type=entry["article_type"],
            desired_word_count=entry["desired_word_count"],
            target_language=TARGET_LANGUAGE,
            tone_override=entry["tone"],
            author_instruction=entry["author_instruction"],
            repository=repo,
        )
        planning_runtime = round(perf_counter() - planning_started, 3)
        entry["plan_id"] = plan.plan_id
        entry["plan_path"] = str(shared_input_dir / "plan.json")
        entry["shared_artifacts_status"] = "completed"
        _write_json(shared_input_dir / "plan.json", _dataclass_dict(plan))
        telemetry.update(
            {
                "preparation_status": "completed",
                "brief_model": brief.model_name,
                "planning_model": plan.model_name,
                "brief_runtime_seconds": brief_runtime,
                "planning_runtime_seconds": planning_runtime,
                "token_usage": {
                    "brief": brief.brief.get("token_usage"),
                    "planning": plan.token_usage,
                },
                "estimated_cost": None,
                "completed_at": _now(),
            }
        )
    except Exception as exc:
        entry["shared_artifacts_status"] = "failed"
        telemetry.update(
            {
                "preparation_status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
                "completed_at": _now(),
            }
        )
        _write_json(shared_input_dir / "preparation_telemetry.json", telemetry)
        raise
    _write_json(shared_input_dir / "preparation_telemetry.json", telemetry)


def _generate_one(
    entry: dict[str, Any],
    provider: str,
    model: str,
    output_dir: Path,
    generation_client: Any,
    eval_client: OpenAIJsonClient,
    *,
    generation_mode: str = "legacy",
    newsroom_prompt_version: str = DEFAULT_NEWSROOM_PROMPT_VERSION,
    retrieval_options: dict[str, Any] | None = None,
    progress_stage: dict[str, str] | None = None,
) -> dict[str, Any]:
    if progress_stage is not None:
        progress_stage["value"] = "LOADING"
    model_dir = _generation_model_dir(
        output_dir,
        provider,
        model,
        generation_mode,
        newsroom_prompt_version=newsroom_prompt_version,
        retrieval_prompt_version=str(
            (retrieval_options or {}).get(
                "retrieval_prompt_version",
                DEFAULT_RETRIEVAL_PROMPT_VERSION,
            )
        ),
    )
    paths = _model_output_paths(model_dir, entry["input_id"])
    input_dir = Path(paths["response"]).parent
    input_dir.mkdir(parents=True, exist_ok=True)
    started_at = _now()
    total_started = perf_counter()
    generation_runtime = None
    evaluation_runtime = None
    retrieval_trace: dict[str, Any] | None = None
    retrieval_leakage: dict[str, Any] | None = None
    draft = None
    evaluation = None
    error: Exception | None = None
    canonical_article: CanonicalArticle | None = None
    try:
        if progress_stage is not None:
            progress_stage["value"] = "GENERATION"
        generation_started = perf_counter()
        draft = _generate_draft_for_mode(
            generation_mode=generation_mode,
            entry=entry,
            model_client=generation_client,
            git_commit=_current_git_commit(),
            newsroom_prompt_version=newsroom_prompt_version,
            retrieval_options=retrieval_options or {},
        )
        generation_runtime = round(perf_counter() - generation_started, 3)
        canonical_article = extract_canonical_article(draft)
        _validate_canonical_article(
            canonical_article,
            provider=provider,
            model=model,
        )
        if progress_stage is not None:
            progress_stage["value"] = "GROUNDING"
        evaluation_started = perf_counter()
        evaluation = evaluate_draft_grounding(
            draft.draft_id,
            model_client=eval_client,
        )
        evaluation_runtime = round(perf_counter() - evaluation_started, 3)
        draft_payload_for_trace = _dict_value(draft.model_dump(mode="json").get("draft"))
        raw_trace = draft_payload_for_trace.get("retrieval_trace")
        if isinstance(raw_trace, dict):
            retrieval_trace = raw_trace
            retrieval_leakage = _run_retrieval_leakage_for_entry(
                entry,
                article=canonical_article.article_body if canonical_article else "",
                retrieval_trace=retrieval_trace,
                retrieval_options=retrieval_options or {},
            )
    except Exception as exc:
        error = exc
    total_runtime = round(perf_counter() - total_started, 3)
    completed_at = _now()
    if progress_stage is not None:
        progress_stage["value"] = "PERSISTING"

    draft_dict = draft.model_dump(mode="json") if draft else {}
    evaluation_dict = evaluation.model_dump(mode="json") if evaluation else {}
    if canonical_article is None and draft is not None:
        try:
            canonical_article = extract_canonical_article(draft)
        except Exception:
            canonical_article = None
    article = canonical_article.article_body if canonical_article else ""
    word_count = canonical_article.word_count if canonical_article else None
    runtime_metadata = _request_metadata(provider, model, generation_client)
    draft_payload = _dict_value(draft_dict.get("draft"))
    prompt_metadata = _generation_prompt_metadata(
        generation_mode,
        newsroom_prompt_version=newsroom_prompt_version,
        retrieval_prompt_version=str(
            (retrieval_options or {}).get(
                "retrieval_prompt_version",
                DEFAULT_RETRIEVAL_PROMPT_VERSION,
            )
        ),
    )
    topic_metadata = _topic_metadata_for_entry(entry)
    git_commit = _current_git_commit()
    token_usage = _dict_value(draft_payload.get("token_usage"))
    eval_payload = _dict_value(evaluation_dict.get("evaluation"))
    eval_usage = _dict_value(eval_payload.get("token_usage"))
    client_ledger = build_client_call_ledger(generation_client)
    generation_ledger = client_ledger or build_generation_call_ledger(
        provider=provider,
        model=model,
        draft=draft_payload,
    )
    cost_payload = build_cost_payload(
        generation_provider=provider,
        generation_model=model,
        generation_usage=token_usage,
        generation_ledger=generation_ledger,
        evaluation_provider=str(evaluation_dict.get("model_provider") or "openai"),
        evaluation_model=str(evaluation_dict.get("model_name") or "gpt-4o-mini"),
        evaluation_usage=eval_usage,
    )
    generation_breakdown = cost_payload["cost_breakdown"]["generation"]
    evaluation_diagnostics = _evaluation_anomaly_diagnostics(eval_payload)
    target_min = _optional_int(_read_plan(entry).get("target_min_word_count"))
    target_max = _optional_int(_read_plan(entry).get("target_max_word_count"))
    response = {
        "input_id": entry["input_id"],
        "source_title": entry["source_title"],
        "source_language": entry["source_language"],
        "topic_metadata": topic_metadata,
        "author_id": entry["author_id"],
        "brief_id": entry["brief_id"],
        "plan_id": entry["plan_id"],
        "provider": provider,
        "generation_model": model,
        "generation_mode": generation_mode,
        "prompt_version": prompt_metadata["prompt_version"],
        "retrieval_prompt_version": prompt_metadata.get("retrieval_prompt_version"),
        "newsroom_profile_version": prompt_metadata.get("newsroom_profile_version"),
        "git_commit": git_commit,
        "input_identifier": entry["input_id"],
        "generated_headline": canonical_article.headline if canonical_article else _dict_value(draft_dict.get("draft")).get("headline"),
        "generated_subheadline": canonical_article.subheadline if canonical_article else _dict_value(draft_dict.get("draft")).get("subheadline"),
        "generated_tamil_article": article,
        "canonical_article_source_field": canonical_article.source_field if canonical_article else None,
        "model_reported_word_count": canonical_article.model_reported_word_count if canonical_article else None,
        "word_count_discrepancy": (
            canonical_article.model_reported_word_count - canonical_article.word_count
            if canonical_article and canonical_article.model_reported_word_count is not None
            else None
        ),
        "word_count": word_count,
        "desired_word_count": entry["desired_word_count"],
        "target_minimum": target_min,
        "target_maximum": target_max,
        "word_count_variance": _word_count_variance(word_count, entry["desired_word_count"]),
        "within_target_range": _within_range(word_count, target_min, target_max),
        "grounding_evaluation_result": eval_payload,
        "retrieval_trace": retrieval_trace,
        "retrieval_leakage_diagnostic": retrieval_leakage,
        "evaluation_diagnostics": evaluation_diagnostics,
        "grounding_score": eval_payload.get("grounding_score"),
        "readiness": (
            eval_payload.get("readiness")
            or eval_payload.get("publication_readiness")
            or eval_payload.get("editorial_readiness")
        ),
        "blockers": _list_value(eval_payload.get("blockers") or eval_payload.get("publication_blockers")),
        "warnings": _list_value(eval_payload.get("warnings") or eval_payload.get("publication_warnings")),
        "unsupported_claims": _list_value(eval_payload.get("unsupported_claims")),
        "claims_to_avoid_violations": _list_value(eval_payload.get("claims_to_avoid_violations")),
        "overclaims": _list_value(eval_payload.get("overclaims")),
        "repetition_indicators": _list_value(eval_payload.get("repetition_indicators")),
        "tamil_word_count_validation": eval_payload.get("tamil_word_count_validation"),
        "completion_status": (
            "completed"
            if error is None and canonical_article is not None
            else "failed"
        ),
        "workflow_settings": {
            "grounding_evaluation": True,
            "auto_revision": False,
            "final_evaluation": False,
        },
        "draft": draft_dict,
        "evaluation": evaluation_dict,
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
    }
    telemetry = {
        "input_id": entry["input_id"],
        "provider": provider,
        "model": model,
        "generation_mode": generation_mode,
        "prompt_version": prompt_metadata["prompt_version"],
        "retrieval_prompt_version": prompt_metadata.get("retrieval_prompt_version"),
        "newsroom_profile_version": prompt_metadata.get("newsroom_profile_version"),
        "git_commit": git_commit,
        "input_identifier": entry["input_id"],
        "topic_metadata": topic_metadata,
        "experiment_type": EXPERIMENT_TYPE,
        "configured_timeout": runtime_metadata["timeout_seconds"],
        "temperature_mode": runtime_metadata["temperature_mode"],
        "requested_temperature": runtime_metadata["temperature_requested"],
        "start_timestamp": started_at,
        "completion_timestamp": completed_at,
        "generation_runtime_seconds": generation_runtime,
        "retrieval_latency_seconds": (
            retrieval_trace.get("retrieval_latency_seconds")
            if retrieval_trace
            else None
        ),
        "retrieval_index_load_seconds": (
            retrieval_trace.get("index_load_seconds") if retrieval_trace else None
        ),
        "retrieval_model_load_seconds": (
            retrieval_trace.get("model_load_seconds") if retrieval_trace else None
        ),
        "grounding_evaluation_runtime_seconds": evaluation_runtime,
        "total_elapsed_runtime_seconds": total_runtime,
        "attempt_count": getattr(generation_client, "last_attempt_count", 1) or 1,
        "retry_count": getattr(generation_client, "last_retry_count", 0) or 0,
        "timeout_count": 1 if isinstance(error, TimeoutError) else 0,
        "transient_error_count": 0,
        "prompt_tokens": generation_breakdown.get("prompt_tokens"),
        "completion_tokens": generation_breakdown.get("completion_tokens"),
        "total_tokens": generation_breakdown.get("provider_total_tokens"),
        "cached_prompt_tokens": generation_breakdown.get("cached_prompt_tokens"),
        "uncached_prompt_tokens": generation_breakdown.get("uncached_prompt_tokens"),
        "estimated_input_cost": generation_breakdown.get("input_cost_usd"),
        "estimated_cached_input_cost": generation_breakdown.get("cached_input_cost_usd"),
        "estimated_output_cost": generation_breakdown.get("output_cost_usd"),
        "estimated_total_cost": generation_breakdown.get("total_cost_usd"),
        "cost_status": generation_breakdown.get("cost_status"),
        **cost_payload,
        "retrieval_trace": retrieval_trace,
        "retrieval_leakage_diagnostic": retrieval_leakage,
        "evaluation_diagnostics": evaluation_diagnostics,
        "billable_failed_call_count": sum(
            1
            for row in generation_ledger
            if row.get("call_status") == "failed"
            and row.get("total_cost_usd") is not None
        ),
        "cost_incurred_before_failure_usd": (
            generation_breakdown.get("total_cost_usd")
            if response["completion_status"] == "failed"
            else None
        ),
        "error_type": type(error).__name__ if error else None,
        "error_message": str(error) if error else None,
        "completion_status": response["completion_status"],
    }
    _write_json(Path(paths["response"]), response)
    _write_json(Path(paths["telemetry"]), telemetry)
    Path(paths["html"]).write_text(_article_html(response, telemetry), encoding="utf-8")
    return _summary_row(response, telemetry, paths)


def _evaluation_anomaly_diagnostics(eval_payload: dict[str, Any]) -> dict[str, Any]:
    anomalies: list[dict[str, Any]] = []
    contradiction_markers = (
        "directly supported",
        "supported by the grounded brief",
        "supported by the brief",
        "not a violation",
        "does not violate",
        "no violation",
    )
    for index, finding in enumerate(
        _list_value(eval_payload.get("claims_to_avoid_violations")),
        start=1,
    ):
        if not isinstance(finding, dict):
            continue
        reason = str(finding.get("reason") or "")
        explanation = " ".join(
            str(finding.get(key) or "")
            for key in ("reason", "explanation", "suggested_fix")
        ).lower()
        if any(marker in explanation for marker in contradiction_markers):
            anomalies.append(
                {
                    "anomaly_type": "claims_to_avoid_self_contradiction",
                    "finding_index": index,
                    "raw_finding": finding,
                    "severity": "review",
                    "explanation": (
                        "Evaluator marked a claims-to-avoid violation, but its "
                        "own explanation appears to describe the claim as "
                        "supported or not violating the rule."
                    ),
                    "reason_text": reason,
                }
            )
    return {
        "status": "review_required" if anomalies else "clear",
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
    }


def _request_metadata(provider: str, model: str, client: Any) -> dict[str, Any]:
    timeout = float(getattr(client, "timeout_seconds", 90.0))
    if provider == "openai":
        return request_runtime_metadata(model, 0.1, timeout)
    return {
        "temperature_mode": "explicit",
        "temperature_requested": 0.1,
        "timeout_seconds": timeout,
    }


def _retrieval_options_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "retrieval_prompt_version": _validate_retrieval_prompt_version(
            getattr(args, "retrieval_prompt_version", DEFAULT_RETRIEVAL_PROMPT_VERSION)
        ),
        "index_path": Path(str(getattr(args, "retrieval_index_path", DEFAULT_INDEX_PATH))),
        "records_path": Path(
            str(getattr(args, "retrieval_records_path", DEFAULT_RECORDS_PATH))
        ),
        "embedding_provider": str(
            getattr(args, "embedding_provider", DEFAULT_EMBEDDING_PROVIDER)
        ),
        "embedding_model": str(
            getattr(args, "embedding_model", DEFAULT_EMBEDDING_MODEL)
        ),
        "rebuild_index": bool(getattr(args, "rebuild_index", False)),
        "reuse_index": bool(getattr(args, "reuse_index", False)),
        "ranking_config": RetrievalRankingConfig(
            top_k=int(getattr(args, "retrieval_top_k", 3)),
            candidate_pool_size=int(getattr(args, "candidate_pool_size", 12)),
            topic_boost_enabled=bool(getattr(args, "topic_boost", False)),
            topic_boost_weight=float(getattr(args, "topic_boost_weight", 0.05)),
            max_examples_per_author=int(
                getattr(args, "max_examples_per_author", 1)
            ),
            max_context_chars=int(
                getattr(args, "max_retrieval_context_chars", 9000)
            ),
        ),
    }


def _retrieval_dry_run_payload(
    *,
    entries: list[dict[str, Any]],
    options: dict[str, Any],
) -> dict[str, Any]:
    _prepare_retrieval_runtime(options)
    index = _retrieval_index(options)
    previews = []
    for entry in entries:
        bundle = _retrieve_for_entry(entry, options, index=index)
        prompt_preview = _retrieval_prompt_preview(entry, bundle, options)
        trace = bundle["trace"]
        previews.append(
            {
                "input_id": entry["input_id"],
                "query_text": trace.get("query_text"),
                "query_topic": trace.get("query_topic"),
                "query_topic_confidence": trace.get("query_topic_confidence"),
                "query_topic_low_confidence": trace.get("query_topic_low_confidence"),
                "query_topic_conflict": trace.get("query_topic_conflict"),
                "retrieved_article_ids": trace.get("retrieved_article_ids"),
                "retrieved_authors": trace.get("retrieved_authors"),
                "retrieved_topics": trace.get("retrieved_topics"),
                "candidate_scores": trace.get("candidate_scores"),
                "selected_scores": trace.get("selected_scores"),
                "exclusions": trace.get("exclusions"),
                "total_retrieval_context_size": trace.get(
                    "total_retrieval_context_size"
                ),
                "prompt_preview": prompt_preview,
            }
        )
    return {
        "index_path": str(options["index_path"]),
        "records_path": str(options["records_path"]),
        "index_version": index.index_version,
        "index_record_count": index.record_count,
        "embedding_provider": index.embedding_provider,
        "embedding_model": index.embedding_model,
        "embedding_dimensions": index.embedding_dimensions,
        "previews": previews,
    }


def _retrieval_index(options: dict[str, Any]) -> Any:
    cached = options.get("_index")
    if cached is not None:
        return cached
    rebuild = bool(options.get("rebuild_index")) and not bool(options.get("reuse_index"))
    return build_or_load_index(
        index_path=Path(options["index_path"]),
        records_path=Path(options["records_path"]),
        rebuild=rebuild,
        embedding_provider=str(options["embedding_provider"]),
        embedding_model=str(options["embedding_model"]),
    )


def _prepare_retrieval_runtime(options: dict[str, Any]) -> None:
    if options.get("_index") is None:
        started = perf_counter()
        options["_index"] = _retrieval_index_without_cache(options)
        options["_index_load_seconds"] = round(perf_counter() - started, 6)
    if options.get("_embedding_provider") is None:
        index = options["_index"]
        started = perf_counter()
        options["_embedding_provider"] = make_embedding_provider(
            str(index.embedding_provider),
            str(index.embedding_model),
        )
        options["_model_load_seconds"] = round(perf_counter() - started, 6)


def _retrieval_index_without_cache(options: dict[str, Any]) -> Any:
    rebuild = bool(options.get("rebuild_index")) and not bool(options.get("reuse_index"))
    return build_or_load_index(
        index_path=Path(options["index_path"]),
        records_path=Path(options["records_path"]),
        rebuild=rebuild,
        embedding_provider=str(options["embedding_provider"]),
        embedding_model=str(options["embedding_model"]),
    )


def _retrieve_for_entry(
    entry: dict[str, Any],
    options: dict[str, Any],
    *,
    index: Any | None = None,
) -> dict[str, Any]:
    resolved_index = index or _retrieval_index(options)
    brief_record = _brief_record_from_entry(entry)
    plan_payload = _read_plan(entry)
    query = build_retrieval_query(
        brief_record=brief_record,
        article_type=str(entry["article_type"]),
        plan_payload=plan_payload,
    )
    result = retrieve_examples(
        index=resolved_index,
        query=query,
        config=options["ranking_config"],
        embedding_provider=options.get("_embedding_provider"),
        source_article_id=_optional_string(entry.get("source_article_id")),
        source_text_hash=_optional_string(entry.get("source_text_hash")),
        source_duplicate_cluster_id=_optional_string(
            entry.get("source_duplicate_cluster_id")
        ),
        source_input_id=str(entry["input_id"]),
    )
    trace = {
        **result.trace,
        "retrieval_prompt_version": options["retrieval_prompt_version"],
        "index_load_seconds": options.get("_index_load_seconds"),
        "model_load_seconds": options.get("_model_load_seconds"),
    }
    return {
        "records": result.selected_records,
        "scores": result.selected_scores,
        "trace": trace,
        "query": result.query,
    }


def _retrieval_prompt_preview(
    entry: dict[str, Any],
    bundle: dict[str, Any],
    options: dict[str, Any],
) -> dict[str, Any]:
    brief_record = _brief_record_from_entry(entry)
    metadata = _generation_prompt_metadata(
        "newsroom_v1_retrieval",
        retrieval_prompt_version=str(options["retrieval_prompt_version"]),
    )
    payload = build_newsroom_retrieval_generation_input(
        brief_record=brief_record,
        author_instruction=str(entry["author_instruction"]),
        target_language=TARGET_LANGUAGE,
        article_type=str(entry["article_type"]),
        desired_word_count=int(entry["desired_word_count"]),
        tone_override=str(entry["tone"]),
        plan_record=None,
        prompt_metadata=metadata,
        retrieved_records=bundle["records"],
        retrieval_scores=bundle["scores"],
        retrieval_trace=bundle["trace"],
    )
    prompt_path, _metadata_path = NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS[
        str(options["retrieval_prompt_version"])
    ]
    return {
        "system_prompt_path": str(prompt_path),
        "system_prompt": prompt_path.read_text(encoding="utf-8"),
        "user_payload": json.loads(payload),
    }


def _brief_record_from_entry(entry: dict[str, Any]) -> GroundedBriefRecord:
    brief_payload = _read_json(Path(str(entry["brief_path"])))
    brief = _dict_value(brief_payload.get("brief"))
    return GroundedBriefRecord(
        brief_id=str(entry["brief_id"] or brief_payload.get("brief_id")),
        source_type="benchmark",
        source_input_hash="benchmark",
        source_url=None,
        source_text_excerpt="",
        source_language=str(entry.get("source_language") or "unknown"),
        target_language=TARGET_LANGUAGE,
        model_provider=str(brief_payload.get("model_provider") or "unknown"),
        model_name=str(brief_payload.get("model_name") or "unknown"),
        status="completed",
        brief_json=StyleScribeRepository.encode_json(brief),
        warnings_json=StyleScribeRepository.encode_warnings([]),
        created_at=str(brief_payload.get("created_at") or _now()),
    )


def _run_retrieval_leakage_for_entry(
    entry: dict[str, Any],
    *,
    article: str,
    retrieval_trace: dict[str, Any],
    retrieval_options: dict[str, Any],
) -> dict[str, Any]:
    index = _retrieval_index(retrieval_options)
    record_by_id = {record.article_id: record for record in index.records}
    retrieved_records = [
        record_by_id[article_id]
        for article_id in _list_value(retrieval_trace.get("retrieved_article_ids"))
        if article_id in record_by_id
    ]
    brief = _dict_value(_read_json(Path(str(entry["brief_path"]))).get("brief"))
    return run_retrieval_leakage_diagnostic(
        grounded_brief=brief,
        generated_article=article,
        retrieved_records=retrieved_records,
    )


def _topic_metadata_for_entry(entry: dict[str, Any]) -> dict[str, object]:
    existing = entry.get("topic_metadata")
    if isinstance(existing, dict):
        return dict(existing)
    try:
        brief = _dict_value(_read_json(Path(str(entry["brief_path"]))).get("brief"))
    except BenchmarkError:
        return {}
    return topic_metadata_from_brief(brief, input_id=str(entry["input_id"]))


def _generate_draft_for_mode(
    *,
    generation_mode: str,
    entry: dict[str, Any],
    model_client: Any,
    git_commit: str | None,
    newsroom_prompt_version: str = DEFAULT_NEWSROOM_PROMPT_VERSION,
    retrieval_options: dict[str, Any] | None = None,
) -> Any:
    kwargs = {
        "author_id": entry["author_id"],
        "brief_id": entry["brief_id"],
        "author_instruction": entry["author_instruction"],
        "target_language": TARGET_LANGUAGE,
        "article_type": entry["article_type"],
        "desired_word_count": entry["desired_word_count"],
        "tone_override": entry["tone"],
        "plan_id": entry["plan_id"],
        "model_client": model_client,
    }
    if generation_mode == "legacy":
        return generate_article_draft(**kwargs)
    if generation_mode == "newsroom_v1":
        return generate_newsroom_article_draft(
            **kwargs,
            input_identifier=str(entry["input_id"]),
            git_commit=git_commit,
            newsroom_prompt_version=newsroom_prompt_version,
        )
    if generation_mode == "newsroom_v1_retrieval":
        options = retrieval_options or {}
        retrieval_bundle = _retrieve_for_entry(entry, options)
        return generate_newsroom_retrieval_article_draft(
            **kwargs,
            retrieved_records=retrieval_bundle["records"],
            retrieval_scores=retrieval_bundle["scores"],
            retrieval_trace=retrieval_bundle["trace"],
            input_identifier=str(entry["input_id"]),
            git_commit=git_commit,
            retrieval_prompt_version=str(
                options.get(
                    "retrieval_prompt_version",
                    DEFAULT_RETRIEVAL_PROMPT_VERSION,
                )
            ),
        )
    raise BenchmarkError(f"Unsupported generation mode: {generation_mode}")


def extract_canonical_article(generation_response: Any) -> CanonicalArticle:
    payload = (
        generation_response.model_dump(mode="json")
        if hasattr(generation_response, "model_dump")
        else generation_response
    )
    response_dict = _dict_value(payload)
    draft = _dict_value(response_dict.get("draft"))
    headline = _string_value(draft.get("headline")).strip()
    subheadline = _string_value(draft.get("subheadline")).strip()
    article_body, source_field = _canonical_article_body(draft)
    word_count = approximate_tamil_word_count(article_body)
    return CanonicalArticle(
        headline=headline,
        subheadline=subheadline,
        article_body=article_body,
        word_count=word_count,
        source_field=source_field,
        model_reported_word_count=_optional_int(
            draft.get("section_assembled_article_word_count")
            or draft.get("word_count")
        ),
    )


def _canonical_article_body(draft: dict[str, Any]) -> tuple[str, str | None]:
    article_body = _string_value(draft.get("article_body")).strip()
    if article_body:
        return article_body, "draft.article_body"
    article = _string_value(draft.get("article")).strip()
    if article:
        return article, "draft.article"
    paragraphs = draft.get("paragraphs")
    if isinstance(paragraphs, list):
        joined = "\n\n".join(str(item).strip() for item in paragraphs if str(item).strip())
        if joined:
            return joined, "draft.paragraphs"
    return "", None


def _validate_canonical_article(
    article: CanonicalArticle,
    *,
    provider: str,
    model: str,
) -> None:
    if not article.headline:
        raise BenchmarkError("missing_generated_headline")
    if not article.article_body.strip():
        raise BenchmarkError("empty_generated_article")
    if len(article.article_body.strip()) < 80:
        raise BenchmarkError("generated_article_below_sanity_threshold")
    if article.word_count <= 0:
        raise BenchmarkError("invalid_generated_article_word_count")
    if not provider or not model:
        raise BenchmarkError("missing_provider_or_model_metadata")


def _write_run_summary(model_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: str(row.get("input_id") or ""))
    payload = {"created_at": _now(), "rows": rows}
    _write_json(model_dir / "run_summary.json", payload)
    _write_csv(model_dir / "run_summary.csv", rows)
    return payload


def _summary_row(response: dict[str, Any], telemetry: dict[str, Any], paths: dict[str, str]) -> dict[str, Any]:
    breakdown = _dict_value(telemetry.get("cost_breakdown"))
    generation_cost = _dict_value(breakdown.get("generation"))
    evaluation_cost = _dict_value(breakdown.get("grounding_evaluation"))
    combined_cost = _dict_value(breakdown.get("combined"))
    return {
        "input_id": response.get("input_id"),
        "provider": response.get("provider"),
        "model": response.get("generation_model"),
        "generation_mode": response.get("generation_mode"),
        "prompt_version": response.get("prompt_version"),
        "newsroom_profile_version": response.get("newsroom_profile_version"),
        "retrieval_prompt_version": response.get("retrieval_prompt_version"),
        "retrieval_index_version": _dict_value(response.get("retrieval_trace")).get(
            "index_version"
        ),
        "git_commit": response.get("git_commit"),
        "status": response.get("completion_status"),
        "source_title": response.get("source_title"),
        "source_language": response.get("source_language"),
        "brief_topic": _dict_value(response.get("topic_metadata")).get(
            "original_brief_topic"
        ),
        "provisional_topic": _dict_value(response.get("topic_metadata")).get(
            "provisional_topic"
        ),
        "provisional_topic_confidence": _dict_value(
            response.get("topic_metadata")
        ).get("provisional_topic_confidence"),
        "topic_low_confidence": _dict_value(response.get("topic_metadata")).get(
            "topic_low_confidence"
        ),
        "topic_multi_category_conflict": _dict_value(
            response.get("topic_metadata")
        ).get("topic_multi_category_conflict"),
        "topic_review_flag": _dict_value(response.get("topic_metadata")).get(
            "topic_review_flag"
        ),
        "author_id": response.get("author_id"),
        "brief_id": response.get("brief_id"),
        "plan_id": response.get("plan_id"),
        "headline": response.get("generated_headline"),
        "word_count": response.get("word_count"),
        "desired_word_count": response.get("desired_word_count"),
        "target_minimum": response.get("target_minimum"),
        "target_maximum": response.get("target_maximum"),
        "word_count_variance": response.get("word_count_variance"),
        "within_target_range": response.get("within_target_range"),
        "grounding_score": response.get("grounding_score"),
        "readiness": response.get("readiness"),
        "unsupported_claim_count": len(_list_value(response.get("unsupported_claims"))),
        "claims_to_avoid_violation_count": len(_list_value(response.get("claims_to_avoid_violations"))),
        "evaluation_anomaly_count": _dict_value(
            response.get("evaluation_diagnostics")
        ).get("anomaly_count"),
        "blocker_count": len(_list_value(response.get("blockers"))),
        "warning_count": len(_list_value(response.get("warnings"))),
        "generation_runtime_seconds": telemetry.get("generation_runtime_seconds"),
        "grounding_evaluation_runtime_seconds": telemetry.get("grounding_evaluation_runtime_seconds"),
        "total_elapsed_runtime_seconds": telemetry.get("total_elapsed_runtime_seconds"),
        "prompt_tokens": telemetry.get("prompt_tokens"),
        "cached_prompt_tokens": telemetry.get("cached_prompt_tokens"),
        "completion_tokens": telemetry.get("completion_tokens"),
        "total_tokens": telemetry.get("total_tokens"),
        "attempt_count": telemetry.get("attempt_count"),
        "retry_count": telemetry.get("retry_count"),
        "retrieval_latency_seconds": telemetry.get("retrieval_latency_seconds"),
        "retrieval_index_load_seconds": telemetry.get(
            "retrieval_index_load_seconds"
        ),
        "retrieval_model_load_seconds": telemetry.get(
            "retrieval_model_load_seconds"
        ),
        "retrieved_article_ids": "|".join(
            str(item)
            for item in _list_value(
                _dict_value(response.get("retrieval_trace")).get(
                    "retrieved_article_ids"
                )
            )
        ),
        "retrieved_authors": "|".join(
            str(item)
            for item in _list_value(
                _dict_value(response.get("retrieval_trace")).get("retrieved_authors")
            )
        ),
        "retrieval_leakage_finding_count": _dict_value(
            response.get("retrieval_leakage_diagnostic")
        ).get("finding_count"),
        "retrieval_leakage_status": _dict_value(
            response.get("retrieval_leakage_diagnostic")
        ).get("status"),
        "estimated_total_cost": telemetry.get("estimated_total_cost"),
        "cost_status": telemetry.get("cost_status"),
        "generation_prompt_tokens": generation_cost.get("prompt_tokens"),
        "generation_cached_prompt_tokens": generation_cost.get("cached_prompt_tokens"),
        "generation_uncached_prompt_tokens": generation_cost.get("uncached_prompt_tokens"),
        "generation_completion_tokens": generation_cost.get("completion_tokens"),
        "generation_provider_total_tokens": generation_cost.get("provider_total_tokens"),
        "generation_reasoning_tokens": generation_cost.get("reasoning_tokens"),
        "generation_accepted_prediction_tokens": generation_cost.get("accepted_prediction_tokens"),
        "generation_rejected_prediction_tokens": generation_cost.get("rejected_prediction_tokens"),
        "generation_total_cost_usd": generation_cost.get("total_cost_usd"),
        "generation_provider_cost_ticks": generation_cost.get("provider_cost_ticks"),
        "generation_provider_reported_cost_usd": generation_cost.get("provider_reported_cost_usd"),
        "generation_provider_cost_conversion_status": generation_cost.get("provider_cost_conversion_status"),
        "generation_cost_status": generation_cost.get("cost_status"),
        "evaluation_prompt_tokens": evaluation_cost.get("prompt_tokens"),
        "evaluation_cached_prompt_tokens": evaluation_cost.get("cached_prompt_tokens"),
        "evaluation_uncached_prompt_tokens": evaluation_cost.get("uncached_prompt_tokens"),
        "evaluation_completion_tokens": evaluation_cost.get("completion_tokens"),
        "evaluation_total_cost_usd": evaluation_cost.get("total_cost_usd"),
        "evaluation_cost_status": evaluation_cost.get("cost_status"),
        "combined_total_cost_usd": combined_cost.get("total_cost_usd"),
        "combined_cost_status": combined_cost.get("cost_status"),
        "cost_incurred_before_failure_usd": telemetry.get("cost_incurred_before_failure_usd"),
        "billable_failed_call_count": telemetry.get("billable_failed_call_count"),
        "pricing_configuration_id": telemetry.get("pricing_configuration_id"),
        "pricing_effective_date": telemetry.get("pricing_effective_date"),
        "token_reconciliation_status": generation_cost.get("token_reconciliation_status"),
        "response_path": paths["response"],
        "html_path": paths["html"],
        "telemetry_path": paths["telemetry"],
        "error_type": response.get("error_type"),
        "error_message": response.get("error_message"),
    }


def _summary_row_from_response(response_path: Path, telemetry_path: Path, html_path: Path) -> dict[str, Any]:
    response = _read_json(response_path)
    telemetry = _read_json(telemetry_path)
    return _summary_row(
        response,
        telemetry,
        {
            "response": str(response_path),
            "telemetry": str(telemetry_path),
            "html": str(html_path),
        },
    )


def _load_existing_summary_rows(model_dir: Path) -> list[dict[str, Any]]:
    summary_path = model_dir / "run_summary.json"
    if summary_path.exists():
        payload = _read_json(summary_path)
        rows = payload.get("rows")
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    csv_path = model_dir / "run_summary.csv"
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def build_comparison_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifest_path = output_dir / "shared" / "manifest.json"
    manifest = load_prepared_or_pending_manifest(manifest_path)
    left_mode = _validate_generation_mode(
        getattr(args, "left_generation_mode", "legacy")
    )
    right_mode = _validate_generation_mode(
        getattr(args, "right_generation_mode", "legacy")
    )
    left_prompt_version = _validate_newsroom_prompt_version(
        getattr(
            args,
            "left_newsroom_prompt_version",
            DEFAULT_NEWSROOM_PROMPT_VERSION,
        )
    )
    left_retrieval_prompt_version = _validate_retrieval_prompt_version(
        getattr(args, "left_retrieval_prompt_version", DEFAULT_RETRIEVAL_PROMPT_VERSION)
    )
    right_prompt_version = _validate_newsroom_prompt_version(
        getattr(
            args,
            "right_newsroom_prompt_version",
            DEFAULT_NEWSROOM_PROMPT_VERSION,
        )
    )
    right_retrieval_prompt_version = _validate_retrieval_prompt_version(
        getattr(
            args,
            "right_retrieval_prompt_version",
            DEFAULT_RETRIEVAL_PROMPT_VERSION,
        )
    )
    models = [
        ComparisonModel(
            args.left_provider,
            args.left_model,
            _generation_model_dir(
                output_dir,
                args.left_provider,
                args.left_model,
                left_mode,
                newsroom_prompt_version=left_prompt_version,
                retrieval_prompt_version=left_retrieval_prompt_version,
            ),
            left_mode,
            left_prompt_version,
            left_retrieval_prompt_version,
        ),
        ComparisonModel(
            args.right_provider,
            args.right_model,
            _generation_model_dir(
                output_dir,
                args.right_provider,
                args.right_model,
                right_mode,
                newsroom_prompt_version=right_prompt_version,
                retrieval_prompt_version=right_retrieval_prompt_version,
            ),
            right_mode,
            right_prompt_version,
            right_retrieval_prompt_version,
        ),
    ]
    third_provider = getattr(args, "third_provider", None)
    third_model = getattr(args, "third_model", None)
    if bool(third_provider) != bool(third_model):
        raise BenchmarkError("--third-provider and --third-model must be provided together.")
    if third_provider and third_model:
        third_mode = _validate_generation_mode(
            getattr(args, "third_generation_mode", "legacy")
        )
        third_prompt_version = _validate_newsroom_prompt_version(
            getattr(
                args,
                "third_newsroom_prompt_version",
                DEFAULT_NEWSROOM_PROMPT_VERSION,
            )
        )
        third_retrieval_prompt_version = _validate_retrieval_prompt_version(
            getattr(
                args,
                "third_retrieval_prompt_version",
                DEFAULT_RETRIEVAL_PROMPT_VERSION,
            )
        )
        models.append(
            ComparisonModel(
                third_provider,
                third_model,
                _generation_model_dir(
                    output_dir,
                    third_provider,
                    third_model,
                    third_mode,
                    newsroom_prompt_version=third_prompt_version,
                    retrieval_prompt_version=third_retrieval_prompt_version,
                ),
                third_mode,
                third_prompt_version,
                third_retrieval_prompt_version,
            )
        )
    comparisons_dir = output_dir / str(
        getattr(args, "comparisons_dir_name", "comparisons")
    )
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    pages: list[dict[str, Any]] = []
    for entry in manifest["inputs"]:
        input_id = str(entry["input_id"])
        outputs = [
            _load_comparison_output(model, input_id, comparisons_dir)
            for model in models
        ]
        if not any(output.exists for output in outputs):
            continue
        integrity = _comparison_integrity(entry, outputs)
        page_path = comparisons_dir / f"{input_id}_comparison.html"
        page_path.write_text(
            _comparison_detail_html(entry, outputs, integrity),
            encoding="utf-8",
        )
        pages.append(
            {
                "input_id": input_id,
                "source_title": entry.get("source_title"),
                "path": page_path,
                "outputs": outputs,
                "left": outputs[0],
                "right": outputs[1],
                "integrity": integrity,
            }
        )

    index_path = comparisons_dir / "index.html"
    index_path.write_text(
        _comparison_index_html(pages, models),
        encoding="utf-8",
    )
    warning_count = sum(1 for page in pages if page["integrity"])
    missing_or_failed = sum(
        1
        for page in pages
        for side in page["outputs"]
        if not side.exists or side.status != "completed"
    )
    print(f"Generated comparison index: {index_path}")
    print(f"Generated detail pages: {len(pages)}")
    print(f"Comparison integrity warnings: {warning_count}")
    print(f"Missing or failed model outputs: {missing_or_failed}")
    _write_newsroom_prompt_comparison_artifacts(comparisons_dir, pages, models)


@dataclass
class ComparisonModel:
    provider: str
    model: str
    model_dir: Path
    generation_mode: str = "legacy"
    newsroom_prompt_version: str = DEFAULT_NEWSROOM_PROMPT_VERSION
    retrieval_prompt_version: str = DEFAULT_RETRIEVAL_PROMPT_VERSION

    @property
    def label(self) -> str:
        mode_label = "" if self.generation_mode == "legacy" else f" {self.generation_mode}"
        return f"{_provider_label(self.provider)}{mode_label}: {self.model}"


@dataclass
class ComparisonOutput:
    model: ComparisonModel
    input_id: str
    input_dir: Path
    response: dict[str, Any]
    telemetry: dict[str, Any]
    summary: dict[str, Any]
    links: dict[str, str]
    exists: bool

    @property
    def status(self) -> str:
        return str(
            self.response.get("completion_status")
            or self.telemetry.get("completion_status")
            or self.summary.get("status")
            or ("missing" if not self.exists else "Not available")
        )


def _load_comparison_output(
    model: ComparisonModel,
    input_id: str,
    comparisons_dir: Path,
) -> ComparisonOutput:
    input_dir = model.model_dir / input_id
    response_path = input_dir / "response.json"
    telemetry_path = input_dir / "telemetry.json"
    html_path = input_dir / "article.html"
    response = _read_json(response_path) if response_path.exists() else {}
    telemetry = _read_json(telemetry_path) if telemetry_path.exists() else {}
    summary = _summary_for_input(model.model_dir, input_id)
    links = {}
    for name, path in (("article", html_path), ("response", response_path), ("telemetry", telemetry_path)):
        if path.exists():
            links[name] = _relative_link(comparisons_dir, path)
    return ComparisonOutput(
        model=model,
        input_id=input_id,
        input_dir=input_dir,
        response=response,
        telemetry=telemetry,
        summary=summary,
        links=links,
        exists=response_path.exists() or telemetry_path.exists() or html_path.exists(),
    )


def _summary_for_input(model_dir: Path, input_id: str) -> dict[str, Any]:
    for row in _load_existing_summary_rows(model_dir):
        if row.get("input_id") == input_id:
            return row
    return {}


def _comparison_integrity(
    entry: dict[str, Any],
    outputs: list[ComparisonOutput],
) -> list[str]:
    mismatches: list[str] = []
    checks = [
        ("input_id", entry.get("input_id")),
        ("author_id", entry.get("author_id")),
        ("brief_id", entry.get("brief_id")),
        ("plan_id", entry.get("plan_id")),
        ("desired_word_count", entry.get("desired_word_count")),
        ("tone", entry.get("tone")),
        ("article_type", entry.get("article_type")),
    ]
    workflow_expected = {
        "grounding_evaluation": True,
        "auto_revision": False,
        "final_evaluation": False,
    }
    for field, expected in checks:
        values = [
            _comparison_field(side, field)
            for side in outputs
            if side.exists and _comparison_field(side, field) is not None
        ]
        if expected is not None:
            values.append(expected)
        if len({str(value) for value in values}) > 1:
            mismatches.append(field)
    workflows = [
        _dict_value(side.response.get("workflow_settings"))
        for side in outputs
        if side.exists and side.response.get("workflow_settings") is not None
    ]
    for workflow in workflows:
        if workflow != workflow_expected:
            mismatches.append("workflow_settings")
    return sorted(set(mismatches))


def _comparison_field(side: ComparisonOutput, field: str) -> object:
    if field in side.response:
        return side.response.get(field)
    if field in side.summary:
        return side.summary.get(field)
    return side.telemetry.get(field)


def _write_newsroom_prompt_comparison_artifacts(
    comparisons_dir: Path,
    pages: list[dict[str, Any]],
    models: list[ComparisonModel],
) -> None:
    manifest = {
        "created_at": _now(),
        "comparison_type": "prompt_only",
        "models": [
            {
                "provider": model.provider,
                "model": model.model,
                "generation_mode": model.generation_mode,
                "output_dir": str(model.model_dir),
                **_generation_prompt_metadata(
                    model.generation_mode,
                    newsroom_prompt_version=model.newsroom_prompt_version,
                ),
            }
            for model in models
        ],
        "input_ids": sorted(str(page["input_id"]) for page in pages),
    }
    per_input = [_comparison_data_row(page) for page in pages]
    aggregate = _comparison_aggregate(models, per_input)
    _write_json(comparisons_dir / "benchmark_manifest.json", manifest)
    _write_json(comparisons_dir / "per_input_comparison.json", per_input)
    _write_json(comparisons_dir / "aggregate_metrics.json", aggregate)
    _write_text_atomic(
        comparisons_dir / "sprint2_benchmark_report.md",
        _sprint2_benchmark_report(manifest, aggregate),
    )


def _comparison_data_row(page: dict[str, Any]) -> dict[str, Any]:
    outputs = page["outputs"]
    return {
        "input_id": page["input_id"],
        "source_title": page.get("source_title"),
        "integrity_warnings": page["integrity"],
        "outputs": [
            {
                "provider": side.model.provider,
                "model": side.model.model,
                "generation_mode": side.model.generation_mode,
                "status": side.status,
                "headline": side.response.get("generated_headline"),
                "word_count": side.response.get("word_count"),
                "grounding_score": side.response.get("grounding_score"),
                "unsupported_claim_count": len(
                    _list_value(side.response.get("unsupported_claims"))
                ),
                "warning_count": len(_list_value(side.response.get("warnings"))),
                "total_elapsed_runtime_seconds": side.telemetry.get(
                    "total_elapsed_runtime_seconds"
                ),
                "prompt_tokens": _dict_value(
                    _dict_value(side.telemetry.get("cost_breakdown")).get(
                        "generation"
                    )
                ).get("prompt_tokens"),
                "completion_tokens": _dict_value(
                    _dict_value(side.telemetry.get("cost_breakdown")).get(
                        "generation"
                    )
                ).get("completion_tokens"),
                "estimated_cost_usd": _dict_value(
                    _dict_value(side.telemetry.get("cost_breakdown")).get(
                        "generation"
                    )
                ).get("total_cost_usd"),
            }
            for side in outputs
        ],
    }


def _comparison_aggregate(
    models: list[ComparisonModel],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    aggregate_rows = []
    for index, model in enumerate(models):
        side_rows = [
            row["outputs"][index]
            for row in rows
            if len(row.get("outputs", [])) > index
        ]
        completed = [
            row for row in side_rows if row.get("status") == "completed"
        ]
        aggregate_rows.append(
            {
                "provider": model.provider,
                "model": model.model,
                "generation_mode": model.generation_mode,
                "completed": len(completed),
                "failed_or_missing": len(side_rows) - len(completed),
                "average_grounding_score": _average_number(
                    row.get("grounding_score") for row in completed
                ),
                "unsupported_additions": sum(
                    _optional_int(row.get("unsupported_claim_count")) or 0
                    for row in completed
                ),
                "average_word_count": _average_number(
                    row.get("word_count") for row in completed
                ),
                "average_latency_seconds": _average_number(
                    row.get("total_elapsed_runtime_seconds") for row in completed
                ),
                "prompt_tokens": sum(
                    _optional_int(row.get("prompt_tokens")) or 0
                    for row in completed
                ),
                "completion_tokens": sum(
                    _optional_int(row.get("completion_tokens")) or 0
                    for row in completed
                ),
                "estimated_cost_usd": _sum_decimal_strings(
                    row.get("estimated_cost_usd") for row in completed
                ),
            }
        )
    return {
        "created_at": _now(),
        "input_count": len(rows),
        "runs": aggregate_rows,
    }


def _sprint2_benchmark_report(
    manifest: dict[str, Any],
    aggregate: dict[str, Any],
) -> str:
    lines = [
        "# Sprint 2 Newsroom V1 Prompt Comparison",
        "",
        "This is a prompt-only comparison. Automated metrics support review; "
        "they are not editorial approval.",
        "",
        f"Inputs: {len(manifest['input_ids'])}",
        "",
        "## Runs",
        "",
    ]
    for row in aggregate["runs"]:
        lines.append(
            "- "
            f"{row['generation_mode']} {row['provider']} {row['model']}: "
            f"completed={row['completed']}; failed_or_missing="
            f"{row['failed_or_missing']}; avg_grounding="
            f"{row['average_grounding_score']}; unsupported_additions="
            f"{row['unsupported_additions']}; avg_latency_seconds="
            f"{row['average_latency_seconds']}; prompt_tokens="
            f"{row['prompt_tokens']}; completion_tokens="
            f"{row['completion_tokens']}; estimated_cost_usd="
            f"{row['estimated_cost_usd']}"
        )
    lines.extend(
        [
            "",
            "## Review Note",
            "",
            "Tamil naturalness, translation-like phrasing, lede quality, "
            "paragraph flow, attribution and repetition require human editorial "
            "review of the side-by-side HTML.",
            "",
        ]
    )
    return "\n".join(lines)


def _average_number(values: object) -> float | None:
    numeric = [
        float(value)
        for value in values
        if isinstance(value, int | float) or str(value).replace(".", "", 1).isdigit()
    ]
    if not numeric:
        return None
    return round(sum(numeric) / len(numeric), 4)


def _sum_decimal_strings(values: object) -> str | None:
    total = Decimal("0")
    seen = False
    for value in values:
        if value is None:
            continue
        try:
            total += Decimal(str(value))
            seen = True
        except Exception:
            continue
    return _money(total) if seen else None


def _comparison_detail_html(
    entry: dict[str, Any],
    outputs: list[ComparisonOutput],
    integrity: list[str],
) -> str:
    header = _comparison_header(entry, outputs, integrity)
    kpi_cards = "\n".join(_kpi_card(output, f"model-{index}") for index, output in enumerate(outputs, start=1))
    article_columns = "\n".join(_article_column(output) for output in outputs)
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>StyleScribe Multi-Model Comparison - {_html_text(entry.get("input_id"))}</title>
  {_comparison_css()}
</head>
<body>
  <main>
    {header}
    <section class="model-grid kpi-grid">
      {kpi_cards}
    </section>
    <section class="model-grid article-grid">
      {article_columns}
    </section>
    {_metric_summary(outputs, entry)}
  </main>
</body>
</html>
"""


def _comparison_header(
    entry: dict[str, Any],
    outputs: list[ComparisonOutput],
    integrity: list[str],
) -> str:
    target_min = _first_available(
        *[output.response.get("target_minimum") for output in outputs],
        *[output.summary.get("target_minimum") for output in outputs],
    )
    target_max = _first_available(
        *[output.response.get("target_maximum") for output in outputs],
        *[output.summary.get("target_maximum") for output in outputs],
    )
    warning = ""
    if integrity:
        fields = ", ".join(_html_text(field) for field in integrity)
        warning = (
            '<div class="integrity warning-red">'
            "Comparison integrity warning: the model runs did not use identical shared inputs or workflow settings."
            f"<br><strong>Mismatched fields:</strong> {fields}</div>"
        )
    return f"""
    <header class="page-header">
      <p class="eyebrow">StyleScribe Multi-Model Comparison</p>
      <h1>{_html_text(entry.get("input_id"))}: {_html_text(entry.get("source_title"))}</h1>
      <div class="shared-meta">
        {_meta_item("Source language", entry.get("source_language"))}
        {_meta_item("Author ID", entry.get("author_id"))}
        {_meta_item("Desired word count", entry.get("desired_word_count"))}
        {_meta_item("Target minimum", target_min)}
        {_meta_item("Target maximum", target_max)}
        {_meta_item("Brief ID", entry.get("brief_id"))}
        {_meta_item("Plan ID", entry.get("plan_id"))}
      </div>
      {warning}
    </header>
"""


def _kpi_card(side: ComparisonOutput, css_class: str) -> str:
    cost = _cost_sections(side)
    generation = cost["generation"]
    evaluation = cost["evaluation"]
    combined = cost["combined"]
    return f"""
      <article class="card model-card {css_class}">
        <h2>{_html_text(_provider_label(side.model.provider))}</h2>
        <p class="model-name">{_html_text(side.model.model)}</p>
        {_kpi_group("Model Identity", [
            ("Provider", side.response.get("provider") or side.model.provider, None),
            ("Model", side.response.get("generation_model") or side.model.model, None),
            ("Completion status", side.status, _status_badge(side.status)),
        ])}
        {_kpi_group("Runtime And Reliability", [
            ("Generation runtime", _seconds_display(side.telemetry.get("generation_runtime_seconds")), None),
            ("Grounding evaluation runtime", _seconds_display(side.telemetry.get("grounding_evaluation_runtime_seconds")), None),
            ("Total elapsed runtime", _seconds_display(side.telemetry.get("total_elapsed_runtime_seconds")), None),
            ("API call count", len(_list_value(side.telemetry.get("generation_call_ledger"))) or None, None),
            ("Attempt count", side.telemetry.get("attempt_count"), None),
            ("Retry count", side.telemetry.get("retry_count"), None),
            ("Invalid JSON retry count", _invalid_json_retry_count(side), None),
            ("Timeout count", side.telemetry.get("timeout_count"), None),
        ])}
        {_kpi_group("Cost", [
            ("Generation cost", _money_display(generation.get("total_cost_usd"), side), "cost-primary"),
            ("Grounding evaluation cost", _money_display(evaluation.get("total_cost_usd"), side), None),
            ("Combined cost", _money_display(combined.get("total_cost_usd"), side), None),
            ("Cost accuracy", side.telemetry.get("cost_accuracy"), None),
            ("Pricing version", side.telemetry.get("pricing_configuration_id"), None),
        ])}
        {_kpi_group("Token Usage", [
            ("Generation prompt tokens", generation.get("prompt_tokens"), None),
            ("Generation cached prompt tokens", generation.get("cached_prompt_tokens"), None),
            ("Generation completion tokens", generation.get("completion_tokens"), None),
            ("Provider-reported generation total tokens", generation.get("provider_total_tokens"), None),
            ("Evaluation prompt tokens", evaluation.get("prompt_tokens"), None),
            ("Evaluation completion tokens", evaluation.get("completion_tokens"), None),
            ("Token reconciliation status", generation.get("token_reconciliation_status"), None),
        ])}
        {_kpi_group("Output And Quality", [
            ("Generated word count", _word_count(side), None),
            ("Desired word count", side.response.get("desired_word_count") or side.summary.get("desired_word_count"), None),
            ("Target range", _target_range(side), None),
            ("Within target range", _within_target_display(side), _boolean_badge(_within_target(side))),
            ("Grounding score", _grounding_score(side), None),
            ("Editorial readiness", _readiness(side), _readiness_badge(_readiness(side))),
            ("Blocker count", len(_attention_items(side, "blockers")), _count_badge(len(_attention_items(side, "blockers")), bad=True)),
            ("Warning count", len(_attention_items(side, "warnings")), _count_badge(len(_attention_items(side, "warnings")), warn=True)),
            ("Unsupported claim count", len(_attention_items(side, "unsupported_claims")), _count_badge(len(_attention_items(side, "unsupported_claims")), bad=True)),
            ("Claims-to-avoid violation count", len(_attention_items(side, "claims_to_avoid_violations")), _count_badge(len(_attention_items(side, "claims_to_avoid_violations")), bad=True)),
            ("Overclaim count", len(_attention_items(side, "overclaims")), _count_badge(len(_attention_items(side, "overclaims")), warn=True)),
            ("Repetition indicator count", len(_attention_items(side, "repetition_indicators")), _count_badge(len(_attention_items(side, "repetition_indicators")), warn=True)),
        ])}
        {_artifact_links(side)}
      </article>
"""


def _article_column(side: ComparisonOutput) -> str:
    if side.status != "completed":
        error = _first_available(
            side.response.get("error_message"),
            side.telemetry.get("error_message"),
            side.summary.get("error_message"),
        )
        error_html = f'<p class="warning-red"><strong>Error:</strong> {_html_text(error)}</p>' if error else ""
        article_body = f'{error_html}<p class="empty-article">No completed article available</p>{_raw_diagnostic_links(side)}'
    else:
        article_body = _render_article_body(side)
    return f"""
      <article class="card article-card">
        <h2>{_html_text(_provider_label(side.model.provider))}</h2>
        <p class="model-name">{_html_text(side.model.model)}</p>
        <p class="model-name">Mode: {_html_text(side.model.generation_mode)} | Prompt: {_html_text(_comparison_field(side, "prompt_version") or "Not available")}</p>
        <p class="model-name">Profile: {_html_text(_comparison_field(side, "newsroom_profile_version") or "Not applicable")}</p>
        <h3>{_html_text(side.response.get("generated_headline"))}</h3>
        <p class="subheadline">{_html_text(side.response.get("generated_subheadline"))}</p>
        <p>{_badge(_grounding_score(side), "neutral")} {_readiness_badge(_readiness(side))}</p>
        {_attention_section(side)}
        <section class="article-body">{article_body}</section>
      </article>
"""


def _raw_diagnostic_links(side: ComparisonOutput) -> str:
    raw_dir = side.input_dir / "raw"
    if not raw_dir.exists():
        return ""
    links = []
    for path in sorted(raw_dir.iterdir()):
        if path.is_file():
            links.append(f'<a href="{escape(Path("..", side.model.model_dir.name, side.input_id, "raw", path.name).as_posix())}">{escape(path.name)}</a>')
    if not links:
        return ""
    return '<p class="links"><strong>Raw diagnostics:</strong> ' + " ".join(links) + "</p>"


def _attention_section(side: ComparisonOutput) -> str:
    blockers = (
        _attention_items(side, "unsupported_claims")
        + _attention_items(side, "claims_to_avoid_violations")
        + _attention_items(side, "invented_facts")
        + _attention_items(side, "contradictions")
        + _attention_items(side, "blockers")
    )
    warnings = (
        _attention_items(side, "overclaims")
        + _attention_items(side, "overclaim_phrases")
        + _attention_items(side, "repetition_indicators")
        + _attention_items(side, "missing_key_facts")
        + _attention_items(side, "warnings")
    )
    generation = _cost_sections(side)["generation"]
    info = [
        {"type": "readiness", "text": _readiness(side)},
        {"type": "grounding_summary", "text": _grounding_summary(side)},
        {"type": "token_reconciliation", "text": generation.get("token_reconciliation_status")},
        {"type": "cost_accuracy", "text": side.telemetry.get("cost_accuracy")},
    ]
    return f"""
        <section class="attention">
          <h4>Blockers</h4>
          {_attention_list(blockers, "blocker")}
          <h4>Warnings</h4>
          {_attention_list(warnings, "warning")}
          <h4>Information</h4>
          {_attention_list(info, "info")}
        </section>
"""


def _attention_items(side: ComparisonOutput, key: str) -> list[Any]:
    direct = _list_value(side.response.get(key))
    if direct:
        return direct
    evaluation = _dict_value(side.response.get("grounding_evaluation_result"))
    return _list_value(evaluation.get(key))


def _attention_list(items: list[Any], kind: str) -> str:
    cleaned = [item for item in items if _attention_text(item)]
    if not cleaned:
        return '<p class="none">None</p>'
    rows = "".join(f'<li class="{kind}">{_attention_item_html(item)}</li>' for item in cleaned)
    return f"<ul class=\"attention-list\">{rows}</ul>"


def _attention_item_html(item: Any) -> str:
    if isinstance(item, dict):
        fields = []
        for label, key in (
            ("Type", "type"),
            ("Claim", "claim"),
            ("Claim", "claim_text"),
            ("Text", "text"),
            ("Phrase", "phrase"),
            ("Reason", "reason"),
            ("Editor action", "editor_action"),
            ("Rewrite guidance", "rewrite_guidance"),
        ):
            value = item.get(key)
            if value not in (None, ""):
                fields.append(f"<strong>{label}:</strong> {_html_text(value)}")
        return "<br>".join(fields) if fields else _html_text(item)
    return _html_text(item)


def _attention_text(item: Any) -> str:
    if isinstance(item, dict):
        for key in ("claim", "claim_text", "text", "phrase", "fact", "summary", "reason"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    return str(item).strip()


def _render_article_body(side: ComparisonOutput) -> str:
    article = str(side.response.get("generated_tamil_article") or "").strip()
    if not article:
        return '<p class="empty-article">No completed article available</p>'
    escaped = escape(article)
    for item in _highlight_items(side, "blocker"):
        escaped = _highlight_exact(escaped, _attention_text(item), "hl-blocker")
    for item in _highlight_items(side, "warning"):
        escaped = _highlight_exact(escaped, _attention_text(item), "hl-warning")
    paragraphs = [part.strip() for part in escaped.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [escaped]
    return "".join(f"<p>{paragraph.replace(chr(10), '<br>')}</p>" for paragraph in paragraphs)


def _highlight_items(side: ComparisonOutput, severity: str) -> list[Any]:
    if severity == "blocker":
        return (
            _attention_items(side, "unsupported_claims")
            + _attention_items(side, "claims_to_avoid_violations")
            + _attention_items(side, "blockers")
        )
    return (
        _attention_items(side, "overclaims")
        + _attention_items(side, "overclaim_phrases")
        + _attention_items(side, "warnings")
    )


def _highlight_exact(escaped_article: str, raw_text: str, css_class: str) -> str:
    if not raw_text:
        return escaped_article
    escaped_text = escape(raw_text)
    if escaped_text not in escaped_article:
        return escaped_article
    return escaped_article.replace(escaped_text, f'<mark class="{css_class}">{escaped_text}</mark>')


def _metric_summary(outputs: list[ComparisonOutput], entry: dict[str, Any]) -> str:
    lines = [
        _compare_lower_many("Runtime", outputs, "generation_runtime_seconds", "seconds"),
        _compare_lower_many("Generation cost", outputs, "generation_total_cost_usd", "cost"),
        _compare_lower_many("Combined cost", outputs, "combined_total_cost_usd", "cost"),
        _compare_length_many(outputs, entry),
        _compare_higher_many("Grounding", outputs, "grounding_score"),
        _compare_count_many("Blockers", outputs, "blockers"),
        _compare_count_many("Warnings", outputs, "warnings"),
        _compare_count_many("Unsupported claims", outputs, "unsupported_claims"),
    ]
    items = "".join(f"<li>{_html_text(line)}</li>" for line in lines)
    return f'<section class="card metric-summary"><h2>Neutral Metric Comparison Summary</h2><ul>{items}</ul></section>'


def _comparison_index_html(
    pages: list[dict[str, Any]],
    models: list[ComparisonModel],
) -> str:
    outputs_by_model = [
        [page["outputs"][index] for page in pages]
        for index, _model in enumerate(models)
    ]
    rows = "".join(_index_row(page) for page in pages)
    aggregate_cards = "\n".join(
        _aggregate_card(model, outputs)
        for model, outputs in zip(models, outputs_by_model, strict=True)
    )
    model_title = " vs ".join(_html_text(model.label) for model in models)
    model_headings = "".join(_index_model_headers(model) for model in models)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>StyleScribe Multi-Model Comparison</title>
  {_comparison_css()}
</head>
<body>
  <main>
    <header class="page-header">
      <p class="eyebrow">StyleScribe Multi-Model Comparison</p>
      <h1>{model_title}</h1>
    </header>
    <section class="model-grid">
      {aggregate_cards}
    </section>
    <section class="card">
      <h2>Inputs</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Input ID</th><th>Source title</th>
              {model_headings}
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>
  </main>
</body>
</html>
"""


def _index_row(page: dict[str, Any]) -> str:
    outputs = page["outputs"]
    href = _html_text(Path(page["path"]).name)
    cells = "".join(_index_model_cells(output) for output in outputs)
    return f"""
            <tr>
              <td>{_html_text(page["input_id"])}</td>
              <td>{_html_text(page.get("source_title"))}</td>
              {cells}
              <td><a href="{href}">Open</a></td>
            </tr>
"""


def _index_model_headers(model: ComparisonModel) -> str:
    label = _html_text(_provider_label(model.provider))
    return (
        f"<th>{label} status</th>"
        f"<th>{label} runtime</th>"
        f"<th>{label} generation cost</th>"
        f"<th>{label} word count</th>"
        f"<th>{label} grounding score</th>"
        f"<th>{label} readiness</th>"
    )


def _index_model_cells(output: ComparisonOutput) -> str:
    generation_cost = _cost_sections(output)["generation"].get("total_cost_usd")
    return (
        f"<td>{_status_badge(output.status)}</td>"
        f"<td>{_html_text(_seconds_display(output.telemetry.get('generation_runtime_seconds')))}</td>"
        f"<td>{_html_text(_money_display(generation_cost, output))}</td>"
        f"<td>{_html_text(_word_count(output))}</td>"
        f"<td>{_html_text(_grounding_score(output))}</td>"
        f"<td>{_readiness_badge(_readiness(output))}</td>"
    )


def _aggregate_card(model: ComparisonModel, outputs: list[ComparisonOutput]) -> str:
    completed = [output for output in outputs if output.status == "completed"]
    failed = [output for output in outputs if output.exists and output.status != "completed"]
    runtime_values = [_number(_cost_or_summary(output, "generation_runtime_seconds")) for output in completed]
    generation_costs = [_number(_cost_sections(output)["generation"].get("total_cost_usd")) for output in completed]
    combined_costs = [_number(_cost_sections(output)["combined"].get("total_cost_usd")) for output in completed]
    word_counts = [_number(_word_count(output)) for output in completed]
    grounding_scores = [_number(_grounding_score(output)) for output in completed]
    within_values = [_within_target(output) for output in completed]
    within_count = sum(1 for value in within_values if value is True)
    readiness_counts: dict[str, int] = {}
    for output in completed:
        readiness = _readiness(output)
        readiness_counts[readiness] = readiness_counts.get(readiness, 0) + 1
    return f"""
      <article class="card aggregate-card">
        <h2>{_html_text(model.label)}</h2>
        {_kpi_group("Aggregate Metrics", [
            ("Completed input count", len(completed), None),
            ("Failed input count", len(failed), None),
            ("Completion rate", _percent(len(completed), len(outputs)), None),
            ("Average generation runtime", _seconds_display(_average(runtime_values)), None),
            ("Median generation runtime", _seconds_display(_median(runtime_values)), None),
            ("Total generation cost", _money_display(_sum_numeric(generation_costs), completed[0] if completed else None), None),
            ("Average generation cost", _money_display(_average(generation_costs), completed[0] if completed else None), None),
            ("Median generation cost", _money_display(_median(generation_costs), completed[0] if completed else None), None),
            ("Average combined cost", _money_display(_average(combined_costs), completed[0] if completed else None), None),
            ("Average word count", _round_display(_average(word_counts)), None),
            ("Within-target percentage", _percent(within_count, len([value for value in within_values if value is not None])), None),
            ("Average grounding score", _round_display(_average(grounding_scores)), None),
            ("Readiness distribution", _readiness_distribution(readiness_counts), None),
            ("Total blockers", sum(len(_attention_items(output, "blockers")) for output in completed), None),
            ("Total warnings", sum(len(_attention_items(output, "warnings")) for output in completed), None),
            ("Total unsupported claims", sum(len(_attention_items(output, "unsupported_claims")) for output in completed), None),
            ("Average API calls per input", _round_display(_average([len(_list_value(output.telemetry.get("generation_call_ledger"))) for output in completed])), None),
            ("Average retries per input", _round_display(_average([_number(output.telemetry.get("retry_count")) for output in completed])), None),
        ])}
      </article>
"""


def _comparison_css() -> str:
    return """<style>
    :root { --blue: #2563eb; --border: #d9e2ec; --bg: #f4f7fb; --text: #18202a; --muted: #5f6b7a; --green: #15803d; --amber: #b45309; --red: #b91c1c; --grey: #64748b; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font-family: "Nirmala UI", "Latha", "Vijaya", "Noto Sans Tamil", Arial, sans-serif; line-height: 1.55; }
    main { max-width: 1480px; margin: 0 auto; padding: 28px; }
    .page-header, .card { background: #fff; border: 1px solid var(--border); border-radius: 8px; padding: 22px; margin-bottom: 18px; }
    .eyebrow { color: var(--blue); font-weight: 700; margin: 0 0 6px; }
    h1, h2, h3, h4 { margin: 0 0 10px; letter-spacing: 0; }
    h1 { font-size: 28px; }
    h2 { font-size: 20px; }
    h3 { font-size: 18px; }
    h4 { font-size: 14px; margin-top: 16px; color: #334155; }
    .two-col, .model-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }
    .shared-meta, .kpi-group { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px 16px; }
    .meta-item, .kpi { border-top: 1px solid #edf2f7; padding-top: 7px; min-width: 0; overflow-wrap: anywhere; }
    .label { color: var(--muted); font-size: 12px; text-transform: uppercase; display: block; }
    .value { font-weight: 600; }
    .model-name { color: var(--muted); margin-top: -4px; overflow-wrap: anywhere; }
    .cost-primary .value { color: var(--blue); font-size: 18px; }
    .badge { display: inline-block; border-radius: 999px; padding: 2px 8px; color: #fff; font-size: 12px; font-weight: 700; }
    .badge.green { background: var(--green); }
    .badge.amber { background: var(--amber); }
    .badge.red { background: var(--red); }
    .badge.neutral { background: var(--grey); }
    .warning-red { border: 1px solid #fecaca; background: #fff1f2; color: var(--red); border-radius: 8px; padding: 12px; margin-top: 16px; }
    .article-body { font-size: 18px; overflow-wrap: anywhere; }
    .article-body p { margin: 0 0 14px; }
    .subheadline { color: var(--muted); font-weight: 600; }
    .attention-list { padding-left: 20px; margin-top: 6px; }
    .attention-list li { margin-bottom: 8px; }
    .blocker { color: var(--red); }
    .warning { color: var(--amber); }
    .info { color: #334155; }
    .none, .empty-article { color: var(--muted); font-style: italic; }
    .hl-blocker { background: #fecaca; color: #7f1d1d; }
    .hl-warning { background: #fde68a; color: #78350f; }
    .links a { margin-right: 12px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    th, td { border-bottom: 1px solid var(--border); padding: 8px; text-align: left; vertical-align: top; }
    th { background: #eef4ff; font-size: 12px; text-transform: uppercase; }
    @media (max-width: 900px) { main { padding: 14px; } .two-col, .model-grid, .shared-meta, .kpi-group { grid-template-columns: 1fr; } }
  </style>"""


def _meta_item(label: str, value: object) -> str:
    return f'<div class="meta-item"><span class="label">{escape(label)}</span><span class="value">{_html_text(value)}</span></div>'


def _kpi_group(title: str, rows: list[tuple[str, object, str | None]]) -> str:
    items = []
    for label, value, css_class in rows:
        rendered = value if isinstance(value, str) and value.startswith("<span class=\"badge") else _html_text(value)
        item_class = f" {css_class}" if css_class else ""
        items.append(
            f'<div class="kpi{item_class}"><span class="label">{escape(label)}</span>'
            f'<span class="value">{rendered}</span></div>'
        )
    return f'<section><h3>{escape(title)}</h3><div class="kpi-group">{"".join(items)}</div></section>'


def _artifact_links(side: ComparisonOutput) -> str:
    if not side.links:
        return '<p class="links none">No saved artifacts available</p>'
    links = " ".join(
        f'<a href="{escape(href)}">{escape(label)}.json</a>' if label != "article" else f'<a href="{escape(href)}">article.html</a>'
        for label, href in side.links.items()
    )
    return f'<p class="links">{links}</p>'


def _badge(text: object, kind: str) -> str:
    return f'<span class="badge {kind}">{_html_text(text)}</span>'


def _status_badge(status: object) -> str:
    normalized = str(status or "").lower()
    if normalized == "completed":
        return _badge("Completed", "green")
    if normalized in {"failed", "missing"}:
        return _badge(normalized.title(), "red")
    if not normalized or normalized == "not available":
        return _badge("Not available", "neutral")
    return _badge(status, "amber")


def _readiness_badge(readiness: object) -> str:
    normalized = str(readiness or "").lower()
    if normalized in {"safe_to_review", "safe to review", "completed"}:
        return _badge(_readiness_label(readiness), "green")
    if normalized in {"revision_required", "needs_revision", "warning"}:
        return _badge(_readiness_label(readiness), "amber")
    if normalized in {"failed", "blocker"}:
        return _badge(_readiness_label(readiness), "red")
    return _badge(_readiness_label(readiness), "neutral")


def _boolean_badge(value: bool | None) -> str:
    if value is True:
        return _badge("Yes", "green")
    if value is False:
        return _badge("No", "red")
    return _badge("Not available", "neutral")


def _count_badge(value: int, *, bad: bool = False, warn: bool = False) -> str:
    if value == 0:
        return _badge("0", "green")
    if bad:
        return _badge(value, "red")
    if warn:
        return _badge(value, "amber")
    return _badge(value, "neutral")


def _cost_sections(side: ComparisonOutput) -> dict[str, dict[str, Any]]:
    breakdown = _dict_value(side.telemetry.get("cost_breakdown"))
    return {
        "generation": _dict_value(breakdown.get("generation")),
        "evaluation": _dict_value(breakdown.get("grounding_evaluation")),
        "combined": _dict_value(breakdown.get("combined")),
    }


def _invalid_json_retry_count(side: ComparisonOutput) -> int | None:
    ledger = _list_value(side.telemetry.get("generation_call_ledger"))
    count = sum(
        1
        for item in ledger
        if isinstance(item, dict)
        and str(item.get("failure_type") or "").lower()
        in {"invalid_json", "schema_validation_failed", "multiple_json_objects", "truncated_response", "empty_response"}
        and int(item.get("attempt") or 1) > 1
    )
    return count if count else _optional_int(side.telemetry.get("invalid_json_retry_count"))


def _word_count(side: ComparisonOutput) -> object:
    return _first_available(side.response.get("word_count"), side.summary.get("word_count"))


def _grounding_score(side: ComparisonOutput) -> object:
    return _first_available(side.response.get("grounding_score"), side.summary.get("grounding_score"))


def _readiness(side: ComparisonOutput) -> str:
    return _readiness_label(_first_available(side.response.get("readiness"), side.summary.get("readiness")))


def _readiness_label(value: object) -> str:
    if value in (None, ""):
        return "Not evaluated"
    return str(value).replace("_", " ")


def _target_range(side: ComparisonOutput) -> str:
    low = _first_available(side.response.get("target_minimum"), side.summary.get("target_minimum"))
    high = _first_available(side.response.get("target_maximum"), side.summary.get("target_maximum"))
    if low is None or high is None:
        return "Not available"
    return f"{low}-{high}"


def _within_target(side: ComparisonOutput) -> bool | None:
    value = _first_available(side.response.get("within_target_range"), side.summary.get("within_target_range"))
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def _within_target_display(side: ComparisonOutput) -> str:
    value = _within_target(side)
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return "Not available"


def _grounding_summary(side: ComparisonOutput) -> object:
    evaluation = _dict_value(side.response.get("grounding_evaluation_result"))
    return _first_available(
        evaluation.get("summary"),
        evaluation.get("grounding_summary"),
        side.response.get("grounding_summary"),
    )


def _seconds_display(value: object) -> str:
    number = _number(value)
    if number is None:
        return "Not available"
    return f"{number:.1f}s"


def _money_display(value: object, side: ComparisonOutput | None) -> str:
    if value is None:
        return "Not available"
    currency = "USD"
    if side is not None:
        currency = str(side.telemetry.get("cost_currency") or "USD")
    return f"{currency} {_money_text(Decimal(str(value)))}"


def _html_text(value: object) -> str:
    if value is None or value == "":
        return "Not available"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return escape(str(value))


def _provider_label(provider: object) -> str:
    labels = {
        "openai": "OpenAI",
        "gemini": "Gemini",
        "grok": "Grok",
    }
    key = str(provider or "").lower()
    return labels.get(key, str(provider or "Not available"))


def _first_available(*values: object) -> object:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cost_or_summary(side: ComparisonOutput, field: str) -> object:
    if field in side.telemetry:
        return side.telemetry.get(field)
    return side.summary.get(field)


def _compare_lower(label: str, left: ComparisonOutput, right: ComparisonOutput, field: str, unit: str) -> str:
    left_value = _metric_value(left, field)
    right_value = _metric_value(right, field)
    if left_value is None or right_value is None:
        return f"{label}: comparison unavailable."
    if left_value == right_value:
        return f"{label}: both models were tied at {_metric_display(left_value, unit)}."
    faster = left if left_value < right_value else right
    slower_value = right_value if left_value < right_value else left_value
    diff = abs(slower_value - min(left_value, right_value))
    return f"{label}: {_provider_label(faster.model.provider)} was lower by {_metric_display(diff, unit)}."


def _compare_lower_many(
    label: str,
    outputs: list[ComparisonOutput],
    field: str,
    unit: str,
) -> str:
    values = [
        (output, _metric_value(output, field))
        for output in outputs
        if _metric_value(output, field) is not None
    ]
    if len(values) < 2:
        return f"{label}: comparison unavailable."
    numbers = [value for _output, value in values if value is not None]
    if len(set(numbers)) == 1:
        return f"{label}: all available models were tied at {_metric_display(numbers[0], unit)}."
    winner, winner_value = min(values, key=lambda item: item[1] if item[1] is not None else float("inf"))
    comparison = ", ".join(
        f"{_provider_label(output.model.provider)} {_metric_display(value, unit)}"
        for output, value in values
        if value is not None
    )
    return (
        f"{label}: {_provider_label(winner.model.provider)} was lowest "
        f"({_metric_display(winner_value, unit)}). {comparison}."
    )


def _compare_higher(label: str, left: ComparisonOutput, right: ComparisonOutput, field: str) -> str:
    left_value = _metric_value(left, field)
    right_value = _metric_value(right, field)
    if left_value is None or right_value is None:
        return f"{label}: comparison unavailable."
    if left_value == right_value:
        return f"{label}: both models scored {_round_display(left_value)}."
    winner = left if left_value > right_value else right
    return f"{label}: {_provider_label(winner.model.provider)} was higher ({_round_display(max(left_value, right_value))} vs {_round_display(min(left_value, right_value))})."


def _compare_higher_many(
    label: str,
    outputs: list[ComparisonOutput],
    field: str,
) -> str:
    values = [
        (output, _metric_value(output, field))
        for output in outputs
        if _metric_value(output, field) is not None
    ]
    if len(values) < 2:
        return f"{label}: comparison unavailable."
    numbers = [value for _output, value in values if value is not None]
    if len(set(numbers)) == 1:
        return f"{label}: all available models scored {_round_display(numbers[0])}."
    winner, winner_value = max(values, key=lambda item: item[1] if item[1] is not None else float("-inf"))
    comparison = ", ".join(
        f"{_provider_label(output.model.provider)} {_round_display(value)}"
        for output, value in values
        if value is not None
    )
    return (
        f"{label}: {_provider_label(winner.model.provider)} was highest "
        f"({_round_display(winner_value)}). {comparison}."
    )


def _compare_count(label: str, left: ComparisonOutput, right: ComparisonOutput, key: str) -> str:
    left_count = len(_attention_items(left, key))
    right_count = len(_attention_items(right, key))
    if left_count == right_count:
        return f"{label}: both models had {left_count}."
    winner = left if left_count < right_count else right
    return f"{label}: {_provider_label(winner.model.provider)} had fewer ({min(left_count, right_count)} vs {max(left_count, right_count)})."


def _compare_count_many(
    label: str,
    outputs: list[ComparisonOutput],
    key: str,
) -> str:
    values = [(output, len(_attention_items(output, key))) for output in outputs if output.exists]
    if len(values) < 2:
        return f"{label}: comparison unavailable."
    counts = [value for _output, value in values]
    if len(set(counts)) == 1:
        return f"{label}: all available models had {counts[0]}."
    winner, winner_count = min(values, key=lambda item: item[1])
    comparison = ", ".join(
        f"{_provider_label(output.model.provider)} {count}"
        for output, count in values
    )
    return (
        f"{label}: {_provider_label(winner.model.provider)} had the fewest "
        f"({winner_count}). {comparison}."
    )


def _compare_length(left: ComparisonOutput, right: ComparisonOutput, entry: dict[str, Any]) -> str:
    desired = _number(entry.get("desired_word_count"))
    left_words = _number(_word_count(left))
    right_words = _number(_word_count(right))
    if desired is None or left_words is None or right_words is None:
        return "Length adherence: comparison unavailable."
    left_delta = abs(left_words - desired)
    right_delta = abs(right_words - desired)
    if left_delta == right_delta:
        return "Length adherence: both outputs were equally close to the target word count."
    winner = left if left_delta < right_delta else right
    return f"Length adherence: {_provider_label(winner.model.provider)} was closer to the target word count."


def _compare_length_many(outputs: list[ComparisonOutput], entry: dict[str, Any]) -> str:
    desired = _number(entry.get("desired_word_count"))
    if desired is None:
        return "Length adherence: comparison unavailable."
    values = []
    for output in outputs:
        word_count = _number(_word_count(output))
        if word_count is not None:
            values.append((output, abs(word_count - desired)))
    if len(values) < 2:
        return "Length adherence: comparison unavailable."
    deltas = [delta for _output, delta in values]
    if len(set(deltas)) == 1:
        return "Length adherence: all available outputs were equally close to the target word count."
    winner, delta = min(values, key=lambda item: item[1])
    comparison = ", ".join(
        f"{_provider_label(output.model.provider)} delta {_round_display(item_delta)}"
        for output, item_delta in values
    )
    return (
        f"Length adherence: {_provider_label(winner.model.provider)} was closest "
        f"to the target word count (delta {_round_display(delta)}). {comparison}."
    )


def _metric_value(side: ComparisonOutput, field: str) -> float | None:
    if field == "generation_total_cost_usd":
        return _number(_cost_sections(side)["generation"].get("total_cost_usd"))
    if field == "combined_total_cost_usd":
        return _number(_cost_sections(side)["combined"].get("total_cost_usd"))
    if field == "grounding_score":
        return _number(_grounding_score(side))
    return _number(_cost_or_summary(side, field))


def _metric_display(value: float, unit: str) -> str:
    if unit == "cost":
        return f"USD {_money_text(Decimal(str(value)))}"
    if unit == "seconds":
        return f"{value:.1f} seconds"
    return _round_display(value)


def _average(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    if not numbers:
        return None
    return sum(numbers) / len(numbers)


def _median(values: list[float | None]) -> float | None:
    numbers = sorted(value for value in values if value is not None)
    if not numbers:
        return None
    middle = len(numbers) // 2
    if len(numbers) % 2:
        return numbers[middle]
    return (numbers[middle - 1] + numbers[middle]) / 2


def _sum_numeric(values: list[float | None]) -> float | None:
    numbers = [value for value in values if value is not None]
    return sum(numbers) if numbers else None


def _percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "Not available"
    return f"{(numerator / denominator) * 100:.1f}%"


def _round_display(value: object) -> str:
    number = _number(value)
    if number is None:
        return "Not available"
    if number.is_integer():
        return str(int(number))
    return f"{number:.2f}"


def _readiness_distribution(counts: dict[str, int]) -> str:
    if not counts:
        return "Not available"
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def _relative_link(from_dir: Path, target: Path) -> str:
    try:
        return target.relative_to(from_dir).as_posix()
    except ValueError:
        try:
            return Path("..", target.relative_to(from_dir.parent).as_posix()).as_posix()
        except ValueError:
            return target.as_posix()


def _load_existing_prepared_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return manifest
    existing = _read_json(path)
    existing_by_id = {
        str(entry.get("input_id")): entry
        for entry in existing.get("inputs", [])
        if isinstance(entry, dict)
    }
    merged = []
    for entry in manifest["inputs"]:
        merged.append({**entry, **existing_by_id.get(entry["input_id"], {})})
    return {"inputs": merged}


def _shared_artifacts_complete(entry: dict[str, Any], output_dir: Path) -> bool:
    return (
        entry.get("shared_artifacts_status") == "completed"
        and bool(entry.get("brief_id"))
        and bool(entry.get("plan_id"))
        and bool(entry.get("brief_path"))
        and bool(entry.get("plan_path"))
        and Path(str(entry["brief_path"])).exists()
        and Path(str(entry["plan_path"])).exists()
        and (output_dir / "shared" / str(entry["input_id"]) / "source.json").exists()
    )


def _validate_shared_artifact_paths(entry: dict[str, Any]) -> None:
    if entry.get("shared_artifacts_status") != "completed":
        raise BenchmarkError(f"Shared artifacts are not complete for {entry.get('input_id')}.")
    for key in ("brief_path", "plan_path"):
        if not entry.get(key) or not Path(str(entry[key])).exists():
            raise BenchmarkError(f"{key} is missing for {entry.get('input_id')}.")


def _model_output_paths(model_dir: Path, input_id: str) -> dict[str, str]:
    input_dir = model_dir / input_id
    return {
        "response": str(input_dir / "response.json"),
        "html": str(input_dir / "article.html"),
        "telemetry": str(input_dir / "telemetry.json"),
    }


def _article_html(response: dict[str, Any], telemetry: dict[str, Any]) -> str:
    blockers = "".join(f"<li>{escape(str(item))}</li>" for item in _list_value(response.get("blockers")))
    warnings = "".join(f"<li>{escape(str(item))}</li>" for item in _list_value(response.get("warnings")))
    unsupported = "".join(f"<li>{escape(str(item))}</li>" for item in _list_value(response.get("unsupported_claims")))
    article_text = str(response.get("generated_tamil_article") or "").strip()
    article = escape(article_text or "Not available").replace("\n", "<br>")
    readiness = response.get("readiness") or "Not evaluated"
    word_count = _display_value(response.get("word_count"))
    grounding_score = _display_value(response.get("grounding_score"))
    runtime = _display_value(telemetry.get("total_elapsed_runtime_seconds"))
    cost_breakdown = _dict_value(telemetry.get("cost_breakdown"))
    generation_cost = _dict_value(cost_breakdown.get("generation"))
    evaluation_cost = _dict_value(cost_breakdown.get("grounding_evaluation"))
    combined_cost = _dict_value(cost_breakdown.get("combined"))
    generation_ledger = _list_value(telemetry.get("generation_call_ledger"))
    generation_thinking_tokens = _sum_optional_int(
        row.get("thinking_tokens")
        for row in generation_ledger
        if isinstance(row, dict)
    )
    generation_reasoning_tokens = _sum_optional_int(
        row.get("reasoning_tokens")
        for row in generation_ledger
        if isinstance(row, dict)
    )
    currency = _display_value(telemetry.get("cost_currency") or generation_cost.get("currency") or evaluation_cost.get("currency"))
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>{escape(str(response.get("input_id")))} - {escape(str(response.get("provider")))} {escape(str(response.get("generation_model")))}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; line-height: 1.55; color: #1f2933; }}
    .meta, .metrics {{ display: grid; grid-template-columns: repeat(2, minmax(220px, 1fr)); gap: 8px 20px; margin-bottom: 24px; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .article {{ border-top: 1px solid #d8dee4; padding-top: 20px; font-size: 18px; }}
    code {{ background: #f6f8fa; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>{escape(str(response.get("generated_headline") or ""))}</h1>
  <p>{escape(str(response.get("generated_subheadline") or ""))}</p>
  <section class="meta">
    <div><strong>Input:</strong> {escape(str(response.get("input_id")))}</div>
    <div><strong>Source:</strong> {escape(str(response.get("source_title")))}</div>
    <div><strong>Provider:</strong> {escape(str(response.get("provider")))}</div>
    <div><strong>Model:</strong> {escape(str(response.get("generation_model")))}</div>
    <div><strong>Generation mode:</strong> {escape(str(response.get("generation_mode") or "legacy"))}</div>
    <div><strong>Prompt version:</strong> {escape(str(response.get("prompt_version") or "Not available"))}</div>
    <div><strong>Newsroom profile:</strong> {escape(str(response.get("newsroom_profile_version") or "Not applicable"))}</div>
    <div><strong>Git commit:</strong> {escape(str(response.get("git_commit") or "Not available"))}</div>
    <div><strong>Author:</strong> {escape(str(response.get("author_id")))}</div>
    <div><strong>Brief ID:</strong> <code>{escape(str(response.get("brief_id")))}</code></div>
    <div><strong>Plan ID:</strong> <code>{escape(str(response.get("plan_id")))}</code></div>
    <div><strong>Status:</strong> {escape(str(response.get("completion_status") or "Not available"))}</div>
  </section>
  <section class="metrics">
    <div><strong>Word count:</strong> {word_count}</div>
    <div><strong>Grounding score:</strong> {grounding_score}</div>
    <div><strong>Readiness:</strong> {escape(str(readiness))}</div>
    <div><strong>Runtime:</strong> {runtime}</div>
  </section>
  <h2>Generation Usage</h2>
  <section class="metrics">
    <div><strong>Generation provider:</strong> {escape(str(generation_cost.get("provider") or response.get("provider") or "Not available"))}</div>
    <div><strong>Generation model:</strong> {escape(str(generation_cost.get("model") or response.get("generation_model") or "Not available"))}</div>
    <div><strong>Generation prompt tokens:</strong> {_display_value(generation_cost.get("prompt_tokens"))}</div>
    <div><strong>Generation cached prompt tokens:</strong> {_display_value(generation_cost.get("cached_prompt_tokens"))}</div>
    <div><strong>Generation completion/output tokens:</strong> {_display_value(generation_cost.get("completion_tokens"))}</div>
    <div><strong>Generation thinking tokens:</strong> {_display_value(generation_thinking_tokens)}</div>
    <div><strong>Generation reasoning tokens:</strong> {_display_value(generation_cost.get("reasoning_tokens") or generation_reasoning_tokens)}</div>
    <div><strong>Provider-reported generation total tokens:</strong> {_display_value(generation_cost.get("provider_total_tokens"))}</div>
    <div><strong>Generation token-reconciliation status:</strong> {escape(str(generation_cost.get("token_reconciliation_status") or "Not available"))}</div>
    <div><strong>Generation cost:</strong> {_cost_display(generation_cost, currency)}</div>
    <div><strong>Provider cost ticks:</strong> {_display_value(generation_cost.get("provider_cost_ticks"))}</div>
    <div><strong>Provider-reported cost:</strong> {_provider_cost_display(generation_cost, currency)}</div>
    <div><strong>Provider cost conversion status:</strong> {escape(str(generation_cost.get("provider_cost_conversion_status") or "Not available"))}</div>
    <div><strong>Reason:</strong> {escape(_cost_reason(generation_cost))}</div>
  </section>
  <h2>Grounding Evaluation Usage</h2>
  <section class="metrics">
    <div><strong>Evaluation provider:</strong> {escape(str(evaluation_cost.get("provider") or "Not available"))}</div>
    <div><strong>Evaluation model:</strong> {escape(str(evaluation_cost.get("model") or "Not available"))}</div>
    <div><strong>Evaluation prompt tokens:</strong> {_display_value(evaluation_cost.get("prompt_tokens"))}</div>
    <div><strong>Evaluation cached prompt tokens:</strong> {_display_value(evaluation_cost.get("cached_prompt_tokens"))}</div>
    <div><strong>Evaluation completion tokens:</strong> {_display_value(evaluation_cost.get("completion_tokens"))}</div>
    <div><strong>Evaluation total tokens:</strong> {_display_value(evaluation_cost.get("provider_total_tokens"))}</div>
    <div><strong>Grounding evaluation cost:</strong> {_cost_display(evaluation_cost, currency)}</div>
    <div><strong>Reason:</strong> {escape(_cost_reason(evaluation_cost))}</div>
  </section>
  <h2>Combined Cost</h2>
  <section class="metrics">
    <div><strong>Combined total cost:</strong> {_cost_display(combined_cost, currency)}</div>
    <div><strong>Cost currency:</strong> {currency}</div>
    <div><strong>Cost accuracy:</strong> {escape(str(telemetry.get("cost_accuracy") or "Not available"))}</div>
    <div><strong>Pricing version:</strong> {escape(str(telemetry.get("pricing_configuration_id") or "Not available"))}</div>
    <div><strong>Pricing effective date:</strong> {escape(str(telemetry.get("pricing_effective_date") or "Not available"))}</div>
  </section>
  <h2>Blockers</h2><ul>{blockers}</ul>
  <h2>Warnings</h2><ul>{warnings}</ul>
  <h2>Unsupported Claims</h2><ul>{unsupported}</ul>
  <section class="article">{article}</section>
</body>
</html>
"""


def _source_text(entry: dict[str, Any]) -> str:
    if entry.get("source_text") is not None:
        return str(entry["source_text"])
    path = Path(str(entry["source_path"]))
    if not path.exists():
        raise BenchmarkError(f"Source path does not exist for {entry['input_id']}: {path}")
    if path.suffix.lower() == ".docx":
        extracted = extract_docx_text(path)
        if not extracted.text.strip():
            raise BenchmarkError(f"DOCX source is empty for {entry['input_id']}: {path}")
        return extracted.text
    if path.suffix.lower() == ".doc":
        raise BenchmarkError(f"DOC source is not supported for {entry['input_id']}: {path}")
    if path.suffix.lower() == ".json":
        payload = _read_json(path)
        text = payload.get("source_text") or payload.get("text") or payload.get("content")
        if not isinstance(text, str) or not text.strip():
            raise BenchmarkError(f"JSON source file lacks source_text/text/content: {path}")
        return text
    return path.read_text(encoding="utf-8")


def _read_plan(entry: dict[str, Any]) -> dict[str, Any]:
    return _read_json(Path(str(entry["plan_path"])))


def _draft_article_text(draft: object) -> str:
    data = _dict_value(draft)
    article = data.get("article")
    if isinstance(article, str):
        return article
    paragraphs = data.get("paragraphs")
    if isinstance(paragraphs, list):
        return "\n\n".join(str(item) for item in paragraphs if str(item).strip())
    return ""


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _pricing_version(path: Path) -> str | None:
    try:
        payload = _read_json(path)
    except BenchmarkError:
        return None
    version = payload.get("version")
    return str(version) if version is not None else None


def pricing_lookup(
    provider: str,
    model: str,
    path: Path = PRICE_CONFIG_PATH,
) -> dict[str, Any] | None:
    try:
        payload = _read_json(path)
    except BenchmarkError:
        return None
    for item in payload.get("models", []):
        if isinstance(item, dict) and item.get("provider") == provider and item.get("model") == model:
            return dict(item)
    return None


def _ledger_entry(
    *,
    provider: str,
    model: str,
    operation: str,
    section_ids: list[str],
    attempt: int,
    usage: dict[str, Any],
    cost_accuracy: str = "per_call_calculated",
    status: str = "parsed",
    failure_type: object = None,
    raw_response_path: object = None,
) -> dict[str, Any]:
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    cached_tokens = _optional_int(usage.get("cached_prompt_tokens")) or 0
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    provider_total = _optional_int(usage.get("total_tokens"))
    cost = price_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )
    return {
        "stage": "generation",
        "operation": operation,
        "section_ids": section_ids,
        "provider": provider,
        "model": model,
        "attempt": attempt,
        "prompt_tokens": prompt_tokens,
        "cached_prompt_tokens": cached_tokens,
        "completion_tokens": completion_tokens,
        "provider_total_tokens": provider_total,
        "thinking_tokens": _optional_int(usage.get("thinking_tokens")),
        "reasoning_tokens": _optional_int(usage.get("reasoning_tokens")),
        "accepted_prediction_tokens": _optional_int(usage.get("accepted_prediction_tokens")),
        "rejected_prediction_tokens": _optional_int(usage.get("rejected_prediction_tokens")),
        "provider_cost_ticks": _optional_int(usage.get("provider_cost_ticks")),
        "provider_reported_cost_usd": usage.get("provider_reported_cost_usd"),
        "provider_cost_conversion_status": usage.get("provider_cost_conversion_status"),
        "input_cost_usd": cost.get("input_cost_usd"),
        "cached_input_cost_usd": cost.get("cached_input_cost_usd"),
        "output_cost_usd": cost.get("output_cost_usd"),
        "total_cost_usd": cost.get("total_cost_usd"),
        "cost_status": cost.get("cost_status"),
        "cost_accuracy": cost_accuracy,
        "call_status": status,
        "failure_type": failure_type,
        "raw_response_path": raw_response_path,
    }


def _cost_breakdown_from_usage(
    *,
    provider: str,
    model: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    prompt_tokens = _optional_int(usage.get("prompt_tokens"))
    cached_tokens = _optional_int(usage.get("cached_prompt_tokens")) or 0
    completion_tokens = _optional_int(usage.get("completion_tokens"))
    cost = price_token_usage(
        provider=provider,
        model=model,
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_tokens,
        completion_tokens=completion_tokens,
    )
    return {
        **cost,
        "provider_total_tokens": _optional_int(usage.get("total_tokens")),
    }


def _unavailable_cost(base: dict[str, Any], status: str) -> dict[str, Any]:
    return {
        **base,
        "input_cost_usd": None,
        "cached_input_cost_usd": None,
        "output_cost_usd": None,
        "total_cost_usd": None,
        "cost_status": status,
    }


def _token_cost(tokens: int, usd_per_million: object) -> Decimal:
    return (Decimal(tokens) * Decimal(str(usd_per_million))) / Decimal(1_000_000)


def _money(value: Decimal) -> float:
    return float(value.quantize(USD_QUANT, rounding=ROUND_HALF_UP))


def _money_text(value: Decimal) -> str:
    return str(value.quantize(USD_QUANT, rounding=ROUND_HALF_UP))


def _sum_money(values: Any) -> Decimal:
    total = Decimal("0")
    for value in values:
        if value is not None:
            total += Decimal(str(value))
    return total


def _uncached_tokens(
    prompt_tokens: int | None,
    cached_prompt_tokens: int | None,
) -> int | None:
    if prompt_tokens is None:
        return None
    return max(prompt_tokens - (cached_prompt_tokens or 0), 0)


def _combined_cost(
    generation_cost: object,
    generation_status: object,
    evaluation_cost: object,
    evaluation_status: object,
) -> tuple[float | None, str]:
    if generation_cost is None or evaluation_cost is None:
        return None, "partial"
    total = Decimal(str(generation_cost)) + Decimal(str(evaluation_cost))
    status = (
        "calculated"
        if generation_status == "calculated" and evaluation_status == "calculated"
        else "partial"
    )
    return _money(total), status


def _token_reconciliation(
    *,
    provider: str,
    model: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    provider_total_tokens: int | None,
) -> dict[str, Any]:
    if prompt_tokens is None or completion_tokens is None or provider_total_tokens is None:
        return {
            "token_reconciliation_status": "partial_usage_metadata",
            "token_reconciliation_difference": None,
            "token_reconciliation_note": "Token metadata is incomplete.",
        }
    category_total = prompt_tokens + completion_tokens
    difference = provider_total_tokens - category_total
    if difference == 0:
        return {
            "token_reconciliation_status": "reconciled",
            "token_reconciliation_difference": 0,
            "token_reconciliation_note": "Provider total equals prompt plus completion tokens.",
        }
    pricing = pricing_lookup(provider, model)
    if difference > 0 and pricing and pricing.get("output_includes_thinking_tokens"):
        return {
            "token_reconciliation_status": "provider_includes_thinking_tokens",
            "token_reconciliation_difference": difference,
            "token_reconciliation_note": (
                "Provider total exceeds prompt plus completion tokens; pricing "
                "entry indicates output includes thinking tokens where applicable."
            ),
        }
    return {
        "token_reconciliation_status": "unexplained_difference",
        "token_reconciliation_difference": difference,
        "token_reconciliation_note": "Provider total does not match billable category tokens.",
    }


def _cost_accuracy(
    ledger: list[dict[str, Any]],
    generation: dict[str, Any],
    evaluation: dict[str, Any],
) -> str:
    if generation.get("cost_status") != "calculated" or evaluation.get("cost_status") != "calculated":
        return "partial"
    if any(row.get("cost_accuracy") == "aggregate_estimate" for row in ledger):
        return "aggregate_estimate"
    return "per_call_calculated" if ledger else "aggregate_estimate"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON file: {path}") from exc
    if not isinstance(payload, dict):
        raise BenchmarkError(f"JSON file must contain an object: {path}")
    return payload


def _read_json_any(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BenchmarkError(f"JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise BenchmarkError(f"Invalid JSON file: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    _write_json(temp_path, payload)
    temp_path.replace(path)


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(content, encoding="utf-8")
    temp_path.replace(path)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "__dict__"):
        return dict(value.__dict__)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return dict(value)


def _title_from_source(source_text: object, source_path: object, fallback: str) -> str:
    if isinstance(source_text, str) and source_text.strip():
        return source_text.strip().splitlines()[0][:120]
    if source_path is not None:
        return Path(str(source_path)).stem
    return fallback


def _word_count_variance(word_count: int | None, desired: int) -> int | None:
    return None if word_count is None else word_count - desired


def _within_range(word_count: int | None, target_min: int | None, target_max: int | None) -> bool | None:
    if word_count is None or target_min is None or target_max is None:
        return None
    return target_min <= word_count <= target_max


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


def _list_value(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None and str(value) else None


def _display_value(value: object) -> str:
    return "Not available" if value is None else escape(str(value))


def _cost_display(cost: dict[str, Any], currency: str) -> str:
    value = cost.get("total_cost_usd")
    if value is None:
        return "Not available"
    return f"{currency} {_money_text(Decimal(str(value)))}"


def _provider_cost_display(cost: dict[str, Any], currency: str) -> str:
    value = cost.get("provider_reported_cost_usd")
    if value is None:
        return "Not available"
    return f"{currency} {_money_text(Decimal(str(value)))}"


def _cost_reason(cost: dict[str, Any]) -> str:
    status = str(cost.get("cost_status") or "")
    if status == "calculated":
        return "calculated"
    return status.replace("_", " ") if status else "Not available"


def _sum_optional_int(values: Any) -> int | None:
    total = 0
    found = False
    for value in values:
        if isinstance(value, int):
            total += value
            found = True
    return total if found else None


def _sum_money_or_none(values: Any) -> float | None:
    total = Decimal("0")
    found = False
    for value in values:
        if value is not None:
            total += Decimal(str(value))
            found = True
    return _money(total) if found else None


def _provider_cost_conversion_status(ledger: list[dict[str, Any]]) -> str:
    statuses = {
        str(row.get("provider_cost_conversion_status"))
        for row in ledger
        if row.get("provider_cost_conversion_status") is not None
    }
    if not statuses:
        return "unavailable"
    if statuses == {"converted"}:
        return "converted"
    if "conversion_unconfirmed" in statuses:
        return "conversion_unconfirmed"
    return "unavailable" if statuses == {"unavailable"} else "partial"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _concise_error(error: object) -> str:
    message = str(error or "")
    message = " ".join(message.split())
    if len(message) > 160:
        return message[:157].rstrip() + "..."
    return message or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
