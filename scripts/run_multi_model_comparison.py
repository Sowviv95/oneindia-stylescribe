# ruff: noqa: E501,I001
"""Run the controlled OpenAI vs Gemini article-generation comparison."""

from __future__ import annotations

import json
import sys
import traceback
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.db.repository import StyleScribeRepository  # noqa: E402
from backend.app.services.article_generation_service import (  # noqa: E402
    generate_article_draft,
)
from backend.app.services.article_revision_service import (  # noqa: E402
    revise_article_grounding,
)
from backend.app.services.draft_grounding_evaluation_service import (  # noqa: E402
    evaluate_draft_grounding,
    evaluate_revision_grounding,
)
import backend.app.services.model_clients.gemini_client as gemini_module  # noqa: E402
from backend.app.services.model_clients.gemini_client import (  # noqa: E402
    GeminiJsonClient,
)
from backend.app.services.model_clients.openai_client import (  # noqa: E402
    OpenAIJsonClient,
    request_runtime_metadata,
)
from backend.app.services.tamil_quality_scanner import (  # noqa: E402
    approximate_tamil_word_count,
)
from backend.app.services.workflow_telemetry import (  # noqa: E402
    estimate_workflow_cost,
    resolve_stage_model,
)

BASE = Path("comparison")
BRIEF_ID = "c126cc9d-5600-4bcd-b7c4-12979e76fdb8"
PLAN_ID = "f26258fe-7e8f-4a3a-938d-5ef9e3f4451d"
AUTHOR_ID = "v_vasanthi"
OPENAI_GENERATION_MODEL = "gpt-5.5"
GEMINI_GENERATION_MODEL = "gemini-3.5-flash"
REQUEST = {
    "author_id": AUTHOR_ID,
    "author_instruction": "Write this as a Tamil news article for Oneindia readers.",
    "target_language": "ta",
    "article_type": "news",
    "desired_word_count": 600,
    "tone_override": "clear, engaging and factual",
    "run_grounding_evaluation": True,
    "export_review": True,
    "export_format": "html",
}


def main() -> int:
    repo = StyleScribeRepository()
    repo.initialize_schema()
    BASE.mkdir(exist_ok=True)
    for provider in ("openai", "gemini"):
        for experiment in ("raw", "final"):
            (BASE / provider / experiment).mkdir(parents=True, exist_ok=True)

    brief, plan = _write_shared_inputs(repo)
    runs: list[dict[str, Any]] = []
    for provider, experiment in (
        ("OpenAI", "raw"),
        ("Gemini", "raw"),
        ("OpenAI", "final"),
        ("Gemini", "final"),
    ):
        print(f"RUN_START {provider} {experiment}", flush=True)
        payload = _run_one(provider, experiment)
        expected_model = (
            OPENAI_GENERATION_MODEL if provider == "OpenAI" else GEMINI_GENERATION_MODEL
        )
        if payload["generation_model"] != expected_model:
            raise RuntimeError(
                f"{provider} {experiment} used {payload['generation_model']}, "
                f"expected {expected_model}"
            )
        out_dir = BASE / provider.lower() / experiment
        (out_dir / "response.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (out_dir / "review.html").write_text(_review_html(payload), encoding="utf-8")
        runs.append(payload)
        print(
            f"RUN_DONE {provider} {experiment} "
            f"model={payload['generation_model']} retries={payload['retry_count']}",
            flush=True,
        )

    raw_runs = [run for run in runs if run["experiment_type"] == "raw"]
    final_runs = [run for run in runs if run["experiment_type"] == "final"]
    html = _comparison_html(raw_runs, final_runs, brief, plan)
    (BASE / "multi_model_comparison.html").write_text(html, encoding="utf-8")
    (BASE / "side_by_side_comparison.html").write_text(html, encoding="utf-8")
    summary = _write_summary(runs)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


def _run_one(provider: str, experiment: str) -> dict[str, Any]:
    retry_delays: list[float] = []
    original_sleep = gemini_module.time.sleep

    def recording_sleep(seconds: float) -> None:
        retry_delays.append(float(seconds))
        original_sleep(seconds)

    if provider == "Gemini":
        gemini_module.time.sleep = recording_sleep
    try:
        generation_client = _generation_client(provider)
        eval_client = _openai_stage_client(
            "evaluation",
            "OPENAI_API_KEY is required for draft evaluation.",
        )
        rev_client = _openai_stage_client(
            "revision",
            "OPENAI_API_KEY is required for article revision.",
        )
        generation_started = perf_counter()
        draft = generate_article_draft(
            author_id=AUTHOR_ID,
            brief_id=BRIEF_ID,
            author_instruction=REQUEST["author_instruction"],
            target_language=REQUEST["target_language"],
            article_type=REQUEST["article_type"],
            desired_word_count=REQUEST["desired_word_count"],
            tone_override=REQUEST["tone_override"],
            plan_id=PLAN_ID,
            model_client=generation_client,
        )
        generation_runtime = round(perf_counter() - generation_started, 3)
    finally:
        if provider == "Gemini":
            gemini_module.time.sleep = original_sleep

    initial_eval_started = perf_counter()
    initial_eval = evaluate_draft_grounding(draft.draft_id, model_client=eval_client)
    initial_eval_runtime = round(perf_counter() - initial_eval_started, 3)
    revision = None
    final_eval = None
    revision_runtime = None
    final_eval_runtime = None
    if experiment == "final":
        revision_started = perf_counter()
        revision = revise_article_grounding(
            draft.draft_id,
            evaluation_id=initial_eval.evaluation_id,
            model_client=rev_client,
        )
        revision_runtime = round(perf_counter() - revision_started, 3)
        final_eval_started = perf_counter()
        final_eval = evaluate_revision_grounding(
            revision.revision_id,
            model_client=eval_client,
        )
        final_eval_runtime = round(perf_counter() - final_eval_started, 3)

    final_article = _article_from_revision(
        revision.revised_draft if revision else None,
        draft.draft,
    )
    final_eval_dict = final_eval.evaluation if final_eval else None
    active_eval = final_eval_dict or initial_eval.evaluation
    gen_model = draft.model_name
    request_metadata = _generation_request_metadata(
        provider,
        gen_model,
        generation_client,
        retry_delays,
    )
    gen_key = f"{_model_key(gen_model)}_generation"
    runtime_by_stage = {gen_key: generation_runtime, "initial_evaluation": initial_eval_runtime}
    if revision_runtime is not None:
        runtime_by_stage["revision"] = revision_runtime
    if final_eval_runtime is not None:
        runtime_by_stage["final_evaluation"] = final_eval_runtime

    return {
        "created_at": datetime.now(UTC).isoformat(),
        "provider": provider,
        "generation_provider": provider,
        "generation_model": gen_model,
        "experiment_type": experiment,
        "mode": experiment,
        "brief_id": BRIEF_ID,
        "plan_id": PLAN_ID,
        "author_id": AUTHOR_ID,
        "request": {
            **REQUEST,
            "run_auto_revision": experiment == "final",
            "run_final_evaluation": experiment == "final",
        },
        "workflow_settings": {
            "run_grounding_evaluation": True,
            "run_auto_revision": experiment == "final",
            "run_final_evaluation": experiment == "final",
        },
        "completion_status": "generated",
        "runtime_seconds": round(sum(runtime_by_stage.values()), 3),
        "generation_time_seconds": generation_runtime,
        "attempt_count": request_metadata["attempt_count"],
        "retry_count": request_metadata["retry_count"],
        "retry_attempts": request_metadata["retry_count"],
        "retry_delays_seconds": retry_delays,
        "configured_timeout_seconds": request_metadata["timeout_seconds"],
        "error": None,
        "draft_id": draft.draft_id,
        "revision_id": revision.revision_id if revision else None,
        "initial_evaluation_id": initial_eval.evaluation_id,
        "final_evaluation_id": final_eval.evaluation_id if final_eval else None,
        "draft": draft.model_dump(mode="json"),
        "initial_evaluation": initial_eval.model_dump(mode="json"),
        "revision": revision.model_dump(mode="json") if revision else None,
        "final_evaluation": final_eval.model_dump(mode="json") if final_eval else None,
        "final_article_text": final_article,
        "final_article_word_count": approximate_tamil_word_count(final_article),
        "summary": {
            "generation_provider_used": provider,
            "generation_model_used": gen_model,
            "headline": draft.draft.get("headline"),
            "word_count": approximate_tamil_word_count(final_article),
            "generation_runtime_seconds": generation_runtime,
            "grounding_score": initial_eval.evaluation.get("grounding_score"),
            "final_evaluation_score": (
                final_eval_dict.get("grounding_score") if final_eval_dict else None
            ),
            "unsupported_claims": _unsupported_count(initial_eval.evaluation),
            "revision_applied": experiment == "final",
            "final_readiness": _readiness(active_eval),
            "blocker_count": _blocker_count(active_eval),
            "warning_count": _warning_count(active_eval),
            "estimated_generation_cost_usd": _generation_cost(gen_model, draft.draft),
            "temperature_mode": request_metadata["temperature_mode"],
            "temperature_requested": request_metadata["temperature_requested"],
            "timeout_seconds": request_metadata["timeout_seconds"],
        },
        "telemetry": {
            "runtime_by_stage": runtime_by_stage,
            "model_used_by_stage": {
                gen_key: gen_model,
                "initial_evaluation": initial_eval.model_name,
                **({"revision": revision.model_name} if revision else {}),
                **({"final_evaluation": final_eval.model_name} if final_eval else {}),
            },
        },
    }


def _generation_client(provider: str) -> OpenAIJsonClient | GeminiJsonClient:
    if provider == "OpenAI":
        return OpenAIJsonClient(
            model_name=OPENAI_GENERATION_MODEL,
            missing_key_message="OPENAI_API_KEY is required for article draft generation.",
        )
    return GeminiJsonClient(
        missing_key_message="GEMINI_API_KEY is required for article draft generation."
    )


def _openai_stage_client(stage: str, message: str) -> OpenAIJsonClient:
    return OpenAIJsonClient(model_name=resolve_stage_model(stage), missing_key_message=message)


def _write_shared_inputs(repo: StyleScribeRepository) -> tuple[dict[str, Any], dict[str, Any]]:
    brief_record = repo.fetch_grounded_brief(BRIEF_ID)
    plan_record = repo.fetch_article_plan(PLAN_ID)
    if brief_record is None:
        raise RuntimeError(f"Shared brief not found: {BRIEF_ID}")
    if plan_record is None:
        raise RuntimeError(f"Shared plan not found: {PLAN_ID}")
    brief = StyleScribeRepository.decode_json_object(brief_record.brief_json)
    plan = {
        "plan_id": plan_record.plan_id,
        "brief_id": plan_record.brief_id,
        "author_id": plan_record.author_id,
        "article_type": plan_record.article_type,
        "desired_word_count": plan_record.desired_word_count,
        "target_min_word_count": plan_record.target_min_word_count,
        "target_max_word_count": plan_record.target_max_word_count,
        "planned_sections": StyleScribeRepository.decode_json_list(
            plan_record.planned_sections_json
        ),
        "plan_summary": plan_record.plan_summary,
        "expansion_items_used": StyleScribeRepository.decode_json_list(
            plan_record.expansion_items_used_json
        ),
        "claims_to_avoid": StyleScribeRepository.decode_json_list(
            plan_record.claims_to_avoid_json
        ),
    }
    (BASE / "shared_inputs.json").write_text(
        json.dumps(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "request": REQUEST,
                "brief_id": BRIEF_ID,
                "plan_id": PLAN_ID,
                "brief": brief,
                "plan": plan,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return brief, plan


def _write_summary(runs: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for payload in runs:
        out_dir = BASE / payload["provider"].lower() / payload["experiment_type"]
        rows.append(
            {
                "provider": payload["provider"],
                "generation_model": payload["generation_model"],
                "experiment_type": payload["experiment_type"],
                "brief_id": payload["brief_id"],
                "plan_id": payload["plan_id"],
                "author_id": payload["author_id"],
                "workflow_settings": payload["workflow_settings"],
                "completion_status": payload["completion_status"],
                "runtime_seconds": payload["runtime_seconds"],
                "generation_runtime_seconds": payload["generation_time_seconds"],
                "configured_timeout_seconds": payload["configured_timeout_seconds"],
                "attempt_count": payload["attempt_count"],
                "word_count": payload["summary"]["word_count"],
                "grounding_score": payload["summary"]["grounding_score"],
                "revision_status": (
                    "applied" if payload["summary"]["revision_applied"] else "not_applied"
                ),
                "final_evaluation_score": payload["summary"]["final_evaluation_score"],
                "json_path": str(out_dir / "response.json").replace("\\", "/"),
                "html_path": str(out_dir / "review.html").replace("\\", "/"),
                "retry_count": payload["retry_count"],
                "retry_delays_seconds": payload["retry_delays_seconds"],
                "temperature_mode": payload["summary"]["temperature_mode"],
                "temperature_requested": payload["summary"]["temperature_requested"],
                "errors": payload["error"],
            }
        )
    summary = {
        "created_at": datetime.now(UTC).isoformat(),
        "shared_brief_id": BRIEF_ID,
        "shared_plan_id": PLAN_ID,
        "rows": rows,
    }
    (BASE / "execution_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    header = (
        "| Provider | Generation Model | Experiment | Brief ID | Plan ID | "
        "Author ID | Status | Timeout | Runtime | Attempts | Word Count | "
        "Grounding Score | Revision | "
        "Final Evaluation Score | Temperature Mode | Temperature Requested | "
        "Retry Count | Output Paths | Errors |"
    )
    lines = [
        header,
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        paths = f"{row['json_path']}<br>{row['html_path']}"
        lines.append(
            f"| {row['provider']} | {row['generation_model']} | "
            f"{row['experiment_type']} | {row['brief_id']} | {row['plan_id']} | "
            f"{row['author_id']} | {row['completion_status']} | "
            f"{row['configured_timeout_seconds']} | {row['runtime_seconds']} | "
            f"{row['attempt_count']} | {row['word_count']} | "
            f"{row['grounding_score']} | {row['revision_status']} | "
            f"{row['final_evaluation_score']} | {row['temperature_mode']} | "
            f"{row['temperature_requested']} | {row['retry_count']} | "
            f"{paths} | {row['errors']} |"
        )
    (BASE / "execution_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def _review_html(payload: dict[str, Any]) -> str:
    draft = payload["draft"]["draft"]
    title = f"{payload['provider']} {payload['experiment_type'].title()} Review"
    return f"""<!doctype html><html lang="ta"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_safe(title)}</title>{_style()}</head><body><main>
<h1>{_safe(title)}</h1>
<section class="card"><h2>Summary</h2><div class="grid">
{_metric("Provider", payload["provider"])}
{_metric("Generation model", payload["generation_model"])}
{_metric("Experiment", payload["experiment_type"])}
{_metric("Word count", payload["summary"]["word_count"])}
{_metric("Generation runtime", payload["generation_time_seconds"])}
{_metric("Configured timeout", payload["configured_timeout_seconds"])}
{_metric("Attempt count", payload["attempt_count"])}
{_metric("Grounding score", payload["summary"]["grounding_score"])}
{_metric("Unsupported claims", payload["summary"]["unsupported_claims"])}
{_metric("Revision applied", payload["summary"]["revision_applied"])}
{_metric("Final evaluation score", payload["summary"]["final_evaluation_score"])}
{_metric("Temperature mode", payload["summary"]["temperature_mode"])}
{_metric("Temperature requested", payload["summary"]["temperature_requested"])}
{_metric("Retry count", payload["retry_count"])}
</div></section>
<section class="card"><h2>Generated Article</h2>
<h3>{_safe(draft.get("headline"))}</h3>
<p><strong>{_safe(draft.get("subheadline"))}</strong></p>
<div class="article-body">{_safe(payload["final_article_text"])}</div></section>
</main></body></html>"""


def _comparison_html(
    raw_runs: list[dict[str, Any]],
    final_runs: list[dict[str, Any]],
    brief: dict[str, Any],
    plan: dict[str, Any],
) -> str:
    raw_a, raw_b = raw_runs
    final_a, final_b = final_runs
    summary = _comparison_summary(final_a, final_b)
    telemetry = {
        f"{_model_key(run['generation_model'])}_generation": run["telemetry"]
        for run in [raw_a, raw_b, final_a, final_b]
    }
    return f"""<!doctype html><html lang="ta"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StyleScribe Multi-Model Comparison</title>{_style()}</head><body><main>
<h1>StyleScribe Multi-Model Comparison</h1>
<section class="card"><h2>Summary</h2><div class="grid">
{_metric("author_id", AUTHOR_ID)}
{_metric("workflow_mode", "standard")}
{_metric("desired_word_count", REQUEST["desired_word_count"])}
{_metric("target_min_word_count", plan.get("target_min_word_count"))}
{_metric("target_max_word_count", plan.get("target_max_word_count"))}
{_metric("models_compared", "OpenAI: gpt-5.5; Gemini: gemini-3.5-flash")}
</div><p><strong>Recommended model output:</strong> {_safe(summary["recommended"])}</p>
<p>{_safe(summary["rationale"])}</p></section>
<section class="card"><h2>Source Summary</h2><div class="grid">
{_metric("brief_id", BRIEF_ID)}
{_metric("plan_id", PLAN_ID)}
{_metric("topic", brief.get("topic"))}
{_metric("source_language", brief.get("source_language"))}
{_metric("target_language", REQUEST["target_language"])}
</div><p>{_safe(brief.get("one_line_summary"))}</p></section>
{_metrics_table(raw_a, raw_b, "Side-by-Side Metrics - Raw Draft")}
{_metrics_table(final_a, final_b, "Side-by-Side Metrics - Full Pipeline")}
<section class="card"><h2>Comparison Summary</h2>
<p><strong>Factual faithfulness:</strong> {_safe(summary["faithfulness"])}</p>
<p><strong>Tamil newsroom quality:</strong> {_safe(summary["newsroom_quality"])}</p>
<p><strong>Readability:</strong> {_safe(summary["readability"])}</p>
<p><strong>Grounding:</strong> {_safe(summary["grounding"])}</p>
<p><strong>Length adherence:</strong> {_safe(summary["length"])}</p>
<p><strong>Editorial attention required:</strong> {_safe(summary["attention"])}</p>
<p><strong>Recommended model output:</strong> {_safe(summary["recommended"])}</p>
</section>
{_attention_table(final_a, final_b, "Editor Attention Items - Full Pipeline")}
{_article_table(raw_a, raw_b, "Generated Articles - Raw Draft")}
{_article_table(final_a, final_b, "Generated Articles - Full Pipeline")}
<section class="card"><h2>Telemetry</h2>
<pre>{_safe(json.dumps(telemetry, ensure_ascii=False, indent=2))}</pre></section>
</main></body></html>"""


def _metrics_table(a: dict[str, Any], b: dict[str, Any], title: str) -> str:
    rows = [
        ("Provider", a["provider"], b["provider"]),
        ("Generation model", a["generation_model"], b["generation_model"]),
        ("Headline", a["summary"].get("headline"), b["summary"].get("headline")),
        ("Word count", a["summary"].get("word_count"), b["summary"].get("word_count")),
        (
            "Generation runtime",
            a["summary"].get("generation_runtime_seconds"),
            b["summary"].get("generation_runtime_seconds"),
        ),
        (
            "Configured timeout",
            a["summary"].get("timeout_seconds"),
            b["summary"].get("timeout_seconds"),
        ),
        ("Attempt count", a.get("attempt_count"), b.get("attempt_count")),
        ("Retry count", a.get("retry_count"), b.get("retry_count")),
        (
            "Grounding score",
            a["summary"].get("grounding_score"),
            b["summary"].get("grounding_score"),
        ),
        (
            "Final readiness",
            a["summary"].get("final_readiness"),
            b["summary"].get("final_readiness"),
        ),
        (
            "Blocker count",
            a["summary"].get("blocker_count"),
            b["summary"].get("blocker_count"),
        ),
        (
            "Warning count",
            a["summary"].get("warning_count"),
            b["summary"].get("warning_count"),
        ),
        (
            "Unsupported claim count",
            a["summary"].get("unsupported_claims"),
            b["summary"].get("unsupported_claims"),
        ),
        (
            "Revision applied",
            a["summary"].get("revision_applied"),
            b["summary"].get("revision_applied"),
        ),
        (
            "Final evaluation score",
            a["summary"].get("final_evaluation_score"),
            b["summary"].get("final_evaluation_score"),
        ),
        (
            "Estimated generation cost",
            a["summary"].get("estimated_generation_cost_usd"),
            b["summary"].get("estimated_generation_cost_usd"),
        ),
        (
            "Temperature mode",
            a["summary"].get("temperature_mode"),
            b["summary"].get("temperature_mode"),
        ),
        (
            "Temperature requested",
            a["summary"].get("temperature_requested"),
            b["summary"].get("temperature_requested"),
        ),
    ]
    body = "".join(
        f"<tr><th>{_safe(label)}</th><td>{_safe(_display(av))}</td>"
        f"<td>{_safe(_display(bv))}</td></tr>"
        for label, av, bv in rows
    )
    return (
        f'<section class="card"><h2>{_safe(title)}</h2><table><thead><tr>'
        "<th>Metric</th><th>OpenAI: gpt-5.5</th>"
        "<th>Gemini: gemini-3.5-flash</th></tr></thead>"
        f"<tbody>{body}</tbody></table></section>"
    )


def _attention_table(a: dict[str, Any], b: dict[str, Any], title: str) -> str:
    return f"""<section class="card"><h2>{_safe(title)}</h2>
<table class="attention-comparison"><thead><tr>
<th>OpenAI: gpt-5.5</th><th>Gemini: gemini-3.5-flash</th>
</tr></thead><tbody><tr><td>{_attention_list(a)}</td>
<td>{_attention_list(b)}</td></tr></tbody></table></section>"""


def _article_table(a: dict[str, Any], b: dict[str, Any], title: str) -> str:
    return f"""<section class="card"><h2>{_safe(title)}</h2>
<p class="attention-legend">Inline highlights appear only for exact matched article text.
Red: blocker attention item. Yellow: warning attention item.</p>
<table class="article-comparison"><thead><tr>
<th>OpenAI: gpt-5.5</th><th>Gemini: gemini-3.5-flash</th>
</tr></thead><tbody><tr>
<td><h2>{_safe(a["summary"].get("headline"))}</h2>
<div class="article-body">{_highlighted_article(a)}</div></td>
<td><h2>{_safe(b["summary"].get("headline"))}</h2>
<div class="article-body">{_highlighted_article(b)}</div></td>
</tr></tbody></table></section>"""


def _attention_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    source = payload.get("final_evaluation") or payload.get("initial_evaluation") or {}
    evaluation = source.get("evaluation") or {}
    items: list[dict[str, Any]] = []
    for key, severity in [
        ("claims_to_avoid_violations", "blocker"),
        ("contradictions", "blocker"),
        ("invented_facts", "blocker"),
        ("unsupported_claims", "warning"),
        ("overclaim_phrases", "warning"),
        ("missing_key_facts", "info"),
    ]:
        for value in _list_value(evaluation.get(key)):
            text = _attention_text(value)
            items.append(
                {
                    "category": key,
                    "severity": severity,
                    "label": key.replace("_", " "),
                    "claim_text": text,
                    "matched_article_text": text,
                    "reason": value.get("reason") if isinstance(value, dict) else None,
                    "editor_action": "Review against grounded brief.",
                }
            )
    return items


def _attention_list(payload: dict[str, Any]) -> str:
    items = _attention_items(payload)
    if not items:
        return '<p class="attention-field">No editor attention items.</p>'
    rendered = []
    for item in items:
        severity = item["severity"]
        fields = "".join(
            f'<div class="attention-field"><strong>{_safe(label)}:</strong>'
            f'<div class="attention-text">{_safe(value)}</div></div>'
            for label, value in [
                ("Claim text", item.get("claim_text")),
                ("Matched article text", item.get("matched_article_text")),
                ("Reason", item.get("reason")),
                ("Editor action", item.get("editor_action")),
            ]
            if value
        )
        rendered.append(
            f'<div class="attention-item {severity}"><div class="attention-title">'
            f"{_safe(item['category'])} · {_safe(severity)} · "
            f"{_safe(item['label'])}</div>{fields}</div>"
        )
    return '<div class="attention-list">' + "".join(rendered) + "</div>"


def _highlighted_article(payload: dict[str, Any]) -> str:
    article = payload.get("final_article_text") or ""
    matches: list[tuple[int, int, str]] = []
    cursor = 0
    for item in _attention_items(payload):
        matched = item.get("matched_article_text")
        if not isinstance(matched, str) or not matched:
            continue
        start = article.find(matched, cursor)
        if start < 0:
            continue
        end = start + len(matched)
        matches.append((start, end, str(item.get("severity") or "info")))
        cursor = end
    if not matches:
        return _safe(article)
    parts: list[str] = []
    cursor = 0
    for start, end, severity in matches:
        parts.append(_safe(article[cursor:start]))
        parts.append(
            f'<mark class="attention-highlight {severity}">'
            f"{_safe(article[start:end])}</mark>"
        )
        cursor = end
    parts.append(_safe(article[cursor:]))
    return "".join(parts)


def _comparison_summary(a: dict[str, Any], b: dict[str, Any]) -> dict[str, str]:
    recommended = a if _deterministic_score(a) >= _deterministic_score(b) else b
    return {
        "faithfulness": (
            f"{a['provider']} grounding={a['summary'].get('grounding_score')}, "
            f"unsupported={a['summary'].get('unsupported_claims')}; "
            f"{b['provider']} grounding={b['summary'].get('grounding_score')}, "
            f"unsupported={b['summary'].get('unsupported_claims')}."
        ),
        "newsroom_quality": (
            "No separate Tamil newsroom quality score was available in these "
            "direct run artifacts."
        ),
        "readability": (
            f"Word counts: {a['provider']}={a['summary'].get('word_count')}, "
            f"{b['provider']}={b['summary'].get('word_count')}."
        ),
        "grounding": (
            f"Final evaluation scores: {a['provider']}="
            f"{a['summary'].get('final_evaluation_score')}, {b['provider']}="
            f"{b['summary'].get('final_evaluation_score')}."
        ),
        "length": (
            f"Target range is 450-690 words; outputs are "
            f"{a['summary'].get('word_count')} and {b['summary'].get('word_count')}."
        ),
        "attention": (
            f"Attention counts: {a['provider']} blockers="
            f"{a['summary'].get('blocker_count')}, warnings="
            f"{a['summary'].get('warning_count')}; {b['provider']} blockers="
            f"{b['summary'].get('blocker_count')}, warnings="
            f"{b['summary'].get('warning_count')}."
        ),
        "recommended": f"{recommended['provider']}: {recommended['generation_model']}",
        "rationale": (
            "Deterministic recommendation from existing grounding, final-evaluation, "
            "unsupported-claim, blocker, and warning counts only."
        ),
    }


def _deterministic_score(payload: dict[str, Any]) -> int:
    summary = payload["summary"]
    return (
        int(summary.get("grounding_score") or 0)
        + int(summary.get("final_evaluation_score") or 0)
        - int(summary.get("unsupported_claims") or 0) * 5
        - int(summary.get("blocker_count") or 0) * 10
        - int(summary.get("warning_count") or 0)
    )


def _style() -> str:
    return """<style>
:root { --bg: #f4f6fa; --card: #ffffff; --text: #172033; --muted: #5c667a; --border: #d9e0ec; --accent: #2457c5; --warning: #fff8e6; --blocker: #fff1f0; }
body { margin: 0; background: var(--bg); color: var(--text); font-family: "Nirmala UI", "Latha", "Vijaya", "Noto Sans Tamil", Arial, sans-serif; line-height: 1.62; }
main { max-width: 1280px; margin: 0 auto; padding: 28px; }
h1, h2, h3 { line-height: 1.25; margin: 0 0 12px; }
h1 { font-size: 30px; margin-bottom: 22px; } h2 { font-size: 21px; } h3 { font-size: 18px; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 18px; margin-bottom: 18px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }
.metric { border: 1px solid var(--border); border-radius: 6px; padding: 9px 11px; background: #fbfcff; }
.label { display: block; color: var(--muted); font-size: 13px; margin-bottom: 3px; }
.value { font-weight: 650; overflow-wrap: anywhere; }
table { border-collapse: collapse; width: 100%; background: var(--card); }
th, td { border: 1px solid var(--border); padding: 10px; text-align: left; vertical-align: top; }
th { background: #eef2f8; }
.article-comparison, .attention-comparison { width: 100%; border-collapse: collapse; table-layout: fixed; }
.article-comparison th, .article-comparison td, .attention-comparison th, .attention-comparison td { width: 50%; vertical-align: top; border: 1px solid var(--border); padding: 16px; overflow-wrap: anywhere; }
.article-comparison th, .attention-comparison th { background: #eef2f8; }
.article-comparison h2 { font-size: 21px; margin-bottom: 10px; }
.article-comparison h3 { color: var(--muted); font-size: 17px; font-weight: 650; margin-bottom: 14px; }
.article-body { white-space: pre-wrap; line-height: 1.65; font-size: 16px; overflow-wrap: anywhere; }
.attention-list { display: grid; gap: 10px; }
.attention-item { border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; background: #f5f7fb; }
.attention-item.blocker { background: var(--blocker); border-color: #f2b8b5; }
.attention-item.warning { background: var(--warning); border-color: #edd48d; }
.attention-item.info { background: #eef5ff; border-color: #bdd7ff; }
.attention-title { font-weight: 700; margin-bottom: 6px; }
.attention-field { margin-top: 6px; font-size: 14px; }
.attention-text { white-space: pre-wrap; }
.attention-highlight { border-radius: 4px; padding: 1px 3px; }
.attention-highlight.blocker { background: #ffd7d4; }
.attention-highlight.warning { background: #ffeaa3; }
.attention-legend { color: var(--muted); font-size: 14px; margin: 0 0 12px; }
.source { white-space: pre-wrap; color: #263247; }
.warning { background: var(--warning); } .blocker { background: var(--blocker); }
</style>"""


def _article_from_revision(revision: dict[str, Any] | None, draft: dict[str, Any]) -> str:
    if revision:
        body = revision.get("article_body")
        if isinstance(body, str) and body.strip():
            return body
    return str(draft.get("article_body") or "")


def _unsupported_count(evaluation: dict[str, Any] | None) -> int | None:
    if not evaluation:
        return None
    return sum(
        len(_list_value(evaluation.get(key)))
        for key in (
            "unsupported_claims",
            "invented_facts",
            "contradictions",
            "claims_to_avoid_violations",
        )
    )


def _blocker_count(evaluation: dict[str, Any] | None) -> int:
    if not evaluation:
        return 0
    return (
        len(_list_value(evaluation.get("invented_facts")))
        + len(_list_value(evaluation.get("contradictions")))
        + len(_list_value(evaluation.get("claims_to_avoid_violations")))
    )


def _warning_count(evaluation: dict[str, Any] | None) -> int:
    if not evaluation:
        return 0
    return (
        len(_list_value(evaluation.get("unsupported_claims")))
        + len(_list_value(evaluation.get("overclaim_phrases")))
        + len(_list_value(evaluation.get("missing_key_facts")))
    )


def _readiness(evaluation: dict[str, Any] | None) -> str | None:
    if not evaluation or evaluation.get("editorial_readiness") is None:
        return None
    return str(evaluation.get("editorial_readiness"))


def _generation_cost(model: str, draft: dict[str, Any]) -> float | None:
    usage = draft.get("token_usage")
    if not isinstance(usage, dict):
        return None
    cost = estimate_workflow_cost({"generation": usage}, {"generation": model})
    value = cost.get("estimated_cost_by_stage_usd", {}).get("generation")
    return float(value) if isinstance(value, int | float) else None


def _generation_request_metadata(
    provider: str,
    model: str,
    client: OpenAIJsonClient | GeminiJsonClient,
    retry_delays: list[float],
) -> dict[str, object]:
    if provider == "OpenAI":
        return {
            **request_runtime_metadata(
                model,
                0.1,
                float(getattr(client, "timeout_seconds", 90.0)),
            ),
            "attempt_count": getattr(client, "last_attempt_count", 1),
            "retry_count": getattr(client, "last_retry_count", 0),
        }
    return {
        "temperature_mode": "explicit",
        "temperature_requested": 0.1,
        "timeout_seconds": getattr(client, "timeout_seconds", None),
        "attempt_count": len(retry_delays) + 1,
        "retry_count": len(retry_delays),
    }


def _model_key(model: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in model.lower()).strip("_")


def _attention_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("claim", "phrase", "fact", "text", "issue"):
            text = value.get(key)
            if isinstance(text, str) and text:
                return text
    return _display(value)


def _list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _metric(label: str, value: Any) -> str:
    return (
        f'<div class="metric"><span class="label">{_safe(label)}</span>'
        f'<span class="value">{_safe(_display(value))}</span></div>'
    )


def _display(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _safe(value: Any) -> str:
    return escape(str(value if value is not None else ""), quote=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"RUN_BLOCKED: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(2) from None
