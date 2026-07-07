"""Run a compact live StyleScribe workflow validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    from backend.app.services.pasted_text_workflow_service import (
        run_pasted_text_to_draft_workflow,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default="manual_request.json")
    parser.add_argument("--desired-word-count", type=int)
    parser.add_argument("--model")
    parser.add_argument("--generation-model")
    parser.add_argument("--workflow-mode", default=None)
    parser.add_argument("--output", default="manual_response_10i_live.json")
    parser.add_argument("--html-output")
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
    if args.generation_model:
        os.environ["OPENAI_MODEL_GENERATION"] = args.generation_model

    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    if args.desired_word_count is not None:
        request["desired_word_count"] = args.desired_word_count
    if args.workflow_mode is not None:
        request["workflow_mode"] = args.workflow_mode

    response = run_pasted_text_to_draft_workflow(**request)
    payload = response.model_dump(mode="json")
    payload["workflow_completed"] = response.status == "completed"
    payload["openai_model"] = os.getenv("OPENAI_MODEL")
    payload["final_grounding_score"] = (
        response.final_evaluation_summary.grounding_score
        if response.final_evaluation_summary
        else None
    )
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.html_output:
        _write_html_output(
            output_path=Path(args.html_output),
            payload=payload,
            request=request,
        )

    report_keys = [
        "workflow_completed",
        "export_paths",
        "desired_word_count",
        "target_min_word_count",
        "target_max_word_count",
        "generation_mode_used",
        "section_assembled_article_word_count",
        "revision_mode",
        "revision_input_word_count",
        "revision_patch_count",
        "revision_patches_applied_count",
        "revision_patches_skipped_count",
        "revision_output_word_count",
        "revision_rejected_for_length_collapse",
        "revision_rejected_reason",
        "generated_headline",
        "generated_subheadline",
        "unsupported_claim_findings_count",
        "unsupported_claim_patch_count",
        "unsupported_claim_patches_applied_count",
        "unsupported_claim_patches_skipped_count",
        "unsupported_claim_patch_skipped_reasons",
        "unsupported_claims_unresolved_count",
        "unsupported_claims_cleared_by_patch",
        "initial_readiness",
        "initial_readiness_reasons",
        "final_readiness",
        "final_readiness_reasons",
        "readiness_decision_source",
        "final_publication_blockers",
        "final_publication_warnings",
        "final_article_word_count",
        "final_word_count_ratio",
        "final_grounding_score",
        "google_signals_available",
        "google_signals_score",
        "google_signals_version",
        "google_signals_risk_flags",
        "google_signals_recommendations",
        "google_signals_error",
        "tamil_quality_warnings",
        "publication_ready_completeness_passed",
        "length_status",
        "section_coverage_status",
        "model_used_by_stage",
        "total_runtime_seconds",
        "llm_call_count_total",
        "llm_call_count_by_stage",
        "runtime_by_stage",
        "slowest_stage",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "cached_prompt_tokens_total",
        "uncached_prompt_tokens_total",
        "prompt_cache_hit_ratio",
        "cached_prompt_tokens_by_stage",
        "prompt_cache_hit_ratio_by_stage",
        "estimated_cost_total_usd",
        "estimated_cost_by_stage_usd",
        "highest_cost_stage",
        "max_concurrent_section_calls",
        "generation_section_group_size",
        "generation_group_call_count",
        "generation_single_section_fallback_count",
        "generation_context_pack_tokens",
        "generation_context_pack_chars",
        "original_generation_context_chars",
        "compressed_generation_context_chars",
        "generation_context_compression_ratio",
        "cost_estimation_available",
        "cost_estimation_notes",
    ]
    print(json.dumps({key: payload.get(key) for key in report_keys}, indent=2))


def _write_html_output(
    output_path: Path,
    payload: dict[str, Any],
    request: dict[str, Any],
) -> None:
    try:
        html, missing_fields = _render_html_output(payload, request)
        output_path.write_text(html, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: HTML output was not written to {output_path}: {exc}")
        return
    if missing_fields:
        print(
            "WARNING: HTML output missing field(s): "
            + ", ".join(sorted(missing_fields))
        )


def _render_html_output(
    payload: dict[str, Any],
    request: dict[str, Any],
) -> tuple[str, list[str]]:
    from backend.app.scripts.review_article_draft import TAMIL_FONT_STACK

    missing_fields: list[str] = []
    article_text, article_source = _best_article_text(payload)
    if not article_text:
        missing_fields.append("final_article_text")
        article_text = "Article text is not available in the response or local records."
        article_source = "missing"

    source_text = request.get("source_text")
    if not isinstance(source_text, str) or not source_text.strip():
        missing_fields.append("source_text")
        source_text = "Source input is not available in the request."
    headline, headline_source = _best_headline(payload, source_text)
    if not headline:
        missing_fields.append("generated_headline")
        headline = "Headline not available"
        headline_source = "missing"

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StyleScribe Manual Validation Output</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --card: #ffffff;
      --text: #172033;
      --muted: #5c667a;
      --border: #d9e0ec;
      --blocker-bg: #fff1f0;
      --blocker-border: #d93025;
      --warning-bg: #fff8e6;
      --warning-border: #d99a00;
      --accent: #2457c5;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: {TAMIL_FONT_STACK};
      line-height: 1.6;
    }}
    main {{
      max-width: 1120px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3 {{
      line-height: 1.25;
      margin: 0 0 14px;
    }}
    h1 {{
      font-size: 30px;
      margin-bottom: 22px;
    }}
    h2 {{
      font-size: 21px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 20px;
      margin-bottom: 18px;
      box-shadow: 0 1px 2px rgba(20, 32, 50, 0.05);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      background: #fbfcff;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 4px;
    }}
    .value {{
      font-weight: 650;
      overflow-wrap: anywhere;
    }}
    .article {{
      font-size: 18px;
      white-space: pre-wrap;
    }}
    .article-headline {{
      font-size: 28px;
      line-height: 1.25;
      margin: 6px 0 14px;
    }}
    .article-subheadline {{
      color: var(--muted);
      font-size: 17px;
      margin: -4px 0 16px;
    }}
    .source {{
      white-space: pre-wrap;
      color: #263247;
    }}
    .list {{
      margin: 0;
      padding-left: 22px;
    }}
    .blockers {{
      background: var(--blocker-bg);
      border-color: var(--blocker-border);
    }}
    .warnings {{
      background: var(--warning-bg);
      border-color: var(--warning-border);
    }}
    .tag {{
      color: var(--accent);
      font-size: 13px;
      font-weight: 650;
    }}
  </style>
</head>
<body>
<main>
  <h1>StyleScribe Manual Validation Output</h1>
  {_summary_section(payload)}
  {_article_section(article_text, article_source, headline, headline_source, payload)}
  {_google_signals_section(payload)}
  {_source_section(source_text)}
  {_telemetry_section(payload)}
</main>
</body>
</html>
"""
    return html, missing_fields


def _summary_section(payload: dict[str, Any]) -> str:
    summary_keys = [
        "workflow_completed",
        "author_id",
        "desired_word_count",
        "target_min_word_count",
        "target_max_word_count",
        "final_article_word_count",
        "length_status",
        "final_readiness",
        "publication_ready_completeness_passed",
    ]
    return f"""<section class="card">
  <h2>Summary</h2>
  <div class="grid">
    {''.join(_metric(key, payload.get(key)) for key in summary_keys)}
  </div>
  {_list_card("Final readiness reasons", payload.get("final_readiness_reasons"))}
  {_list_card(
      "Final publication blockers",
      payload.get("final_publication_blockers"),
      class_name="blockers",
  )}
  {_list_card(
      "Final publication warnings",
      payload.get("final_publication_warnings"),
      class_name="warnings",
  )}
</section>"""


def _article_section(
    article_text: str,
    article_source: str,
    headline: str,
    headline_source: str,
    payload: dict[str, Any],
) -> str:
    subheadline = payload.get("generated_subheadline")
    subheadline_html = (
        f'<div class="article-subheadline">{_safe(subheadline)}</div>'
        if isinstance(subheadline, str) and subheadline.strip()
        else ""
    )
    return f"""<section class="card">
  <h2>Generated Tamil Article</h2>
  <div class="tag">Headline source: {_safe(headline_source)}</div>
  <h3 class="article-headline">{_safe(headline)}</h3>
  {subheadline_html}
  <div class="tag">Article source: {_safe(article_source)}</div>
  <div class="article">{_safe(article_text)}</div>
</section>"""


def _source_section(source_text: str) -> str:
    return f"""<section class="card">
  <h2>Source Input</h2>
  <div class="source">{_safe(source_text)}</div>
</section>"""


def _google_signals_section(payload: dict[str, Any]) -> str:
    google_signals = payload.get("google_signals")
    if isinstance(google_signals, dict):
        score = google_signals.get("score")
        version = google_signals.get("version")
        components = google_signals.get("components")
        risk_flags = google_signals.get("risk_flags")
        recommendations = google_signals.get("recommendations")
        metadata = google_signals.get("metadata")
        error = payload.get("google_signals_error")
    else:
        score = payload.get("google_signals_score")
        version = payload.get("google_signals_version")
        components = payload.get("google_signals_components")
        risk_flags = payload.get("google_signals_risk_flags")
        recommendations = payload.get("google_signals_recommendations")
        metadata = payload.get("google_signals_metadata")
        error = payload.get("google_signals_error")
    metadata = metadata if isinstance(metadata, dict) else {}
    component_rows = ""
    if isinstance(components, list):
        component_rows = "".join(
            _google_signal_component_row(component) for component in components
        )
    if not component_rows:
        component_rows = "<p>Component scores are not available.</p>"
    return f"""<section class="card">
  <h2>Google Signals</h2>
  <div class="grid">
    {_metric("google_signals_score", score)}
    {_metric("google_signals_version", version)}
    {_metric("primary_search_intent", metadata.get("primary_search_intent"))}
    {_metric("suggested_slug", metadata.get("suggested_slug"))}
  </div>
  {_list_card("Google Signals risk flags", risk_flags, class_name="warnings")}
  {_list_card("Google Signals recommendations", recommendations)}
  <div class="card">
    <h3>Component Scores</h3>
    {component_rows}
  </div>
  {_metric("google_signals_error", error)}
</section>"""


def _google_signal_component_row(component: Any) -> str:
    if not isinstance(component, dict):
        return ""
    label = (
        f"{component.get('name')}: {component.get('score')}/100 "
        f"(weight {component.get('weight')}, risk {component.get('risk_level')})"
    )
    rationale = component.get("rationale")
    return f"""<div class="metric">
  <span class="label">{_safe(label)}</span>
  <span class="value">{_safe(_display_value(rationale))}</span>
</div>"""


def _telemetry_section(payload: dict[str, Any]) -> str:
    telemetry_keys = [
        "total_runtime_seconds",
        "llm_call_count_total",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "cached_prompt_tokens_total",
        "prompt_cache_hit_ratio",
        "estimated_cost_total_usd",
        "slowest_stage",
        "highest_cost_stage",
        "max_concurrent_section_calls",
        "generation_section_group_size",
        "generation_group_call_count",
        "generation_single_section_fallback_count",
        "generation_context_compression_ratio",
    ]
    return f"""<section class="card">
  <h2>Telemetry</h2>
  <div class="grid">
    {''.join(_metric(key, payload.get(key)) for key in telemetry_keys)}
  </div>
</section>"""


def _metric(label: str, value: Any) -> str:
    return f"""<div class="metric">
  <span class="label">{_safe(label)}</span>
  <span class="value">{_safe(_display_value(value))}</span>
</div>"""


def _list_card(title: str, values: Any, class_name: str = "") -> str:
    items = values if isinstance(values, list) else []
    class_attr = f" {class_name}" if class_name else ""
    if not items:
        body = "<p>None</p>"
    else:
        body = "<ul class=\"list\">" + "".join(
            f"<li>{_safe(_display_value(item))}</li>" for item in items
        ) + "</ul>"
    return f"""<div class="card{class_attr}">
  <h3>{_safe(title)}</h3>
  {body}
</div>"""


def _best_article_text(payload: dict[str, Any]) -> tuple[str, str]:
    from backend.app.db.repository import StyleScribeRepository

    for key in ("final_article", "revised_article", "section_assembled_article"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value, key

    repository = StyleScribeRepository()
    repository.initialize_schema()
    revision_id = payload.get("revision_id")
    if isinstance(revision_id, str) and revision_id:
        revision = repository.fetch_article_revision(revision_id)
        if revision and revision.revised_article_body.strip():
            return (
                revision.revised_article_body,
                "article_revisions.revised_article_body",
            )

    draft_id = payload.get("draft_id")
    if isinstance(draft_id, str) and draft_id:
        draft = repository.fetch_article_draft(draft_id)
        if draft:
            draft_payload = StyleScribeRepository.decode_json_object(draft.draft_json)
            article_body = draft_payload.get("article_body")
            if isinstance(article_body, str) and article_body.strip():
                return article_body, "article_drafts.draft_json.article_body"

    return "", ""


def _best_headline(payload: dict[str, Any], source_text: str) -> tuple[str, str]:
    generated = payload.get("generated_headline")
    if isinstance(generated, str) and generated.strip():
        return generated, "generated_headline"
    draft_summary = payload.get("draft_summary")
    if isinstance(draft_summary, dict):
        draft_headline = draft_summary.get("headline")
        if isinstance(draft_headline, str) and draft_headline.strip():
            return draft_headline, "draft_summary.headline"
    source_headline = _source_headline(source_text)
    if source_headline:
        return source_headline, "source_text.first_line"
    return "", ""


def _source_headline(source_text: str) -> str:
    for line in source_text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _display_value(value: Any) -> str:
    if value is None:
        return "Not available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _safe(value: Any) -> str:
    return escape(str(value), quote=True)


if __name__ == "__main__":
    main()
