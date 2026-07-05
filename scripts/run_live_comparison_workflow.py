"""Run a live StyleScribe multi-author comparison workflow."""

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
    from backend.app.services.multi_author_comparison_service import (
        run_multi_author_comparison_workflow,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default="manual_request.json")
    parser.add_argument("--author-id-a", required=True)
    parser.add_argument("--author-id-b", required=True)
    parser.add_argument("--desired-word-count", type=int)
    parser.add_argument("--model")
    parser.add_argument("--generation-model")
    parser.add_argument("--workflow-mode", default=None)
    parser.add_argument("--output", default="manual_comparison_response.json")
    parser.add_argument("--html-output")
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
    if args.generation_model:
        os.environ["OPENAI_MODEL_GENERATION"] = args.generation_model

    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    request["author_id_a"] = args.author_id_a
    request["author_id_b"] = args.author_id_b
    if args.desired_word_count is not None:
        request["desired_word_count"] = args.desired_word_count
    if args.workflow_mode is not None:
        request["workflow_mode"] = args.workflow_mode
    if args.generation_model:
        request["generation_model"] = args.generation_model

    response = run_multi_author_comparison_workflow(
        source_text=request["source_text"],
        author_id_a=request["author_id_a"],
        author_id_b=request["author_id_b"],
        author_instruction=request.get("author_instruction"),
        target_language=request.get("target_language", "ta"),
        article_type=request.get("article_type", "news"),
        desired_word_count=request.get("desired_word_count", 600),
        tone_override=request.get("tone_override"),
        workflow_mode=request.get("workflow_mode", "standard"),
    )
    payload = response.model_dump(mode="json")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.html_output:
        html_path = Path(args.html_output)
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(_render_html(payload, request), encoding="utf-8")

    report_keys = {
        "workflow_completed": payload.get("workflow_completed"),
        "workflow_mode": payload.get("workflow_mode"),
        "desired_word_count": payload.get("desired_word_count"),
        "target_min_word_count": payload.get("target_min_word_count"),
        "target_max_word_count": payload.get("target_max_word_count"),
        "brief_id": payload.get("shared_grounded_brief", {}).get("brief_id"),
        "author_a": _author_report(payload.get("author_a")),
        "author_b": _author_report(payload.get("author_b")),
        "recommendation": payload.get("comparison_summary", {}).get(
            "recommended_draft"
        ),
        "aggregate_runtime_seconds": payload.get("aggregate_runtime_seconds"),
        "aggregate_token_usage": payload.get("aggregate_token_usage"),
        "aggregate_estimated_cost_usd": payload.get(
            "aggregate_estimated_cost_usd"
        ),
    }
    print(json.dumps(report_keys, indent=2))


def _author_report(value: Any) -> dict[str, Any]:
    author = value if isinstance(value, dict) else {}
    return {
        "author_id": author.get("author_id"),
        "draft_id": author.get("draft_id"),
        "plan_id": author.get("plan_id"),
        "evaluation_id": author.get("evaluation_id"),
        "generated_headline": author.get("generated_headline"),
        "word_count": author.get("word_count"),
        "grounding_score": author.get("grounding_score"),
        "final_readiness": author.get("final_readiness"),
        "blockers": author.get("blockers"),
        "warnings": author.get("warnings"),
    }


def _render_html(payload: dict[str, Any], request: dict[str, Any]) -> str:
    from backend.app.scripts.review_article_draft import TAMIL_FONT_STACK

    author_a = _dict_value(payload.get("author_a"))
    author_b = _dict_value(payload.get("author_b"))
    summary = _dict_value(payload.get("comparison_summary"))
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>StyleScribe Multi-Author Comparison</title>
  <style>
    :root {{
      --bg: #f4f6fa;
      --card: #ffffff;
      --text: #172033;
      --muted: #5c667a;
      --border: #d9e0ec;
      --accent: #2457c5;
      --warning: #fff8e6;
      --blocker: #fff1f0;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: {TAMIL_FONT_STACK};
      line-height: 1.62;
    }}
    main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px;
    }}
    h1, h2, h3 {{
      line-height: 1.25;
      margin: 0 0 12px;
    }}
    h1 {{ font-size: 30px; margin-bottom: 22px; }}
    h2 {{ font-size: 21px; }}
    h3 {{ font-size: 18px; }}
    .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 18px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 9px 11px;
      background: #fbfcff;
    }}
    .label {{
      display: block;
      color: var(--muted);
      font-size: 13px;
      margin-bottom: 3px;
    }}
    .value {{ font-weight: 650; overflow-wrap: anywhere; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: var(--card);
    }}
    th, td {{
      border: 1px solid var(--border);
      padding: 10px;
      text-align: left;
      vertical-align: top;
    }}
    th {{ background: #eef2f8; }}
    .article-comparison {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    .article-comparison th,
    .article-comparison td {{
      width: 50%;
      vertical-align: top;
      border: 1px solid var(--border);
      padding: 16px;
      overflow-wrap: anywhere;
    }}
    .article-comparison th {{
      background: #eef2f8;
    }}
    .article-comparison h2 {{
      font-size: 21px;
      margin-bottom: 10px;
    }}
    .article-comparison h3 {{
      color: var(--muted);
      font-size: 17px;
      font-weight: 650;
      margin-bottom: 14px;
    }}
    .article-body {{
      white-space: pre-wrap;
      line-height: 1.65;
      font-size: 16px;
      overflow-wrap: anywhere;
    }}
    .attention-comparison {{
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    .attention-comparison th,
    .attention-comparison td {{
      width: 50%;
      vertical-align: top;
      border: 1px solid var(--border);
      padding: 14px;
      overflow-wrap: anywhere;
    }}
    .attention-list {{
      display: grid;
      gap: 10px;
    }}
    .attention-item {{
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 10px 12px;
      background: #f5f7fb;
    }}
    .attention-item.blocker {{
      background: var(--blocker);
      border-color: #f2b8b5;
    }}
    .attention-item.warning {{
      background: var(--warning);
      border-color: #edd48d;
    }}
    .attention-item.info {{
      background: #eef5ff;
      border-color: #bdd7ff;
    }}
    .attention-title {{
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .attention-field {{
      margin-top: 6px;
      font-size: 14px;
    }}
    .attention-text {{
      white-space: pre-wrap;
    }}
    .attention-highlight {{
      border-radius: 4px;
      padding: 1px 3px;
    }}
    .attention-highlight.blocker {{ background: #ffd7d4; }}
    .attention-highlight.warning {{ background: #ffeaa3; }}
    .attention-legend {{
      color: var(--muted);
      font-size: 14px;
      margin: 0 0 12px;
    }}
    .source {{
      white-space: pre-wrap;
      color: #263247;
    }}
    .warning {{ background: var(--warning); }}
    .blocker {{ background: var(--blocker); }}
  </style>
</head>
<body>
<main>
  <h1>StyleScribe Multi-Author Comparison</h1>
  {_summary_section(payload, summary)}
  {_source_section(payload, request)}
  {_metrics_table(author_a, author_b)}
  {_comparison_section(summary)}
  {_editor_attention_section(author_a, author_b)}
  {_article_comparison_section(author_a, author_b)}
  {_telemetry_section(payload)}
</main>
</body>
</html>
"""


def _summary_section(payload: dict[str, Any], summary: dict[str, Any]) -> str:
    keys = [
        "workflow_completed",
        "workflow_mode",
        "desired_word_count",
        "target_min_word_count",
        "target_max_word_count",
        "aggregate_runtime_seconds",
        "aggregate_estimated_cost_usd",
    ]
    return f"""<section class="card">
  <h2>Summary</h2>
  <div class="grid">{''.join(_metric(key, payload.get(key)) for key in keys)}</div>
  <p><strong>Recommended draft:</strong> {_safe(summary.get("recommended_draft"))}</p>
  <p>{_safe(summary.get("recommendation_rationale"))}</p>
</section>"""


def _source_section(payload: dict[str, Any], request: dict[str, Any]) -> str:
    brief = _dict_value(payload.get("shared_grounded_brief"))
    brief_summary = _dict_value(payload.get("brief_summary"))
    return f"""<section class="card">
  <h2>Source Summary</h2>
  <div class="grid">
    {_metric("brief_id", brief.get("brief_id"))}
    {_metric("topic", brief_summary.get("topic"))}
    {_metric("source_language", brief.get("source_language"))}
    {_metric("brief_model", brief.get("model_name"))}
  </div>
  <p>{_safe(brief_summary.get("one_line_summary"))}</p>
  <h3>Source Input</h3>
  <div class="source">{_safe(request.get("source_text"))}</div>
</section>"""


def _metrics_table(author_a: dict[str, Any], author_b: dict[str, Any]) -> str:
    rows = [
        ("Author ID", author_a.get("author_id"), author_b.get("author_id")),
        (
            "Headline",
            author_a.get("generated_headline"),
            author_b.get("generated_headline"),
        ),
        ("Word count", author_a.get("word_count"), author_b.get("word_count")),
        (
            "Grounding score",
            author_a.get("grounding_score"),
            author_b.get("grounding_score"),
        ),
        (
            "Final readiness",
            author_a.get("final_readiness"),
            author_b.get("final_readiness"),
        ),
        (
            "Blocker count",
            _count_items(author_a.get("blockers")),
            _count_items(author_b.get("blockers")),
        ),
        (
            "Warning count",
            _count_items(author_a.get("warnings")),
            _count_items(author_b.get("warnings")),
        ),
    ]
    body = "".join(
        f"<tr><th>{_safe(label)}</th><td>{_safe(_display(a))}</td><td>{_safe(_display(b))}</td></tr>"
        for label, a, b in rows
    )
    return f"""<section class="card">
  <h2>Side-By-Side Metrics</h2>
  <table>
    <thead><tr><th>Metric</th><th>Author A</th><th>Author B</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</section>"""


def _article_comparison_section(
    author_a: dict[str, Any],
    author_b: dict[str, Any],
) -> str:
    legend = (
        "Inline highlights appear only for exact matched article text. "
        "Red: blocker attention item. Yellow: warning attention item."
    )
    return f"""<section class="card">
  <h2>Generated Articles</h2>
  <p class="attention-legend">{_safe(legend)}</p>
  <table class="article-comparison">
    <thead>
      <tr>
        <th>Author A: {_safe(_display(author_a.get("author_id")))}</th>
        <th>Author B: {_safe(_display(author_b.get("author_id")))}</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        {_author_article_cell(author_a)}
        {_author_article_cell(author_b)}
      </tr>
    </tbody>
  </table>
</section>"""


def _author_article_cell(author: dict[str, Any]) -> str:
    return f"""<td>
  <h2>{_safe(_display(author.get("generated_headline")))}</h2>
  <h3>{_safe(_display(author.get("generated_subheadline")))}</h3>
  {_list_block("Blockers", author.get("blockers"), "blocker")}
  {_list_block("Warnings", author.get("warnings"), "warning")}
  <div class="article-body">{_highlighted_article_body(author)}</div>
</td>"""


def _editor_attention_section(
    author_a: dict[str, Any],
    author_b: dict[str, Any],
) -> str:
    return f"""<section class="card">
  <h2>Editor Attention Items</h2>
  <table class="attention-comparison">
    <thead>
      <tr>
        <th>Author A: {_safe(_display(author_a.get("author_id")))}</th>
        <th>Author B: {_safe(_display(author_b.get("author_id")))}</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>{_attention_list(author_a.get("editor_attention_items"))}</td>
        <td>{_attention_list(author_b.get("editor_attention_items"))}</td>
      </tr>
    </tbody>
  </table>
</section>"""


def _attention_list(values: Any) -> str:
    items = values if isinstance(values, list) else []
    if not items:
        return '<p class="attention-field">No editor attention items.</p>'
    rendered = "".join(_attention_item(item) for item in items)
    return f'<div class="attention-list">{rendered}</div>'


def _attention_item(value: Any) -> str:
    item = _dict_value(value)
    severity = _attention_severity(item.get("severity"))
    title = (
        f"{_display(item.get('category'))} · "
        f"{_display(item.get('severity'))} · "
        f"{_display(item.get('label'))}"
    )
    fields = [
        ("Claim text", item.get("claim_text")),
        ("Matched article text", item.get("matched_article_text")),
        ("Avoid rule", item.get("avoid_rule")),
        ("Reason", item.get("reason")),
        ("Editor action", item.get("editor_action")),
    ]
    details = "".join(
        _attention_field(label, field_value)
        for label, field_value in fields
        if field_value
    )
    return f"""<div class="attention-item {severity}">
  <div class="attention-title">{_safe(title)}</div>
  {details}
</div>"""


def _attention_field(label: str, value: Any) -> str:
    return f"""<div class="attention-field">
  <strong>{_safe(label)}:</strong>
  <div class="attention-text">{_safe(_display(value))}</div>
</div>"""


def _comparison_section(summary: dict[str, Any]) -> str:
    factual = summary.get("factual_faithfulness_comparison")
    return f"""<section class="card">
  <h2>Comparison Summary</h2>
  <p><strong>Factual faithfulness:</strong> {_safe(factual)}</p>
  <p><strong>Author style:</strong> {_safe(summary.get("author_style_difference"))}</p>
  <p><strong>Readability:</strong> {_safe(summary.get("readability_difference"))}</p>
  <p><strong>Recommendation:</strong> {_safe(summary.get("recommended_draft"))}</p>
  <p>{_safe(summary.get("recommendation_rationale"))}</p>
</section>"""


def _telemetry_section(payload: dict[str, Any]) -> str:
    telemetry = _dict_value(payload.get("telemetry"))
    runtime_json = json.dumps(
        telemetry.get("runtime_by_stage"),
        ensure_ascii=False,
        indent=2,
    )
    cost_json = json.dumps(
        telemetry.get("estimated_cost_by_stage_usd"),
        ensure_ascii=False,
        indent=2,
    )
    keys = [
        "llm_call_count_total",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "cached_prompt_tokens_total",
        "prompt_cache_hit_ratio",
        "slowest_stage",
        "highest_cost_stage",
    ]
    return f"""<section class="card">
  <h2>Telemetry</h2>
  <div class="grid">{''.join(_metric(key, telemetry.get(key)) for key in keys)}</div>
  <h3>Runtime By Stage</h3>
  <pre>{_safe(runtime_json)}</pre>
  <h3>Cost By Stage</h3>
  <pre>{_safe(cost_json)}</pre>
</section>"""


def _metric(label: str, value: Any) -> str:
    return f"""<div class="metric">
  <span class="label">{_safe(label)}</span>
  <span class="value">{_safe(_display(value))}</span>
</div>"""


def _list_block(title: str, values: Any, class_name: str) -> str:
    items = values if isinstance(values, list) else []
    if not items:
        return ""
    list_items = "".join(f"<li>{_safe(_display(item))}</li>" for item in items)
    return f"""<div class="card {class_name}">
  <h3>{_safe(title)}</h3>
  <ul>{list_items}</ul>
</div>"""


def _count_items(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0


def _highlighted_article_body(author: dict[str, Any]) -> str:
    article_body = _display(author.get("article_body"))
    attention_items = author.get("editor_attention_items")
    if not isinstance(attention_items, list):
        return _safe(article_body)

    matches: list[tuple[int, int, str]] = []
    cursor = 0
    for value in attention_items:
        item = _dict_value(value)
        matched = item.get("matched_article_text")
        if not isinstance(matched, str) or not matched:
            continue
        start = article_body.find(matched, cursor)
        if start < 0:
            continue
        end = start + len(matched)
        matches.append((start, end, _attention_severity(item.get("severity"))))
        cursor = end

    if not matches:
        return _safe(article_body)

    parts: list[str] = []
    cursor = 0
    for start, end, severity in matches:
        parts.append(_safe(article_body[cursor:start]))
        parts.append(
            f'<mark class="attention-highlight {severity}">'
            f"{_safe(article_body[start:end])}</mark>"
        )
        cursor = end
    parts.append(_safe(article_body[cursor:]))
    return "".join(parts)


def _attention_severity(value: Any) -> str:
    severity = str(value or "info")
    if severity in {"blocker", "warning", "info"}:
        return severity
    return "info"


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _display(value: Any) -> str:
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
