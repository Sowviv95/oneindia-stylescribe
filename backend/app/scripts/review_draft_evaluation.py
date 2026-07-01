"""Print or export draft grounding evaluation results."""

import argparse
import sys
from html import escape
from pathlib import Path
from typing import Any

from backend.app.models.draft_evaluation_models import DraftEvaluationResponse
from backend.app.scripts.review_article_draft import TAMIL_FONT_STACK
from backend.app.services.draft_grounding_evaluation_service import (
    DraftEvaluationError,
    get_draft_evaluation,
    get_latest_draft_evaluation,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--draft-id")
    group.add_argument("--evaluation-id")
    parser.add_argument("--output")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown")
    args = parser.parse_args()

    try:
        response = (
            get_draft_evaluation(args.evaluation_id)
            if args.evaluation_id
            else get_latest_draft_evaluation(args.draft_id)
        )
    except DraftEvaluationError as exc:
        raise SystemExit(str(exc)) from exc

    markdown = render_markdown(response)
    print(markdown)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = render_html(response) if args.format == "html" else markdown
        output_path.write_text(content, encoding="utf-8")


def render_markdown(response: DraftEvaluationResponse) -> str:
    evaluation = response.evaluation
    lines = [
        "# Draft Grounding Evaluation",
        "",
        f"- Evaluation ID: {response.evaluation_id}",
        f"- Draft ID: {response.draft_id}",
        f"- Brief ID: {response.brief_id}",
        f"- Author ID: {response.author_id}",
        f"- Model: {response.model_provider}/{response.model_name}",
        f"- Status: {response.status}",
        "",
        "## Scores",
        "",
        f"- Grounding score: {evaluation.get('grounding_score')}",
        f"- Claim safety score: {evaluation.get('claim_safety_score')}",
        f"- Fact preservation score: {evaluation.get('fact_preservation_score')}",
        f"- Overall risk: {evaluation.get('overall_risk')}",
        f"- Editorial readiness: {evaluation.get('editorial_readiness')}",
        "",
        "## Unsupported Claims",
        "",
    ]
    lines.extend(_markdown_items(evaluation.get("unsupported_claims")))
    lines.extend(["", "## Overclaim Phrases", ""])
    lines.extend(_markdown_items(evaluation.get("overclaim_phrases")))
    lines.extend(["", "## Claims To Avoid Violations", ""])
    lines.extend(_markdown_items(evaluation.get("claims_to_avoid_violations")))
    lines.extend(["", "## Rewrite Guidance", ""])
    lines.extend(_markdown_items(evaluation.get("rewrite_guidance")))
    lines.extend(["", "## Summary", "", str(evaluation.get("summary") or "")])
    return "\n".join(lines) + "\n"


def render_html(response: DraftEvaluationResponse) -> str:
    markdown = render_markdown(response)
    body_lines = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            body_lines.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body_lines.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.startswith("- "):
            body_lines.append(f"<p>{escape(line)}</p>")
        elif line:
            body_lines.append(f"<p>{escape(line)}</p>")
        else:
            body_lines.append("")
    body = "\n".join(body_lines)
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>Draft Grounding Evaluation</title>
  <style>
    body {{
      font-family: {TAMIL_FONT_STACK};
      line-height: 1.65;
      margin: 32px;
      max-width: 980px;
    }}
    h1, h2 {{ line-height: 1.3; }}
    p {{ white-space: pre-wrap; }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _markdown_items(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        return ["- None"]
    return [f"- {_stringify(item)}" for item in value]


def _stringify(item: Any) -> str:
    if isinstance(item, dict):
        return "; ".join(f"{key}: {value}" for key, value in item.items())
    return str(item)


if __name__ == "__main__":
    main()
