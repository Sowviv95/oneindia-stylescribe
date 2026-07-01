"""Print or export a reviewable article draft with source/style highlights."""

import argparse
import sys
import textwrap
from html import escape
from pathlib import Path
from typing import Any

from backend.app.db.repository import StyleScribeRepository
from backend.app.models.article_draft_models import ArticleDraftResponse
from backend.app.services.article_generation_service import (
    ArticleGenerationError,
    get_article_draft,
)

TAMIL_FONT_STACK = (
    '"Nirmala UI", "Latha", "Vijaya", "Noto Sans Tamil", Arial, sans-serif'
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", required=True)
    parser.add_argument("--output")
    parser.add_argument("--format", choices=["markdown", "html"], default="markdown")
    args = parser.parse_args()

    context = load_review_context(args.draft_id)
    console_output = render_console(context)
    print(console_output)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = (
            render_html(context)
            if args.format == "html"
            else render_markdown(context)
        )
        output_path.write_text(content, encoding="utf-8")


def load_review_context(draft_id: str) -> dict[str, Any]:
    repo = StyleScribeRepository()
    try:
        draft_response = get_article_draft(draft_id, repo)
    except ArticleGenerationError as exc:
        raise SystemExit(str(exc)) from exc

    profile_record = repo.fetch_author_style_profile(draft_response.profile_id)
    brief_record = repo.fetch_grounded_brief(draft_response.brief_id)
    profile = (
        StyleScribeRepository.decode_json_object(profile_record.profile_json)
        if profile_record
        else {}
    )
    brief = (
        StyleScribeRepository.decode_json_object(brief_record.brief_json)
        if brief_record
        else {}
    )
    return {
        "draft": draft_response,
        "profile": profile,
        "brief": brief,
    }


def render_console(context: dict[str, Any]) -> str:
    draft_response = _draft_response(context)
    lines = _metadata_lines(draft_response)
    lines.extend(["", "Source brief highlights:"])
    lines.extend(_brief_lines(context["brief"]))
    lines.extend(["", "Style profile highlights:"])
    lines.extend(_style_lines(context["profile"]))
    lines.extend(["", "Generated draft:"])
    lines.extend(_draft_lines(draft_response.draft, wrap=True))
    return "\n".join(lines)


def render_markdown(context: dict[str, Any]) -> str:
    draft_response = _draft_response(context)
    lines = ["# Article Draft Review", ""]
    lines.extend(f"- {line}" for line in _metadata_lines(draft_response))
    lines.extend(["", "## Source Brief Highlights", ""])
    lines.extend(_markdown_list(_brief_lines(context["brief"])))
    lines.extend(["", "## Style Profile Highlights", ""])
    lines.extend(_markdown_list(_style_lines(context["profile"])))
    lines.extend(["", "## Generated Draft", ""])
    lines.extend(_draft_markdown_lines(draft_response.draft))
    return "\n".join(lines) + "\n"


def render_html(context: dict[str, Any]) -> str:
    markdown_sections = render_markdown(context).splitlines()
    body_lines = []
    for line in markdown_sections:
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
  <title>Article Draft Review</title>
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


def _metadata_lines(draft_response: ArticleDraftResponse) -> list[str]:
    lines = [
        f"Draft ID: {draft_response.draft_id}",
        f"Author ID: {draft_response.author_id}",
        f"Model: {draft_response.model_provider}/{draft_response.model_name}",
        f"Brief ID: {draft_response.brief_id}",
        f"Style profile ID: {draft_response.profile_id}",
        f"Article type: {draft_response.article_type}",
        f"Desired word count: {draft_response.desired_word_count}",
        f"Tone override: {draft_response.tone_override}",
        f"Include SEO: {draft_response.include_seo}",
    ]
    if draft_response.warnings:
        lines.append("Warnings:")
        lines.extend(f"- {warning}" for warning in draft_response.warnings)
    return lines


def _brief_lines(brief: dict[str, Any]) -> list[str]:
    lines = [
        f"Topic: {brief.get('topic')}",
        f"One-line summary: {brief.get('one_line_summary')}",
        "Confirmed facts:",
    ]
    lines.extend(f"- {fact}" for fact in _list_value(brief.get("confirmed_facts")))
    lines.append("Claims to avoid:")
    lines.extend(f"- {claim}" for claim in _list_value(brief.get("claims_to_avoid")))
    return lines


def _style_lines(profile: dict[str, Any]) -> list[str]:
    lines = []
    for key in (
        "overall_tone",
        "headline_style",
        "intro_style",
        "paragraph_style",
        "tamil_register",
    ):
        lines.append(f"{key.replace('_', ' ').title()}: {profile.get(key)}")
    lines.append("Dos:")
    lines.extend(f"- {item}" for item in _list_value(profile.get("dos")))
    lines.append("Donts:")
    lines.extend(f"- {item}" for item in _list_value(profile.get("donts")))
    return lines


def _draft_lines(draft: dict[str, Any], wrap: bool) -> list[str]:
    lines: list[str] = []
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        value = str(draft.get(key) or "")
        lines.extend(["", f"{key.replace('_', ' ').title()}:", _wrap(value, wrap)])
    lines.extend(["", "Tags:"])
    lines.extend(f"- {tag}" for tag in _list_value(draft.get("suggested_tags")))
    return lines


def _draft_markdown_lines(draft: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        lines.extend([f"### {key.replace('_', ' ').title()}", ""])
        lines.extend([str(draft.get(key) or ""), ""])
    lines.extend(["### Tags", ""])
    lines.extend(f"- {tag}" for tag in _list_value(draft.get("suggested_tags")))
    return lines


def _markdown_list(lines: list[str]) -> list[str]:
    return [f"- {line}" if not line.startswith("- ") else f"  {line}" for line in lines]


def _draft_response(context: dict[str, Any]) -> ArticleDraftResponse:
    value = context["draft"]
    if not isinstance(value, ArticleDraftResponse):
        raise TypeError("draft context must contain ArticleDraftResponse")
    return value


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _wrap(value: str, wrap: bool) -> str:
    return textwrap.fill(value, width=100) if wrap else value


if __name__ == "__main__":
    main()
