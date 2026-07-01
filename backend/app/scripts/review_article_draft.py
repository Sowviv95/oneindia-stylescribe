"""Print a reviewable article draft with source/style highlights."""

import argparse
import sys
import textwrap
from typing import Any

from backend.app.db.repository import StyleScribeRepository
from backend.app.services.article_generation_service import (
    ArticleGenerationError,
    get_article_draft,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--draft-id", required=True)
    args = parser.parse_args()

    repo = StyleScribeRepository()
    try:
        draft_response = get_article_draft(args.draft_id, repo)
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

    print(f"Draft ID: {draft_response.draft_id}")
    print(f"Author ID: {draft_response.author_id}")
    print(f"Model: {draft_response.model_provider}/{draft_response.model_name}")
    print(f"Brief ID: {draft_response.brief_id}")
    print(f"Style profile ID: {draft_response.profile_id}")
    if draft_response.warnings:
        print("\nWarnings:")
        for warning in draft_response.warnings:
            print(f"- {warning}")

    print("\nSource brief highlights:")
    _print_brief_highlights(brief)

    print("\nStyle profile highlights:")
    _print_style_highlights(profile)

    print("\nGenerated draft:")
    _print_draft(draft_response.draft)


def _print_brief_highlights(brief: dict[str, Any]) -> None:
    print(f"Topic: {brief.get('topic')}")
    print(f"One-line summary: {brief.get('one_line_summary')}")
    print("Confirmed facts:")
    for fact in _list_value(brief.get("confirmed_facts")):
        print(f"- {fact}")
    print("Claims to avoid:")
    for claim in _list_value(brief.get("claims_to_avoid")):
        print(f"- {claim}")


def _print_style_highlights(profile: dict[str, Any]) -> None:
    for key in (
        "overall_tone",
        "headline_style",
        "intro_style",
        "paragraph_style",
        "tamil_register",
    ):
        print(f"{key.replace('_', ' ').title()}: {profile.get(key)}")
    print("Dos:")
    for item in _list_value(profile.get("dos")):
        print(f"- {item}")
    print("Donts:")
    for item in _list_value(profile.get("donts")):
        print(f"- {item}")


def _print_draft(draft: dict[str, Any]) -> None:
    for key in (
        "headline",
        "subheadline",
        "article_body",
        "seo_title",
        "meta_description",
    ):
        print(f"\n{key.replace('_', ' ').title()}:")
        print(textwrap.fill(str(draft.get(key) or ""), width=100))
    print("\nTags:")
    for tag in _list_value(draft.get("suggested_tags")):
        print(f"- {tag}")


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


if __name__ == "__main__":
    main()
