"""Print bounded source excerpts and the latest style profile for review."""

import argparse
import sys
import textwrap
from typing import Any

from backend.app.services.author_style_profile_service import (
    AuthorStyleProfileError,
    get_latest_author_style_profile,
)

MAX_PRINT_CHARS = 700


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--author-id", required=True)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    try:
        profile = get_latest_author_style_profile(args.author_id)
    except AuthorStyleProfileError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Profile ID: {profile.profile_id}")
    print(f"Author ID: {profile.author_id}")
    print(f"Snapshot ID: {profile.snapshot_id}")
    print(f"Model: {profile.model_provider}/{profile.model_name}")
    print(f"Status: {profile.status}")
    if profile.warnings:
        print("\nWarnings:")
        for warning in profile.warnings:
            print(f"- {warning}")

    print("\nSource excerpts used for profile:")
    review_refs = _reviewable_refs(profile.source_excerpt_refs)
    for index, ref in enumerate(review_refs[: args.limit], start=1):
        print(f"\n[{index}] {ref.get('filename')}")
        print(f"Title/Heading: {ref.get('title_or_heading')}")
        print(f"Category: {ref.get('category')}")
        print(f"Excerpt type: {ref.get('excerpt_type')}")
        print(_bounded_text(str(ref.get("excerpt_text") or "")))

    print("\nGenerated style profile:")
    _print_profile(profile.profile)


def _print_profile(profile: dict[str, Any]) -> None:
    for key, value in profile.items():
        title = key.replace("_", " ").title()
        print(f"\n{title}:")
        if isinstance(value, list):
            for item in value:
                print(f"- {item}")
        else:
            print(textwrap.fill(str(value), width=100))


def _bounded_text(text: str) -> str:
    if len(text) <= MAX_PRINT_CHARS:
        return text
    return text[: MAX_PRINT_CHARS - 3].rstrip() + "..."


def _reviewable_refs(refs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    priority = {
        "intro": 0,
        "body": 1,
        "closing": 2,
        "long_article": 3,
        "short_article": 4,
        "headline": 5,
    }
    return sorted(
        refs,
        key=lambda ref: priority.get(str(ref.get("excerpt_type")), 99),
    )


if __name__ == "__main__":
    main()
