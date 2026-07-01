"""Print a bounded grounded brief review."""

import argparse
import sys
import textwrap
from typing import Any

from backend.app.services.grounded_brief_service import (
    GroundedBriefError,
    get_grounded_brief,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--brief-id", required=True)
    args = parser.parse_args()

    try:
        response = get_grounded_brief(args.brief_id)
    except GroundedBriefError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Brief ID: {response.brief_id}")
    print(f"Source type: {response.source_type}")
    if response.source_url:
        print(f"Source URL: {response.source_url}")
    print(f"Detected source language: {response.source_language}")
    print(f"Target language: {response.target_language}")
    print(f"Model: {response.model_provider}/{response.model_name}")
    print(f"Status: {response.status}")
    if response.warnings:
        print("\nWarnings:")
        for warning in response.warnings:
            print(f"- {warning}")

    print("\nSource excerpt:")
    print(response.source_text_excerpt)

    print("\nGenerated grounded brief:")
    _print_json_like(response.brief)


def _print_json_like(value: dict[str, Any]) -> None:
    for key, item in value.items():
        print(f"\n{key.replace('_', ' ').title()}:")
        if isinstance(item, list):
            for entry in item:
                print(f"- {entry}")
        else:
            print(textwrap.fill(str(item), width=100))


if __name__ == "__main__":
    main()
