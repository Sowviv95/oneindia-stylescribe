"""Run inventory and DOCX extraction for the Oneindia newsroom corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.newsroom_corpus_preparation_service import (  # noqa: E402
    DEFAULT_ARTICLES_JSONL,
    DEFAULT_CLASSIFIED_DIR,
    DEFAULT_CLEANED_DIR,
    PreparationPathConfig,
    run_newsroom_corpus_preparation,
)
from backend.app.services.newsroom_corpus_service import (  # noqa: E402
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_RAW_DIR,
    DEFAULT_REJECTED_DIR,
    DEFAULT_REPORTS_DIR,
    SHORT_WORD_THRESHOLD,
    CorpusPathConfig,
    run_newsroom_corpus_extraction,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inventory and extract the Oneindia Tamil newsroom DOCX corpus."
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--extracted-dir", type=Path, default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument("--cleaned-dir", type=Path, default=DEFAULT_CLEANED_DIR)
    parser.add_argument("--rejected-dir", type=Path, default=DEFAULT_REJECTED_DIR)
    parser.add_argument("--classified-dir", type=Path, default=DEFAULT_CLASSIFIED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument(
        "--mode",
        choices=[
            "inventory",
            "extract",
            "profile",
            "duplicates",
            "clean",
            "classify",
            "prepare",
            "full-run",
        ],
        default="extract",
        help=(
            "inventory/extract operate on DOCX files; profile/duplicates/clean/"
            "classify/prepare use extracted articles.jsonl; full-run does both."
        ),
    )
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--short-word-threshold",
        type=int,
        default=SHORT_WORD_THRESHOLD,
        help="Word count below which extracted documents are reported as very_short.",
    )
    args = parser.parse_args()

    extraction_paths = CorpusPathConfig(
        raw_dir=args.raw_dir,
        extracted_dir=args.extracted_dir,
        rejected_dir=args.rejected_dir,
        reports_dir=args.reports_dir,
    )
    preparation_paths = PreparationPathConfig(
        articles_jsonl=args.extracted_dir / DEFAULT_ARTICLES_JSONL.name,
        cleaned_dir=args.cleaned_dir,
        rejected_dir=args.rejected_dir,
        classified_dir=args.classified_dir,
        reports_dir=args.reports_dir,
    )
    extraction_mode = args.mode in {"inventory", "extract", "full-run"}
    preparation_mode = args.mode in {
        "profile",
        "duplicates",
        "clean",
        "classify",
        "prepare",
        "full-run",
    }

    if args.inventory_only:
        extraction_mode = True
        preparation_mode = False

    if extraction_mode:
        extraction_result = run_newsroom_corpus_extraction(
            paths=extraction_paths,
            inventory_only=args.inventory_only or args.mode == "inventory",
            short_word_threshold=args.short_word_threshold,
        )
        _print_summary("extraction", extraction_result.summary)
        _print_outputs(extraction_result.output_paths)

    if preparation_mode:
        preparation_result = run_newsroom_corpus_preparation(
            paths=preparation_paths,
            mode=args.mode,
        )
        _print_summary("preparation", preparation_result.summary)
        _print_outputs(preparation_result.output_paths)


def _print_summary(label: str, summary: dict[str, int]) -> None:
    print(f"{label}:")
    for key, value in summary.items():
        print(f"{key}: {value}")


def _print_outputs(output_paths: dict[str, Path]) -> None:
    print("outputs:")
    for key, path in output_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
