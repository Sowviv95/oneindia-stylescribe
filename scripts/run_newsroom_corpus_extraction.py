"""Run inventory and DOCX extraction for the Oneindia newsroom corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    parser.add_argument("--rejected-dir", type=Path, default=DEFAULT_REJECTED_DIR)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument(
        "--short-word-threshold",
        type=int,
        default=SHORT_WORD_THRESHOLD,
        help="Word count below which extracted documents are reported as very_short.",
    )
    args = parser.parse_args()

    result = run_newsroom_corpus_extraction(
        paths=CorpusPathConfig(
            raw_dir=args.raw_dir,
            extracted_dir=args.extracted_dir,
            rejected_dir=args.rejected_dir,
            reports_dir=args.reports_dir,
        ),
        inventory_only=args.inventory_only,
        short_word_threshold=args.short_word_threshold,
    )

    for key, value in result.summary.items():
        print(f"{key}: {value}")
    print("outputs:")
    for key, path in result.output_paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
