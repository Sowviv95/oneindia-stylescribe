"""Build deterministic editorial review artifacts for newsroom prompt comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.newsroom_benchmark_review_service import (  # noqa: E402
    ReviewPackConfig,
    generate_editorial_review_pack,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-root", required=True)
    parser.add_argument("--legacy-dir-name", default="gemini_3_5_flash")
    parser.add_argument(
        "--newsroom-dir-name",
        default="newsroom_v1_gemini_gemini_3_5_flash",
    )
    parser.add_argument("--output-dir-name", default="editorial_review_pack")
    args = parser.parse_args(argv)
    paths = generate_editorial_review_pack(
        ReviewPackConfig(
            comparison_root=Path(args.comparison_root),
            legacy_dir_name=args.legacy_dir_name,
            newsroom_dir_name=args.newsroom_dir_name,
            output_dir_name=args.output_dir_name,
        )
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
