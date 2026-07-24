"""Build deterministic editorial review artifacts for newsroom prompt comparisons."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.newsroom_benchmark_review_service import (  # noqa: E402
    ReviewModeSpec,
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
    parser.add_argument(
        "--extra-mode",
        action="append",
        default=[],
        help="Extra mode as mode_key|Label|directory_name. May be repeated.",
    )
    args = parser.parse_args(argv)
    paths = generate_editorial_review_pack(
        ReviewPackConfig(
            comparison_root=Path(args.comparison_root),
            legacy_dir_name=args.legacy_dir_name,
            newsroom_dir_name=args.newsroom_dir_name,
            output_dir_name=args.output_dir_name,
            extra_modes=tuple(_parse_extra_mode(value) for value in args.extra_mode),
        )
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


def _parse_extra_mode(value: str) -> ReviewModeSpec:
    parts = value.split("|", 2)
    if len(parts) != 3 or not all(part.strip() for part in parts):
        raise SystemExit("--extra-mode must use mode_key|Label|directory_name")
    return ReviewModeSpec(
        mode_key=parts[0].strip(),
        label=parts[1].strip(),
        dir_name=parts[2].strip(),
    )


if __name__ == "__main__":
    raise SystemExit(main())
