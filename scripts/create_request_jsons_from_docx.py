"""Create StyleScribe manual workflow request JSON files from DOCX inputs."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.services.docx_extractor import (  # noqa: E402
    DocxExtractionError,
    extract_docx_text,
)

DEFAULT_INPUT_DIR = Path("manual_tests/inputs/docx")
DEFAULT_OUTPUT_DIR = Path("manual_tests/inputs/json")
DEFAULT_AUTHOR_ID = "v_vasanthi"
DEFAULT_DESIRED_WORD_COUNT = 600
DEFAULT_WORKFLOW_MODE = "standard"

ConversionStatus = Literal["created", "skipped", "overwritten", "failed"]


@dataclass(frozen=True)
class ConversionResult:
    source_path: Path
    output_path: Path | None
    char_count: int
    word_count: int
    status: ConversionStatus
    message: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DOCX article inputs into StyleScribe request JSON files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--author-id", default=DEFAULT_AUTHOR_ID)
    parser.add_argument(
        "--desired-word-count",
        type=int,
        default=DEFAULT_DESIRED_WORD_COUNT,
    )
    parser.add_argument("--workflow-mode", default=DEFAULT_WORKFLOW_MODE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = create_request_jsons(
        input_dir=args.input_dir,
        input_file=args.input_file,
        output_dir=args.output_dir,
        author_id=args.author_id,
        desired_word_count=args.desired_word_count,
        workflow_mode=args.workflow_mode,
        overwrite=args.overwrite,
    )
    for result in results:
        print(_result_line(result))
    print(_summary_line(results))


def create_request_jsons(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    input_file: Path | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    author_id: str = DEFAULT_AUTHOR_ID,
    desired_word_count: int = DEFAULT_DESIRED_WORD_COUNT,
    workflow_mode: str = DEFAULT_WORKFLOW_MODE,
    overwrite: bool = False,
) -> list[ConversionResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docx_paths = _docx_paths(input_dir=input_dir, input_file=input_file)
    return [
        convert_docx_to_request_json(
            docx_path=docx_path,
            output_dir=output_dir,
            author_id=author_id,
            desired_word_count=desired_word_count,
            workflow_mode=workflow_mode,
            overwrite=overwrite,
        )
        for docx_path in docx_paths
    ]


def convert_docx_to_request_json(
    *,
    docx_path: Path,
    output_dir: Path,
    author_id: str = DEFAULT_AUTHOR_ID,
    desired_word_count: int = DEFAULT_DESIRED_WORD_COUNT,
    workflow_mode: str = DEFAULT_WORKFLOW_MODE,
    overwrite: bool = False,
) -> ConversionResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / request_filename_for_docx(docx_path)
    try:
        extracted = extract_docx_text(docx_path)
    except DocxExtractionError as exc:
        return ConversionResult(
            source_path=docx_path,
            output_path=output_path,
            char_count=0,
            word_count=0,
            status="failed",
            message=str(exc),
        )

    if not extracted.text.strip():
        return ConversionResult(
            source_path=docx_path,
            output_path=output_path,
            char_count=0,
            word_count=0,
            status="skipped",
            message="empty extracted text",
        )

    output_exists = output_path.exists()
    if output_exists and not overwrite:
        return ConversionResult(
            source_path=docx_path,
            output_path=output_path,
            char_count=extracted.char_count,
            word_count=_wordish_count(extracted.text),
            status="skipped",
            message="output exists; pass --overwrite to replace",
        )

    payload = {
        "author_id": author_id,
        "source_text": extracted.text,
        "desired_word_count": desired_word_count,
        "workflow_mode": workflow_mode,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return ConversionResult(
        source_path=docx_path,
        output_path=output_path,
        char_count=extracted.char_count,
        word_count=_wordish_count(extracted.text),
        status="overwritten" if output_exists and overwrite else "created",
    )


def request_filename_for_docx(docx_path: Path) -> str:
    return f"request_{slugify(docx_path.stem)}.json"


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_value.lower()).strip("_")
    slug = re.sub(r"_+", "_", slug)
    return slug or "docx_input"


def _docx_paths(*, input_dir: Path, input_file: Path | None) -> list[Path]:
    if input_file is not None:
        return [] if _is_temporary_word_file(input_file) else [input_file]
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.glob("*.docx")
        if not _is_temporary_word_file(path)
    )


def _is_temporary_word_file(path: Path) -> bool:
    return path.name.startswith("~$")


def _wordish_count(text: str) -> int:
    return len(text.split())


def _result_line(result: ConversionResult) -> str:
    output = str(result.output_path) if result.output_path else "not available"
    message = f" | message={result.message}" if result.message else ""
    return (
        f"source={result.source_path} | output={output} | chars={result.char_count} "
        f"| words={result.word_count} | status={result.status}{message}"
    )


def _summary_line(results: list[ConversionResult]) -> str:
    converted = sum(1 for item in results if item.status in {"created", "overwritten"})
    skipped = sum(1 for item in results if item.status == "skipped")
    failed = sum(1 for item in results if item.status == "failed")
    return (
        f"summary: files_seen={len(results)} | files_converted={converted} "
        f"| files_skipped={skipped} | files_failed={failed}"
    )


if __name__ == "__main__":
    main()
