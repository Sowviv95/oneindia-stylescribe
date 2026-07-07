"""Run Google Signals calibration workflow outputs from DOCX source cases."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.models.pasted_text_workflow_models import (  # noqa: E402
    PastedTextWorkflowResponse,
)
from backend.app.services.docx_extractor import (  # noqa: E402
    DocxExtractionError,
    extract_docx_text,
)
from backend.app.services.pasted_text_workflow_service import (  # noqa: E402
    run_pasted_text_to_draft_workflow,
)

DEFAULT_INPUT_DIR = Path("manual_tests/google_signals_calibration/source_docs")
DEFAULT_OUTPUT_DIR = Path("manual_tests/google_signals_calibration/inputs")
DEFAULT_AUTHOR_ID = "v_vasanthi"
DEFAULT_DESIRED_WORD_COUNT = 600
DEFAULT_WORKFLOW_MODE = "standard"
DEFAULT_ARTICLE_TYPE = "news"
DEFAULT_TARGET_LANGUAGE = "ta"
DEFAULT_AUTHOR_INSTRUCTION = "Write this as a Tamil news article for Oneindia readers."
DEFAULT_TONE_OVERRIDE = "clear, engaging and factual"

BatchStatus = Literal["created", "overwritten", "skipped", "failed"]
WorkflowRunner = Callable[..., PastedTextWorkflowResponse]


@dataclass(frozen=True)
class BatchResult:
    source_path: Path
    output_path: Path
    status: BatchStatus
    generated_headline: str | None = None
    google_signals_score: int | None = None
    message: str = ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run StyleScribe pasted-text workflows for DOCX Google Signals "
            "calibration cases."
        )
    )
    parser.add_argument(
        "--source-dir",
        "--input-dir",
        dest="source_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--author-id", default=DEFAULT_AUTHOR_ID)
    parser.add_argument("--desired-word-count", type=int, default=600)
    parser.add_argument("--workflow-mode", default=DEFAULT_WORKFLOW_MODE)
    parser.add_argument("--article-type", default=DEFAULT_ARTICLE_TYPE)
    parser.add_argument("--target-language", default=DEFAULT_TARGET_LANGUAGE)
    parser.add_argument("--author-instruction", default=DEFAULT_AUTHOR_INSTRUCTION)
    parser.add_argument("--tone-override", default=DEFAULT_TONE_OVERRIDE)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results = run_docx_batch(
        input_dir=args.source_dir,
        output_dir=args.output_dir,
        author_id=args.author_id,
        desired_word_count=args.desired_word_count,
        workflow_mode=args.workflow_mode,
        article_type=args.article_type,
        target_language=args.target_language,
        author_instruction=args.author_instruction,
        tone_override=args.tone_override,
        overwrite=args.overwrite,
        progress_callback=print,
    )
    print(_summary_line(results))


def run_docx_batch(
    *,
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    author_id: str = DEFAULT_AUTHOR_ID,
    desired_word_count: int = DEFAULT_DESIRED_WORD_COUNT,
    workflow_mode: str = DEFAULT_WORKFLOW_MODE,
    article_type: str = DEFAULT_ARTICLE_TYPE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    author_instruction: str | None = DEFAULT_AUTHOR_INSTRUCTION,
    tone_override: str | None = DEFAULT_TONE_OVERRIDE,
    overwrite: bool = False,
    workflow_runner: WorkflowRunner = run_pasted_text_to_draft_workflow,
    progress_callback: Callable[[str], None] | None = None,
) -> list[BatchResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[BatchResult] = []
    for docx_path in docx_paths(input_dir):
        if progress_callback is not None:
            progress_callback(f"Processing {docx_path}")
        result = run_docx_case(
            docx_path=docx_path,
            output_dir=output_dir,
            author_id=author_id,
            desired_word_count=desired_word_count,
            workflow_mode=workflow_mode,
            article_type=article_type,
            target_language=target_language,
            author_instruction=author_instruction,
            tone_override=tone_override,
            overwrite=overwrite,
            workflow_runner=workflow_runner,
        )
        if progress_callback is not None:
            progress_callback(_result_line(result))
        results.append(result)
    return results


def run_docx_case(
    *,
    docx_path: Path,
    output_dir: Path,
    author_id: str = DEFAULT_AUTHOR_ID,
    desired_word_count: int = DEFAULT_DESIRED_WORD_COUNT,
    workflow_mode: str = DEFAULT_WORKFLOW_MODE,
    article_type: str = DEFAULT_ARTICLE_TYPE,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    author_instruction: str | None = DEFAULT_AUTHOR_INSTRUCTION,
    tone_override: str | None = DEFAULT_TONE_OVERRIDE,
    overwrite: bool = False,
    workflow_runner: WorkflowRunner = run_pasted_text_to_draft_workflow,
) -> BatchResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / workflow_json_filename_for_docx(docx_path)
    output_exists = output_path.exists()
    if output_exists and not overwrite:
        return BatchResult(
            source_path=docx_path,
            output_path=output_path,
            status="skipped",
            message="output exists; pass --overwrite to replace",
        )

    try:
        extracted = extract_docx_text(docx_path)
    except DocxExtractionError as exc:
        return BatchResult(
            source_path=docx_path,
            output_path=output_path,
            status="failed",
            message=str(exc),
        )
    if not extracted.text.strip():
        return BatchResult(
            source_path=docx_path,
            output_path=output_path,
            status="failed",
            message="empty extracted text",
        )

    try:
        response = workflow_runner(
            author_id=author_id,
            source_text=extracted.text,
            author_instruction=author_instruction,
            target_language=target_language,
            article_type=article_type,
            desired_word_count=desired_word_count,
            tone_override=tone_override,
            run_grounding_evaluation=True,
            run_auto_revision=True,
            run_final_evaluation=True,
            export_review=False,
            workflow_mode=workflow_mode,
        )
    except Exception as exc:  # noqa: BLE001
        return BatchResult(
            source_path=docx_path,
            output_path=output_path,
            status="failed",
            message=f"workflow failed: {exc}",
        )

    payload = _workflow_payload(response)
    payload["case_name"] = docx_path.stem
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return BatchResult(
        source_path=docx_path,
        output_path=output_path,
        status="overwritten" if output_exists and overwrite else "created",
        generated_headline=response.generated_headline,
        google_signals_score=response.google_signals_score,
    )


def workflow_json_filename_for_docx(docx_path: Path) -> str:
    return f"{docx_path.stem}.json"


def docx_paths(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        return []
    return sorted(
        path
        for path in input_dir.glob("*.docx")
        if not path.name.startswith("~$")
        and path.name != "00_google_signals_test_case_index.docx"
    )


def _workflow_payload(response: PastedTextWorkflowResponse) -> dict[str, Any]:
    payload = response.model_dump(mode="json")
    payload["workflow_completed"] = response.status == "completed"
    payload["openai_model"] = os.getenv("OPENAI_MODEL")
    payload["final_grounding_score"] = (
        response.final_evaluation_summary.grounding_score
        if response.final_evaluation_summary
        else None
    )
    return payload


def _result_line(result: BatchResult) -> str:
    score = (
        str(result.google_signals_score)
        if result.google_signals_score is not None
        else "not available"
    )
    headline = result.generated_headline or "not available"
    message = f" | message={result.message}" if result.message else ""
    return (
        f"file={result.source_path} | status={result.status} | "
        f"generated_headline={headline} | google_signals_score={score} | "
        f"output={result.output_path}{message}"
    )


def _summary_line(results: list[BatchResult]) -> str:
    created = sum(1 for item in results if item.status == "created")
    overwritten = sum(1 for item in results if item.status == "overwritten")
    skipped = sum(1 for item in results if item.status == "skipped")
    failed = sum(1 for item in results if item.status == "failed")
    return (
        f"summary: files_seen={len(results)} | files_created={created} "
        f"| files_overwritten={overwritten} | files_skipped={skipped} "
        f"| files_failed={failed}"
    )


if __name__ == "__main__":
    main()
