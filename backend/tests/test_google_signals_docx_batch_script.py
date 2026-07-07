import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from docx import Document

from backend.app.models.pasted_text_workflow_models import (
    PastedTextWorkflowResponse,
    SourceCleanupSummary,
    WorkflowBriefSummary,
    WorkflowDraftSummary,
    WorkflowEvaluationSummary,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "run_google_signals_docx_batch.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docx_batch_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["docx_batch_script"] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_json_filename_for_docx() -> None:
    script = _load_script()

    assert (
        script.workflow_json_filename_for_docx(
            Path("01_strong_source_clear_google_intent.docx")
        )
        == "01_strong_source_clear_google_intent.json"
    )


def test_docx_paths_skips_index_file(tmp_path: Path) -> None:
    script = _load_script()
    _write_docx(tmp_path / "00_google_signals_test_case_index.docx", ["Index"])
    _write_docx(tmp_path / "01_case.docx", ["Case"])

    assert script.docx_paths(tmp_path) == [tmp_path / "01_case.docx"]


def test_docx_case_runs_workflow_and_writes_full_response_json(
    tmp_path: Path,
) -> None:
    script = _load_script()
    docx_path = tmp_path / "01_case.docx"
    output_dir = tmp_path / "inputs"
    _write_docx(docx_path, ["First source paragraph", "Second source paragraph"])
    calls: list[dict[str, object]] = []

    def fake_runner(**kwargs: object) -> PastedTextWorkflowResponse:
        calls.append(kwargs)
        return _workflow_response(google_signals_score=81)

    result = script.run_docx_case(
        docx_path=docx_path,
        output_dir=output_dir,
        workflow_runner=fake_runner,
    )

    output_path = output_dir / "01_case.json"
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert result.status == "created"
    assert result.google_signals_score == 81
    assert result.generated_headline == "Generated headline"
    assert calls[0]["source_text"] == "First source paragraph\nSecond source paragraph"
    assert calls[0]["author_id"] == "v_vasanthi"
    assert calls[0]["run_grounding_evaluation"] is True
    assert calls[0]["run_auto_revision"] is True
    assert calls[0]["run_final_evaluation"] is True
    assert payload["workflow_completed"] is True
    assert payload["google_signals_score"] == 81
    assert payload["case_name"] == "01_case"


def test_docx_case_does_not_overwrite_without_flag(tmp_path: Path) -> None:
    script = _load_script()
    docx_path = tmp_path / "existing.docx"
    output_dir = tmp_path / "inputs"
    output_dir.mkdir()
    output_path = output_dir / "existing.json"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    _write_docx(docx_path, ["Replacement text"])

    result = script.run_docx_case(
        docx_path=docx_path,
        output_dir=output_dir,
        workflow_runner=lambda **kwargs: _workflow_response(google_signals_score=90),
    )

    assert result.status == "skipped"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"existing": True}

    overwritten = script.run_docx_case(
        docx_path=docx_path,
        output_dir=output_dir,
        overwrite=True,
        workflow_runner=lambda **kwargs: _workflow_response(google_signals_score=90),
    )

    assert overwritten.status == "overwritten"
    assert json.loads(output_path.read_text(encoding="utf-8"))[
        "google_signals_score"
    ] == 90


def test_docx_batch_continues_after_workflow_failure(tmp_path: Path) -> None:
    script = _load_script()
    input_dir = tmp_path / "source_docs"
    output_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_docx(input_dir / "01_ok.docx", ["Good source"])
    _write_docx(input_dir / "02_fail.docx", ["Bad source"])

    def fake_runner(**kwargs: object) -> PastedTextWorkflowResponse:
        if kwargs["source_text"] == "Bad source":
            raise RuntimeError("boom")
        return _workflow_response(google_signals_score=76)

    results = script.run_docx_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        workflow_runner=fake_runner,
    )

    assert [result.status for result in results] == ["created", "failed"]
    assert (output_dir / "01_ok.json").exists()
    assert not (output_dir / "02_fail.json").exists()
    assert "workflow failed: boom" in results[1].message


def test_result_line_handles_missing_google_signals() -> None:
    script = _load_script()
    result = script.BatchResult(
        source_path=Path("source.docx"),
        output_path=Path("source.json"),
        status="created",
        generated_headline="Generated headline",
        google_signals_score=None,
    )

    line = script._result_line(result)

    assert "generated_headline=Generated headline" in line
    assert "google_signals_score=not available" in line


def _workflow_response(google_signals_score: int | None) -> PastedTextWorkflowResponse:
    return PastedTextWorkflowResponse(
        workflow_id="workflow-1",
        status="completed",
        author_id="v_vasanthi",
        brief_id="brief-1",
        draft_id="draft-1",
        evaluation_id="evaluation-1",
        source_cleanup=SourceCleanupSummary(
            original_char_count=100,
            cleaned_char_count=90,
            removed_line_count=0,
            warnings=[],
        ),
        brief_summary=WorkflowBriefSummary(
            topic="Topic",
            one_line_summary="Summary",
            confirmed_facts=["Fact"],
            claims_to_avoid=[],
        ),
        draft_summary=WorkflowDraftSummary(
            headline="Generated headline",
            subheadline="Generated subheadline",
            seo_title="SEO",
            tags=[],
        ),
        evaluation_summary=WorkflowEvaluationSummary(
            grounding_score=91,
            claim_safety_score=90,
            overall_risk="low",
            editorial_readiness="safe_to_review",
        ),
        final_evaluation_summary=WorkflowEvaluationSummary(
            grounding_score=92,
            claim_safety_score=91,
            overall_risk="low",
            editorial_readiness="safe_to_review",
        ),
        generated_headline="Generated headline",
        google_signals_available=True,
        google_signals_score=google_signals_score,
        google_signals_version="google_signals_v1",
        export_paths=[],
        warnings=[],
    )


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
