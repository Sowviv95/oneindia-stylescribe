import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

from docx import Document

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "create_request_jsons_from_docx.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("docx_request_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["docx_request_script"] = module
    spec.loader.exec_module(module)
    return module


def test_slug_filename_generation() -> None:
    script = _load_script()

    assert (
        script.request_filename_for_docx(Path("Visa Processing Review.docx"))
        == "request_visa_processing_review.json"
    )
    assert (
        script.request_filename_for_docx(
            Path("Coastal Districts Prepare For Cyclone Watch.docx")
        )
        == "request_coastal_districts_prepare_for_cyclone_watch.json"
    )


def test_creates_expected_request_shape(tmp_path: Path) -> None:
    script = _load_script()
    docx_path = tmp_path / "Visa Processing Review.docx"
    output_dir = tmp_path / "json"
    _write_docx(docx_path, ["First paragraph", "Second paragraph"])

    result = script.convert_docx_to_request_json(
        docx_path=docx_path,
        output_dir=output_dir,
    )

    assert result.status == "created"
    payload = json.loads(
        (output_dir / "request_visa_processing_review.json").read_text()
    )
    assert payload == {
        "author_id": "v_vasanthi",
        "source_text": "First paragraph\nSecond paragraph",
        "desired_word_count": 600,
        "workflow_mode": "standard",
    }


def test_existing_output_is_not_overwritten_without_flag(tmp_path: Path) -> None:
    script = _load_script()
    docx_path = tmp_path / "Tamil Nadu Rain Advisory.docx"
    output_dir = tmp_path / "json"
    output_dir.mkdir()
    output_path = output_dir / "request_tamil_nadu_rain_advisory.json"
    output_path.write_text('{"existing": true}\n', encoding="utf-8")
    _write_docx(docx_path, ["Replacement text"])

    result = script.convert_docx_to_request_json(
        docx_path=docx_path,
        output_dir=output_dir,
    )

    assert result.status == "skipped"
    assert json.loads(output_path.read_text(encoding="utf-8")) == {"existing": True}

    overwritten = script.convert_docx_to_request_json(
        docx_path=docx_path,
        output_dir=output_dir,
        overwrite=True,
    )

    assert overwritten.status == "overwritten"
    assert json.loads(output_path.read_text(encoding="utf-8"))["source_text"] == (
        "Replacement text"
    )


def test_empty_docx_is_skipped(tmp_path: Path) -> None:
    script = _load_script()
    docx_path = tmp_path / "Empty.docx"
    output_dir = tmp_path / "json"
    _write_docx(docx_path, ["   "])

    result = script.convert_docx_to_request_json(
        docx_path=docx_path,
        output_dir=output_dir,
    )

    assert result.status == "skipped"
    assert result.message == "empty extracted text"
    assert not (output_dir / "request_empty.json").exists()


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
