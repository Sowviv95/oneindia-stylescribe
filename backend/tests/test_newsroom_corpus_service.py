import csv
import json
import shutil
from pathlib import Path

from docx import Document

from backend.app.services.newsroom_corpus_service import (
    CorpusPathConfig,
    article_id_for_source,
    build_inventory,
    run_newsroom_corpus_extraction,
)


def test_build_inventory_preserves_provenance_and_ignores_temp_files(
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "00_raw"
    author_dir = raw_dir / "v_vasanthi"
    author_dir.mkdir(parents=True)
    _write_docx(author_dir / "article.docx", ["Headline", "Body text"])
    _write_docx(author_dir / "~$article.docx", ["Temporary"])

    records = build_inventory(raw_dir)

    assert len(records) == 1
    assert records[0].author_id == "v_vasanthi"
    assert records[0].source_filename == "article.docx"
    assert records[0].relative_source_path == "v_vasanthi/article.docx"
    assert records[0].article_id == article_id_for_source(
        "v_vasanthi",
        "v_vasanthi/article.docx",
    )
    assert records[0].file_size_bytes > 0
    assert len(records[0].file_sha256) == 64


def test_run_newsroom_corpus_extraction_writes_records_and_reports(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    author_dir = paths.raw_dir / "v_vasanthi"
    author_dir.mkdir(parents=True)
    _write_docx(
        author_dir / "article.docx",
        ["Headline", "Subheadline", "Body paragraph one", "Body paragraph two"],
    )

    result = run_newsroom_corpus_extraction(paths=paths, short_word_threshold=3)

    article_lines = _read_jsonl(result.output_paths["articles_jsonl"])
    assert result.summary["raw_docx_count"] == 1
    assert result.summary["article_records_written"] == 1
    assert result.summary["rejection_records_written"] == 0
    assert article_lines[0]["author_id"] == "v_vasanthi"
    assert article_lines[0]["source_filename"] == "article.docx"
    assert article_lines[0]["relative_source_path"] == "v_vasanthi/article.docx"
    assert article_lines[0]["headline"] == "Headline"
    assert article_lines[0]["subheadline"] == "Subheadline"
    assert article_lines[0]["body_text"] == "Body paragraph one\nBody paragraph two"
    assert article_lines[0]["paragraph_sequence"] == [
        "Headline",
        "Subheadline",
        "Body paragraph one",
        "Body paragraph two",
    ]
    assert article_lines[0]["total_word_count"] == 8
    assert Path(result.output_paths["summary_markdown"]).exists()


def test_run_newsroom_corpus_extraction_reports_malformed_empty_and_short(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    author_dir = paths.raw_dir / "hema_vandhana"
    author_dir.mkdir(parents=True)
    (author_dir / "malformed.docx").write_bytes(b"not a docx")
    _write_docx(author_dir / "empty.docx", [])
    _write_docx(author_dir / "short.docx", ["Headline", "tiny"])

    result = run_newsroom_corpus_extraction(paths=paths, short_word_threshold=10)

    rejection_lines = _read_jsonl(result.output_paths["rejections_jsonl"])
    statuses = {
        record["source_filename"]: record["extraction_status"]
        for record in rejection_lines
    }
    assert statuses == {
        "empty.docx": "empty",
        "malformed.docx": "failed",
        "short.docx": "very_short",
    }
    assert result.summary["failed_count"] == 1
    assert result.summary["empty_count"] == 1
    assert result.summary["very_short_count"] == 1


def test_run_newsroom_corpus_extraction_reports_exact_duplicates(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    author_dir = paths.raw_dir / "shyamsundar_i"
    author_dir.mkdir(parents=True)
    original = author_dir / "original.docx"
    copied = author_dir / "copied.docx"
    same_text = author_dir / "same-text.docx"
    _write_docx(original, ["Headline", "Subheadline", "Same body"])
    shutil.copyfile(original, copied)
    _write_docx(
        same_text,
        ["Headline", "Subheadline", "Same body"],
        comments="different docx metadata",
    )

    result = run_newsroom_corpus_extraction(paths=paths, short_word_threshold=1)

    assert result.summary["duplicate_file_groups"] == 1
    assert result.summary["duplicate_file_records"] == 2
    assert result.summary["duplicate_text_groups"] == 1
    assert result.summary["duplicate_text_records"] == 3
    duplicate_file_rows = _read_csv(result.output_paths["duplicate_files_csv"])
    duplicate_text_rows = _read_csv(result.output_paths["duplicate_text_csv"])
    assert duplicate_file_rows[0]["duplicate_count"] == "2"
    assert duplicate_text_rows[0]["duplicate_count"] == "3"


def test_inventory_only_mode_skips_text_extraction(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    author_dir = paths.raw_dir / "v_vasanthi"
    author_dir.mkdir(parents=True)
    (author_dir / "malformed.docx").write_bytes(b"not a docx")

    result = run_newsroom_corpus_extraction(paths=paths, inventory_only=True)

    assert result.summary["inventory_only"] == 1
    assert result.summary["raw_docx_count"] == 1
    assert result.summary["article_records_written"] == 0
    assert result.summary["rejection_records_written"] == 0
    assert _read_jsonl(result.output_paths["articles_jsonl"]) == []


def _paths(tmp_path: Path) -> CorpusPathConfig:
    return CorpusPathConfig(
        raw_dir=tmp_path / "00_raw",
        extracted_dir=tmp_path / "01_extracted",
        rejected_dir=tmp_path / "03_rejected",
        reports_dir=tmp_path / "reports",
    )


def _write_docx(
    path: Path,
    paragraphs: list[str],
    comments: str | None = None,
) -> None:
    document = Document()
    if comments is not None:
        document.core_properties.comments = comments
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
