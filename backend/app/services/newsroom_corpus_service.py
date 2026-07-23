"""Inventory and DOCX extraction foundation for the newsroom corpus."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from backend.app.services.docx_extractor import (
    DocxExtractionError,
    ExtractedDocx,
    extract_docx_text,
)

DEFAULT_RAW_DIR = Path("data/newsroom_corpus/00_raw")
DEFAULT_EXTRACTED_DIR = Path("data/newsroom_corpus/01_extracted")
DEFAULT_REJECTED_DIR = Path("data/newsroom_corpus/03_rejected")
DEFAULT_REPORTS_DIR = Path("data/newsroom_corpus/reports")
SHORT_WORD_THRESHOLD = 80


@dataclass(frozen=True)
class CorpusPathConfig:
    raw_dir: Path = DEFAULT_RAW_DIR
    extracted_dir: Path = DEFAULT_EXTRACTED_DIR
    rejected_dir: Path = DEFAULT_REJECTED_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR


@dataclass(frozen=True)
class InventoryRecord:
    article_id: str
    author_id: str
    source_filename: str
    relative_source_path: str
    file_size_bytes: int
    file_sha256: str


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    author_id: str
    source_filename: str
    relative_source_path: str
    file_size_bytes: int
    file_sha256: str
    text_sha256: str
    extraction_status: str
    extraction_warnings: list[str]
    headline: str | None
    subheadline: str | None
    body_text: str
    paragraph_sequence: list[str]
    total_word_count: int
    char_count: int


@dataclass(frozen=True)
class RejectionRecord:
    article_id: str
    author_id: str
    source_filename: str
    relative_source_path: str
    file_size_bytes: int
    file_sha256: str
    extraction_status: str
    extraction_warnings: list[str]
    total_word_count: int | None = None
    text_sha256: str | None = None


@dataclass(frozen=True)
class DuplicateGroup:
    duplicate_type: str
    digest: str
    duplicate_count: int
    article_ids: list[str]
    relative_source_paths: list[str]


@dataclass(frozen=True)
class CorpusExtractionResult:
    inventory_records: list[InventoryRecord]
    article_records: list[ArticleRecord]
    rejection_records: list[RejectionRecord]
    duplicate_file_groups: list[DuplicateGroup]
    duplicate_text_groups: list[DuplicateGroup]
    output_paths: dict[str, Path]
    summary: dict[str, int]


def run_newsroom_corpus_extraction(
    *,
    paths: CorpusPathConfig | None = None,
    inventory_only: bool = False,
    short_word_threshold: int = SHORT_WORD_THRESHOLD,
) -> CorpusExtractionResult:
    """Create repeatable corpus inventory, extraction artifacts and reports."""

    paths = paths or CorpusPathConfig()
    inventory_records = build_inventory(paths.raw_dir)
    duplicate_file_groups = _duplicate_groups(
        digest_by_record={
            record.article_id: record.file_sha256 for record in inventory_records
        },
        path_by_record={
            record.article_id: record.relative_source_path
            for record in inventory_records
        },
        duplicate_type="file_sha256",
    )

    article_records: list[ArticleRecord] = []
    rejection_records: list[RejectionRecord] = []
    duplicate_text_groups: list[DuplicateGroup] = []
    if not inventory_only:
        for inventory_record in inventory_records:
            docx_path = paths.raw_dir / inventory_record.relative_source_path
            try:
                extracted = extract_docx_text(docx_path)
            except DocxExtractionError as exc:
                rejection_records.append(
                    _failed_rejection(inventory_record, str(exc))
                )
                continue

            article_record = _article_record(
                inventory_record,
                extracted,
                short_word_threshold=short_word_threshold,
            )
            article_records.append(article_record)
            if article_record.extraction_status in {"empty", "very_short"}:
                rejection_records.append(_rejection_from_article(article_record))

        duplicate_text_groups = _duplicate_groups(
            digest_by_record={
                record.article_id: record.text_sha256
                for record in article_records
                if record.text_sha256
            },
            path_by_record={
                record.article_id: record.relative_source_path
                for record in article_records
            },
            duplicate_type="text_sha256",
        )

    output_paths = write_corpus_outputs(
        paths=paths,
        inventory_records=inventory_records,
        article_records=article_records,
        rejection_records=rejection_records,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_text_groups=duplicate_text_groups,
        inventory_only=inventory_only,
    )
    summary = _summary(
        inventory_records=inventory_records,
        article_records=article_records,
        rejection_records=rejection_records,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_text_groups=duplicate_text_groups,
        inventory_only=inventory_only,
    )
    return CorpusExtractionResult(
        inventory_records=inventory_records,
        article_records=article_records,
        rejection_records=rejection_records,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_text_groups=duplicate_text_groups,
        output_paths=output_paths,
        summary=summary,
    )


def build_inventory(raw_dir: Path) -> list[InventoryRecord]:
    """Return deterministic inventory records for all non-temporary DOCX files."""

    if not raw_dir.exists():
        return []
    records: list[InventoryRecord] = []
    for docx_path in sorted(raw_dir.rglob("*.docx")):
        if _is_temporary_word_file(docx_path):
            continue
        relative_source_path = docx_path.relative_to(raw_dir).as_posix()
        author_id = Path(relative_source_path).parts[0]
        file_bytes = docx_path.read_bytes()
        records.append(
            InventoryRecord(
                article_id=article_id_for_source(author_id, relative_source_path),
                author_id=author_id,
                source_filename=docx_path.name,
                relative_source_path=relative_source_path,
                file_size_bytes=len(file_bytes),
                file_sha256=sha256(file_bytes).hexdigest(),
            )
        )
    return records


def article_id_for_source(author_id: str, relative_source_path: str) -> str:
    """Create a stable article ID without depending on extracted text."""

    digest = sha256(f"{author_id}:{relative_source_path}".encode()).hexdigest()
    return f"{author_id}:{digest[:16]}"


def write_corpus_outputs(
    *,
    paths: CorpusPathConfig,
    inventory_records: list[InventoryRecord],
    article_records: list[ArticleRecord],
    rejection_records: list[RejectionRecord],
    duplicate_file_groups: list[DuplicateGroup],
    duplicate_text_groups: list[DuplicateGroup],
    inventory_only: bool,
) -> dict[str, Path]:
    paths.extracted_dir.mkdir(parents=True, exist_ok=True)
    paths.rejected_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "articles_jsonl": paths.extracted_dir / "articles.jsonl",
        "rejections_jsonl": paths.rejected_dir / "extraction_rejections.jsonl",
        "inventory_csv": paths.reports_dir / "raw_corpus_inventory.csv",
        "duplicate_files_csv": paths.reports_dir / "exact_duplicate_files.csv",
        "duplicate_text_csv": paths.reports_dir / "exact_duplicate_text.csv",
        "summary_markdown": paths.reports_dir / "extraction_summary.md",
        "author_counts_csv": paths.reports_dir / "author_counts.csv",
    }

    _write_jsonl(output_paths["articles_jsonl"], article_records)
    _write_jsonl(output_paths["rejections_jsonl"], rejection_records)
    _write_inventory_csv(output_paths["inventory_csv"], inventory_records)
    _write_duplicate_csv(output_paths["duplicate_files_csv"], duplicate_file_groups)
    _write_duplicate_csv(output_paths["duplicate_text_csv"], duplicate_text_groups)
    _write_author_counts_csv(
        output_paths["author_counts_csv"],
        inventory_records,
        article_records,
        rejection_records,
    )
    _write_summary_markdown(
        output_paths["summary_markdown"],
        inventory_records=inventory_records,
        article_records=article_records,
        rejection_records=rejection_records,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_text_groups=duplicate_text_groups,
        inventory_only=inventory_only,
    )
    return output_paths


def _article_record(
    inventory_record: InventoryRecord,
    extracted: ExtractedDocx,
    *,
    short_word_threshold: int,
) -> ArticleRecord:
    warnings: list[str] = []
    status = "extracted"
    if not extracted.text.strip():
        status = "empty"
        warnings.append("empty_document")
    elif extracted.word_count < short_word_threshold:
        status = "very_short"
        warnings.append(f"very_short_document_under_{short_word_threshold}_words")
    if extracted.headline is None:
        warnings.append("missing_headline")
    if extracted.subheadline is None:
        warnings.append("missing_subheadline")

    return ArticleRecord(
        article_id=inventory_record.article_id,
        author_id=inventory_record.author_id,
        source_filename=inventory_record.source_filename,
        relative_source_path=inventory_record.relative_source_path,
        file_size_bytes=inventory_record.file_size_bytes,
        file_sha256=inventory_record.file_sha256,
        text_sha256=(
            sha256(extracted.text.encode()).hexdigest() if extracted.text else ""
        ),
        extraction_status=status,
        extraction_warnings=warnings,
        headline=extracted.headline,
        subheadline=extracted.subheadline,
        body_text=extracted.body_text,
        paragraph_sequence=extracted.paragraphs,
        total_word_count=extracted.word_count,
        char_count=extracted.char_count,
    )


def _failed_rejection(
    inventory_record: InventoryRecord,
    warning: str,
) -> RejectionRecord:
    return RejectionRecord(
        article_id=inventory_record.article_id,
        author_id=inventory_record.author_id,
        source_filename=inventory_record.source_filename,
        relative_source_path=inventory_record.relative_source_path,
        file_size_bytes=inventory_record.file_size_bytes,
        file_sha256=inventory_record.file_sha256,
        extraction_status="failed",
        extraction_warnings=[warning],
    )


def _rejection_from_article(article_record: ArticleRecord) -> RejectionRecord:
    return RejectionRecord(
        article_id=article_record.article_id,
        author_id=article_record.author_id,
        source_filename=article_record.source_filename,
        relative_source_path=article_record.relative_source_path,
        file_size_bytes=article_record.file_size_bytes,
        file_sha256=article_record.file_sha256,
        extraction_status=article_record.extraction_status,
        extraction_warnings=article_record.extraction_warnings,
        total_word_count=article_record.total_word_count,
        text_sha256=article_record.text_sha256,
    )


def _duplicate_groups(
    *,
    digest_by_record: dict[str, str],
    path_by_record: dict[str, str],
    duplicate_type: str,
) -> list[DuplicateGroup]:
    article_ids_by_digest: dict[str, list[str]] = defaultdict(list)
    for article_id, digest in digest_by_record.items():
        if digest:
            article_ids_by_digest[digest].append(article_id)

    groups: list[DuplicateGroup] = []
    for digest, article_ids in sorted(article_ids_by_digest.items()):
        if len(article_ids) < 2:
            continue
        sorted_article_ids = sorted(article_ids)
        groups.append(
            DuplicateGroup(
                duplicate_type=duplicate_type,
                digest=digest,
                duplicate_count=len(sorted_article_ids),
                article_ids=sorted_article_ids,
                relative_source_paths=[
                    path_by_record[article_id] for article_id in sorted_article_ids
                ],
            )
        )
    return groups


def _write_jsonl(
    path: Path,
    records: list[ArticleRecord] | list[RejectionRecord],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _write_inventory_csv(path: Path, records: list[InventoryRecord]) -> None:
    fieldnames = [
        "article_id",
        "author_id",
        "source_filename",
        "relative_source_path",
        "file_size_bytes",
        "file_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def _write_duplicate_csv(path: Path, groups: list[DuplicateGroup]) -> None:
    fieldnames = [
        "duplicate_type",
        "digest",
        "duplicate_count",
        "article_ids",
        "relative_source_paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            writer.writerow(
                {
                    "duplicate_type": group.duplicate_type,
                    "digest": group.digest,
                    "duplicate_count": group.duplicate_count,
                    "article_ids": "|".join(group.article_ids),
                    "relative_source_paths": "|".join(group.relative_source_paths),
                }
            )


def _write_author_counts_csv(
    path: Path,
    inventory_records: list[InventoryRecord],
    article_records: list[ArticleRecord],
    rejection_records: list[RejectionRecord],
) -> None:
    authors = sorted({record.author_id for record in inventory_records})
    inventory_counts = Counter(record.author_id for record in inventory_records)
    extracted_counts = Counter(
        record.author_id
        for record in article_records
        if record.extraction_status == "extracted"
    )
    failed_counts = Counter(
        record.author_id
        for record in rejection_records
        if record.extraction_status == "failed"
    )
    empty_counts = Counter(
        record.author_id
        for record in rejection_records
        if record.extraction_status == "empty"
    )
    very_short_counts = Counter(
        record.author_id
        for record in rejection_records
        if record.extraction_status == "very_short"
    )
    fieldnames = [
        "author_id",
        "raw_docx_count",
        "extracted_count",
        "failed_count",
        "empty_count",
        "very_short_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for author_id in authors:
            writer.writerow(
                {
                    "author_id": author_id,
                    "raw_docx_count": inventory_counts[author_id],
                    "extracted_count": extracted_counts[author_id],
                    "failed_count": failed_counts[author_id],
                    "empty_count": empty_counts[author_id],
                    "very_short_count": very_short_counts[author_id],
                }
            )


def _write_summary_markdown(
    path: Path,
    *,
    inventory_records: list[InventoryRecord],
    article_records: list[ArticleRecord],
    rejection_records: list[RejectionRecord],
    duplicate_file_groups: list[DuplicateGroup],
    duplicate_text_groups: list[DuplicateGroup],
    inventory_only: bool,
) -> None:
    summary = _summary(
        inventory_records=inventory_records,
        article_records=article_records,
        rejection_records=rejection_records,
        duplicate_file_groups=duplicate_file_groups,
        duplicate_text_groups=duplicate_text_groups,
        inventory_only=inventory_only,
    )
    author_counts = Counter(record.author_id for record in inventory_records)
    lines = [
        "# Newsroom Corpus Extraction Summary",
        "",
        f"- mode: {'inventory_only' if inventory_only else 'full_extraction'}",
        f"- raw_docx_count: {summary['raw_docx_count']}",
        f"- article_records_written: {summary['article_records_written']}",
        f"- rejection_records_written: {summary['rejection_records_written']}",
        f"- failed_count: {summary['failed_count']}",
        f"- empty_count: {summary['empty_count']}",
        f"- very_short_count: {summary['very_short_count']}",
        f"- duplicate_file_groups: {summary['duplicate_file_groups']}",
        f"- duplicate_file_records: {summary['duplicate_file_records']}",
        f"- duplicate_text_groups: {summary['duplicate_text_groups']}",
        f"- duplicate_text_records: {summary['duplicate_text_records']}",
        "",
        "## Author Counts",
        "",
        "| author_id | raw_docx_count |",
        "| --- | ---: |",
    ]
    for author_id, count in sorted(author_counts.items()):
        lines.append(f"| {author_id} | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _summary(
    *,
    inventory_records: list[InventoryRecord],
    article_records: list[ArticleRecord],
    rejection_records: list[RejectionRecord],
    duplicate_file_groups: list[DuplicateGroup],
    duplicate_text_groups: list[DuplicateGroup],
    inventory_only: bool,
) -> dict[str, int]:
    rejection_status_counts = Counter(
        record.extraction_status for record in rejection_records
    )
    return {
        "inventory_only": int(inventory_only),
        "raw_docx_count": len(inventory_records),
        "article_records_written": len(article_records),
        "rejection_records_written": len(rejection_records),
        "failed_count": rejection_status_counts["failed"],
        "empty_count": rejection_status_counts["empty"],
        "very_short_count": rejection_status_counts["very_short"],
        "duplicate_file_groups": len(duplicate_file_groups),
        "duplicate_file_records": sum(
            group.duplicate_count for group in duplicate_file_groups
        ),
        "duplicate_text_groups": len(duplicate_text_groups),
        "duplicate_text_records": sum(
            group.duplicate_count for group in duplicate_text_groups
        ),
    }


def _is_temporary_word_file(path: Path) -> bool:
    return path.name.startswith("~$")
