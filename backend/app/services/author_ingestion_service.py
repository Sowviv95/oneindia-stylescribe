"""Deterministic local author sample ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from uuid import uuid4

from backend.app.db.repository import (
    ArticleRecord,
    AuthorRecord,
    IngestionRunRecord,
    StyleScribeRepository,
)
from backend.app.models.ingestion_models import IngestionSummary
from backend.app.services.docx_extractor import DocxExtractionError, extract_docx_text
from backend.app.services.metadata_reader import MetadataRow, read_metadata_rows


@dataclass(frozen=True)
class MetadataMatch:
    row: MetadataRow | None
    warning: str | None = None


def ingest_author_samples(
    author_id: str,
    display_name: str,
    language: str,
    articles_dir: Path,
    metadata_path: Path | None,
    repository: StyleScribeRepository | None = None,
) -> IngestionSummary:
    """Ingest local DOCX author samples and optional Excel metadata."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    started_at = _utc_now()
    run_id = str(uuid4())
    warnings: list[str] = []
    articles_ingested = 0
    articles_failed = 0

    repo.upsert_author(
        AuthorRecord(
            author_id=author_id,
            display_name=display_name,
            language=language,
            created_at=started_at,
            updated_at=started_at,
        )
    )

    docx_paths = sorted(articles_dir.glob("*.docx"))
    if not articles_dir.exists():
        warnings.append(f"Articles directory does not exist: {articles_dir}")
    if not docx_paths:
        warnings.append(f"No DOCX files found in: {articles_dir}")

    metadata_rows = _read_metadata(metadata_path, warnings)
    match_by_filename, unmatched_rows = _build_metadata_filename_index(metadata_rows)
    if unmatched_rows and metadata_rows:
        warnings.append(
            f"{len(unmatched_rows)} metadata rows do not contain filename fields"
        )

    use_sequential_fallback = (
        bool(metadata_rows)
        and len(metadata_rows) == len(docx_paths)
        and len(match_by_filename) < len(docx_paths)
    )
    if use_sequential_fallback:
        warnings.append(
            "Sequential metadata fallback used because DOCX and metadata row "
            "counts match."
        )

    for index, docx_path in enumerate(docx_paths):
        now = _utc_now()
        try:
            extracted = extract_docx_text(docx_path)
            metadata_match = _match_metadata(
                docx_path=docx_path,
                index=index,
                metadata_rows=metadata_rows,
                match_by_filename=match_by_filename,
                use_sequential_fallback=use_sequential_fallback,
            )
            if metadata_match.warning:
                warnings.append(metadata_match.warning)

            metadata = metadata_match.row or {}
            article = ArticleRecord(
                article_id=_article_id(author_id, docx_path),
                author_id=author_id,
                filename=docx_path.name,
                title=metadata.get("title"),
                heading=metadata.get("heading"),
                url=metadata.get("url"),
                category=metadata.get("category"),
                tags=metadata.get("tags"),
                keywords=metadata.get("keywords"),
                meta_description=metadata.get("meta_description"),
                added_date=metadata.get("added_date"),
                content_from_metadata=metadata.get("content"),
                extracted_text=extracted.text,
                text_char_count=extracted.char_count,
                source_path=str(docx_path),
                created_at=now,
                updated_at=now,
            )
            repo.upsert_article(article)
            articles_ingested += 1
        except DocxExtractionError as exc:
            warnings.append(str(exc))
            articles_failed += 1

    completed_at = _utc_now()
    status = "completed" if articles_failed == 0 else "completed_with_errors"
    repo.create_ingestion_run(
        IngestionRunRecord(
            run_id=run_id,
            author_id=author_id,
            status=status,
            articles_seen=len(docx_paths),
            articles_ingested=articles_ingested,
            articles_failed=articles_failed,
            metadata_rows_seen=len(metadata_rows),
            warnings_json=StyleScribeRepository.encode_warnings(warnings),
            started_at=started_at,
            completed_at=completed_at,
        )
    )

    return IngestionSummary(
        run_id=run_id,
        author_id=author_id,
        status=status,
        articles_seen=len(docx_paths),
        articles_ingested=articles_ingested,
        articles_failed=articles_failed,
        metadata_rows_seen=len(metadata_rows),
        warnings=warnings,
    )


def _read_metadata(
    metadata_path: Path | None,
    warnings: list[str],
) -> list[MetadataRow]:
    if metadata_path is None:
        return []
    if not metadata_path.exists():
        warnings.append(f"Metadata file does not exist: {metadata_path}")
        return []
    try:
        return read_metadata_rows(metadata_path)
    except Exception as exc:
        warnings.append(f"Unable to read metadata file: {metadata_path}: {exc}")
        return []


def _build_metadata_filename_index(
    metadata_rows: list[MetadataRow],
) -> tuple[dict[str, MetadataRow], list[MetadataRow]]:
    by_filename: dict[str, MetadataRow] = {}
    unmatched: list[MetadataRow] = []
    for row in metadata_rows:
        filename = row.get("filename")
        if filename:
            by_filename[_normalize_match_text(Path(filename).stem)] = row
        else:
            unmatched.append(row)
    return by_filename, unmatched


def _match_metadata(
    docx_path: Path,
    index: int,
    metadata_rows: list[MetadataRow],
    match_by_filename: dict[str, MetadataRow],
    use_sequential_fallback: bool,
) -> MetadataMatch:
    if not metadata_rows:
        return MetadataMatch(row=None)

    filename_key = _normalize_match_text(docx_path.stem)
    if filename_key in match_by_filename:
        return MetadataMatch(row=match_by_filename[filename_key])

    similar_row = _match_by_title_or_heading(filename_key, metadata_rows)
    if similar_row is not None:
        return MetadataMatch(row=similar_row)

    if use_sequential_fallback:
        return MetadataMatch(row=metadata_rows[index])

    return MetadataMatch(row=None, warning=f"No metadata match for {docx_path.name}")


def _match_by_title_or_heading(
    filename_key: str,
    metadata_rows: list[MetadataRow],
) -> MetadataRow | None:
    best_score = 0.0
    best_row: MetadataRow | None = None
    for row in metadata_rows:
        for key in ("title", "heading"):
            value = row.get(key)
            if not value:
                continue
            score = SequenceMatcher(
                None,
                filename_key,
                _normalize_match_text(value),
            ).ratio()
            if score > best_score:
                best_score = score
                best_row = row
    return best_row if best_score >= 0.72 else None


def _normalize_match_text(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _article_id(author_id: str, path: Path) -> str:
    return f"{author_id}:{path.stem}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
