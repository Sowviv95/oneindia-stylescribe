"""Repository functions for StyleScribe SQLite storage."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from backend.app.db.connection import get_connection

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


@dataclass(frozen=True)
class AuthorRecord:
    author_id: str
    display_name: str
    language: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ArticleRecord:
    article_id: str
    author_id: str
    filename: str
    title: str | None
    heading: str | None
    url: str | None
    category: str | None
    tags: str | None
    keywords: str | None
    meta_description: str | None
    added_date: str | None
    content_from_metadata: str | None
    extracted_text: str
    text_char_count: int
    source_path: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class IngestionRunRecord:
    run_id: str
    author_id: str
    status: str
    articles_seen: int
    articles_ingested: int
    articles_failed: int
    metadata_rows_seen: int
    warnings_json: str
    started_at: str
    completed_at: str


@dataclass(frozen=True)
class ArticleListItem:
    article_id: str
    filename: str
    title: str | None
    heading: str | None
    category: str | None
    text_char_count: int
    url: str | None


class StyleScribeRepository:
    """SQLite repository for authors, articles, and ingestion runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    def initialize_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with get_connection(self.db_path) as connection:
            connection.executescript(schema_sql)

    def upsert_author(self, author: AuthorRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO authors (
                    author_id, display_name, language, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(author_id) DO UPDATE SET
                    display_name = excluded.display_name,
                    language = excluded.language,
                    updated_at = excluded.updated_at
                """,
                (
                    author.author_id,
                    author.display_name,
                    author.language,
                    author.created_at,
                    author.updated_at,
                ),
            )

    def upsert_article(self, article: ArticleRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO author_articles (
                    article_id, author_id, filename, title, heading, url,
                    category, tags, keywords, meta_description, added_date,
                    content_from_metadata, extracted_text, text_char_count,
                    source_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article.article_id,
                    article.author_id,
                    article.filename,
                    article.title,
                    article.heading,
                    article.url,
                    article.category,
                    article.tags,
                    article.keywords,
                    article.meta_description,
                    article.added_date,
                    article.content_from_metadata,
                    article.extracted_text,
                    article.text_char_count,
                    article.source_path,
                    article.created_at,
                    article.updated_at,
                ),
            )

    def create_ingestion_run(self, run: IngestionRunRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO ingestion_runs (
                    run_id, author_id, status, articles_seen, articles_ingested,
                    articles_failed, metadata_rows_seen, warnings_json,
                    started_at, completed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run.run_id,
                    run.author_id,
                    run.status,
                    run.articles_seen,
                    run.articles_ingested,
                    run.articles_failed,
                    run.metadata_rows_seen,
                    run.warnings_json,
                    run.started_at,
                    run.completed_at,
                ),
            )

    def list_articles_for_author(self, author_id: str) -> list[ArticleListItem]:
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    article_id, filename, title, heading, category,
                    text_char_count, url
                FROM author_articles
                WHERE author_id = ?
                ORDER BY filename
                """,
                (author_id,),
            ).fetchall()

        return [self._map_article_list_item(row) for row in rows]

    @staticmethod
    def encode_warnings(warnings: list[str]) -> str:
        return json.dumps(warnings, ensure_ascii=False)

    @staticmethod
    def _map_article_list_item(row: sqlite3.Row) -> ArticleListItem:
        return ArticleListItem(
            article_id=str(row["article_id"]),
            filename=str(row["filename"]),
            title=row["title"],
            heading=row["heading"],
            category=row["category"],
            text_char_count=int(row["text_char_count"]),
            url=row["url"],
        )
