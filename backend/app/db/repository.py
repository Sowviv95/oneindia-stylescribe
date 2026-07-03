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


@dataclass(frozen=True)
class ArticleForAnalysis:
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


@dataclass(frozen=True)
class StyleSnapshotRecord:
    snapshot_id: str
    author_id: str
    article_count: int
    language: str
    status: str
    stats_json: str
    excerpt_pack_json: str
    warnings_json: str
    created_at: str


@dataclass(frozen=True)
class AuthorStyleProfileRecord:
    profile_id: str
    author_id: str
    snapshot_id: str
    language: str
    model_provider: str
    model_name: str
    status: str
    profile_json: str
    source_excerpt_refs_json: str
    warnings_json: str
    created_at: str


@dataclass(frozen=True)
class GroundedBriefRecord:
    brief_id: str
    source_type: str
    source_input_hash: str
    source_url: str | None
    source_text_excerpt: str
    source_language: str
    target_language: str
    model_provider: str
    model_name: str
    status: str
    brief_json: str
    warnings_json: str
    created_at: str


@dataclass(frozen=True)
class ArticlePlanRecord:
    plan_id: str
    brief_id: str
    author_id: str
    article_type: str
    desired_word_count: int | None
    target_min_word_count: int | None
    target_max_word_count: int | None
    planned_sections_json: str
    expansion_items_used_json: str
    claims_to_avoid_json: str
    plan_summary: str
    model_provider: str
    model_name: str
    token_usage_json: str
    created_at: str


@dataclass(frozen=True)
class ArticleDraftRecord:
    draft_id: str
    author_id: str
    profile_id: str
    brief_id: str
    target_language: str
    model_provider: str
    model_name: str
    status: str
    author_instruction: str | None
    article_type: str | None
    desired_word_count: int | None
    tone_override: str | None
    include_seo: bool
    draft_json: str
    warnings_json: str
    created_at: str


@dataclass(frozen=True)
class DraftEvaluationRecord:
    evaluation_id: str
    draft_id: str
    brief_id: str
    author_id: str
    model_provider: str
    model_name: str
    status: str
    evaluation_json: str
    warnings_json: str
    created_at: str


@dataclass(frozen=True)
class ArticleRevisionRecord:
    revision_id: str
    draft_id: str
    evaluation_id: str
    author_id: str
    revised_headline: str
    revised_subheadline: str
    revised_article_body: str
    revised_seo_title: str
    revised_meta_description: str
    revised_tags_json: str
    revision_summary: str
    removed_or_softened_claims_json: str
    model_provider: str
    model_name: str
    token_usage_json: str
    created_at: str


@dataclass(frozen=True)
class WorkflowRunRecord:
    workflow_id: str
    workflow_type: str
    author_id: str
    brief_id: str | None
    draft_id: str | None
    evaluation_id: str | None
    status: str
    input_summary_json: str
    output_summary_json: str
    warnings_json: str
    created_at: str


class StyleScribeRepository:
    """SQLite repository for authors, articles, and ingestion runs."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = db_path

    def initialize_schema(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with get_connection(self.db_path) as connection:
            connection.executescript(schema_sql)
            self._ensure_article_draft_columns(connection)
            self._ensure_article_revisions_table(connection)
            self._ensure_article_plans_table(connection)

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

    def fetch_articles_for_analysis(
        self,
        author_id: str,
    ) -> list[ArticleForAnalysis]:
        with get_connection(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT
                    article_id, author_id, filename, title, heading, url,
                    category, tags, keywords, meta_description, added_date,
                    content_from_metadata, extracted_text, text_char_count
                FROM author_articles
                WHERE author_id = ?
                ORDER BY filename
                """,
                (author_id,),
            ).fetchall()

        return [self._map_article_for_analysis(row) for row in rows]

    def fetch_author_language(self, author_id: str) -> str | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                "SELECT language FROM authors WHERE author_id = ?",
                (author_id,),
            ).fetchone()
        return str(row["language"]) if row else None

    def save_style_snapshot(self, snapshot: StyleSnapshotRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO author_style_snapshots (
                    snapshot_id, author_id, article_count, language, status,
                    stats_json, excerpt_pack_json, warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot.snapshot_id,
                    snapshot.author_id,
                    snapshot.article_count,
                    snapshot.language,
                    snapshot.status,
                    snapshot.stats_json,
                    snapshot.excerpt_pack_json,
                    snapshot.warnings_json,
                    snapshot.created_at,
                ),
            )

    def fetch_latest_style_snapshot(
        self,
        author_id: str,
    ) -> StyleSnapshotRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    snapshot_id, author_id, article_count, language, status,
                    stats_json, excerpt_pack_json, warnings_json, created_at
                FROM author_style_snapshots
                WHERE author_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (author_id,),
            ).fetchone()

        return self._map_style_snapshot(row) if row else None

    def save_author_style_profile(self, profile: AuthorStyleProfileRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO author_style_profiles (
                    profile_id, author_id, snapshot_id, language, model_provider,
                    model_name, status, profile_json, source_excerpt_refs_json,
                    warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.profile_id,
                    profile.author_id,
                    profile.snapshot_id,
                    profile.language,
                    profile.model_provider,
                    profile.model_name,
                    profile.status,
                    profile.profile_json,
                    profile.source_excerpt_refs_json,
                    profile.warnings_json,
                    profile.created_at,
                ),
            )

    def fetch_latest_author_style_profile(
        self,
        author_id: str,
    ) -> AuthorStyleProfileRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    profile_id, author_id, snapshot_id, language, model_provider,
                    model_name, status, profile_json, source_excerpt_refs_json,
                    warnings_json, created_at
                FROM author_style_profiles
                WHERE author_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (author_id,),
            ).fetchone()

        return self._map_author_style_profile(row) if row else None

    def fetch_author_style_profile(
        self,
        profile_id: str,
    ) -> AuthorStyleProfileRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    profile_id, author_id, snapshot_id, language, model_provider,
                    model_name, status, profile_json, source_excerpt_refs_json,
                    warnings_json, created_at
                FROM author_style_profiles
                WHERE profile_id = ?
                """,
                (profile_id,),
            ).fetchone()

        return self._map_author_style_profile(row) if row else None

    def save_grounded_brief(self, brief: GroundedBriefRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO grounded_briefs (
                    brief_id, source_type, source_input_hash, source_url,
                    source_text_excerpt, source_language, target_language,
                    model_provider, model_name, status, brief_json,
                    warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    brief.brief_id,
                    brief.source_type,
                    brief.source_input_hash,
                    brief.source_url,
                    brief.source_text_excerpt,
                    brief.source_language,
                    brief.target_language,
                    brief.model_provider,
                    brief.model_name,
                    brief.status,
                    brief.brief_json,
                    brief.warnings_json,
                    brief.created_at,
                ),
            )

    def fetch_grounded_brief(self, brief_id: str) -> GroundedBriefRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    brief_id, source_type, source_input_hash, source_url,
                    source_text_excerpt, source_language, target_language,
                    model_provider, model_name, status, brief_json,
                    warnings_json, created_at
                FROM grounded_briefs
                WHERE brief_id = ?
                """,
                (brief_id,),
            ).fetchone()

        return self._map_grounded_brief(row) if row else None

    def fetch_latest_grounded_brief(self) -> GroundedBriefRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    brief_id, source_type, source_input_hash, source_url,
                    source_text_excerpt, source_language, target_language,
                    model_provider, model_name, status, brief_json,
                    warnings_json, created_at
                FROM grounded_briefs
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()

        return self._map_grounded_brief(row) if row else None

    def save_article_plan(self, plan: ArticlePlanRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO article_plans (
                    plan_id, brief_id, author_id, article_type,
                    desired_word_count, target_min_word_count,
                    target_max_word_count, planned_sections_json,
                    expansion_items_used_json, claims_to_avoid_json,
                    plan_summary, model_provider, model_name,
                    token_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    plan.brief_id,
                    plan.author_id,
                    plan.article_type,
                    plan.desired_word_count,
                    plan.target_min_word_count,
                    plan.target_max_word_count,
                    plan.planned_sections_json,
                    plan.expansion_items_used_json,
                    plan.claims_to_avoid_json,
                    plan.plan_summary,
                    plan.model_provider,
                    plan.model_name,
                    plan.token_usage_json,
                    plan.created_at,
                ),
            )

    def fetch_article_plan(self, plan_id: str) -> ArticlePlanRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    plan_id, brief_id, author_id, article_type,
                    desired_word_count, target_min_word_count,
                    target_max_word_count, planned_sections_json,
                    expansion_items_used_json, claims_to_avoid_json,
                    plan_summary, model_provider, model_name,
                    token_usage_json, created_at
                FROM article_plans
                WHERE plan_id = ?
                """,
                (plan_id,),
            ).fetchone()

        return self._map_article_plan(row) if row else None

    def save_article_draft(self, draft: ArticleDraftRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO article_drafts (
                    draft_id, author_id, profile_id, brief_id, target_language,
                    model_provider, model_name, status, author_instruction,
                    article_type, desired_word_count, tone_override, include_seo,
                    draft_json, warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    draft.draft_id,
                    draft.author_id,
                    draft.profile_id,
                    draft.brief_id,
                    draft.target_language,
                    draft.model_provider,
                    draft.model_name,
                    draft.status,
                    draft.author_instruction,
                    draft.article_type,
                    draft.desired_word_count,
                    draft.tone_override,
                    int(draft.include_seo),
                    draft.draft_json,
                    draft.warnings_json,
                    draft.created_at,
                ),
            )

    def fetch_article_draft(self, draft_id: str) -> ArticleDraftRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    draft_id, author_id, profile_id, brief_id, target_language,
                    model_provider, model_name, status, author_instruction,
                    article_type, desired_word_count, tone_override, include_seo,
                    draft_json, warnings_json, created_at
                FROM article_drafts
                WHERE draft_id = ?
                """,
                (draft_id,),
            ).fetchone()

        return self._map_article_draft(row) if row else None

    def fetch_latest_article_draft_for_author(
        self,
        author_id: str,
    ) -> ArticleDraftRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    draft_id, author_id, profile_id, brief_id, target_language,
                    model_provider, model_name, status, author_instruction,
                    article_type, desired_word_count, tone_override, include_seo,
                    draft_json, warnings_json, created_at
                FROM article_drafts
                WHERE author_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (author_id,),
            ).fetchone()

        return self._map_article_draft(row) if row else None

    def save_draft_evaluation(self, evaluation: DraftEvaluationRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO draft_evaluations (
                    evaluation_id, draft_id, brief_id, author_id, model_provider,
                    model_name, status, evaluation_json, warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation.evaluation_id,
                    evaluation.draft_id,
                    evaluation.brief_id,
                    evaluation.author_id,
                    evaluation.model_provider,
                    evaluation.model_name,
                    evaluation.status,
                    evaluation.evaluation_json,
                    evaluation.warnings_json,
                    evaluation.created_at,
                ),
            )

    def fetch_draft_evaluation(
        self,
        evaluation_id: str,
    ) -> DraftEvaluationRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    evaluation_id, draft_id, brief_id, author_id, model_provider,
                    model_name, status, evaluation_json, warnings_json, created_at
                FROM draft_evaluations
                WHERE evaluation_id = ?
                """,
                (evaluation_id,),
            ).fetchone()

        return self._map_draft_evaluation(row) if row else None

    def fetch_latest_draft_evaluation(
        self,
        draft_id: str,
    ) -> DraftEvaluationRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    evaluation_id, draft_id, brief_id, author_id, model_provider,
                    model_name, status, evaluation_json, warnings_json, created_at
                FROM draft_evaluations
                WHERE draft_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (draft_id,),
            ).fetchone()

        return self._map_draft_evaluation(row) if row else None

    def fetch_draft_with_grounded_brief(
        self,
        draft_id: str,
    ) -> tuple[ArticleDraftRecord, GroundedBriefRecord] | None:
        draft = self.fetch_article_draft(draft_id)
        if draft is None:
            return None
        brief = self.fetch_grounded_brief(draft.brief_id)
        if brief is None:
            return None
        return draft, brief

    def save_article_revision(self, revision: ArticleRevisionRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO article_revisions (
                    revision_id, draft_id, evaluation_id, author_id,
                    revised_headline, revised_subheadline, revised_article_body,
                    revised_seo_title, revised_meta_description, revised_tags_json,
                    revision_summary, removed_or_softened_claims_json,
                    model_provider, model_name, token_usage_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    revision.revision_id,
                    revision.draft_id,
                    revision.evaluation_id,
                    revision.author_id,
                    revision.revised_headline,
                    revision.revised_subheadline,
                    revision.revised_article_body,
                    revision.revised_seo_title,
                    revision.revised_meta_description,
                    revision.revised_tags_json,
                    revision.revision_summary,
                    revision.removed_or_softened_claims_json,
                    revision.model_provider,
                    revision.model_name,
                    revision.token_usage_json,
                    revision.created_at,
                ),
            )

    def fetch_article_revision(
        self,
        revision_id: str,
    ) -> ArticleRevisionRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    revision_id, draft_id, evaluation_id, author_id,
                    revised_headline, revised_subheadline, revised_article_body,
                    revised_seo_title, revised_meta_description, revised_tags_json,
                    revision_summary, removed_or_softened_claims_json,
                    model_provider, model_name, token_usage_json, created_at
                FROM article_revisions
                WHERE revision_id = ?
                """,
                (revision_id,),
            ).fetchone()

        return self._map_article_revision(row) if row else None

    def fetch_latest_article_revision(
        self,
        draft_id: str,
    ) -> ArticleRevisionRecord | None:
        with get_connection(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT
                    revision_id, draft_id, evaluation_id, author_id,
                    revised_headline, revised_subheadline, revised_article_body,
                    revised_seo_title, revised_meta_description, revised_tags_json,
                    revision_summary, removed_or_softened_claims_json,
                    model_provider, model_name, token_usage_json, created_at
                FROM article_revisions
                WHERE draft_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (draft_id,),
            ).fetchone()

        return self._map_article_revision(row) if row else None

    def save_workflow_run(self, workflow: WorkflowRunRecord) -> None:
        with get_connection(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO workflow_runs (
                    workflow_id, workflow_type, author_id, brief_id, draft_id,
                    evaluation_id, status, input_summary_json, output_summary_json,
                    warnings_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    workflow.workflow_id,
                    workflow.workflow_type,
                    workflow.author_id,
                    workflow.brief_id,
                    workflow.draft_id,
                    workflow.evaluation_id,
                    workflow.status,
                    workflow.input_summary_json,
                    workflow.output_summary_json,
                    workflow.warnings_json,
                    workflow.created_at,
                ),
            )

    @staticmethod
    def encode_warnings(warnings: list[str]) -> str:
        return json.dumps(warnings, ensure_ascii=False)

    @staticmethod
    def encode_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def decode_json_object(value: str) -> dict[str, object]:
        decoded = json.loads(value)
        return decoded if isinstance(decoded, dict) else {}

    @staticmethod
    def decode_json_list(value: str) -> list[str]:
        decoded = json.loads(value)
        if not isinstance(decoded, list):
            return []
        return [str(item) for item in decoded]

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

    @staticmethod
    def _map_article_for_analysis(row: sqlite3.Row) -> ArticleForAnalysis:
        return ArticleForAnalysis(
            article_id=str(row["article_id"]),
            author_id=str(row["author_id"]),
            filename=str(row["filename"]),
            title=row["title"],
            heading=row["heading"],
            url=row["url"],
            category=row["category"],
            tags=row["tags"],
            keywords=row["keywords"],
            meta_description=row["meta_description"],
            added_date=row["added_date"],
            content_from_metadata=row["content_from_metadata"],
            extracted_text=str(row["extracted_text"]),
            text_char_count=int(row["text_char_count"]),
        )

    @staticmethod
    def _map_style_snapshot(row: sqlite3.Row) -> StyleSnapshotRecord:
        return StyleSnapshotRecord(
            snapshot_id=str(row["snapshot_id"]),
            author_id=str(row["author_id"]),
            article_count=int(row["article_count"]),
            language=str(row["language"]),
            status=str(row["status"]),
            stats_json=str(row["stats_json"]),
            excerpt_pack_json=str(row["excerpt_pack_json"]),
            warnings_json=str(row["warnings_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_author_style_profile(row: sqlite3.Row) -> AuthorStyleProfileRecord:
        return AuthorStyleProfileRecord(
            profile_id=str(row["profile_id"]),
            author_id=str(row["author_id"]),
            snapshot_id=str(row["snapshot_id"]),
            language=str(row["language"]),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            status=str(row["status"]),
            profile_json=str(row["profile_json"]),
            source_excerpt_refs_json=str(row["source_excerpt_refs_json"]),
            warnings_json=str(row["warnings_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_grounded_brief(row: sqlite3.Row) -> GroundedBriefRecord:
        return GroundedBriefRecord(
            brief_id=str(row["brief_id"]),
            source_type=str(row["source_type"]),
            source_input_hash=str(row["source_input_hash"]),
            source_url=row["source_url"],
            source_text_excerpt=str(row["source_text_excerpt"]),
            source_language=str(row["source_language"]),
            target_language=str(row["target_language"]),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            status=str(row["status"]),
            brief_json=str(row["brief_json"]),
            warnings_json=str(row["warnings_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_article_plan(row: sqlite3.Row) -> ArticlePlanRecord:
        return ArticlePlanRecord(
            plan_id=str(row["plan_id"]),
            brief_id=str(row["brief_id"]),
            author_id=str(row["author_id"]),
            article_type=str(row["article_type"]),
            desired_word_count=row["desired_word_count"],
            target_min_word_count=row["target_min_word_count"],
            target_max_word_count=row["target_max_word_count"],
            planned_sections_json=str(row["planned_sections_json"]),
            expansion_items_used_json=str(row["expansion_items_used_json"]),
            claims_to_avoid_json=str(row["claims_to_avoid_json"]),
            plan_summary=str(row["plan_summary"]),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            token_usage_json=str(row["token_usage_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_article_draft(row: sqlite3.Row) -> ArticleDraftRecord:
        return ArticleDraftRecord(
            draft_id=str(row["draft_id"]),
            author_id=str(row["author_id"]),
            profile_id=str(row["profile_id"]),
            brief_id=str(row["brief_id"]),
            target_language=str(row["target_language"]),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            status=str(row["status"]),
            author_instruction=row["author_instruction"],
            article_type=row["article_type"],
            desired_word_count=row["desired_word_count"],
            tone_override=row["tone_override"],
            include_seo=bool(row["include_seo"]),
            draft_json=str(row["draft_json"]),
            warnings_json=str(row["warnings_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_draft_evaluation(row: sqlite3.Row) -> DraftEvaluationRecord:
        return DraftEvaluationRecord(
            evaluation_id=str(row["evaluation_id"]),
            draft_id=str(row["draft_id"]),
            brief_id=str(row["brief_id"]),
            author_id=str(row["author_id"]),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            status=str(row["status"]),
            evaluation_json=str(row["evaluation_json"]),
            warnings_json=str(row["warnings_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _map_article_revision(row: sqlite3.Row) -> ArticleRevisionRecord:
        return ArticleRevisionRecord(
            revision_id=str(row["revision_id"]),
            draft_id=str(row["draft_id"]),
            evaluation_id=str(row["evaluation_id"]),
            author_id=str(row["author_id"]),
            revised_headline=str(row["revised_headline"]),
            revised_subheadline=str(row["revised_subheadline"]),
            revised_article_body=str(row["revised_article_body"]),
            revised_seo_title=str(row["revised_seo_title"]),
            revised_meta_description=str(row["revised_meta_description"]),
            revised_tags_json=str(row["revised_tags_json"]),
            revision_summary=str(row["revision_summary"]),
            removed_or_softened_claims_json=str(
                row["removed_or_softened_claims_json"]
            ),
            model_provider=str(row["model_provider"]),
            model_name=str(row["model_name"]),
            token_usage_json=str(row["token_usage_json"]),
            created_at=str(row["created_at"]),
        )

    @staticmethod
    def _ensure_article_draft_columns(connection: sqlite3.Connection) -> None:
        rows = connection.execute("PRAGMA table_info(article_drafts)").fetchall()
        existing_columns = {str(row["name"]) for row in rows}
        migrations = {
            "article_type": "ALTER TABLE article_drafts ADD COLUMN article_type TEXT",
            "desired_word_count": (
                "ALTER TABLE article_drafts ADD COLUMN desired_word_count INTEGER"
            ),
            "tone_override": "ALTER TABLE article_drafts ADD COLUMN tone_override TEXT",
            "include_seo": (
                "ALTER TABLE article_drafts ADD COLUMN include_seo INTEGER "
                "NOT NULL DEFAULT 1"
            ),
        }
        for column_name, sql in migrations.items():
            if column_name not in existing_columns:
                connection.execute(sql)

    @staticmethod
    def _ensure_article_revisions_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS article_revisions (
                revision_id TEXT PRIMARY KEY,
                draft_id TEXT NOT NULL,
                evaluation_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                revised_headline TEXT NOT NULL,
                revised_subheadline TEXT NOT NULL,
                revised_article_body TEXT NOT NULL,
                revised_seo_title TEXT NOT NULL,
                revised_meta_description TEXT NOT NULL,
                revised_tags_json TEXT NOT NULL,
                revision_summary TEXT NOT NULL,
                removed_or_softened_claims_json TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                token_usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(draft_id) REFERENCES article_drafts(draft_id),
                FOREIGN KEY(evaluation_id) REFERENCES draft_evaluations(evaluation_id)
            )
            """
        )

    @staticmethod
    def _ensure_article_plans_table(connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS article_plans (
                plan_id TEXT PRIMARY KEY,
                brief_id TEXT NOT NULL,
                author_id TEXT NOT NULL,
                article_type TEXT NOT NULL,
                desired_word_count INTEGER,
                target_min_word_count INTEGER,
                target_max_word_count INTEGER,
                planned_sections_json TEXT NOT NULL,
                expansion_items_used_json TEXT NOT NULL,
                claims_to_avoid_json TEXT NOT NULL,
                plan_summary TEXT NOT NULL,
                model_provider TEXT NOT NULL,
                model_name TEXT NOT NULL,
                token_usage_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(brief_id) REFERENCES grounded_briefs(brief_id)
            )
            """
        )
