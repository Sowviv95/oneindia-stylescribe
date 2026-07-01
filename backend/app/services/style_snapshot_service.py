"""Build and persist deterministic author style snapshots."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.app.db.repository import StyleScribeRepository, StyleSnapshotRecord
from backend.app.models.style_models import AuthorStyleSnapshotResponse
from backend.app.services.excerpt_pack_service import build_excerpt_pack
from backend.app.services.style_statistics_service import calculate_style_statistics


class AuthorStyleSnapshotError(RuntimeError):
    """Raised when a style snapshot cannot be built."""


def build_author_style_snapshot(
    author_id: str,
    repository: StyleScribeRepository | None = None,
) -> AuthorStyleSnapshotResponse:
    """Build, save, and return a deterministic style snapshot."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    articles = repo.fetch_articles_for_analysis(author_id)
    if not articles:
        raise AuthorStyleSnapshotError(
            f"No ingested articles found for author_id: {author_id}"
        )

    language = repo.fetch_author_language(author_id) or "unknown"
    stats = calculate_style_statistics(articles)
    excerpt_pack = build_excerpt_pack(articles)
    warnings: list[str] = []
    created_at = datetime.now(UTC).isoformat()
    snapshot_id = str(uuid4())

    snapshot = StyleSnapshotRecord(
        snapshot_id=snapshot_id,
        author_id=author_id,
        article_count=len(articles),
        language=language,
        status="completed",
        stats_json=StyleScribeRepository.encode_json(stats),
        excerpt_pack_json=StyleScribeRepository.encode_json(excerpt_pack),
        warnings_json=StyleScribeRepository.encode_warnings(warnings),
        created_at=created_at,
    )
    repo.save_style_snapshot(snapshot)

    return AuthorStyleSnapshotResponse(
        snapshot_id=snapshot_id,
        author_id=author_id,
        article_count=len(articles),
        language=language,
        status="completed",
        stats=stats,
        excerpt_pack=excerpt_pack,
        warnings=warnings,
        created_at=created_at,
    )


def get_latest_author_style_snapshot(
    author_id: str,
    repository: StyleScribeRepository | None = None,
) -> AuthorStyleSnapshotResponse:
    """Return the latest persisted style snapshot for an author."""

    repo = repository or StyleScribeRepository()
    repo.initialize_schema()
    snapshot = repo.fetch_latest_style_snapshot(author_id)
    if snapshot is None:
        raise AuthorStyleSnapshotError(
            f"No style snapshot found for author_id: {author_id}"
        )

    return AuthorStyleSnapshotResponse(
        snapshot_id=snapshot.snapshot_id,
        author_id=snapshot.author_id,
        article_count=snapshot.article_count,
        language=snapshot.language,
        status=snapshot.status,
        stats=StyleScribeRepository.decode_json_object(snapshot.stats_json),
        excerpt_pack=StyleScribeRepository.decode_json_object(
            snapshot.excerpt_pack_json
        ),
        warnings=StyleScribeRepository.decode_json_list(snapshot.warnings_json),
        created_at=snapshot.created_at,
    )
