"""Article-level retrieval support for generic newsroom generation."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib import import_module, metadata
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol, cast

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.services.newsroom_corpus_preparation_service import (
    classify_topic,
    normalize_for_comparison,
)

DEFAULT_RETRIEVAL_DIR = Path("data/newsroom_corpus/03_retrieval")
DEFAULT_CLEANED_ARTICLES_JSONL = Path(
    "data/newsroom_corpus/02_cleaned/cleaned_articles.jsonl"
)
DEFAULT_NEAR_DUPLICATES_JSONL = Path(
    "data/newsroom_corpus/reports/near_duplicate_clusters.jsonl"
)
DEFAULT_RECORDS_PATH = DEFAULT_RETRIEVAL_DIR / "retrieval_records.jsonl"
DEFAULT_INDEX_PATH = DEFAULT_RETRIEVAL_DIR / "newsroom_retrieval_index.json"
RETRIEVAL_CORPUS_VERSION = "oneindia_newsroom_retrieval_corpus_v1"
RETRIEVAL_INDEX_VERSION = "article_hashing_index_v1"
DEFAULT_PROFILE_VERSION = "oneindia_tamil_generic_newsroom_sprint2"
DEFAULT_EMBEDDING_PROVIDER = "local_hashing"
DEFAULT_EMBEDDING_MODEL = "word_hashing_256_v1"
DEFAULT_EMBEDDING_DIMENSIONS = 256
SENTENCE_TRANSFORMERS_PROVIDER = "sentence_transformers"
DEFAULT_SENTENCE_TRANSFORMERS_MODEL = "intfloat/multilingual-e5-small"
TOKEN_RE = re.compile(r"[\w\u0b80-\u0bff]+", re.UNICODE)


@dataclass(frozen=True)
class RetrievalRecord:
    article_id: str
    author_id: str
    source_path: str
    cleaned_article_text: str
    paragraphs: list[str]
    word_count: int
    paragraph_count: int
    topic: str
    topic_confidence: float
    topic_review_flag: bool
    topic_low_confidence: bool
    topic_multi_category_conflict: bool
    file_sha256: str | None
    text_sha256: str | None
    normalized_text_sha256: str | None
    duplicate_cluster_id: str | None
    canonical_article_id: str | None
    near_duplicate_article_ids: list[str]
    cleaning_decision: dict[str, object]
    corpus_version: str
    newsroom_profile_version: str
    embedding_text: str
    embedding_metadata: dict[str, object]


@dataclass(frozen=True)
class RetrievalIndex:
    index_version: str
    corpus_version: str
    newsroom_profile_version: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    normalization_method: str
    similarity_metric: str
    dependency_version: str | None
    record_count: int
    created_at: str
    input_hash: str
    records: list[RetrievalRecord]
    embeddings: list[list[float]]


@dataclass(frozen=True)
class RetrievalRankingConfig:
    top_k: int = 3
    candidate_pool_size: int = 12
    minimum_similarity: float | None = None
    topic_boost_enabled: bool = False
    topic_boost_weight: float = 0.05
    max_examples_per_author: int = 1
    max_context_chars: int = 9000
    exclude_exact_duplicates: bool = True
    exclude_near_duplicates: bool = True
    author_diversity: bool = True


@dataclass(frozen=True)
class RetrievalQuery:
    text: str
    topic: str
    topic_confidence: float
    topic_review_flag: bool
    topic_low_confidence: bool
    topic_multi_category_conflict: bool
    metadata: dict[str, object]


@dataclass(frozen=True)
class CandidateScore:
    article_id: str
    author_id: str
    topic: str
    topic_confidence: float
    topic_low_confidence: bool
    topic_multi_category_conflict: bool
    similarity_score: float
    topic_boost: float
    final_score: float
    selected_rank: int | None = None
    context_chars: int = 0


@dataclass(frozen=True)
class RetrievalResult:
    query: RetrievalQuery
    selected_records: list[RetrievalRecord]
    selected_scores: list[CandidateScore]
    candidate_scores: list[CandidateScore]
    exclusions: list[dict[str, object]]
    trace: dict[str, object]


class EmbeddingProvider(Protocol):
    provider: str
    model: str
    dimensions: int
    normalization_method: str
    similarity_metric: str
    dependency_version: str | None

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per text."""


class HashingEmbeddingProvider:
    """Deterministic sparse lexical embedder for local tests and dry runs."""

    provider = DEFAULT_EMBEDDING_PROVIDER
    normalization_method = "l2"
    similarity_metric = "cosine"
    dependency_version: str | None = None

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    ) -> None:
        self.model = model
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            _normalize_vector(_hash_vector(text, self.dimensions))
            for text in texts
        ]


class SentenceTransformersEmbeddingProvider:
    """Optional CPU local semantic embedder for retrieval indexing."""

    provider = SENTENCE_TRANSFORMERS_PROVIDER
    normalization_method = "l2"
    similarity_metric = "cosine"

    def __init__(
        self,
        model: str = DEFAULT_SENTENCE_TRANSFORMERS_MODEL,
        *,
        batch_size: int = 16,
        show_progress_bar: bool = True,
    ) -> None:
        try:
            sentence_transformers = import_module("sentence_transformers")
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for "
                "sentence_transformers retrieval embeddings."
            ) from exc
        sentence_transformer = sentence_transformers.SentenceTransformer
        self.model = model
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.dependency_version = _package_version("sentence-transformers")
        self._model = sentence_transformer(model, device="cpu")
        dimensions = self._model.get_sentence_embedding_dimension()
        if not isinstance(dimensions, int):
            raise RuntimeError(f"Could not determine embedding dimensions for {model}.")
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=False,
            normalize_embeddings=True,
            show_progress_bar=self.show_progress_bar,
        )
        return [
            [float(value) for value in vector]
            for vector in cast(list[Any], embeddings)
        ]


def make_embedding_provider(
    provider: str = DEFAULT_EMBEDDING_PROVIDER,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> EmbeddingProvider:
    if provider == SENTENCE_TRANSFORMERS_PROVIDER:
        return SentenceTransformersEmbeddingProvider(model=model)
    if provider != DEFAULT_EMBEDDING_PROVIDER:
        raise ValueError(f"Unsupported retrieval embedding provider: {provider}")
    dimensions = _dimensions_from_model(model)
    return HashingEmbeddingProvider(model=model, dimensions=dimensions)


def build_retrieval_records(
    *,
    cleaned_articles_jsonl: Path = DEFAULT_CLEANED_ARTICLES_JSONL,
    near_duplicates_jsonl: Path = DEFAULT_NEAR_DUPLICATES_JSONL,
    output_path: Path | None = DEFAULT_RECORDS_PATH,
    corpus_version: str = RETRIEVAL_CORPUS_VERSION,
    newsroom_profile_version: str = DEFAULT_PROFILE_VERSION,
) -> list[RetrievalRecord]:
    near_duplicate_ids = _near_duplicate_members(near_duplicates_jsonl)
    records: list[RetrievalRecord] = []
    for article in _load_jsonl(cleaned_articles_jsonl):
        if str(article.get("preparation_status") or "") != "accepted":
            continue
        paragraphs = _string_list(article.get("paragraph_sequence"))
        text = str(article.get("body_text") or "").strip()
        embedding_text = _embedding_text(article, text, paragraphs)
        record = RetrievalRecord(
            article_id=str(article["article_id"]),
            author_id=str(article["author_id"]),
            source_path=str(article.get("relative_source_path") or ""),
            cleaned_article_text=text,
            paragraphs=paragraphs,
            word_count=_int_value(article.get("word_count")),
            paragraph_count=_int_value(article.get("paragraph_count")),
            topic=str(article.get("topic") or "other"),
            topic_confidence=_float_value(article.get("topic_confidence")),
            topic_review_flag=bool(article.get("topic_review_flag")),
            topic_low_confidence=bool(article.get("topic_low_confidence")),
            topic_multi_category_conflict=bool(
                article.get("topic_multi_category_conflict")
            ),
            file_sha256=_optional_string(article.get("file_sha256")),
            text_sha256=_optional_string(article.get("text_sha256")),
            normalized_text_sha256=_optional_string(
                article.get("normalized_text_sha256")
            ),
            duplicate_cluster_id=_optional_string(article.get("duplicate_cluster_id")),
            canonical_article_id=_optional_string(article.get("canonical_article_id")),
            near_duplicate_article_ids=near_duplicate_ids.get(
                str(article["article_id"]), []
            ),
            cleaning_decision={
                "preparation_status": article.get("preparation_status"),
                "article_usable": article.get("article_usable"),
                "decision_reasons": article.get("decision_reasons") or [],
                "review_flags": article.get("review_flags") or [],
                "informational_flags": article.get("informational_flags") or [],
            },
            corpus_version=corpus_version,
            newsroom_profile_version=newsroom_profile_version,
            embedding_text=embedding_text,
            embedding_metadata={
                "unit": "article",
                "fields": [
                    "topic",
                    "headline_candidate",
                    "paragraph_sequence",
                    "body_text",
                ],
                "char_count": len(embedding_text),
            },
        )
        records.append(record)

    records = sorted(records, key=lambda item: item.article_id)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return records


def build_or_load_index(
    *,
    index_path: Path = DEFAULT_INDEX_PATH,
    records_path: Path = DEFAULT_RECORDS_PATH,
    rebuild: bool = False,
    embedding_provider: str = DEFAULT_EMBEDDING_PROVIDER,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> RetrievalIndex:
    if index_path.exists() and not rebuild:
        index = load_retrieval_index(index_path)
        _validate_index_matches_request(
            index=index,
            embedding_provider=embedding_provider,
            embedding_model=embedding_model,
        )
        return index
    records = (
        load_retrieval_records(records_path)
        if records_path.exists() and not rebuild
        else build_retrieval_records(output_path=records_path)
    )
    provider = make_embedding_provider(embedding_provider, embedding_model)
    embeddings = provider.embed([record.embedding_text for record in records])
    index = RetrievalIndex(
        index_version=_index_version_for_provider(provider.provider, provider.model),
        corpus_version=RETRIEVAL_CORPUS_VERSION,
        newsroom_profile_version=DEFAULT_PROFILE_VERSION,
        embedding_provider=provider.provider,
        embedding_model=provider.model,
        embedding_dimensions=provider.dimensions,
        normalization_method=provider.normalization_method,
        similarity_metric=provider.similarity_metric,
        dependency_version=provider.dependency_version,
        record_count=len(records),
        created_at=datetime.now(UTC).isoformat(),
        input_hash=_records_hash(records),
        records=records,
        embeddings=embeddings,
    )
    save_retrieval_index(index, index_path)
    return index


def save_retrieval_index(index: RetrievalIndex, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                **{
                    key: value
                    for key, value in asdict(index).items()
                    if key not in {"records", "embeddings"}
                },
                "records": [asdict(record) for record in index.records],
                "embeddings": index.embeddings,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_retrieval_index(path: Path) -> RetrievalIndex:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [RetrievalRecord(**item) for item in payload["records"]]
    return RetrievalIndex(
        index_version=str(payload["index_version"]),
        corpus_version=str(payload["corpus_version"]),
        newsroom_profile_version=str(payload["newsroom_profile_version"]),
        embedding_provider=str(payload["embedding_provider"]),
        embedding_model=str(payload["embedding_model"]),
        embedding_dimensions=int(payload["embedding_dimensions"]),
        normalization_method=str(payload.get("normalization_method", "l2")),
        similarity_metric=str(payload.get("similarity_metric", "cosine")),
        dependency_version=_optional_string(payload.get("dependency_version")),
        record_count=int(payload["record_count"]),
        created_at=str(payload["created_at"]),
        input_hash=str(payload["input_hash"]),
        records=records,
        embeddings=[
            [float(value) for value in vector] for vector in payload["embeddings"]
        ],
    )


def load_retrieval_records(path: Path) -> list[RetrievalRecord]:
    return [RetrievalRecord(**item) for item in _load_jsonl(path)]


def build_retrieval_query(
    *,
    brief_record: GroundedBriefRecord,
    article_type: str,
    plan_payload: dict[str, object] | None = None,
) -> RetrievalQuery:
    brief = StyleScribeRepository.decode_json_object(brief_record.brief_json)
    text = retrieval_query_text(
        brief=brief,
        article_type=article_type,
        plan_payload=plan_payload,
    )
    topic_result = classify_topic(
        {"article_id": brief_record.brief_id, "body_text": text}
    )
    return RetrievalQuery(
        text=text,
        topic=topic_result.topic,
        topic_confidence=topic_result.confidence,
        topic_review_flag=topic_result.review_flag,
        topic_low_confidence=topic_result.low_confidence,
        topic_multi_category_conflict=topic_result.multi_category_conflict,
        metadata={
            "brief_topic": brief.get("topic"),
            "brief_id": brief_record.brief_id,
            "source_language": brief_record.source_language,
            "target_language": brief_record.target_language,
        },
    )


def retrieval_query_text(
    *,
    brief: dict[str, object],
    article_type: str,
    plan_payload: dict[str, object] | None = None,
) -> str:
    parts: list[str] = [
        f"article_type: {article_type}",
        f"topic: {brief.get('topic') or ''}",
        f"summary: {brief.get('one_line_summary') or ''}",
    ]
    for key in (
        "confirmed_facts",
        "key_entities",
        "places",
        "dates_or_timeline",
        "numbers_and_statistics",
        "background_from_source",
        "policy_or_legal_context",
        "affected_groups",
        "suggested_tamil_angle",
    ):
        parts.extend(_flatten_value(brief.get(key), key))
    if plan_payload:
        parts.extend(_flatten_value(plan_payload.get("plan_summary"), "plan_summary"))
        parts.extend(_flatten_value(plan_payload.get("planned_sections"), "sections"))
    return "\n".join(part for part in parts if part.strip())


def retrieve_examples(
    *,
    index: RetrievalIndex,
    query: RetrievalQuery,
    config: RetrievalRankingConfig,
    embedding_provider: EmbeddingProvider | None = None,
    source_article_id: str | None = None,
    source_text_hash: str | None = None,
    source_duplicate_cluster_id: str | None = None,
    source_input_id: str | None = None,
) -> RetrievalResult:
    started = perf_counter()
    provider = embedding_provider or make_embedding_provider(
        index.embedding_provider,
        index.embedding_model,
    )
    query_embedding = provider.embed([query.text])[0]
    raw_scores: list[CandidateScore] = []
    exclusions: list[dict[str, object]] = []
    for record, embedding in zip(index.records, index.embeddings, strict=True):
        reason = _exclusion_reason(
            record,
            source_article_id=source_article_id,
            source_text_hash=source_text_hash,
            source_duplicate_cluster_id=source_duplicate_cluster_id,
            config=config,
        )
        if reason:
            exclusions.append({"article_id": record.article_id, "reason": reason})
            continue
        similarity = _cosine(query_embedding, embedding)
        if (
            config.minimum_similarity is not None
            and similarity < config.minimum_similarity
        ):
            exclusions.append(
                {
                    "article_id": record.article_id,
                    "reason": "below_minimum_similarity",
                    "similarity_score": round(similarity, 6),
                }
            )
            continue
        topic_boost = _topic_boost(query, record, config)
        raw_scores.append(
            CandidateScore(
                article_id=record.article_id,
                author_id=record.author_id,
                topic=record.topic,
                topic_confidence=record.topic_confidence,
                topic_low_confidence=record.topic_low_confidence,
                topic_multi_category_conflict=record.topic_multi_category_conflict,
                similarity_score=round(similarity, 6),
                topic_boost=round(topic_boost, 6),
                final_score=round(similarity + topic_boost, 6),
            )
        )
    ranked = sorted(
        raw_scores,
        key=lambda item: (-item.final_score, -item.similarity_score, item.article_id),
    )[: max(config.candidate_pool_size, config.top_k)]
    selected: list[CandidateScore] = []
    author_counts: Counter[str] = Counter()
    context_chars = 0
    record_by_id = {record.article_id: record for record in index.records}
    for score in ranked:
        record = record_by_id[score.article_id]
        if (
            config.author_diversity
            and author_counts[score.author_id] >= config.max_examples_per_author
            and _has_available_author_alternative(
                ranked,
                selected,
                author_counts,
                config,
            )
        ):
            exclusions.append(
                {
                    "article_id": score.article_id,
                    "reason": "author_diversity_limit",
                    "author_id": score.author_id,
                }
            )
            continue
        example_chars = _retrieval_context_chars(record)
        if selected and context_chars + example_chars > config.max_context_chars:
            exclusions.append(
                {
                    "article_id": score.article_id,
                    "reason": "retrieval_context_limit",
                    "context_chars": example_chars,
                }
            )
            continue
        selected_score = CandidateScore(
            **{
                **asdict(score),
                "selected_rank": len(selected) + 1,
                "context_chars": example_chars,
            }
        )
        selected.append(selected_score)
        author_counts[score.author_id] += 1
        context_chars += example_chars
        if len(selected) >= config.top_k:
            break
    selected_records = [record_by_id[score.article_id] for score in selected]
    latency = round(perf_counter() - started, 6)
    trace: dict[str, object] = {
        "source_input_id": source_input_id,
        "retrieval_mode": "newsroom_v1_retrieval",
        "corpus_version": index.corpus_version,
        "profile_version": index.newsroom_profile_version,
        "embedding_provider": index.embedding_provider,
        "embedding_model": index.embedding_model,
        "embedding_dimensions": index.embedding_dimensions,
        "index_version": index.index_version,
        "index_record_count": index.record_count,
        "index_input_hash": index.input_hash,
        "query_text": query.text,
        "query_topic": query.topic,
        "query_topic_confidence": query.topic_confidence,
        "query_topic_low_confidence": query.topic_low_confidence,
        "query_topic_conflict": query.topic_multi_category_conflict,
        "query_topic_review_flag": query.topic_review_flag,
        "candidate_pool_size": config.candidate_pool_size,
        "final_top_k": config.top_k,
        "topic_boost_enabled": config.topic_boost_enabled,
        "topic_boost_weight": config.topic_boost_weight,
        "diversity_configuration": {
            "author_diversity": config.author_diversity,
            "max_examples_per_author": config.max_examples_per_author,
            "max_context_chars": config.max_context_chars,
        },
        "ranking_configuration": asdict(config),
        "retrieved_article_ids": [score.article_id for score in selected],
        "retrieved_authors": [score.author_id for score in selected],
        "retrieved_topics": [
            {
                "article_id": score.article_id,
                "topic": score.topic,
                "topic_confidence": score.topic_confidence,
                "topic_low_confidence": score.topic_low_confidence,
                "topic_multi_category_conflict": score.topic_multi_category_conflict,
            }
            for score in selected
        ],
        "candidate_scores": [asdict(score) for score in ranked],
        "selected_scores": [asdict(score) for score in selected],
        "exclusions": exclusions,
        "total_retrieval_context_size": context_chars,
        "retrieval_latency_seconds": latency,
        "retrieval_cost": {
            "cost_status": "no_external_cost",
            "total_cost_usd": 0.0,
        },
    }
    return RetrievalResult(
        query=query,
        selected_records=selected_records,
        selected_scores=selected,
        candidate_scores=ranked,
        exclusions=exclusions,
        trace=trace,
    )


def build_retrieved_examples_payload(
    records: list[RetrievalRecord],
    scores: list[CandidateScore],
    *,
    max_context_chars: int,
) -> list[dict[str, object]]:
    payload: list[dict[str, object]] = []
    remaining = max_context_chars
    score_by_id = {score.article_id: score for score in scores}
    for record in records:
        if remaining <= 0:
            break
        text = _truncate_text(record.cleaned_article_text, min(remaining, 3000))
        remaining -= len(text)
        score = score_by_id[record.article_id]
        payload.append(
            {
                "article_id": record.article_id,
                "author_id": record.author_id,
                "topic": record.topic,
                "word_count": record.word_count,
                "paragraph_count": record.paragraph_count,
                "similarity_score": score.similarity_score,
                "topic_boost": score.topic_boost,
                "final_score": score.final_score,
                "selected_rank": score.selected_rank,
                "editorial_example_text": text,
            }
        )
    return payload


def topic_metadata_from_brief(
    brief: dict[str, object],
    *,
    input_id: str,
) -> dict[str, object]:
    text = retrieval_query_text(brief=brief, article_type="news")
    result = classify_topic({"article_id": input_id, "body_text": text})
    return {
        "original_brief_topic": brief.get("topic"),
        "provisional_topic": result.topic,
        "provisional_topic_confidence": result.confidence,
        "topic_low_confidence": result.low_confidence,
        "topic_multi_category_conflict": result.multi_category_conflict,
        "topic_review_flag": result.review_flag,
    }


def _embedding_text(
    article: dict[str, object],
    body_text: str,
    paragraphs: list[str],
) -> str:
    return "\n".join(
        item
        for item in [
            f"topic: {article.get('topic') or ''}",
            f"headline_candidate: {article.get('headline_candidate') or ''}",
            "\n".join(paragraphs[:8]),
            body_text,
        ]
        if item.strip()
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _near_duplicate_members(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for cluster in _load_jsonl(path):
        ids = [str(item) for item in cluster.get("article_ids", [])]
        for article_id in ids:
            result[article_id] = [item for item in ids if item != article_id]
    return result


def _records_hash(records: list[RetrievalRecord]) -> str:
    digest_input = "\n".join(
        f"{record.article_id}:{record.normalized_text_sha256 or record.text_sha256}"
        for record in records
    )
    return sha256(digest_input.encode("utf-8")).hexdigest()


def _hash_vector(text: str, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        digest = sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    return vector


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _tokens(text: str) -> list[str]:
    normalized = normalize_for_comparison(text)
    return [token for token in TOKEN_RE.findall(normalized) if len(token) > 1]


def _topic_boost(
    query: RetrievalQuery,
    record: RetrievalRecord,
    config: RetrievalRankingConfig,
) -> float:
    if not config.topic_boost_enabled:
        return 0.0
    if query.topic != record.topic:
        return 0.0
    if query.topic_low_confidence or query.topic_multi_category_conflict:
        return 0.0
    if record.topic_low_confidence or record.topic_multi_category_conflict:
        return 0.0
    return max(0.0, min(config.topic_boost_weight, 0.2))


def _exclusion_reason(
    record: RetrievalRecord,
    *,
    source_article_id: str | None,
    source_text_hash: str | None,
    source_duplicate_cluster_id: str | None,
    config: RetrievalRankingConfig,
) -> str | None:
    if source_article_id and record.article_id == source_article_id:
        return "source_article"
    if (
        config.exclude_exact_duplicates
        and source_text_hash
        and source_text_hash
        in {record.text_sha256, record.normalized_text_sha256}
    ):
        return "exact_duplicate"
    if (
        config.exclude_near_duplicates
        and source_article_id
        and source_article_id in record.near_duplicate_article_ids
    ):
        return "near_duplicate_sibling"
    if (
        config.exclude_near_duplicates
        and source_duplicate_cluster_id
        and record.duplicate_cluster_id == source_duplicate_cluster_id
    ):
        return "near_duplicate_cluster"
    return None


def _has_available_author_alternative(
    ranked: list[CandidateScore],
    selected: list[CandidateScore],
    author_counts: Counter[str],
    config: RetrievalRankingConfig,
) -> bool:
    selected_ids = {score.article_id for score in selected}
    for score in ranked:
        if score.article_id in selected_ids:
            continue
        if author_counts[score.author_id] < config.max_examples_per_author:
            return True
    return False


def _retrieval_context_chars(record: RetrievalRecord) -> int:
    return min(len(record.cleaned_article_text), 3000)


def _truncate_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _flatten_value(value: object, key: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [f"{key}: {value}"]
    if isinstance(value, dict):
        return [f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True)}"]
    if isinstance(value, list):
        return [
            f"{key}: {json.dumps(item, ensure_ascii=False, sort_keys=True)}"
            if isinstance(item, dict)
            else f"{key}: {item}"
            for item in value
        ]
    return [f"{key}: {value}"]


def _dimensions_from_model(model: str) -> int:
    match = re.search(r"(\d+)", model)
    if not match:
        return DEFAULT_EMBEDDING_DIMENSIONS
    return max(32, min(int(match.group(1)), 2048))


def _index_version_for_provider(provider: str, model: str) -> str:
    if provider == DEFAULT_EMBEDDING_PROVIDER and model == DEFAULT_EMBEDDING_MODEL:
        return RETRIEVAL_INDEX_VERSION
    safe_provider = _safe_identifier(provider)
    safe_model = _safe_identifier(model)
    return f"article_{safe_provider}_{safe_model}_index_v1"


def _validate_index_matches_request(
    *,
    index: RetrievalIndex,
    embedding_provider: str,
    embedding_model: str,
) -> None:
    if (
        index.embedding_provider == embedding_provider
        and index.embedding_model == embedding_model
    ):
        return
    raise ValueError(
        "Retrieval index embedding mismatch: "
        f"index has {index.embedding_provider}/{index.embedding_model}, "
        f"request asked for {embedding_provider}/{embedding_model}. "
        "Use a model-specific index path or rebuild explicitly."
    )


def _safe_identifier(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower() or "unknown"


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _int_value(value: object) -> int:
    return value if isinstance(value, int) else 0


def _float_value(value: object) -> float:
    return float(value) if isinstance(value, int | float) else 0.0
