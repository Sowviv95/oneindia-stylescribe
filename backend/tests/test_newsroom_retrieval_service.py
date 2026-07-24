import json
from pathlib import Path

import backend.app.services.newsroom_retrieval_service as retrieval_service
from backend.app.services.newsroom_retrieval_service import (
    RetrievalIndex,
    RetrievalQuery,
    RetrievalRankingConfig,
    RetrievalRecord,
    build_or_load_index,
    build_retrieval_records,
    load_retrieval_index,
    make_embedding_provider,
    retrieve_examples,
    save_retrieval_index,
)


def test_retrieval_records_include_accepted_only_and_preserve_topic_fields(
    tmp_path: Path,
) -> None:
    cleaned = tmp_path / "cleaned.jsonl"
    cleaned.write_text(
        "\n".join(
            [
                json.dumps(
                    _article(
                        "a1",
                        preparation_status="accepted",
                        topic="technology",
                        topic_confidence=0.9,
                        topic_review_flag=False,
                    )
                ),
                json.dumps(
                    _article("a2", preparation_status="review_required"),
                ),
                json.dumps(
                    _article("a3", preparation_status="rejected"),
                ),
            ]
        ),
        encoding="utf-8",
    )

    records = build_retrieval_records(
        cleaned_articles_jsonl=cleaned,
        near_duplicates_jsonl=tmp_path / "missing.jsonl",
        output_path=tmp_path / "records.jsonl",
    )

    assert [record.article_id for record in records] == ["a1"]
    record = records[0]
    assert record.topic == "technology"
    assert record.topic_confidence == 0.9
    assert record.topic_review_flag is False
    assert record.topic_low_confidence is False
    assert record.topic_multi_category_conflict is False
    assert record.paragraphs == ["First paragraph.", "Second paragraph."]
    assert record.file_sha256 == "file-a1"
    assert record.text_sha256 == "text-a1"


def test_ranking_uses_similarity_before_topic_boost_and_never_hard_filters() -> None:
    index = _index(
        [
            _record("similar_other", "author_a", "business", confidence=0.9),
            _record("topic_match_weaker", "author_b", "technology", confidence=0.9),
        ],
        embeddings=[[1.0, 0.0], [0.92, 0.0]],
    )
    result = retrieve_examples(
        index=index,
        query=_query("technology", confidence=0.9),
        config=RetrievalRankingConfig(
            top_k=2,
            candidate_pool_size=2,
            topic_boost_enabled=True,
            topic_boost_weight=0.03,
            max_examples_per_author=2,
        ),
        embedding_provider=_FakeProvider([1.0, 0.0]),
    )

    assert [score.article_id for score in result.selected_scores] == [
        "similar_other",
        "topic_match_weaker",
    ]
    assert result.selected_scores[0].topic_boost == 0.0
    assert result.selected_scores[1].topic_boost == 0.03


def test_topic_boost_requires_high_confidence_non_conflict_match() -> None:
    index = _index(
        [
            _record("high", "author_a", "technology", confidence=0.9),
            _record("low", "author_b", "technology", confidence=0.2, low=True),
            _record(
                "conflict",
                "author_c",
                "technology",
                confidence=0.9,
                conflict=True,
            ),
        ],
        embeddings=[[0.7, 0.0], [0.7, 0.0], [0.7, 0.0]],
    )

    result = retrieve_examples(
        index=index,
        query=_query("technology", confidence=0.9),
        config=RetrievalRankingConfig(
            top_k=3,
            candidate_pool_size=3,
            topic_boost_enabled=True,
            topic_boost_weight=0.05,
            max_examples_per_author=3,
        ),
        embedding_provider=_FakeProvider([1.0, 0.0]),
    )

    boosts = {score.article_id: score.topic_boost for score in result.selected_scores}
    assert boosts == {"high": 0.05, "low": 0.0, "conflict": 0.0}


def test_source_duplicate_near_duplicate_and_author_diversity_exclusions() -> None:
    records = [
        _record("source", "author_a", "technology"),
        _record("exact", "author_b", "technology", text_hash="same"),
        _record("near", "author_c", "technology", near=["source"]),
        _record("a1", "author_a", "technology"),
        _record("a2", "author_a", "technology"),
        _record("b1", "author_b", "technology"),
    ]
    index = _index(records, embeddings=[[1.0, 0.0]] * len(records))

    result = retrieve_examples(
        index=index,
        query=_query("technology"),
        config=RetrievalRankingConfig(
            top_k=2,
            candidate_pool_size=6,
            max_examples_per_author=1,
        ),
        embedding_provider=_FakeProvider([1.0, 0.0]),
        source_article_id="source",
        source_text_hash="same",
    )

    assert [score.author_id for score in result.selected_scores] == [
        "author_a",
        "author_b",
    ]
    reasons = {item["article_id"]: item["reason"] for item in result.exclusions}
    assert reasons["source"] == "source_article"
    assert reasons["exact"] == "exact_duplicate"
    assert reasons["near"] == "near_duplicate_sibling"
    assert reasons["a2"] == "author_diversity_limit"


def test_index_metadata_persistence_and_reuse_validation(tmp_path: Path) -> None:
    index_path = tmp_path / "semantic" / "index.json"
    index = _index(
        [_record("a1", "author_a", "technology")],
        embeddings=[[1.0, 0.0]],
        provider="sentence_transformers",
        model="intfloat/multilingual-e5-small",
        dimensions=384,
        index_version="article_sentence_transformers_index_v1",
    )
    save_retrieval_index(index, index_path)

    loaded = build_or_load_index(
        index_path=index_path,
        records_path=tmp_path / "records.jsonl",
        embedding_provider="sentence_transformers",
        embedding_model="intfloat/multilingual-e5-small",
    )

    assert loaded.embedding_provider == "sentence_transformers"
    assert loaded.embedding_model == "intfloat/multilingual-e5-small"
    assert loaded.embedding_dimensions == 384
    assert loaded.normalization_method == "l2"
    assert loaded.similarity_metric == "cosine"
    assert loaded.dependency_version == "test-version"


def test_index_mismatch_requires_model_specific_path_or_rebuild(tmp_path: Path) -> None:
    index_path = tmp_path / "index.json"
    save_retrieval_index(
        _index([_record("a1", "author_a", "technology")], [[1.0, 0.0]]),
        index_path,
    )

    try:
        build_or_load_index(
            index_path=index_path,
            records_path=tmp_path / "records.jsonl",
            embedding_provider="sentence_transformers",
            embedding_model="intfloat/multilingual-e5-small",
        )
    except ValueError as exc:
        assert "embedding mismatch" in str(exc)
    else:
        raise AssertionError("Expected index embedding mismatch.")


def test_legacy_index_metadata_loads_with_backward_compatible_defaults(
    tmp_path: Path,
) -> None:
    index = _index([_record("a1", "author_a", "technology")], [[1.0, 0.0]])
    payload = {
        "index_version": index.index_version,
        "corpus_version": index.corpus_version,
        "newsroom_profile_version": index.newsroom_profile_version,
        "embedding_provider": index.embedding_provider,
        "embedding_model": index.embedding_model,
        "embedding_dimensions": index.embedding_dimensions,
        "record_count": index.record_count,
        "created_at": index.created_at,
        "input_hash": index.input_hash,
        "records": [record.__dict__ for record in index.records],
        "embeddings": index.embeddings,
    }
    path = tmp_path / "legacy_index.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_retrieval_index(path)

    assert loaded.normalization_method == "l2"
    assert loaded.similarity_metric == "cosine"
    assert loaded.dependency_version is None


def test_sentence_transformers_provider_selection_uses_fake_dependency(
    monkeypatch,
) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model: str, *, device: str) -> None:
            self.model = model
            self.device = device

        def get_sentence_embedding_dimension(self) -> int:
            return 3

        def encode(self, texts: list[str], **kwargs) -> list[list[float]]:
            assert kwargs["normalize_embeddings"] is True
            return [[1.0, 0.0, 0.0] for _text in texts]

    class FakeModule:
        SentenceTransformer = FakeSentenceTransformer

    monkeypatch.setattr(retrieval_service, "import_module", lambda name: FakeModule)
    monkeypatch.setattr(
        retrieval_service,
        "_package_version",
        lambda package: "fake-version",
    )

    provider = make_embedding_provider(
        "sentence_transformers",
        "intfloat/multilingual-e5-small",
    )

    assert provider.provider == "sentence_transformers"
    assert provider.model == "intfloat/multilingual-e5-small"
    assert provider.dimensions == 3
    assert provider.dependency_version == "fake-version"
    assert provider.embed(["query"]) == [[1.0, 0.0, 0.0]]


def _article(
    article_id: str,
    *,
    preparation_status: str,
    topic: str = "other",
    topic_confidence: float = 0.0,
    topic_review_flag: bool = True,
) -> dict[str, object]:
    return {
        "article_id": article_id,
        "author_id": "author",
        "relative_source_path": f"{article_id}.docx",
        "body_text": "First paragraph.\nSecond paragraph.",
        "paragraph_sequence": ["First paragraph.", "Second paragraph."],
        "word_count": 4,
        "paragraph_count": 2,
        "preparation_status": preparation_status,
        "article_usable": preparation_status != "rejected",
        "topic": topic,
        "topic_confidence": topic_confidence,
        "topic_review_flag": topic_review_flag,
        "topic_low_confidence": topic_confidence < 0.5,
        "topic_multi_category_conflict": False,
        "file_sha256": f"file-{article_id}",
        "text_sha256": f"text-{article_id}",
        "normalized_text_sha256": f"norm-{article_id}",
        "duplicate_cluster_id": None,
        "canonical_article_id": article_id,
    }


def _record(
    article_id: str,
    author_id: str,
    topic: str,
    *,
    confidence: float = 0.9,
    low: bool = False,
    conflict: bool = False,
    text_hash: str | None = None,
    near: list[str] | None = None,
) -> RetrievalRecord:
    return RetrievalRecord(
        article_id=article_id,
        author_id=author_id,
        source_path=f"{article_id}.docx",
        cleaned_article_text=f"{article_id} text",
        paragraphs=[f"{article_id} text"],
        word_count=2,
        paragraph_count=1,
        topic=topic,
        topic_confidence=confidence,
        topic_review_flag=low or conflict,
        topic_low_confidence=low,
        topic_multi_category_conflict=conflict,
        file_sha256=f"file-{article_id}",
        text_sha256=text_hash or f"text-{article_id}",
        normalized_text_sha256=text_hash or f"norm-{article_id}",
        duplicate_cluster_id=None,
        canonical_article_id=article_id,
        near_duplicate_article_ids=near or [],
        cleaning_decision={"preparation_status": "accepted"},
        corpus_version="corpus",
        newsroom_profile_version="profile",
        embedding_text=f"{article_id} text",
        embedding_metadata={},
    )


def _query(topic: str, *, confidence: float = 0.9) -> RetrievalQuery:
    return RetrievalQuery(
        text="query",
        topic=topic,
        topic_confidence=confidence,
        topic_review_flag=False,
        topic_low_confidence=False,
        topic_multi_category_conflict=False,
        metadata={},
    )


def _index(
    records: list[RetrievalRecord],
    embeddings: list[list[float]],
    *,
    provider: str = "fake",
    model: str = "fake",
    dimensions: int = 2,
    index_version: str = "idx",
) -> RetrievalIndex:
    return RetrievalIndex(
        index_version=index_version,
        corpus_version="corpus",
        newsroom_profile_version="profile",
        embedding_provider=provider,
        embedding_model=model,
        embedding_dimensions=dimensions,
        normalization_method="l2",
        similarity_metric="cosine",
        dependency_version="test-version" if provider != "fake" else None,
        record_count=len(records),
        created_at="now",
        input_hash="hash",
        records=records,
        embeddings=embeddings,
    )


class _FakeProvider:
    provider = "fake"
    model = "fake"
    dimensions = 2
    normalization_method = "l2"
    similarity_metric = "cosine"
    dependency_version = None

    def __init__(self, vector: list[float]) -> None:
        self.vector = vector

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vector for _text in texts]
