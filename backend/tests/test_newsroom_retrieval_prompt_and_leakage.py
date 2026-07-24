import json

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.services.article_generation_service import (
    NEWSROOM_PROMPT_VERSION_PATHS,
    NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS,
    _newsroom_retrieval_prompt_metadata,
    build_newsroom_retrieval_generation_input,
)
from backend.app.services.newsroom_retrieval_service import (
    CandidateScore,
    RetrievalRecord,
)
from backend.app.services.retrieval_leakage_diagnostic_service import (
    run_retrieval_leakage_diagnostic,
)


def test_retrieval_prompt_payload_separates_examples_and_isolation_rules() -> None:
    record = _record("r1", "author_a", "technology")
    payload = json.loads(
        build_newsroom_retrieval_generation_input(
            brief_record=_brief_record(),
            author_instruction="Write for Oneindia.",
            target_language="ta",
            retrieved_records=[record],
            retrieval_scores=[
                CandidateScore(
                    article_id="r1",
                    author_id="author_a",
                    topic="technology",
                    topic_confidence=0.9,
                    topic_low_confidence=False,
                    topic_multi_category_conflict=False,
                    similarity_score=0.8,
                    topic_boost=0.0,
                    final_score=0.8,
                    selected_rank=1,
                )
            ],
            retrieval_trace={
                "retrieval_mode": "newsroom_v1_retrieval",
                "index_version": "idx",
                "corpus_version": "corpus",
                "retrieved_article_ids": ["r1"],
                "selected_scores": [],
                "ranking_configuration": {"max_context_chars": 1000},
            },
        )
    )

    assert "factual_source_brief" in payload
    assert "retrieved_editorial_examples_for_structure_only" in payload
    isolation = " ".join(payload["retrieval_factual_isolation_rules"])
    assert "Do not copy facts from retrieved examples" in isolation
    assert "names, dates, numbers, places, quotations or events" in isolation
    assert payload["retrieval_prompt_version"] == (
        "oneindia_newsroom_v1.0_retrieval_v1"
    )


def test_existing_newsroom_prompt_files_are_not_retrieval_prompt_files() -> None:
    v1_prompt_path, _ = NEWSROOM_PROMPT_VERSION_PATHS["oneindia_newsroom_v1.0"]
    retrieval_prompt_path, _ = NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS[
        "oneindia_newsroom_v1.0_retrieval_v1"
    ]

    v1_prompt = v1_prompt_path.read_text(encoding="utf-8")
    retrieval_prompt = retrieval_prompt_path.read_text(encoding="utf-8")

    assert "Retrieved examples are not factual evidence" not in v1_prompt
    assert "Retrieved examples are not factual evidence" in retrieval_prompt


def test_retrieval_impact_guard_prompt_is_separate_and_preserves_v1() -> None:
    v1_prompt_path, _ = NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS[
        "oneindia_newsroom_v1.0_retrieval_v1"
    ]
    guard_prompt_path, _ = NEWSROOM_RETRIEVAL_PROMPT_VERSION_PATHS[
        "oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard"
    ]

    v1_prompt = v1_prompt_path.read_text(encoding="utf-8")
    guard_prompt = guard_prompt_path.read_text(encoding="utf-8")

    assert v1_prompt_path != guard_prompt_path
    assert "Impact and implementation guard" not in v1_prompt
    assert "Impact and implementation guard" in guard_prompt
    assert "Do not convert expected outcomes into confirmed outcomes" in guard_prompt
    metadata = _newsroom_retrieval_prompt_metadata(
        "oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard"
    )
    assert metadata["supersedes_retrieval_prompt_version"] == (
        "oneindia_newsroom_v1.0_retrieval_v1"
    )


def test_leakage_diagnostic_flags_copied_date_number_proper_noun_and_quote() -> None:
    retrieved = _record("r1", "author_a", "politics")
    retrieved = _replace_text(
        retrieved,
        (
            'Minister Arjun Rao said "special route opens soon" on 12/06/2026 '
            "with 45 buses."
        ),
    )
    result = run_retrieval_leakage_diagnostic(
        grounded_brief={
            "topic": "Transport update",
            "confirmed_facts": ["The city announced a transport review."],
        },
        generated_article=(
            'Minister Arjun Rao said "special route opens soon" on 12/06/2026 '
            "with 45 buses."
        ),
        retrieved_records=[retrieved],
    )

    types = {item["finding_type"] for item in result["findings"]}
    assert {"date", "number", "proper_noun", "quotation"} <= types
    assert result["status"] == "review_required"


def test_leakage_diagnostic_ignores_brief_supported_and_common_phrasing() -> None:
    retrieved = _replace_text(
        _record("r1", "author_a", "politics"),
        "according to officials the review happened on 12/06/2026",
    )
    result = run_retrieval_leakage_diagnostic(
        grounded_brief={
            "topic": "Transport update",
            "confirmed_facts": [
                "According to officials, the review happened on 12/06/2026."
            ],
        },
        generated_article="according to officials the review happened on 12/06/2026",
        retrieved_records=[retrieved],
    )

    assert result["finding_count"] == 0


def _brief_record() -> GroundedBriefRecord:
    return GroundedBriefRecord(
        brief_id="brief-1",
        source_type="manual",
        source_input_hash="hash",
        source_url=None,
        source_text_excerpt="source",
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief_json=StyleScribeRepository.encode_json(
            {
                "topic": "Digital queue system",
                "one_line_summary": "Hospitals are testing a queue system.",
                "confirmed_facts": ["Hospitals are testing a queue system."],
            }
        ),
        warnings_json=StyleScribeRepository.encode_warnings([]),
        created_at="2026-07-24T00:00:00+00:00",
    )


def _replace_text(record, text: str):
    return type(record)(**{**record.__dict__, "cleaned_article_text": text})


def _record(article_id: str, author_id: str, topic: str) -> RetrievalRecord:
    return RetrievalRecord(
        article_id=article_id,
        author_id=author_id,
        source_path=f"{article_id}.docx",
        cleaned_article_text=f"{article_id} text",
        paragraphs=[f"{article_id} text"],
        word_count=2,
        paragraph_count=1,
        topic=topic,
        topic_confidence=0.9,
        topic_review_flag=False,
        topic_low_confidence=False,
        topic_multi_category_conflict=False,
        file_sha256=f"file-{article_id}",
        text_sha256=f"text-{article_id}",
        normalized_text_sha256=f"norm-{article_id}",
        duplicate_cluster_id=None,
        canonical_article_id=article_id,
        near_duplicate_article_ids=[],
        cleaning_decision={"preparation_status": "accepted"},
        corpus_version="corpus",
        newsroom_profile_version="profile",
        embedding_text=f"{article_id} text",
        embedding_metadata={},
    )
