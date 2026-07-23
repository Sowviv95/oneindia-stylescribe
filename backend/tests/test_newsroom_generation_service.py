import json

from backend.app.db.repository import GroundedBriefRecord, StyleScribeRepository
from backend.app.services.article_generation_service import (
    NEWSROOM_PROMPT_VERSION_PATHS,
    build_newsroom_generation_input,
)


def test_newsroom_input_separates_source_facts_from_editorial_guidance() -> None:
    brief = _brief_record()

    payload = json.loads(
        build_newsroom_generation_input(
            brief_record=brief,
            author_instruction="Write a careful Tamil news article.",
            target_language="ta",
            desired_word_count=600,
            prompt_metadata={
                "prompt_version": "oneindia_newsroom_v1.0",
                "newsroom_profile_version": (
                    "oneindia_tamil_generic_newsroom_sprint2"
                ),
            },
        )
    )

    assert "factual_source_brief" in payload
    assert "generic_newsroom_editorial_rules" in payload
    assert "style_profile_for_voice_only" not in payload
    assert payload["separation_rule"].startswith("factual_source_brief")


def test_newsroom_phrase_bank_wording_is_optional_not_mandatory() -> None:
    payload = json.loads(
        build_newsroom_generation_input(
            brief_record=_brief_record(),
            author_instruction=None,
            target_language="ta",
        )
    )

    guidance = payload["generic_newsroom_editorial_rules"]["phrase_bank_policy"]
    prohibited = " ".join(payload["prohibited_behaviours"])
    assert "optional" in guidance
    assert "Do not force" in prohibited
    assert "preferred_phrase_bank" not in payload


def test_newsroom_v1_prompt_file_remains_unmodified_by_length_calibration() -> None:
    v1_prompt_path, _ = NEWSROOM_PROMPT_VERSION_PATHS["oneindia_newsroom_v1.0"]
    v1_1_prompt_path, _ = NEWSROOM_PROMPT_VERSION_PATHS[
        "oneindia_newsroom_v1.1_length_calibrated"
    ]

    v1_prompt = v1_prompt_path.read_text(encoding="utf-8")
    v1_1_prompt = v1_1_prompt_path.read_text(encoding="utf-8")

    assert "Length-control rules:" not in v1_prompt
    assert "Length-control rules:" in v1_1_prompt
    assert "Do not pad with repeated lede facts" in v1_1_prompt


def _brief_record() -> GroundedBriefRecord:
    return GroundedBriefRecord(
        brief_id="brief-1",
        source_type="manual",
        source_input_hash="hash",
        source_url=None,
        source_text_excerpt="The civic body announced a new schedule.",
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief_json=StyleScribeRepository.encode_json(
            {
                "confirmed_facts": [
                    "The civic body announced a new waste collection schedule."
                ],
                "claims_to_avoid": ["Do not claim the issue is solved."],
            }
        ),
        warnings_json=StyleScribeRepository.encode_warnings([]),
        created_at="2026-07-23T00:00:00+00:00",
    )
