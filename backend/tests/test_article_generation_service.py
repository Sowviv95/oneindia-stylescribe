from pathlib import Path

import pytest

from backend.app.db.repository import (
    ArticlePlanRecord,
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_generation_service import (
    ArticleGenerationError,
    build_article_generation_input,
    generate_article_draft,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError


def test_generate_article_draft_saves_to_sqlite(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    client = MockDraftClient()

    response = generate_article_draft(
        author_id="v_vasanthi",
        brief_id="brief-1",
        author_instruction="Write a concise Tamil article.",
        repository=repository,
        model_client=client,
    )
    saved = repository.fetch_article_draft(response.draft_id)

    assert response.draft_id
    assert response.profile_id == "profile-1"
    assert response.brief_id == "brief-1"
    assert response.article_type == "news"
    assert response.desired_word_count == 600
    assert response.include_seo is True
    assert response.draft["headline"] == "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி"
    assert saved is not None
    assert saved.article_type == "news"


def test_missing_style_profile_error(tmp_path: Path) -> None:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()

    with pytest.raises(ArticleGenerationError, match="Generate style profile first"):
        generate_article_draft("missing", "brief-1", repository=repository)


def test_missing_grounded_brief_error(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    with pytest.raises(ArticleGenerationError, match="No grounded brief found"):
        generate_article_draft(
            "v_vasanthi",
            "missing",
            repository=repository,
            model_client=MockDraftClient(),
        )


def test_invalid_json_from_openai_handled(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    with pytest.raises(OpenAIClientError, match="invalid JSON"):
        generate_article_draft(
            "v_vasanthi",
            "brief-1",
            repository=repository,
            model_client=InvalidDraftClient(),
        )


def test_target_language_warning_when_not_ta(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)

    response = generate_article_draft(
        "v_vasanthi",
        "brief-1",
        target_language="en",
        repository=repository,
        model_client=MockDraftClient(),
    )

    assert any("Tamil-focused" in warning for warning in response.warnings)


def test_prompt_input_separates_style_and_facts(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    profile = repository.fetch_latest_author_style_profile("v_vasanthi")
    brief = repository.fetch_grounded_brief("brief-1")
    assert profile is not None
    assert brief is not None

    payload = build_article_generation_input(
        profile_record=profile,
        brief_record=brief,
        author_instruction=None,
        target_language="ta",
        article_type="public_interest",
        desired_word_count=700,
        tone_override="measured public-interest",
        include_seo=True,
    )

    assert "style_profile_for_voice_only" in payload
    assert "grounded_brief_for_facts_only" in payload
    assert "public_interest" in payload
    assert "700" in payload
    assert "article_body_target_word_count_range" in payload
    assert "minimum_75_percent" in payload
    assert "maximum_115_percent" in payload
    assert "measured public-interest" in payload
    assert "outside knowledge" in payload
    assert "style_adaptation_rule" in payload
    assert "FULL_AUTHOR_SAMPLE_CORPUS" not in payload
    assert "FULL_SOURCE_ARTICLE_TEXT" not in payload


def test_prompt_input_includes_article_plan(tmp_path: Path) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    profile = repository.fetch_latest_author_style_profile("v_vasanthi")
    brief = repository.fetch_grounded_brief("brief-1")
    assert profile is not None
    assert brief is not None
    plan = ArticlePlanRecord(
        plan_id="plan-1",
        brief_id="brief-1",
        author_id="v_vasanthi",
        article_type="news",
        desired_word_count=600,
        target_min_word_count=450,
        target_max_word_count=690,
        planned_sections_json=StyleScribeRepository.encode_json(
            [{"section_name": "lead", "target_words": 90}]
        ),
        expansion_items_used_json=StyleScribeRepository.encode_json(["Fact"]),
        claims_to_avoid_json=StyleScribeRepository.encode_json(["Avoid"]),
        plan_summary="Plan summary.",
        model_provider="openai",
        model_name="gpt-4o-mini",
        token_usage_json="{}",
        created_at="2026-01-01T00:00:00+00:00",
    )

    payload = build_article_generation_input(
        profile_record=profile,
        brief_record=brief,
        author_instruction=None,
        target_language="ta",
        desired_word_count=600,
        plan_record=plan,
    )

    assert "grounded_article_plan" in payload
    assert "planned_sections" in payload
    assert "section target words" in payload
    assert "not a short summary" in payload
    assert "article_sections" in payload


def test_plan_based_generation_assembles_article_body_from_sections(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile_and_brief(tmp_path)
    plan = _save_plan(repository)

    response = generate_article_draft(
        author_id="v_vasanthi",
        brief_id="brief-1",
        desired_word_count=600,
        plan_id=plan.plan_id,
        repository=repository,
        model_client=SectionDraftClient(),
    )

    assert "தொடக்க பகுதி" in response.draft["article_body"]
    assert "எண்ணிக்கை பகுதி" in response.draft["article_body"]
    assert response.draft["article_body"] != "சுருக்கம் மட்டும்."
    assert len(response.draft["article_sections"]) == 10
    assert response.draft["generation_mode_used"] == "section_assembled"
    assert response.draft["original_draft_source"] == "section_assembled"


def test_article_generation_prompt_has_tamil_quality_and_length_guidance() -> None:
    prompt = Path("backend/app/prompts/article_generation_prompt.txt").read_text(
        encoding="utf-8"
    )

    assert "native Tamil news phrasing" in prompt
    assert "desired_word_count" in prompt
    assert "450-690 words" in prompt
    assert "7-10 developed paragraphs" in prompt
    assert "complete article" in prompt
    assert "article_sections" in prompt
    assert "section_text" in prompt
    assert "ruling" in prompt
    assert "தீர்ப்பு" in prompt
    assert "இந்திய குடியுரிமை பெற்றவர்கள்" in prompt


class MockDraftClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        if "backend-assembled section" in system_prompt:
            return _section_response()
        assert "Use only the grounded brief for facts" in system_prompt
        assert "Use the author style profile as a writing influence" in system_prompt
        assert "style_profile_for_voice_only" in user_payload
        assert "grounded_brief_for_facts_only" in user_payload
        assert "desired_word_count" in user_payload
        assert "article_body_target_word_count_range" in user_payload
        assert "article_type" in user_payload
        return {
            "headline": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "மூன்று பகுதிகளில் அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி தொடங்க உள்ளது.",
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை திட்டம்",
            "meta_description": "சென்னையில் புதிய வெள்ள எச்சரிக்கை முயற்சி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "fact_usage_notes": ["18 sensors preserved."],
            "style_usage_notes": ["Used emotional but grounded tone."],
        }


class InvalidDraftClient(MockDraftClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI returned invalid JSON.")


class SectionDraftClient(MockDraftClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        if "backend-assembled section" in system_prompt:
            return _section_response()
        assert "article_sections" in system_prompt
        assert "grounded_article_plan" in user_payload
        return {
            "headline": "திட்டமிட்ட கட்டுரை",
            "subheadline": "பிரிவுகளின் அடிப்படையில் எழுதப்பட்டது.",
            "article_body": "சுருக்கம் மட்டும்.",
            "article_sections": [
                {
                    "section_name": "lead",
                    "target_words": 90,
                    "section_text": "தொடக்க பகுதி உறுதிப்படுத்தப்பட்ட தகவல்களுடன் எழுதப்பட்டது.",
                    "grounded_facts_used": ["18 sensors"],
                },
                {
                    "section_name": "numbers",
                    "target_words": 90,
                    "section_text": (
                        "எண்ணிக்கை பகுதி 18 சென்சார்கள் பற்றிய தகவலை "
                        "பயன்படுத்துகிறது."
                    ),
                    "grounded_facts_used": ["18 sensors"],
                },
            ],
            "seo_title": "திட்டமிட்ட கட்டுரை",
            "meta_description": "பிரிவுகளின் அடிப்படையில் எழுதப்பட்ட செய்தி.",
            "suggested_tags": ["சென்னை"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
        }


def _repository_with_profile_and_brief(tmp_path: Path) -> StyleScribeRepository:
    repository = StyleScribeRepository(tmp_path / "stylescribe.db")
    repository.initialize_schema()
    repository.save_author_style_profile(
        AuthorStyleProfileRecord(
            profile_id="profile-1",
            author_id="v_vasanthi",
            snapshot_id="snapshot-1",
            language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            profile_json=StyleScribeRepository.encode_json(
                {
                    "overall_tone": "Measured",
                    "headline_style": "Direct",
                    "intro_style": "Context first",
                    "paragraph_style": "Compact",
                    "tamil_register": "Conversational Tamil",
                    "dos": ["Stay grounded"],
                    "donts": ["Do not invent facts"],
                }
            ),
            source_excerpt_refs_json="[]",
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_grounded_brief(
        GroundedBriefRecord(
            brief_id="brief-1",
            source_type="text",
            source_input_hash="hash",
            source_url=None,
            source_text_excerpt="Short excerpt only",
            source_language="en",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            brief_json=StyleScribeRepository.encode_json(
                {
                    "topic": "Flood warning pilot",
                    "one_line_summary": "Chennai starts a flood warning pilot.",
                    "confirmed_facts": ["18 sensors will be installed."],
                    "claims_to_avoid": ["Do not claim it has succeeded."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository


def _save_plan(repository: StyleScribeRepository) -> ArticlePlanRecord:
    plan = ArticlePlanRecord(
        plan_id="plan-1",
        brief_id="brief-1",
        author_id="v_vasanthi",
        article_type="news",
        desired_word_count=600,
        target_min_word_count=450,
        target_max_word_count=690,
        planned_sections_json=StyleScribeRepository.encode_json(
            [
                {"section_name": "lead", "target_words": 90},
                {"section_name": "numbers", "target_words": 90},
            ]
        ),
        expansion_items_used_json=StyleScribeRepository.encode_json(["18 sensors"]),
        claims_to_avoid_json=StyleScribeRepository.encode_json(["Avoid claims"]),
        plan_summary="Two-section plan.",
        model_provider="openai",
        model_name="gpt-4o-mini",
        token_usage_json="{}",
        created_at="2026-01-01T00:00:00+00:00",
    )
    repository.save_article_plan(plan)
    return plan


def _section_response() -> dict[str, object]:
    return {
        "section_name": "lead",
        "target_words": 100,
        "section_text": (
            "தொடக்க பகுதி உறுதிப்படுத்தப்பட்ட தகவல்களுடன் எழுதப்பட்டது. "
            "எண்ணிக்கை பகுதி 18 சென்சார்கள் பற்றிய தகவலை பயன்படுத்துகிறது. "
            "இந்த பகுதி திட்டத்தின் முக்கிய விவரங்களை வாசகர்களுக்கு தெளிவாக "
            "சொல்கிறது. ஆறு மாத சோதனை முயற்சி பற்றிய பின்னணியும் இதில் "
            "இணைக்கப்படுகிறது. அதனால் செய்தி சுருக்கமாக இல்லாமல் "
            "தொடர்புடைய விவரங்களுடன் அமைகிறது. குடிமை அமைப்பு கூறிய "
            "தகவல்கள் மட்டுமே இங்கு பயன்படுத்தப்படுகின்றன. தாழ்வான "
            "தெருக்கள், மழை எச்சரிக்கை, குடியிருப்பாளர்களுக்கான எஸ்.எம்.எஸ் "
            "அறிவிப்பு ஆகியவை கவனமாக இணைக்கப்படுகின்றன. நகரமெங்கும் "
            "விரிவாக்கம் குறித்து முடிவு பின்னர் எடுக்கப்படும் என்பதும் "
            "கட்டுரையின் வரம்பை தெளிவுபடுத்துகிறது. அதிகாரிகள் கூறிய "
            "ஆறு மாத கால எல்லை வாசகர்களுக்கு திட்டத்தின் தற்போதைய நிலையை "
            "புரியவைக்கிறது. எந்த பலனும் முன்கூட்டியே உறுதியாக கூறப்படாமல் "
            "சோதனை, சென்சார், எச்சரிக்கை, குடியிருப்பாளர் தகவல் ஆகிய "
            "உறுதிப்படுத்தப்பட்ட அம்சங்கள் மட்டும் இந்த பகுதியில் "
            "விரிவாக பயன்படுத்தப்படுகின்றன."
        ),
        "grounded_facts_used": ["18 sensors"],
        "warnings": [],
    }
