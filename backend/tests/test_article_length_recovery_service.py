import json
from pathlib import Path

from backend.app.db.repository import (
    ArticleRevisionRecord,
    AuthorStyleProfileRecord,
    DraftEvaluationRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_length_recovery_service import (
    assess_length_recovery_need,
    build_article_length_recovery_input,
    expand_article_to_target_length,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count


def test_under_75_with_sufficient_items_requires_recovery() -> None:
    decision = assess_length_recovery_need(
        {"article_body": "சிறிய செய்தி."},
        _rich_brief(),
        desired_word_count=600,
    )

    assert decision.length_recovery_required is True
    assert decision.short_output_invalid is True
    assert decision.expansion_items_available >= 5
    assert decision.target_min_word_count == 450


def test_under_75_with_thin_brief_does_not_force_recovery() -> None:
    decision = assess_length_recovery_need(
        {"article_body": "சிறிய செய்தி."},
        {"confirmed_facts": ["18 sensors"]},
        desired_word_count=600,
    )

    assert decision.length_recovery_required is False
    assert decision.short_output_invalid is False


def test_expansion_prompt_and_payload_include_rules(tmp_path: Path) -> None:
    repository = _repository_with_recovery_inputs(tmp_path)
    revision = repository.fetch_article_revision("revision-1")
    brief = repository.fetch_grounded_brief("brief-1")
    evaluation = repository.fetch_draft_evaluation("evaluation-1")
    profile = repository.fetch_author_style_profile("profile-1")
    assert revision is not None
    assert brief is not None
    assert evaluation is not None
    assert profile is not None

    prompt = Path("backend/app/prompts/article_length_recovery_prompt.txt").read_text(
        encoding="utf-8"
    )
    payload = build_article_length_recovery_input(
        current_revision=revision,
        brief=brief,
        evaluation=evaluation,
        profile=profile,
        desired_word_count=600,
        article_type="news",
        target_language="ta",
        tone_override="clear",
    )
    decoded = json.loads(payload)

    assert decoded["target_word_count_range"]["minimum_75_percent"] == 450
    assert "Do not add generic filler" in prompt
    assert "Do not invent facts" in prompt
    assert "unsupported future benefits" in prompt
    assert "affected groups" in prompt
    assert "legal/policy context" in prompt
    assert "structured_article_guidance" in decoded


def test_expand_article_to_target_length_saves_expanded_revision(
    tmp_path: Path,
) -> None:
    repository = _repository_with_recovery_inputs(tmp_path)
    revision = repository.fetch_article_revision("revision-1")
    brief = repository.fetch_grounded_brief("brief-1")
    evaluation = repository.fetch_draft_evaluation("evaluation-1")
    profile = repository.fetch_author_style_profile("profile-1")
    assert revision is not None
    assert brief is not None
    assert evaluation is not None
    assert profile is not None

    response = expand_article_to_target_length(
        current_revision=revision,
        brief=brief,
        evaluation=evaluation,
        profile=profile,
        desired_word_count=600,
        article_type="news",
        target_language="ta",
        tone_override="clear",
        repository=repository,
        model_client=SuccessfulExpansionClient(),
    )

    saved = repository.fetch_article_revision(response.revision_id)
    assert saved is not None
    assert approximate_tamil_word_count(saved.revised_article_body) >= 450
    assert response.expansion_items_used


class SuccessfulExpansionClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert (
            "target word range" in user_payload
            or "target_word_count_range" in user_payload
        )
        paragraph = (
            "சென்னை நகரில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் தொடங்கும் என்று "
            "அதிகாரிகள் தெரிவித்துள்ளனர். குறைந்த உயரப்பகுதி சாலைகளின் அருகே "
            "18 சென்சார்கள் அமைக்கப்படும் என்பதும், கனமழை நேரங்களில் குடியிருப்போருக்கு "
            "SMS எச்சரிக்கை அனுப்பப்படும் என்பதும் உறுதிப்படுத்தப்பட்ட தகவல்களாகும்."
        )
        return {
            "headline": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி தொடங்குகிறது.",
            "article_body": "\n\n".join([paragraph for _ in range(17)]),
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "meta_description": "சென்னை வெள்ள எச்சரிக்கை முயற்சி குறித்த செய்தி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "expansion_summary": "Expanded using confirmed facts and SMS context.",
            "expansion_items_used": ["18 sensors", "SMS alerts"],
            "warnings": [],
        }


def _rich_brief() -> dict[str, object]:
    return {
        "confirmed_facts": ["Pilot begins.", "18 sensors."],
        "numbers_and_statistics": ["18 sensors"],
        "affected_groups": ["residents"],
        "dates_or_timeline": ["next month"],
        "quotes": ["officials said"],
        "policy_or_legal_context": ["pilot runs six months"],
    }


def _repository_with_recovery_inputs(tmp_path: Path) -> StyleScribeRepository:
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
            profile_json=StyleScribeRepository.encode_json({"overall_tone": "clear"}),
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
            source_text_excerpt="Officials said a pilot will begin next month.",
            source_language="en",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            brief_json=StyleScribeRepository.encode_json(_rich_brief()),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_draft_evaluation(
        DraftEvaluationRecord(
            evaluation_id="evaluation-1",
            draft_id="draft-1",
            brief_id="brief-1",
            author_id="v_vasanthi",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            evaluation_json=StyleScribeRepository.encode_json(
                {"unsupported_claims": [], "overclaim_phrases": []}
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_article_revision(
        ArticleRevisionRecord(
            revision_id="revision-1",
            draft_id="draft-1",
            evaluation_id="evaluation-1",
            author_id="v_vasanthi",
            revised_headline="சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            revised_subheadline="18 சென்சார்கள் அமைக்கும் முயற்சி.",
            revised_article_body="சிறிய செய்தி.",
            revised_seo_title="சென்னை வெள்ள எச்சரிக்கை",
            revised_meta_description="சுருக்கம்.",
            revised_tags_json="[]",
            revision_summary="Short revision.",
            removed_or_softened_claims_json="[]",
            model_provider="openai",
            model_name="gpt-4o-mini",
            token_usage_json="{}",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository
