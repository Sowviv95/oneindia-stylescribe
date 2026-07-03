import json
from pathlib import Path

from backend.app.db.repository import (
    ArticleDraftRecord,
    AuthorStyleProfileRecord,
    DraftEvaluationRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.article_revision_service import (
    ArticleRevisionError,
    _apply_revision_patch_mode,
    build_article_revision_input,
    cleanup_revised_article_tamil,
    render_revision_review_markdown,
    revise_article_grounding,
)
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count


def test_revision_prompt_uses_evaluation_feedback_and_grounded_brief(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path)
    draft = repository.fetch_article_draft("draft-1")
    brief = repository.fetch_grounded_brief("brief-1")
    evaluation = repository.fetch_draft_evaluation("evaluation-1")
    profile = repository.fetch_author_style_profile("profile-1")
    assert draft is not None
    assert brief is not None
    assert evaluation is not None
    assert profile is not None

    payload = build_article_revision_input(draft, brief, evaluation, profile)

    assert "grounding_evaluation_feedback" in payload
    assert "grounding_findings_to_patch" in payload
    assert "unsupported_claims" in payload
    assert "grounded_brief_for_facts_only" in payload
    assert "original_generated_draft" in payload
    assert "desired_word_count" in payload
    assert "original_draft_approximate_word_count" in payload
    assert "revised_article_target_word_count_range" in payload
    assert "source_excerpt" in payload
    assert "length_preservation_rule" in payload
    assert "Do not introduce new facts" in payload
    assert "75% to 115%" in payload
    assert "rewrite unsupported sentences into neutral grounded context" in payload
    assert "FULL_AUTHOR_SAMPLE_CORPUS" not in payload
    assert "FULL_SOURCE_ARTICLE_TEXT" not in payload

    decoded = json.loads(payload)
    assert decoded["desired_word_count"] == 600
    assert decoded["original_draft_approximate_word_count"] > 0
    assert decoded["revised_article_target_word_count_range"] == {
        "minimum_75_percent": 450,
        "target": 600,
        "maximum_115_percent": 690,
    }
    assert decoded["grounded_brief_for_facts_only"]["source_excerpt"]
    findings = decoded["grounding_evaluation_feedback"]["grounding_findings_to_patch"]
    assert findings[0]["issue_type"] == "unsupported_claim"
    assert findings[0]["finding_id"] == "unsupported_claim_1"


def test_revision_prompt_instructs_english_leftover_cleanup() -> None:
    prompt = Path("backend/app/prompts/article_revision_prompt.txt").read_text(
        encoding="utf-8"
    )

    assert "ruling" in prompt
    assert "தீர்ப்பு" in prompt
    assert "H-1B or SMS" in prompt
    assert "Do not simply delete all risky paragraphs" in prompt
    assert "Do not shrink a full article into a short summary" in prompt
    assert "unsupported paragraphs into neutral grounded context" in prompt
    assert "450-690 words" in prompt
    assert "Do not add filler or repeat points" in prompt
    assert "publication-quality Tamil" in prompt
    assert "Preserve the requested article length" in prompt
    assert "Correct contextual mistranslations" in prompt
    assert "Do not rewrite the full article" in prompt
    assert "Do not produce a complete article body" in prompt
    assert "Return only JSON patch instructions" in prompt
    assert "First handle unsupported claims" in prompt
    assert "source_finding_id" in prompt


def test_revision_payload_handles_missing_desired_word_count(tmp_path: Path) -> None:
    repository = _repository_with_revision_inputs(tmp_path, desired_word_count=None)
    draft = repository.fetch_article_draft("draft-1")
    brief = repository.fetch_grounded_brief("brief-1")
    evaluation = repository.fetch_draft_evaluation("evaluation-1")
    profile = repository.fetch_author_style_profile("profile-1")
    assert draft is not None
    assert brief is not None
    assert evaluation is not None
    assert profile is not None

    payload = json.loads(
        build_article_revision_input(draft, brief, evaluation, profile)
    )

    assert payload["desired_word_count"] is None
    assert payload["revised_article_target_word_count_range"] is None
    assert payload["original_draft_approximate_word_count"] > 0


def test_revise_article_grounding_saves_revision_and_softens_claims(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path)

    response = revise_article_grounding(
        "draft-1",
        repository=repository,
        model_client=MockRevisionClient(),
    )
    saved = repository.fetch_latest_article_revision("draft-1")

    assert response.revision_id
    assert response.evaluation_id == "evaluation-1"
    assert "பாதுகாப்பு உறுதி" not in response.revised_draft["headline"]
    assert "முக்கியமான மாற்றம்" not in response.revised_draft["article_body"]
    assert "ruling" not in response.revised_draft["headline"]
    assert "ruling" not in response.revised_draft["subheadline"]
    assert "ruling" not in response.revised_draft["meta_description"]
    assert response.removed_or_softened_claims
    assert saved is not None
    assert saved.revision_id == response.revision_id


def test_revise_article_grounding_applies_safe_patch_mode(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path, desired_word_count=None)

    response = revise_article_grounding(
        "draft-1",
        repository=repository,
        model_client=PatchRevisionClient(),
    )

    assert response.revision_mode == "patch"
    assert response.revision_patch_count == 3
    assert response.revision_patches_applied_count == 1
    assert response.revision_patches_skipped_count == 2
    assert response.unsupported_claim_findings_count == 1
    assert response.unsupported_claim_patch_count == 1
    assert response.unsupported_claim_patches_applied_count == 1
    assert response.unsupported_claims_cleared_by_patch is True
    assert response.revision_rejected_for_length_collapse is False
    assert response.revised_article_source == "patch_applied"
    assert (
        "à®®à¯à®•à¯à®•à®¿à®¯à®®à®¾à®© à®®à®¾à®±à¯à®±à®®à¯"
        not in response.revised_draft["article_body"]
    )
    assert "18 à®šà¯†à®©à¯à®šà®¾à®°à¯à®•à®³à¯" in response.revised_draft["article_body"]
    assert any(
        "low_confidence" in reason
        for reason in response.revision_patch_skipped_reasons
    )
    assert any(
        "target_not_found" in reason
        for reason in response.revision_patch_skipped_reasons
    )


def test_unsupported_claim_patch_uses_normalized_match() -> None:
    result = _apply_revision_patch_mode(
        revision_instruction={
            "patches": [
                {
                    "issue_type": "unsupported_claim",
                    "target_text": "unsupported   safety claim",
                    "replacement_text": "grounded cautious claim",
                    "confidence": "high",
                    "source_finding_id": "unsupported_claim_1",
                    "resolves_blocker": True,
                    "blocker_type": "unsupported_claim",
                }
            ],
            "notes": [],
        },
        original_draft={
            "article_body": "Opening.\nunsupported safety claim continues.",
            "headline": "",
            "subheadline": "",
            "seo_title": "",
            "meta_description": "",
            "suggested_tags": [],
        },
        desired_word_count=None,
        unsupported_claim_findings=[
            {
                "finding_id": "unsupported_claim_1",
                "claim_text": "unsupported   safety claim",
                "issue_type": "unsupported_claim",
            }
        ],
    )

    assert "grounded cautious claim continues" in result.revision["article_body"]
    assert result.metadata["unsupported_claim_patches_applied_count"] == 1
    details = result.metadata["unsupported_claim_patch_details"]
    assert isinstance(details, list)
    assert details[0]["match_mode"] == "normalized"


def test_unsupported_claim_patch_skips_multiple_matches() -> None:
    result = _apply_revision_patch_mode(
        revision_instruction={
            "patches": [
                {
                    "issue_type": "unsupported_claim",
                    "target_text": "unsupported claim",
                    "replacement_text": "grounded claim",
                    "confidence": "high",
                    "source_finding_id": "unsupported_claim_1",
                    "resolves_blocker": True,
                    "blocker_type": "unsupported_claim",
                }
            ],
            "notes": [],
        },
        original_draft={
            "article_body": "unsupported claim. unsupported claim.",
            "headline": "",
            "subheadline": "",
            "seo_title": "",
            "meta_description": "",
            "suggested_tags": [],
        },
        desired_word_count=None,
        unsupported_claim_findings=[
            {
                "finding_id": "unsupported_claim_1",
                "claim_text": "unsupported claim",
                "issue_type": "unsupported_claim",
            }
        ],
    )

    assert result.metadata["unsupported_claim_patches_applied_count"] == 0
    assert result.metadata["unsupported_claim_patches_skipped_count"] == 1
    skipped = result.metadata["unsupported_claim_patch_skipped_reasons"]
    assert isinstance(skipped, list)
    assert "target_found_multiple_times" in skipped[0]


def test_low_confidence_unsupported_claim_patch_is_skipped() -> None:
    result = _apply_revision_patch_mode(
        revision_instruction={
            "patches": [
                {
                    "issue_type": "unsupported_claim",
                    "target_text": "unsupported claim",
                    "replacement_text": "grounded claim",
                    "confidence": "low",
                    "source_finding_id": "unsupported_claim_1",
                    "resolves_blocker": True,
                    "blocker_type": "unsupported_claim",
                }
            ],
            "notes": [],
        },
        original_draft={
            "article_body": "unsupported claim.",
            "headline": "",
            "subheadline": "",
            "seo_title": "",
            "meta_description": "",
            "suggested_tags": [],
        },
        desired_word_count=None,
        unsupported_claim_findings=[
            {
                "finding_id": "unsupported_claim_1",
                "claim_text": "unsupported claim",
                "issue_type": "unsupported_claim",
            }
        ],
    )

    assert result.metadata["unsupported_claim_patches_applied_count"] == 0
    skipped = result.metadata["unsupported_claim_patch_skipped_reasons"]
    assert isinstance(skipped, list)
    assert "low_confidence" in skipped[0]


def test_unsupported_claim_patch_can_use_source_finding_target() -> None:
    result = _apply_revision_patch_mode(
        revision_instruction={
            "patches": [
                {
                    "issue_type": "unsupported_claim",
                    "target_text": "model paraphrased target",
                    "replacement_text": "grounded claim",
                    "confidence": "high",
                    "source_finding_id": "unsupported_claim_1",
                    "resolves_blocker": True,
                    "blocker_type": "unsupported_claim",
                }
            ],
            "notes": [],
        },
        original_draft={
            "article_body": "Opening. unsupported claim.",
            "headline": "",
            "subheadline": "",
            "seo_title": "",
            "meta_description": "",
            "suggested_tags": [],
        },
        desired_word_count=None,
        unsupported_claim_findings=[
            {
                "finding_id": "unsupported_claim_1",
                "claim_text": "unsupported claim",
                "issue_type": "unsupported_claim",
            }
        ],
    )

    assert "grounded claim" in result.revision["article_body"]
    details = result.metadata["unsupported_claim_patch_details"]
    assert isinstance(details, list)
    assert details[0]["match_mode"] == "finding_exact"


def test_revise_article_grounding_rejects_evaluation_for_other_draft(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path)
    repository.save_draft_evaluation(
        DraftEvaluationRecord(
            evaluation_id="evaluation-other",
            draft_id="other-draft",
            brief_id="brief-1",
            author_id="v_vasanthi",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            evaluation_json=StyleScribeRepository.encode_json(
                {"overall_risk": "high"}
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:01:00+00:00",
        )
    )

    try:
        revise_article_grounding(
            "draft-1",
            evaluation_id="evaluation-other",
            repository=repository,
            model_client=MockRevisionClient(),
        )
    except ArticleRevisionError as exc:
        assert "does not belong" in str(exc)
    else:
        raise AssertionError("Expected ArticleRevisionError")


def test_revise_article_grounding_cleans_ruling_in_persisted_fields(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path)

    response = revise_article_grounding(
        "draft-1",
        repository=repository,
        model_client=RulingRevisionClient(),
    )
    combined = "\n".join(
        str(response.revised_draft[field])
        for field in (
            "headline",
            "subheadline",
            "article_body",
            "seo_title",
            "meta_description",
        )
    )

    assert "ruling" not in combined
    assert "இந்த தீர்ப்பு" in combined
    assert "H-1B" in combined
    assert "SMS" in combined
    assert "Tamil-English mixed phrasing was cleaned" in response.revision_summary


def test_revise_article_grounding_passes_length_context_to_model(
    tmp_path: Path,
) -> None:
    repository = _repository_with_revision_inputs(tmp_path)
    client = LengthAwareRevisionClient()

    response = revise_article_grounding(
        "draft-1",
        repository=repository,
        model_client=client,
    )

    assert client.payload is not None
    assert client.payload["desired_word_count"] == 600
    assert client.payload["original_generated_draft"]["article_body_word_count"] > 0
    target_range = client.payload["revised_article_target_word_count_range"]
    assert target_range["minimum_75_percent"] == 450
    assert approximate_tamil_word_count(response.revised_draft["article_body"]) >= 450
    assert "18" in response.revised_draft["article_body"]


def test_revision_review_export_includes_length_details() -> None:
    markdown = render_revision_review_markdown(
        cleaned_source_excerpt="source",
        brief={
            "topic": "Flood warning pilot",
            "one_line_summary": "Pilot starts.",
            "confirmed_facts": ["18 sensors"],
            "claims_to_avoid": ["Do not claim success."],
        },
        original_draft={"article_body": "original"},
        initial_evaluation={
            "grounding_score": 75,
            "editorial_readiness": "revision_required",
        },
        revised_draft={"article_body": "revised"},
        revision_summary="Softened claims.",
        final_evaluation={
            "grounding_score": 82,
            "editorial_readiness": "review_required",
        },
        tamil_quality_status="warning",
        tamil_quality_warnings=["short"],
        requested_word_count=600,
        original_draft_word_count=520,
        final_article_word_count=430,
        length_status="warning",
        length_warning_reason=(
            "Final article body is materially shorter than requested."
        ),
        final_article_word_count_ratio=0.717,
        generation_metadata={
            "revision_mode": "patch",
            "revision_patch_count": 2,
            "revision_patches_applied_count": 1,
            "revision_patches_skipped_count": 1,
            "revision_patch_skipped_reasons": ["patch 2: low confidence"],
            "revision_output_word_count": 520,
            "revision_rejected_for_length_collapse": False,
            "revision_rejected_reason": None,
            "revised_article_source": "patch_applied",
            "unsupported_claim_findings_count": 1,
            "unsupported_claim_patch_count": 1,
            "unsupported_claim_patches_applied_count": 1,
            "unsupported_claim_patches_skipped_count": 0,
            "unsupported_claim_patch_skipped_reasons": [],
            "unsupported_claims_unresolved_count": 0,
            "unsupported_claims_cleared_by_patch": True,
        },
    )

    assert "Original draft approximate word count: 520" in markdown
    assert "Final article approximate word count: 430" in markdown
    assert "Requested desired_word_count: 600" in markdown
    assert "Length warning reason: Final article body is materially shorter" in markdown
    assert "Final article word count ratio: 0.717" in markdown
    assert "Revision mode: patch" in markdown
    assert "Revision patches applied: 1" in markdown
    assert "Revision patch skipped reasons" in markdown
    assert "Unsupported Claim Closure" in markdown
    assert "Unsupported claim patches applied: 1" in markdown


def test_tamil_cleanup_replaces_ruling_and_preserves_technical_terms() -> None:
    revised = cleanup_revised_article_tamil(
        {
            "headline": "இந்த ruling H-1B குடும்பங்களுக்கு தொடர்புடையது",
            "subheadline": "ruling வழங்கியுள்ளது; SMS அறிவிப்பு அல்ல",
            "article_body": (
                "இந்த ruling குறித்து விசாவில் உள்ளோர் கவனிக்கலாம். "
                "இந்த தீர்ப்பு சமூகத்தில் முக்கியமானது. "
                "அமைப்புகளின் ஆதரவுடன் வந்துள்ளது. "
                "உரிமைகளை உறுதிப்படுத்துவதில் தொடர்புடையது. "
                "வாழ்வில் தொடர்புடையது எனக் கூறலாம். "
                "எதிர்காலத்தை உறுதிப்படுத்துகிறது. "
                "குடும்பங்களுக்கு தொடர்புடையது. "
                "சமூகத்தில் தாக்கத்தை ஏற்படுத்தும் என்று எதிர்பார்க்கப்படுகிறது. "
                "புதிய அத்தியாயத்தை தொடங்குகிறது. "
                "முக்கியமான சட்ட முடிவாகும்."
            ),
            "seo_title": "ruling மற்றும் H-1B விசா",
            "meta_description": "இந்த ruling குறித்து SMS விவரம் இல்லை.",
            "suggested_tags": ["H-1B", "SMS"],
        }
    )

    combined = "\n".join(
        str(revised[field])
        for field in (
            "headline",
            "subheadline",
            "article_body",
            "seo_title",
            "meta_description",
        )
    )
    assert "ruling" not in combined
    assert "இந்த தீர்ப்பு" in combined
    assert "தீர்ப்பு வழங்கியுள்ளது" in combined
    assert "தீர்ப்பு குறித்து" in combined
    assert "முக்கியமானது" not in combined
    assert "தொடர்புடையது" in combined
    assert "ஆதரவுடன் வந்துள்ளது" not in combined
    assert "கருத்துகளுடன் பதிவாகியுள்ளது" in combined
    assert "உரிமைகளை உறுதிப்படுத்துவதில் தொடர்புடையது" not in combined
    assert "பிறப்புரிமை குடியுரிமை தொடர்பானதாகும்" in combined
    assert "வாழ்வில் தொடர்புடையது எனக் கூறலாம்" not in combined
    assert "குடும்பங்களுடன் தொடர்புடையதாக பார்க்கப்படுகிறது" in combined
    assert "எதிர்காலத்தை உறுதிப்படுத்துகிறது" not in combined
    assert "பிறப்புரிமை குடியுரிமையை குறிப்பிடுகிறது" in combined
    assert "குடும்பங்களுக்கு தொடர்புடையது" not in combined
    assert "குழந்தைகளின் பிறப்புரிமை குடியுரிமை தொடர்பானதாகும்" in combined
    assert "சமூகத்தில் தாக்கத்தை ஏற்படுத்தும்" not in combined
    assert "சமூகத்துடன் தொடர்புடையதாக பார்க்கப்படுகிறது" in combined
    assert "புதிய அத்தியாயத்தை தொடங்குகிறது" not in combined
    assert "தொடர்புடைய சட்ட முடிவாகும்" in combined
    assert "முக்கியமான சட்ட முடிவாகும்" not in combined
    assert "H-1B" in combined
    assert "SMS" in combined


def test_tamil_cleanup_replaces_corrupted_belongs_token() -> None:
    revised = cleanup_revised_article_tamil(
        {
            "article_body": (
                "அமெரிக்காவில் யார் pertencிக்கிறார்கள் என்பதற்கான கருத்து."
            )
        }
    )

    assert "pertenc" not in str(revised["article_body"])
    assert "சேர்ந்துள்ளனர்" in str(revised["article_body"])


class MockRevisionClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "Use only the grounded brief as the factual source" in system_prompt
        assert "grounding_evaluation_feedback" in user_payload
        return {
            "headline": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": (
                "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் தொடங்க உள்ளது.\n"
                "குறைந்த உயரப் பகுதிகளின் அருகே 18 சென்சார்கள் அமைக்கப்படும்."
            ),
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "meta_description": (
                "சென்னையில் அடுத்த மாதம் தொடங்கும் வெள்ள எச்சரிக்கை முயற்சி "
                "குறித்த செய்தி."
            ),
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "revision_summary": "Unsupported safety and impact claims were removed.",
            "removed_or_softened_claims": ["பாதுகாப்பு உறுதி", "முக்கியமான மாற்றம்"],
            "fact_usage_notes": ["18 sensors preserved."],
            "style_usage_notes": ["Kept concise Tamil news style."],
        }


class PatchRevisionClient(MockRevisionClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "Do not rewrite the full article" in system_prompt
        assert "grounding_evaluation_feedback" in user_payload
        payload = json.loads(user_payload)
        article_body = str(
            payload["original_generated_draft"]["draft"]["article_body"]
        )
        return {
            "revision_required": True,
            "patches": [
                {
                    "issue_type": "unsupported_claim",
                    "target_text": article_body,
                    "replacement_text": (
                        "à®‡à®¤à¯ à®…à®Ÿà¯à®¤à¯à®¤ à®®à®¾à®¤à®®à¯ "
                        "à®¤à¯Šà®Ÿà®™à¯à®•à¯à®®à¯ à®µà¯†à®³à¯à®³ "
                        "à®Žà®šà¯à®šà®°à®¿à®•à¯à®•à¯ˆ à®®à¯à®¯à®±à¯à®šà®¿; "
                        "18 à®šà¯†à®©à¯à®šà®¾à®°à¯à®•à®³à¯ "
                        "à®…à®®à¯ˆà®•à¯à®•à®ªà¯à®ªà®Ÿà¯à®®à¯."
                    ),
                    "reason": "Replace unsupported impact claim with confirmed facts.",
                    "confidence": "high",
                },
                {
                    "issue_type": "clarity",
                    "target_text": (
                        "à®®à®•à¯à®•à®³à®¿à®©à¯ à®ªà®¾à®¤à¯à®•à®¾à®ªà¯à®ªà¯ "
                        "à®®à¯‡à®®à¯à®ªà®Ÿà¯à®®à¯."
                    ),
                    "replacement_text": (
                        "à®•à®©à®®à®´à¯ˆ à®¨à¯‡à®°à®™à¯à®•à®³à®¿à®²à¯ SMS "
                        "à®Žà®šà¯à®šà®°à®¿à®•à¯à®•à¯ˆ "
                        "à®…à®©à¯à®ªà¯à®ªà®ªà¯à®ªà®Ÿà¯à®®à¯."
                    ),
                    "reason": "Potentially useful but not automatic.",
                    "confidence": "low",
                },
                {
                    "issue_type": "style",
                    "target_text": "missing exact text",
                    "replacement_text": "replacement",
                    "reason": "Should be skipped because target is absent.",
                    "confidence": "high",
                },
            ],
            "notes": ["No broad rewrite was needed."],
        }


class RulingRevisionClient(MockRevisionClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounding_evaluation_feedback" in user_payload
        return {
            "headline": "இந்த ruling H-1B குடும்பங்களுக்கு தொடர்புடையது",
            "subheadline": "ruling வழங்கியுள்ளது; SMS விவரம் இல்லை",
            "article_body": "இந்த ruling குறித்து விசாவில் உள்ளோர் கவனிக்கலாம்.",
            "seo_title": "ruling மற்றும் H-1B விசா",
            "meta_description": "இந்த ruling குறித்து SMS விவரம் இல்லை.",
            "suggested_tags": ["H-1B", "SMS"],
            "revision_summary": (
                "Unsupported claims were softened and SEO fields were revised."
            ),
            "removed_or_softened_claims": ["புதிய நம்பிக்கை"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
        }


class LengthAwareRevisionClient(MockRevisionClient):
    def __init__(self) -> None:
        self.payload: dict[str, object] | None = None

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        self.payload = json.loads(user_payload)
        assert "Do not shrink a full article into a short summary" in system_prompt
        paragraph = (
            "சென்னை நகரில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் தொடங்கும் என்று "
            "அதிகாரிகள் தெரிவித்துள்ளனர். குறைந்த உயரப்பகுதி சாலைகளின் அருகே "
            "18 சென்சார்கள் அமைக்கப்படும் என்பதும், கனமழை நேரங்களில் குடியிருப்போருக்கு "
            "SMS எச்சரிக்கை அனுப்பப்படும் என்பதும் உறுதிப்படுத்தப்பட்ட தகவல்களாகும்."
        )
        body = "\n\n".join([paragraph for _ in range(17)])
        return {
            "headline": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": body,
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "meta_description": (
                "சென்னையில் அடுத்த மாதம் தொடங்கும் வெள்ள எச்சரிக்கை முயற்சி "
                "குறித்த செய்தி."
            ),
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "revision_summary": (
                "Unsupported benefit claims were replaced with grounded neutral "
                "context while preserving length."
            ),
            "removed_or_softened_claims": ["பாதுகாப்பு உறுதி"],
            "fact_usage_notes": ["18 sensors and SMS alerts preserved."],
            "style_usage_notes": ["Kept news style."],
        }


def _repository_with_revision_inputs(
    tmp_path: Path,
    desired_word_count: int | None = 600,
) -> StyleScribeRepository:
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
                    "paragraph_style": "Compact",
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
            source_text_excerpt="Chennai officials announced a pilot.",
            source_language="en",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            brief_json=StyleScribeRepository.encode_json(
                {
                    "topic": "Flood warning pilot",
                    "one_line_summary": "Pilot starts next month.",
                    "confirmed_facts": ["18 sensors will be installed."],
                    "claims_to_avoid": ["Do not claim effectiveness."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    repository.save_article_draft(
        ArticleDraftRecord(
            draft_id="draft-1",
            author_id="v_vasanthi",
            profile_id="profile-1",
            brief_id="brief-1",
            target_language="ta",
            model_provider="openai",
            model_name="gpt-4o-mini",
            status="completed",
            author_instruction="Write as Tamil news.",
            article_type="news",
            desired_word_count=desired_word_count,
            tone_override="clear",
            include_seo=True,
            draft_json=StyleScribeRepository.encode_json(
                {
                    "headline": "பாதுகாப்பு உறுதி தரும் முயற்சி",
                    "subheadline": "மக்களின் பாதுகாப்பு மேம்படும்.",
                    "article_body": "இது வாழ்வில் முக்கியமான மாற்றம் தரும்.",
                    "seo_title": "பாதுகாப்பு உறுதி",
                    "meta_description": "மக்களுக்கு புதிய நம்பிக்கை.",
                    "suggested_tags": ["சென்னை"],
                }
            ),
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
                {
                    "overall_risk": "high",
                    "editorial_readiness": "revision_required",
                    "unsupported_claims": [
                        {
                            "claim": "பாதுகாப்பு உறுதி",
                            "suggested_fix": "State only that the pilot begins.",
                        }
                    ],
                    "overclaim_phrases": ["முக்கியமான மாற்றம்"],
                    "rewrite_guidance": ["Remove benefit language."],
                }
            ),
            warnings_json="[]",
            created_at="2026-01-01T00:00:00+00:00",
        )
    )
    return repository
