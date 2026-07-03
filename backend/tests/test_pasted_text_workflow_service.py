import logging
from pathlib import Path

import pytest

from backend.app.db.repository import AuthorStyleProfileRecord, StyleScribeRepository
from backend.app.services.model_clients.openai_client import OpenAIClientError
from backend.app.services.pasted_text_workflow_service import (
    _final_readiness_decision,
    run_pasted_text_to_draft_workflow,
)


def test_pasted_text_workflow_success_with_mocked_models(tmp_path: Path) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        author_instruction="Write for Oneindia readers.",
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    assert response.workflow_id
    assert response.brief_id
    assert response.draft_id
    assert response.evaluation_id
    assert response.source_cleanup.removed_line_count >= 3
    assert response.brief_summary.topic == "Flood warning pilot"
    assert response.draft_summary.headline == "சென்னை வெள்ள எச்சரிக்கை முயற்சி"
    assert response.evaluation_summary is not None
    assert response.evaluation_summary.overall_risk == "medium"
    assert response.tamil_quality_status
    assert response.requested_word_count == 600
    assert response.original_draft_word_count is not None
    assert response.final_article_word_count is not None
    assert response.length_status in {"pass", "warning"}
    assert response.final_article_word_count_ratio is not None
    assert response.article_plan_used is True
    assert response.planned_section_count == 6
    assert response.planned_target_word_count == 600
    assert response.section_coverage_status in {"pass", "warning"}
    assert repository.fetch_latest_draft_evaluation(response.draft_id) is not None
    assert response.total_runtime_seconds is not None
    assert "generation" in response.runtime_by_stage
    assert response.llm_call_count_total >= 1
    assert response.generation_model_used == "gpt-4o-mini"
    assert "generation" in response.token_usage_by_stage
    assert response.cost_estimation_available is False


def test_pasted_text_workflow_without_grounding_evaluation(tmp_path: Path) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_grounding_evaluation=False,
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
    )

    assert response.evaluation_id is None
    assert response.evaluation_summary is None
    assert response.revision_id is None


def test_pasted_text_workflow_exports_utf8_html_and_markdown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repository = _repository_with_profile(tmp_path)

    html_response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        export_review=True,
        export_format="html",
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )
    md_response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        export_review=True,
        export_format="markdown",
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    html = Path(html_response.export_paths[0]).read_text(encoding="utf-8")
    markdown = Path(md_response.export_paths[0]).read_text(encoding="utf-8")
    assert '<meta charset="utf-8">' in html
    assert "Nirmala UI" in html
    assert "சென்னை வெள்ள எச்சரிக்கை முயற்சி" in html
    assert "Grounding Evaluation" in markdown


def test_pasted_text_workflow_with_auto_revision(tmp_path: Path) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_auto_revision=True,
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
        revision_model_client=MockRevisionClient(),
    )

    assert response.initial_evaluation_id
    assert response.revision_id
    assert response.final_evaluation_id is None
    assert response.initial_readiness == "review_required"
    assert repository.fetch_latest_article_revision(response.draft_id) is not None


def test_pasted_text_workflow_runs_final_evaluation_when_requested(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)
    evaluation_client = CountingEvaluationClient()

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_auto_revision=True,
        run_final_evaluation=True,
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=evaluation_client,
        revision_model_client=MockRevisionClient(),
    )

    assert evaluation_client.calls == 2
    assert response.final_evaluation_id
    assert response.final_readiness == "revision_required"
    assert response.readiness_decision_source == "final_article_state"
    assert "length_below_minimum" in response.final_publication_blockers
    assert response.publication_ready_completeness_passed is False


def test_pasted_text_workflow_revision_export_contains_before_and_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_auto_revision=True,
        run_final_evaluation=True,
        export_review=True,
        export_format="html",
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=CountingEvaluationClient(),
        revision_model_client=RulingWorkflowRevisionClient(),
    )

    html = Path(response.export_paths[0]).read_text(encoding="utf-8")
    assert "Original Draft" in html
    assert "Initial Grounding Evaluation" in html
    assert "Unsupported claims" in html
    assert "Overclaim phrases" in html
    assert "Revision Summary" in html
    assert "Revised Draft" in html
    assert "Final Grounding Evaluation" in html
    assert "Tamil Quality And Length" in html
    assert "Tamil quality status" in html
    assert "Length Recovery" in html
    assert "Article Plan" in html
    assert "Section coverage status" in html
    assert "Final Article Used For Evaluation" in html
    assert "Original draft approximate word count" in html
    assert "Final article approximate word count" in html
    assert "Length warning reason" in html
    assert "Length status" in html
    assert "Readiness Decision" in html
    assert "Final publication blockers" in html
    assert "Allowed English terms block readiness" in html
    assert "Workflow Runtime Summary" in html
    assert "Estimated cost total USD" in html
    assert "This is an editor-assisted draft" in html
    assert "ruling" not in html
    assert "H-1B" in html
    assert "SMS" in html


def test_pasted_text_workflow_warns_when_cleanup_removes_too_much(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)
    noisy_source = "\n".join(["Advertisement"] * 20) + "\n" + _source_text()

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=noisy_source,
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    assert any("removed more than 50%" in warning for warning in response.warnings)


def test_pasted_text_workflow_response_does_not_expose_api_keys(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    assert "OPENAI_API_KEY" not in response.model_dump_json()


def test_pasted_text_workflow_logs_failed_stage(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    repository = _repository_with_profile(tmp_path)
    caplog.set_level(
        logging.INFO,
        logger="backend.app.services.pasted_text_workflow_service",
    )

    with pytest.raises(OpenAIClientError, match="initial grounding evaluation"):
        run_pasted_text_to_draft_workflow(
            author_id="v_vasanthi",
            source_text=_source_text(),
            repository=repository,
            brief_model_client=MockBriefClient(),
            plan_model_client=MockPlanClient(),
            draft_model_client=MockDraftClient(),
            evaluation_model_client=TimeoutEvaluationClient(),
        )

    assert "stage=pasted text cleanup status=completed" in caplog.text
    assert "stage=grounded brief generation status=completed" in caplog.text
    assert "stage=initial draft generation status=completed" in caplog.text
    assert "stage=initial grounding evaluation status=failed" in caplog.text


def test_pasted_text_workflow_reports_length_warning_reason(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        desired_word_count=600,
        repository=repository,
        brief_model_client=MockBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
    )

    assert response.length_status == "warning"
    assert response.length_warning_reason is not None
    assert "materially shorter" in response.length_warning_reason
    assert "Grounded brief appears too limited" in response.length_warning_reason
    assert response.final_article_word_count_ratio is not None
    assert response.final_article_word_count_ratio < 0.75


def test_pasted_text_workflow_uses_expanded_article_as_final(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)
    evaluation_client = TrackingEvaluationClient()

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_auto_revision=True,
        run_final_evaluation=True,
        desired_word_count=600,
        repository=repository,
        brief_model_client=RichBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=evaluation_client,
        revision_model_client=MockRevisionClient(),
        length_recovery_model_client=SuccessfulExpansionClient(),
    )

    assert response.length_recovery_required is True
    assert response.length_recovery_attempted is True
    assert response.length_recovery_succeeded is True
    assert response.length_recovery_failed is False
    assert response.short_output_invalid is True
    assert response.expansion_items_available >= 5
    assert response.expansion_items_used
    assert response.revised_word_count_before_expansion is not None
    assert response.revised_word_count_before_expansion < 450
    assert response.final_article_word_count is not None
    assert response.final_article_word_count >= 450
    assert response.length_status == "pass"
    assert response.final_evaluation_id
    assert evaluation_client.calls == 2
    assert response.revision_id in evaluation_client.payloads[-1]


def test_pasted_text_workflow_reports_failed_length_recovery(
    tmp_path: Path,
) -> None:
    repository = _repository_with_profile(tmp_path)

    response = run_pasted_text_to_draft_workflow(
        author_id="v_vasanthi",
        source_text=_source_text(),
        run_auto_revision=True,
        run_final_evaluation=False,
        desired_word_count=600,
        repository=repository,
        brief_model_client=RichBriefClient(),
        plan_model_client=MockPlanClient(),
        draft_model_client=MockDraftClient(),
        evaluation_model_client=MockEvaluationClient(),
        revision_model_client=MockRevisionClient(),
        length_recovery_model_client=FailedExpansionClient(),
    )

    assert response.length_recovery_required is True
    assert response.length_recovery_attempted is True
    assert response.length_recovery_succeeded is False
    assert response.length_recovery_failed is True
    assert response.length_status == "warning"
    assert response.length_warning_reason == (
        "Length recovery failed despite sufficient grounded expansion material."
    )


def test_final_readiness_allows_allowed_english_warning() -> None:
    decision = _final_readiness_decision(
        final_evaluation={
            "grounding_score": 85,
            "editorial_readiness": "revision_required",
            "unsupported_claims": [],
        },
        initial_evaluation=None,
        final_article_word_count=536,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="warning",
        tamil_quality_warnings=["Allowed English term(s) present: H-1B"],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=[],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "review_required"
    assert decision.blockers == []
    assert "allowed_english_terms_only" in decision.warnings
    assert decision.publication_ready_completeness_passed is True


def test_final_readiness_blocks_below_minimum_length() -> None:
    decision = _final_readiness_decision(
        final_evaluation={"grounding_score": 90, "unsupported_claims": []},
        initial_evaluation=None,
        final_article_word_count=280,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="warning",
        tamil_quality_warnings=[
            "Final article body is materially shorter than requested."
        ],
        length_status="warning",
        section_coverage_status="warning",
        revision_patch_skipped_reasons=[],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "revision_required"
    assert "length_below_minimum" in decision.blockers


def test_final_readiness_blocks_unresolved_unsupported_claim() -> None:
    decision = _final_readiness_decision(
        final_evaluation={
            "grounding_score": 88,
            "unsupported_claims": [{"claim": "unsupported benefit"}],
        },
        initial_evaluation=None,
        final_article_word_count=500,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="pass",
        tamil_quality_warnings=[],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=[],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "revision_required"
    assert "unsupported_claims_remaining" in decision.blockers


def test_skipped_noncritical_patch_does_not_block_publication() -> None:
    decision = _final_readiness_decision(
        final_evaluation={"grounding_score": 92, "unsupported_claims": []},
        initial_evaluation=None,
        final_article_word_count=500,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="pass",
        tamil_quality_warnings=[],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=["patch 2: low confidence"],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "review_required"
    assert decision.blockers == []
    assert "revision_patch_skipped_non_critical" in decision.warnings


def test_skipped_unsupported_patch_does_not_block_when_final_claims_clear() -> None:
    decision = _final_readiness_decision(
        final_evaluation={"grounding_score": 92, "unsupported_claims": []},
        initial_evaluation=None,
        final_article_word_count=500,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="pass",
        tamil_quality_warnings=[],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=[
            "patch 1: unsupported_claim: unsupported_claim_patch_unmatched"
        ],
        unsupported_claim_patch_skipped_reasons=[
            "patch 1: unsupported_claim: unsupported_claim_patch_unmatched"
        ],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "review_required"
    assert decision.blockers == []


def test_skipped_unsupported_patch_blocks_when_final_claim_remains() -> None:
    decision = _final_readiness_decision(
        final_evaluation={
            "grounding_score": 92,
            "unsupported_claims": [{"claim": "still unsupported"}],
        },
        initial_evaluation=None,
        final_article_word_count=500,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="pass",
        tamil_quality_warnings=[],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=[
            "patch 1: unsupported_claim: unsupported_claim_patch_unmatched"
        ],
        unsupported_claim_patch_skipped_reasons=[
            "patch 1: unsupported_claim: unsupported_claim_patch_unmatched"
        ],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "revision_required"
    assert "unsupported_claims_remaining" in decision.blockers
    assert "revision_patch_pending" in decision.blockers


def test_final_readiness_uses_final_evaluation_not_initial_status() -> None:
    decision = _final_readiness_decision(
        final_evaluation={
            "grounding_score": 94,
            "editorial_readiness": "safe_to_review",
            "unsupported_claims": [],
        },
        initial_evaluation={
            "grounding_score": 60,
            "editorial_readiness": "revision_required",
            "unsupported_claims": [{"claim": "old issue"}],
        },
        final_article_word_count=500,
        target_min_word_count=450,
        target_max_word_count=690,
        tamil_quality_status="pass",
        tamil_quality_warnings=[],
        length_status="pass",
        section_coverage_status="pass",
        revision_patch_skipped_reasons=[],
        revision_rejected_for_length_collapse=False,
        revision_rejected_reason=None,
        run_final_evaluation=True,
    )

    assert decision.readiness == "safe_to_review"
    assert decision.source == "final_article_state"
    assert decision.blockers == []


class MockBriefClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "source_text" in user_payload
        assert "Advertisement" not in user_payload
        return {
            "topic": "Flood warning pilot",
            "one_line_summary": "Chennai officials announced a pilot.",
            "source_language": "en",
            "target_language": "ta",
            "confirmed_facts": [
                "A flood-warning pilot will begin next month.",
                "18 sensors will be installed.",
            ],
            "key_entities": [],
            "places": [],
            "dates_or_timeline": ["next month"],
            "numbers_and_statistics": ["18 sensors"],
            "quotes": [],
            "background_from_source": [],
            "missing_or_unclear_information": [],
            "claims_to_avoid": ["Do not claim effectiveness before the pilot begins."],
            "suggested_tamil_angle": "Flood warning pilot",
            "editorial_risk_notes": [],
        }


class MockPlanClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "desired_word_count" in user_payload
        assert "target_word_count_range" in user_payload
        assert "unsupported benefit" in system_prompt
        return {
            "target_word_count": 600,
            "target_min_word_count": 450,
            "target_max_word_count": 690,
            "planned_sections": [
                {
                    "section_name": "lead",
                    "purpose": "core news",
                    "target_words": 90,
                    "grounded_facts_to_use": ["Pilot begins next month"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": ["Do not claim success"],
                    "must_not_add": ["No filler"],
                },
                {
                    "section_name": "details",
                    "purpose": "official action",
                    "target_words": 90,
                    "grounded_facts_to_use": ["18 sensors"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": [],
                    "must_not_add": [],
                },
                {
                    "section_name": "affected groups",
                    "purpose": "residents",
                    "target_words": 90,
                    "grounded_facts_to_use": ["low-lying streets"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": [],
                    "must_not_add": [],
                },
                {
                    "section_name": "numbers",
                    "purpose": "statistics",
                    "target_words": 90,
                    "grounded_facts_to_use": ["18 sensors"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": [],
                    "must_not_add": [],
                },
                {
                    "section_name": "timeline",
                    "purpose": "next month",
                    "target_words": 90,
                    "grounded_facts_to_use": ["next month"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": [],
                    "must_not_add": [],
                },
                {
                    "section_name": "closing",
                    "purpose": "grounded close",
                    "target_words": 90,
                    "grounded_facts_to_use": ["six month pilot"],
                    "quotes_or_attributions_to_use": [],
                    "claims_to_avoid": [],
                    "must_not_add": [],
                },
            ],
            "expansion_items_used": ["18 sensors", "next month"],
            "claims_to_avoid": ["Do not claim effectiveness."],
            "plan_summary": "Six-section grounded plan.",
            "warnings": [],
        }


class RichBriefClient(MockBriefClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        brief = super().generate_structured_json(system_prompt, user_payload)
        brief["confirmed_facts"] = [
            "A flood-warning pilot will begin next month.",
            "18 sensors will be installed.",
            "Sensors will be installed near low-lying streets.",
        ]
        brief["affected_groups"] = ["Residents in low-lying streets"]
        brief["quotes"] = ["Officials said residents will receive SMS alerts."]
        brief["policy_or_legal_context"] = ["The pilot will run for six months."]
        brief["background_from_source"] = ["Citywide expansion will be decided later."]
        return brief


class MockDraftClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        if "one section of a grounded Tamil news article" in system_prompt:
            return {
                "section_name": "lead",
                "target_words": 90,
                "section_text": (
                    "சென்னை நகரில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் "
                    "தொடங்கும் என்று அதிகாரிகள் தெரிவித்துள்ளனர். குறைந்த "
                    "உயரப்பகுதி சாலைகளின் அருகே 18 சென்சார்கள் அமைக்கப்படும்."
                ),
                "grounded_facts_used": ["18 sensors"],
                "warnings": [],
            }
        assert "grounded_brief_for_facts_only" in user_payload
        return {
            "headline": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி தொடங்க உள்ளது.",
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை திட்டம்",
            "meta_description": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி குறித்த செய்தி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
        }


class MockEvaluationClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounded_brief_only_factual_source" in user_payload
        return {
            "grounding_score": 82,
            "claim_safety_score": 80,
            "fact_preservation_score": 90,
            "overall_risk": "medium",
            "editorial_readiness": "review_required",
            "unsupported_claims": [],
            "overclaim_phrases": [],
            "invented_facts": [],
            "contradictions": [],
            "claims_to_avoid_violations": [],
            "missing_key_facts": [],
            "preserved_facts": ["18 sensors"],
            "number_date_name_checks": [],
            "rewrite_guidance": [],
            "summary": "Mostly grounded.",
        }


class TimeoutEvaluationClient(MockEvaluationClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        raise OpenAIClientError("OpenAI request timed out after 3 seconds.")


class CountingEvaluationClient(MockEvaluationClient):
    def __init__(self) -> None:
        self.calls = 0

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        self.calls += 1
        evaluation = super().generate_structured_json(system_prompt, user_payload)
        if self.calls == 2:
            evaluation = dict(evaluation)
            evaluation["grounding_score"] = 94
            evaluation["claim_safety_score"] = 95
            evaluation["overall_risk"] = "low"
            evaluation["editorial_readiness"] = "safe_to_review"
        return evaluation


class TrackingEvaluationClient(CountingEvaluationClient):
    def __init__(self) -> None:
        super().__init__()
        self.payloads: list[str] = []

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        self.payloads.append(user_payload)
        return super().generate_structured_json(system_prompt, user_payload)


class MockRevisionClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounding_evaluation_feedback" in user_payload
        return {
            "headline": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி அடுத்த மாதம் தொடங்குகிறது.",
            "article_body": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி தொடங்க உள்ளது.",
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "meta_description": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி குறித்த செய்தி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "revision_summary": "Unsupported benefit language was removed.",
            "removed_or_softened_claims": ["மக்களின் பாதுகாப்பு மேம்படும்"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
        }


class RulingWorkflowRevisionClient(MockRevisionClient):
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
            "revision_summary": "Unsupported benefit language was removed.",
            "removed_or_softened_claims": ["புதிய நம்பிக்கை"],
            "fact_usage_notes": [],
            "style_usage_notes": [],
        }


class SuccessfulExpansionClient:
    provider = "openai"
    model_name = "gpt-4o-mini"

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        assert "grounded brief and source excerpt" in system_prompt
        assert "target_word_count_range" in user_payload
        paragraph = (
            "சென்னை நகரில் வெள்ள எச்சரிக்கை முயற்சி அடுத்த மாதம் தொடங்கும் என்று "
            "அதிகாரிகள் தெரிவித்துள்ளனர். குறைந்த உயரப்பகுதி சாலைகளின் அருகே "
            "18 சென்சார்கள் அமைக்கப்படும் என்பதும், கனமழை நேரங்களில் குடியிருப்போருக்கு "
            "SMS எச்சரிக்கை அனுப்பப்படும் என்பதும் உறுதிப்படுத்தப்பட்ட தகவல்களாகும்."
        )
        return {
            "headline": "சென்னையில் வெள்ள எச்சரிக்கை முயற்சி",
            "subheadline": "18 சென்சார்கள் அமைக்கும் முயற்சி தொடங்குகிறது.",
            "article_body": "\n\n".join([paragraph for _ in range(17)]),
            "seo_title": "சென்னை வெள்ள எச்சரிக்கை முயற்சி",
            "meta_description": "சென்னை வெள்ள எச்சரிக்கை முயற்சி குறித்த செய்தி.",
            "suggested_tags": ["சென்னை", "வெள்ள எச்சரிக்கை"],
            "expansion_summary": "Expanded with grounded facts.",
            "expansion_items_used": ["18 sensors", "SMS alerts"],
            "warnings": [],
        }


class FailedExpansionClient(SuccessfulExpansionClient):
    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
    ) -> dict[str, object]:
        expanded = super().generate_structured_json(system_prompt, user_payload)
        expanded["article_body"] = "சிறிய செய்தி மட்டுமே."
        expanded["warnings"] = ["Could not expand."]
        return expanded


def _repository_with_profile(tmp_path: Path) -> StyleScribeRepository:
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
    return repository


def _source_text() -> str:
    return """
    Advertisement
    Share
    Chennai city officials said a new flood-warning pilot will begin next month.
    Read more
    The civic body said 18 sensors will be installed near low-lying streets.
    Residents will receive SMS alerts during heavy rain.
    Subscribe
    """
