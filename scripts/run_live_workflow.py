"""Run a compact live StyleScribe workflow validation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from backend.app.services.pasted_text_workflow_service import (
    run_pasted_text_to_draft_workflow,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", default="manual_request.json")
    parser.add_argument("--desired-word-count", type=int)
    parser.add_argument("--model")
    parser.add_argument("--generation-model")
    parser.add_argument("--workflow-mode", default=None)
    parser.add_argument("--output", default="manual_response_10i_live.json")
    args = parser.parse_args()

    if args.model:
        os.environ["OPENAI_MODEL"] = args.model
    if args.generation_model:
        os.environ["OPENAI_MODEL_GENERATION"] = args.generation_model

    request = json.loads(Path(args.request).read_text(encoding="utf-8-sig"))
    if args.desired_word_count is not None:
        request["desired_word_count"] = args.desired_word_count
    if args.workflow_mode is not None:
        request["workflow_mode"] = args.workflow_mode

    response = run_pasted_text_to_draft_workflow(**request)
    payload = response.model_dump(mode="json")
    payload["workflow_completed"] = response.status == "completed"
    payload["openai_model"] = os.getenv("OPENAI_MODEL")
    payload["final_grounding_score"] = (
        response.final_evaluation_summary.grounding_score
        if response.final_evaluation_summary
        else None
    )
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    report_keys = [
        "workflow_completed",
        "export_paths",
        "desired_word_count",
        "target_min_word_count",
        "target_max_word_count",
        "generation_mode_used",
        "section_assembled_article_word_count",
        "revision_mode",
        "revision_input_word_count",
        "revision_patch_count",
        "revision_patches_applied_count",
        "revision_patches_skipped_count",
        "revision_output_word_count",
        "revision_rejected_for_length_collapse",
        "revision_rejected_reason",
        "unsupported_claim_findings_count",
        "unsupported_claim_patch_count",
        "unsupported_claim_patches_applied_count",
        "unsupported_claim_patches_skipped_count",
        "unsupported_claim_patch_skipped_reasons",
        "unsupported_claims_unresolved_count",
        "unsupported_claims_cleared_by_patch",
        "initial_readiness",
        "initial_readiness_reasons",
        "final_readiness",
        "final_readiness_reasons",
        "readiness_decision_source",
        "final_publication_blockers",
        "final_publication_warnings",
        "final_article_word_count",
        "final_word_count_ratio",
        "final_grounding_score",
        "tamil_quality_warnings",
        "publication_ready_completeness_passed",
        "length_status",
        "section_coverage_status",
        "model_used_by_stage",
        "total_runtime_seconds",
        "llm_call_count_total",
        "llm_call_count_by_stage",
        "runtime_by_stage",
        "slowest_stage",
        "total_prompt_tokens",
        "total_completion_tokens",
        "total_tokens",
        "cached_prompt_tokens_total",
        "uncached_prompt_tokens_total",
        "prompt_cache_hit_ratio",
        "cached_prompt_tokens_by_stage",
        "prompt_cache_hit_ratio_by_stage",
        "estimated_cost_total_usd",
        "estimated_cost_by_stage_usd",
        "highest_cost_stage",
        "max_concurrent_section_calls",
        "generation_section_group_size",
        "generation_group_call_count",
        "generation_single_section_fallback_count",
        "generation_context_pack_tokens",
        "generation_context_pack_chars",
        "original_generation_context_chars",
        "compressed_generation_context_chars",
        "generation_context_compression_ratio",
        "cost_estimation_available",
        "cost_estimation_notes",
    ]
    print(json.dumps({key: payload.get(key) for key in report_keys}, indent=2))


if __name__ == "__main__":
    main()
