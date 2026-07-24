# ruff: noqa: E501
"""Deterministic editorial review pack for newsroom prompt comparisons."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReviewModeSpec:
    mode_key: str
    label: str
    dir_name: str


@dataclass(frozen=True)
class ReviewPackConfig:
    comparison_root: Path
    legacy_dir_name: str = "gemini_3_5_flash"
    newsroom_dir_name: str = "newsroom_v1_gemini_gemini_3_5_flash"
    output_dir_name: str = "editorial_review_pack"
    extra_modes: tuple[ReviewModeSpec, ...] = ()


def generate_editorial_review_pack(config: ReviewPackConfig) -> dict[str, Path]:
    manifest = _read_json(config.comparison_root / "shared" / "manifest.json")
    output_dir = config.comparison_root / config.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        _review_case(config, entry)
        for entry in manifest.get("inputs", [])
        if isinstance(entry, dict)
    ]
    paths = {
        "review_manifest_json": output_dir / "review_manifest.json",
        "review_manifest_csv": output_dir / "review_manifest.csv",
        "length_adherence_csv": output_dir / "length_adherence_report.csv",
        "factual_coverage_jsonl": output_dir / "factual_coverage_report.jsonl",
        "editorial_review_sheet_csv": output_dir / "editorial_review_sheet.csv",
        "editorial_review_html": output_dir / "editorial_review_pack.html",
    }
    _write_json(paths["review_manifest_json"], {"cases": cases})
    _write_manifest_csv(paths["review_manifest_csv"], cases)
    _write_length_csv(paths["length_adherence_csv"], cases)
    _write_coverage_jsonl(paths["factual_coverage_jsonl"], cases)
    _write_review_sheet_csv(paths["editorial_review_sheet_csv"], cases)
    paths["editorial_review_html"].write_text(
        _review_html(cases),
        encoding="utf-8",
    )
    return paths


def _review_case(config: ReviewPackConfig, entry: dict[str, Any]) -> dict[str, Any]:
    input_id = str(entry["input_id"])
    brief = _read_json(Path(str(entry["brief_path"])))
    source = _brief_payload(brief)
    mode_specs = _review_mode_specs(config)
    modes = {
        spec.mode_key: _mode_review(
            spec.mode_key,
            config.comparison_root / spec.dir_name / input_id,
            entry,
            source,
        )
        for spec in mode_specs
    }
    legacy = modes["legacy"]
    newsroom = modes["newsroom_v1"]
    return {
        "input_id": input_id,
        "source_title": entry.get("source_title"),
        "source_path": entry.get("source_path"),
        "brief_path": entry.get("brief_path"),
        "legacy_output_path": legacy["response_path"],
        "newsroom_v1_output_path": newsroom["response_path"],
        "requested_length": _target(entry),
        "source_brief": {
            "topic": source.get("topic"),
            "one_line_summary": source.get("one_line_summary"),
            "confirmed_facts": _string_list(source.get("confirmed_facts")),
            "quotes": _list_of_dicts(source.get("quotes")),
            "dates_or_timeline": _string_list(source.get("dates_or_timeline")),
            "numbers_and_statistics": _string_list(
                source.get("numbers_and_statistics")
            ),
            "affected_groups": _string_list(source.get("affected_groups")),
            "policy_or_legal_context": _string_list(
                source.get("policy_or_legal_context")
            ),
        },
        "source_information_density": _information_density(source),
        "mode_order": [spec.mode_key for spec in mode_specs],
        "mode_labels": {spec.mode_key: spec.label for spec in mode_specs},
        "modes": modes,
        "legacy": legacy,
        "newsroom_v1": newsroom,
        "comparison_notes": _comparison_notes(legacy, newsroom),
        "editorial_review_status": "pending",
    }


def _review_mode_specs(config: ReviewPackConfig) -> tuple[ReviewModeSpec, ...]:
    return (
        ReviewModeSpec("legacy", "Legacy", config.legacy_dir_name),
        ReviewModeSpec("newsroom_v1", "Newsroom v1", config.newsroom_dir_name),
        *config.extra_modes,
    )


def _mode_review(
    mode: str,
    input_dir: Path,
    entry: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    response = _read_json(input_dir / "response.json")
    evaluation = _evaluation_payload(response)
    article = str(response.get("generated_tamil_article") or "")
    target = _target(entry)
    coverage = _coverage(source, evaluation)
    actual = _int_or_none(response.get("word_count"))
    return {
        "mode": mode,
        "status": response.get("completion_status"),
        "prompt_version": response.get("prompt_version"),
        "generation_model": response.get("generation_model"),
        "headline": response.get("generated_headline"),
        "word_count": actual,
        "target_length": target,
        "target_minimum": _int_or_none(response.get("target_minimum")),
        "target_maximum": _int_or_none(response.get("target_maximum")),
        "variance_from_target": None if actual is None else actual - target,
        "percent_variance_from_target": (
            None if actual is None else round(((actual - target) / target) * 100, 2)
        ),
        "length_assessment": _length_assessment(actual, response, coverage),
        "grounding_score": response.get("grounding_score"),
        "unsupported_claims": _list_value(response.get("unsupported_claims")),
        "warnings": _list_value(response.get("warnings")),
        "claims_to_avoid_violations": _list_value(
            response.get("claims_to_avoid_violations")
        ),
        "overclaim_phrases": _list_value(
            evaluation.get("overclaim_phrases") or response.get("overclaims")
        ),
        "evaluation_diagnostics": response.get("evaluation_diagnostics"),
        "retrieval_trace": response.get("retrieval_trace"),
        "retrieval_leakage_diagnostic": response.get(
            "retrieval_leakage_diagnostic"
        ),
        "source_claims_covered": coverage["covered_claims"],
        "source_claims_omitted": coverage["omitted_claims"],
        "coverage_summary": {
            "source_claim_count": coverage["source_claim_count"],
            "covered_claim_count": len(coverage["covered_claims"]),
            "omitted_claim_count": len(coverage["omitted_claims"]),
            "coverage_ratio": coverage["coverage_ratio"],
        },
        "names_dates_numbers_preserved": _names_dates_numbers(source, article),
        "attribution_coverage": _attribution_coverage(source, evaluation, article),
        "article_path": str(input_dir / "article.html"),
        "response_path": str(input_dir / "response.json"),
        "telemetry_path": str(input_dir / "telemetry.json"),
    }


def _coverage(source: dict[str, Any], evaluation: dict[str, Any]) -> dict[str, Any]:
    claims = _string_list(source.get("confirmed_facts"))
    preserved = _string_list(evaluation.get("preserved_facts"))
    missing = _string_list(evaluation.get("missing_key_facts"))
    covered: list[str] = []
    omitted: list[str] = []
    for claim in claims:
        if _best_similarity(claim, preserved) >= 0.42:
            covered.append(claim)
        elif _best_similarity(claim, missing) >= 0.42:
            omitted.append(claim)
        else:
            omitted.append(claim)
    return {
        "source_claim_count": len(claims),
        "covered_claims": covered,
        "omitted_claims": omitted,
        "coverage_ratio": round(len(covered) / len(claims), 4) if claims else None,
    }


def _length_assessment(
    actual: int | None,
    response: dict[str, Any],
    coverage: dict[str, Any],
) -> str:
    if actual is None:
        return "not_available"
    minimum = _int_or_none(response.get("target_minimum"))
    maximum = _int_or_none(response.get("target_maximum"))
    omitted = len(coverage["omitted_claims"])
    unsupported = len(_list_value(response.get("unsupported_claims")))
    if minimum is not None and actual < minimum:
        if omitted:
            return "short_with_possible_factual_omissions"
        return "short_but_potentially_useful_concision"
    if maximum is not None and actual > maximum:
        if unsupported:
            return "long_with_possible_unsupported_or_verbose_content"
        return "long_review_for_unnecessary_verbosity"
    if unsupported:
        return "in_range_but_content_review_needed"
    return "in_target_range"


def _comparison_notes(legacy: dict[str, Any], newsroom: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    if (
        isinstance(newsroom.get("word_count"), int)
        and isinstance(newsroom.get("target_minimum"), int)
        and newsroom["word_count"] < newsroom["target_minimum"]
    ):
        notes.append("newsroom_v1_below_target_minimum")
    if len(newsroom["source_claims_omitted"]) > len(legacy["source_claims_omitted"]):
        notes.append("newsroom_v1_may_omit_more_source_claims")
    if len(legacy["unsupported_claims"]) > len(newsroom["unsupported_claims"]):
        notes.append("legacy_has_more_unsupported_claim_warnings")
    if (
        isinstance(legacy.get("word_count"), int)
        and isinstance(newsroom.get("word_count"), int)
        and legacy["word_count"] - newsroom["word_count"] >= 120
    ):
        notes.append("legacy_substantially_longer_review_for_verbosity")
    return notes


def _information_density(source: dict[str, Any]) -> dict[str, Any]:
    counts = {
        "confirmed_facts": len(_string_list(source.get("confirmed_facts"))),
        "quotes": len(_list_of_dicts(source.get("quotes"))),
        "entities": len(_list_of_dicts(source.get("key_entities"))),
        "dates_or_timeline": len(_string_list(source.get("dates_or_timeline"))),
        "numbers_and_statistics": len(
            _string_list(source.get("numbers_and_statistics"))
        ),
        "affected_groups": len(_string_list(source.get("affected_groups"))),
        "policy_or_legal_context": len(
            _string_list(source.get("policy_or_legal_context"))
        ),
    }
    total = sum(counts.values())
    if total >= 14:
        label = "high"
    elif total >= 8:
        label = "medium"
    else:
        label = "low"
    return {"label": label, "score": total, "counts": counts}


def _names_dates_numbers(source: dict[str, Any], article: str) -> dict[str, Any]:
    candidates: list[str] = []
    for entity in source.get("key_entities", []):
        if isinstance(entity, dict):
            candidates.extend(
                str(entity.get(key) or "")
                for key in ("name_original", "name_tamil")
                if entity.get(key)
            )
    candidates.extend(_string_list(source.get("dates_or_timeline")))
    candidates.extend(_string_list(source.get("numbers_and_statistics")))
    candidates.extend(re.findall(r"\d+(?:[.,-]\d+)*", " ".join(candidates)))
    preserved = [item for item in candidates if item and item in article]
    return {
        "candidate_count": len(candidates),
        "preserved_count": len(preserved),
        "preserved_items": preserved[:20],
    }


def _attribution_coverage(
    source: dict[str, Any],
    evaluation: dict[str, Any],
    article: str,
) -> dict[str, Any]:
    quotes = _list_of_dicts(source.get("quotes"))
    preserved_facts = " ".join(_string_list(evaluation.get("preserved_facts"))).lower()
    speakers = [
        str(quote.get("speaker"))
        for quote in quotes
        if isinstance(quote, dict) and quote.get("speaker")
    ]
    covered_speakers = [
        speaker
        for speaker in speakers
        if speaker.lower() in preserved_facts or speaker in article
    ]
    return {
        "source_quote_count": len(quotes),
        "covered_speakers": covered_speakers,
        "has_direct_quote_marks": '"' in article or "'" in article,
    }


def _review_html(cases: list[dict[str, Any]]) -> str:
    body = "\n".join(_case_html(case) for case in cases)
    return f"""<!doctype html>
<html lang="ta">
<head>
  <meta charset="utf-8">
  <title>Newsroom V1 Editorial Review Pack</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    article {{ border: 1px solid #d8dee4; padding: 16px; margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
    pre {{ white-space: pre-wrap; background: #f6f8fa; padding: 12px; }}
    textarea {{ width: 100%; min-height: 72px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d8dee4; padding: 6px; vertical-align: top; }}
  </style>
</head>
<body>
<h1>Newsroom V1 Editorial Review Pack</h1>
{body}
</body>
</html>
"""


def _case_html(case: dict[str, Any]) -> str:
    source = case["source_brief"]
    modes = _list_value(case.get("mode_order")) or ["legacy", "newsroom_v1"]
    raw_labels = case.get("mode_labels")
    labels = raw_labels if isinstance(raw_labels, dict) else {}
    mode_html = "\n".join(
        _mode_html(str(labels.get(mode_key) or mode_key), case["modes"][mode_key])
        for mode_key in modes
        if mode_key in case["modes"]
    )
    preferred_options = " / ".join(str(mode_key) for mode_key in modes) + " / neither"
    return f"""
<article>
  <h2>{escape(str(case['input_id']))}: {escape(str(case.get('source_title') or ''))}</h2>
  <h3>Source Brief</h3>
  <pre>{escape(json.dumps(source, ensure_ascii=False, indent=2))}</pre>
  <div class="grid">
    {mode_html}
  </div>
  <h3>Editorial Review Fields</h3>
  <table>
    <tr><th>Preferred version</th><td>{escape(preferred_options)}</td></tr>
    <tr><th>Corrected publishable text</th><td><textarea></textarea></td></tr>
    <tr><th>Missing facts</th><td><textarea></textarea></td></tr>
    <tr><th>Unnecessary content</th><td><textarea></textarea></td></tr>
    <tr><th>Supporting-detail quality</th><td><textarea></textarea></td></tr>
    <tr><th>Attribution correction</th><td><textarea></textarea></td></tr>
    <tr><th>Contextual sequencing</th><td><textarea></textarea></td></tr>
    <tr><th>Background placement</th><td><textarea></textarea></td></tr>
    <tr><th>Impact-framing concern</th><td><textarea></textarea></td></tr>
    <tr><th>Factual correction</th><td><textarea></textarea></td></tr>
    <tr><th>Copied wording</th><td><textarea></textarea></td></tr>
    <tr><th>Factual leakage</th><td><textarea></textarea></td></tr>
    <tr><th>Unnatural Tamil</th><td><textarea></textarea></td></tr>
    <tr><th>Translation-like phrasing</th><td><textarea></textarea></td></tr>
    <tr><th>Headline correction</th><td><textarea></textarea></td></tr>
    <tr><th>Structural correction</th><td><textarea></textarea></td></tr>
    <tr><th>Overall editing effort</th><td>low / medium / high</td></tr>
  </table>
</article>
"""


def _mode_html(label: str, mode: dict[str, Any]) -> str:
    return f"""
<section>
  <h3>{escape(label)}</h3>
  <p><strong>Prompt:</strong> {escape(str(mode.get('prompt_version')))}</p>
  <p><strong>Words:</strong> {mode.get('word_count')} | <strong>Variance:</strong> {mode.get('percent_variance_from_target')}%</p>
  <p><strong>Grounding:</strong> {mode.get('grounding_score')} | <strong>Assessment:</strong> {escape(str(mode.get('length_assessment')))}</p>
  <p><strong>Covered:</strong> {mode['coverage_summary']['covered_claim_count']} / {mode['coverage_summary']['source_claim_count']}</p>
  <p><strong>Unsupported:</strong> {len(mode['unsupported_claims'])} | <strong>Overclaim:</strong> {len(mode['overclaim_phrases'])} | <strong>Leakage:</strong> {_dict_value(mode.get('retrieval_leakage_diagnostic')).get('finding_count')}</p>
  <p><a href="{escape(Path(mode['article_path']).as_posix())}">Article HTML</a> | <a href="{escape(Path(mode['response_path']).as_posix())}">Response JSON</a></p>
  <h4>Omitted Source Claims</h4>
  <pre>{escape(json.dumps(mode['source_claims_omitted'], ensure_ascii=False, indent=2))}</pre>
</section>
"""


def _write_manifest_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        rows.append(_flat_case_row(case))
    _write_csv(path, rows, list(rows[0]) if rows else [])


def _write_length_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        mode_order = _list_value(case.get("mode_order")) or ["legacy", "newsroom_v1"]
        for mode_key in mode_order:
            mode = case["modes"][str(mode_key)]
            rows.append(
                {
                    "input_id": case["input_id"],
                    "mode": mode_key,
                    "requested_length": case["requested_length"],
                    "actual_words": mode["word_count"],
                    "percent_variance_from_target": mode["percent_variance_from_target"],
                    "source_density": case["source_information_density"]["label"],
                    "length_assessment": mode["length_assessment"],
                    "covered_claim_count": mode["coverage_summary"]["covered_claim_count"],
                    "omitted_claim_count": mode["coverage_summary"]["omitted_claim_count"],
                    "unsupported_claim_count": len(mode["unsupported_claims"]),
                }
            )
    _write_csv(path, rows, list(rows[0]) if rows else [])


def _write_coverage_jsonl(path: Path, cases: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")


def _write_review_sheet_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    rows = []
    for case in cases:
        row = _flat_case_row(case)
        row.update(
            {
                "preferred_version": "",
                "corrected_publishable_text": "",
                "missing_facts": "",
                "unnecessary_content": "",
                "supporting_detail_quality": "",
                "attribution_quality": "",
                "contextual_sequencing": "",
                "background_placement": "",
                "impact_framing_concern": "",
                "copied_wording": "",
                "factual_leakage": "",
                "unnatural_tamil": "",
                "translation_like_phrasing": "",
                "headline_correction": "",
                "factual_correction": "",
                "structural_correction": "",
                "overall_editing_effort": "",
                "reviewer_notes": "",
            }
        )
        rows.append(row)
    _write_csv(path, rows, list(rows[0]) if rows else [])


def _flat_case_row(case: dict[str, Any]) -> dict[str, Any]:
    row = {
        "input_id": case["input_id"],
        "source_path": case.get("source_path"),
        "legacy_output_path": case["legacy_output_path"],
        "newsroom_v1_output_path": case["newsroom_v1_output_path"],
        "requested_length": case["requested_length"],
        "legacy_words": case["legacy"]["word_count"],
        "newsroom_v1_words": case["newsroom_v1"]["word_count"],
        "legacy_percent_variance": case["legacy"]["percent_variance_from_target"],
        "newsroom_v1_percent_variance": case["newsroom_v1"]["percent_variance_from_target"],
        "legacy_coverage": case["legacy"]["coverage_summary"]["coverage_ratio"],
        "newsroom_v1_coverage": case["newsroom_v1"]["coverage_summary"]["coverage_ratio"],
        "legacy_warnings": len(case["legacy"]["warnings"]),
        "newsroom_v1_warnings": len(case["newsroom_v1"]["warnings"]),
        "comparison_notes": "|".join(case["comparison_notes"]),
        "editorial_review_status": case["editorial_review_status"],
    }
    for mode_key in _list_value(case.get("mode_order")):
        mode_key = str(mode_key)
        if mode_key in ("legacy", "newsroom_v1") or mode_key not in case["modes"]:
            continue
        mode = case["modes"][mode_key]
        row[f"{mode_key}_output_path"] = mode["response_path"]
        row[f"{mode_key}_words"] = mode["word_count"]
        row[f"{mode_key}_grounding"] = mode["grounding_score"]
        row[f"{mode_key}_unsupported"] = len(mode["unsupported_claims"])
        row[f"{mode_key}_overclaim"] = len(mode["overclaim_phrases"])
        row[f"{mode_key}_leakage"] = _dict_value(
            mode.get("retrieval_leakage_diagnostic")
        ).get("finding_count")
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _brief_payload(brief: dict[str, Any]) -> dict[str, Any]:
    payload = brief.get("brief")
    return payload if isinstance(payload, dict) else {}


def _evaluation_payload(response: dict[str, Any]) -> dict[str, Any]:
    evaluation = response.get("grounding_evaluation_result")
    if isinstance(evaluation, dict):
        return evaluation
    nested = response.get("evaluation")
    if isinstance(nested, dict) and isinstance(nested.get("evaluation"), dict):
        nested_evaluation = nested["evaluation"]
        return nested_evaluation if isinstance(nested_evaluation, dict) else {}
    return {}


def _target(entry: dict[str, Any]) -> int:
    return int(entry.get("desired_word_count") or 600)


def _int_or_none(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _list_value(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _dict_value(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_of_dicts(value: object) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _string_list(value: object) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _tokens(text: str) -> set[str]:
    return {token for token in re.split(r"\W+", text.lower()) if len(token) > 2}


def _best_similarity(claim: str, candidates: list[str]) -> float:
    claim_tokens = _tokens(claim)
    if not claim_tokens:
        return 0.0
    best = 0.0
    for candidate in candidates:
        candidate_tokens = _tokens(candidate)
        if not candidate_tokens:
            continue
        intersection = len(claim_tokens & candidate_tokens)
        union = len(claim_tokens | candidate_tokens)
        best = max(best, intersection / union if union else 0.0)
    return best
