import csv
import json
from pathlib import Path

from scripts.run_google_signals_calibration import (
    summarize_google_signals_folder,
    summarize_workflow_json,
    write_summary_csv,
)


def test_google_signals_calibration_valid_object(tmp_path: Path) -> None:
    path = tmp_path / "valid.json"
    _write_json(
        path,
        {
            "case_name": "case-a",
            "author_id": "v_vasanthi",
            "generated_headline": "Chennai flood warning pilot",
            "google_signals": {
                "score": 82,
                "version": "google_signals_v1",
                "components": [
                    {"name": "search_intent_clarity", "score": 85},
                    {"name": "headline_search_clarity", "score": 80},
                    {"name": "freshness_timeliness", "score": 78},
                    {"name": "originality_angle", "score": 72},
                    {"name": "eeat_trust", "score": 88},
                    {"name": "snippet_meta_readiness", "score": 81},
                    {"name": "structured_data_readiness", "score": 79},
                ],
                "risk_flags": ["Headline could be sharper."],
                "recommendations": ["Add a clearer first-paragraph summary."],
            },
        },
    )

    row = summarize_workflow_json(path)

    assert row["case_name"] == "case-a"
    assert row["filename"] == "valid.json"
    assert row["author_id"] == "v_vasanthi"
    assert row["google_signals_score"] == "82"
    assert row["search_intent_clarity"] == "85"
    assert row["structured_data_readiness"] == "79"
    assert row["google_signals_risk_flags"] == "Headline could be sharper."


def test_google_signals_calibration_missing_object(tmp_path: Path) -> None:
    path = tmp_path / "missing.json"
    _write_json(
        path,
        {
            "author_id": "author-1",
            "draft_summary": {"headline": "Fallback headline"},
        },
    )

    row = summarize_workflow_json(path)

    assert row["case_name"] == "missing"
    assert row["generated_headline"] == "Fallback headline"
    assert row["google_signals_score"] == ""
    assert row["search_intent_clarity"] == ""
    assert row["google_signals_error"] == ""


def test_google_signals_calibration_evaluator_error_case(tmp_path: Path) -> None:
    path = tmp_path / "error.json"
    _write_json(
        path,
        {
            "generated_headline": "Headline",
            "google_signals_available": False,
            "google_signals_error": "Google Signals evaluator failed: invalid JSON",
        },
    )

    row = summarize_workflow_json(path)

    assert row["generated_headline"] == "Headline"
    assert row["google_signals_error"] == (
        "Google Signals evaluator failed: invalid JSON"
    )
    assert row["google_signals_score"] == ""


def test_google_signals_calibration_missing_individual_component(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    _write_json(
        input_dir / "partial.json",
        {
            "google_signals_score": 70,
            "google_signals_version": "google_signals_v1",
            "google_signals_components": [
                {"name": "search_intent_clarity", "score": 76},
                {"name": "headline_search_clarity", "score": 70},
            ],
            "google_signals_risk_flags": ["Intent is slightly broad."],
            "google_signals_recommendations": ["Sharpen the headline."],
        },
    )
    output_path = tmp_path / "summary.csv"

    rows = summarize_google_signals_folder(input_dir)
    write_summary_csv(rows, output_path)
    csv_rows = list(csv.DictReader(output_path.open(encoding="utf-8")))

    assert len(rows) == 1
    assert csv_rows[0]["search_intent_clarity"] == "76"
    assert csv_rows[0]["headline_search_clarity"] == "70"
    assert csv_rows[0]["freshness_timeliness"] == ""
    assert csv_rows[0]["google_signals_risk_flags"] == "Intent is slightly broad."
    assert csv_rows[0]["google_signals_recommendations"] == "Sharpen the headline."


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

