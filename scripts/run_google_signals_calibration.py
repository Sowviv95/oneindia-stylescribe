from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path

DEFAULT_INPUT_DIR = Path("manual_tests/google_signals_calibration/inputs")
DEFAULT_OUTPUT_PATH = Path(
    "manual_tests/google_signals_calibration/google_signals_summary.csv"
)

COMPONENT_NAMES = [
    "search_intent_clarity",
    "headline_search_clarity",
    "freshness_timeliness",
    "originality_angle",
    "eeat_trust",
    "snippet_meta_readiness",
    "structured_data_readiness",
]

CSV_HEADERS = [
    "case_name",
    "filename",
    "author_id",
    "generated_headline",
    "google_signals_score",
    "google_signals_version",
    *COMPONENT_NAMES,
    "google_signals_risk_flags",
    "google_signals_recommendations",
    "google_signals_error",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()

    rows = summarize_google_signals_folder(args.input_dir)
    write_summary_csv(rows, args.output)
    print(f"Wrote {len(rows)} Google Signals calibration row(s) to {args.output}")


def summarize_google_signals_folder(input_dir: Path) -> list[dict[str, str]]:
    if not input_dir.exists():
        return []
    rows: list[dict[str, str]] = []
    for path in sorted(input_dir.glob("*.json")):
        rows.append(summarize_workflow_json(path))
    return rows


def summarize_workflow_json(path: Path) -> dict[str, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return _empty_row(
            path,
            google_signals_error=f"Invalid JSON: {exc}",
        )
    if not isinstance(payload, dict):
        return _empty_row(path, google_signals_error="Workflow JSON is not an object.")

    google_signals = _dict_value(payload.get("google_signals"))
    components = _component_scores(
        google_signals.get("components")
        if google_signals
        else payload.get("google_signals_components")
    )
    row = _empty_row(
        path,
        case_name=_case_name(payload, path),
        author_id=_string_value(payload.get("author_id")),
        generated_headline=_headline(payload),
        google_signals_error=_string_value(payload.get("google_signals_error")),
    )
    row["google_signals_score"] = _string_value(
        google_signals.get("score")
        if google_signals
        else payload.get("google_signals_score")
    )
    row["google_signals_version"] = _string_value(
        google_signals.get("version")
        if google_signals
        else payload.get("google_signals_version")
    )
    for component_name in COMPONENT_NAMES:
        row[component_name] = components.get(component_name, "")
    row["google_signals_risk_flags"] = _joined_values(
        google_signals.get("risk_flags")
        if google_signals
        else payload.get("google_signals_risk_flags")
    )
    row["google_signals_recommendations"] = _joined_values(
        google_signals.get("recommendations")
        if google_signals
        else payload.get("google_signals_recommendations")
    )
    return row


def write_summary_csv(rows: Iterable[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in CSV_HEADERS})


def _empty_row(
    path: Path,
    *,
    case_name: str | None = None,
    author_id: str = "",
    generated_headline: str = "",
    google_signals_error: str = "",
) -> dict[str, str]:
    return {
        "case_name": case_name or path.stem,
        "filename": path.name,
        "author_id": author_id,
        "generated_headline": generated_headline,
        "google_signals_score": "",
        "google_signals_version": "",
        **{component_name: "" for component_name in COMPONENT_NAMES},
        "google_signals_risk_flags": "",
        "google_signals_recommendations": "",
        "google_signals_error": google_signals_error,
    }


def _case_name(payload: dict[str, object], path: Path) -> str:
    value = payload.get("case_name")
    return value if isinstance(value, str) and value else path.stem


def _headline(payload: dict[str, object]) -> str:
    headline = payload.get("generated_headline")
    if isinstance(headline, str):
        return headline
    draft_summary = _dict_value(payload.get("draft_summary"))
    return _string_value(draft_summary.get("headline"))


def _component_scores(value: object) -> dict[str, str]:
    if not isinstance(value, list):
        return {}
    scores: dict[str, str] = {}
    for component in value:
        if not isinstance(component, dict):
            continue
        name = component.get("name")
        if isinstance(name, str) and name:
            scores[name] = _string_value(component.get("score"))
    return scores


def _joined_values(value: object) -> str:
    if isinstance(value, list):
        return " | ".join(_string_value(item) for item in value if item is not None)
    return _string_value(value)


def _dict_value(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _string_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


if __name__ == "__main__":
    main()
