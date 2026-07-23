# ruff: noqa: E501
from pathlib import Path

from backend.app.services.newsroom_benchmark_review_service import (
    ReviewPackConfig,
    generate_editorial_review_pack,
)


def test_review_pack_generates_deterministic_manifest(tmp_path: Path) -> None:
    root = _comparison_root(tmp_path)

    first = generate_editorial_review_pack(ReviewPackConfig(root))
    first_text = first["review_manifest_json"].read_text(encoding="utf-8")
    second = generate_editorial_review_pack(ReviewPackConfig(root))
    second_text = second["review_manifest_json"].read_text(encoding="utf-8")

    assert first_text == second_text
    assert first["editorial_review_sheet_csv"].exists()
    assert "preferred_version" in first["editorial_review_sheet_csv"].read_text(
        encoding="utf-8"
    )


def test_factual_coverage_marks_covered_and_omitted_claims(tmp_path: Path) -> None:
    root = _comparison_root(tmp_path)

    paths = generate_editorial_review_pack(ReviewPackConfig(root))
    coverage = paths["factual_coverage_jsonl"].read_text(encoding="utf-8")

    assert "AI-powered customer support systems" in coverage
    assert "source_claims_covered" in coverage
    assert "source_claims_omitted" in coverage
    assert "short_with_possible_factual_omissions" in coverage


def _comparison_root(tmp_path: Path) -> Path:
    root = tmp_path / "comparison"
    shared = root / "shared" / "input_01"
    shared.mkdir(parents=True)
    brief = shared / "brief.json"
    brief.write_text(
        """
{
  "brief": {
    "topic": "AI support",
    "one_line_summary": "Companies test AI support.",
    "confirmed_facts": [
      "A group of Indian technology companies is testing AI-powered customer support systems.",
      "The pilot is being conducted in Tamil and Hindi."
    ],
    "quotes": [{"speaker": "Industry experts"}],
    "dates_or_timeline": [],
    "numbers_and_statistics": [],
    "affected_groups": ["Banks"],
    "policy_or_legal_context": []
  }
}
""",
        encoding="utf-8",
    )
    manifest = root / "shared" / "manifest.json"
    manifest.write_text(
        f"""
{{
  "inputs": [
    {{
      "input_id": "input_01",
      "source_title": "Source",
      "source_path": "source.docx",
      "brief_path": "{brief.as_posix()}",
      "desired_word_count": 600
    }}
  ]
}}
""",
        encoding="utf-8",
    )
    _write_response(
        root / "gemini_3_5_flash" / "input_01",
        "legacy",
        520,
        [
            "A group of Indian technology companies is testing AI-powered customer support systems.",
            "The pilot is being conducted in Tamil and Hindi.",
        ],
    )
    _write_response(
        root / "newsroom_v1_gemini_gemini_3_5_flash" / "input_01",
        "newsroom_v1",
        320,
        [
            "A group of Indian technology companies is testing AI-powered customer support systems.",
        ],
    )
    return root


def _write_response(
    input_dir: Path,
    mode: str,
    word_count: int,
    preserved_facts: list[str],
) -> None:
    input_dir.mkdir(parents=True)
    (input_dir / "response.json").write_text(
        f"""
{{
  "completion_status": "completed",
  "generation_mode": "{mode}",
  "prompt_version": "{mode}_prompt",
  "generation_model": "gemini-3.5-flash",
  "generated_headline": "Headline",
  "generated_tamil_article": "article",
  "word_count": {word_count},
  "target_minimum": 450,
  "target_maximum": 690,
  "grounding_score": 90,
  "unsupported_claims": [],
  "warnings": [],
  "grounding_evaluation_result": {{
    "preserved_facts": {preserved_facts!r},
    "missing_key_facts": ["The pilot is being conducted in Tamil and Hindi."]
  }}
}}
""".replace("'", '"'),
        encoding="utf-8",
    )
    (input_dir / "article.html").write_text("<html></html>", encoding="utf-8")
    (input_dir / "telemetry.json").write_text("{}", encoding="utf-8")
