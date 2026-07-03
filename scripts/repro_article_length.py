"""Minimal direct OpenAI bake-off for Tamil article length behavior."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.config import get_settings
from backend.app.db.repository import (
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.services.model_clients.openai_client import OpenAIClientError
from backend.app.services.tamil_quality_scanner import approximate_tamil_word_count

DESIRED_WORD_COUNT = 600
SUCCESS_MIN_WORDS = 500
SUCCESS_MAX_WORDS = 700
SECTION_COUNT = 8
SECTION_TARGET_WORDS = 75
SECTION_MIN_WORDS = 70
PARTS = [
    ("part_1_opening_background", "opening and background", 180, 150),
    ("part_2_key_details_context", "key details and context", 180, 150),
    ("part_3_implications_conclusion", "implications and conclusion", 180, 150),
]


@dataclass(frozen=True)
class DirectResponse:
    content: str
    runtime_seconds: float


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    if settings.openai_api_key is None:
        raise OpenAIClientError("OPENAI_API_KEY is required.")

    repo = StyleScribeRepository()
    repo.initialize_schema()
    brief = _load_brief(repo, args.brief_id)
    profile = _load_profile(repo, args.author_id)
    model = args.model or settings.openai_model or "gpt-4o-mini"
    client = OpenAI(
        api_key=settings.openai_api_key.get_secret_value(),
        timeout=settings.openai_timeout_seconds,
        max_retries=settings.openai_max_retries,
    )

    print("StyleScribe article length bake-off")
    print(f"model={model}")
    print("max_tokens_configured=false")
    print(f"brief_id={brief.brief_id}")
    print(f"profile_id={profile.profile_id}")
    print(f"author_id={profile.author_id}")
    print(json.dumps(_brief_sufficiency_report(brief), ensure_ascii=False))
    print("")

    grounding_rule = _grounding_rule(args.relaxed_grounding)
    reports = [
        _run_baseline(client, model, brief, profile, grounding_rule, args.debug),
        _run_hard_section_contract(
            client,
            model,
            brief,
            profile,
            grounding_rule,
            args.debug,
        ),
        _run_multi_part_article(
            client,
            model,
            brief,
            profile,
            grounding_rule,
            args.debug,
        ),
    ]

    for report in reports:
        print(json.dumps(report, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct OpenAI article length bake-off."
    )
    parser.add_argument("--brief-id", help="Grounded brief ID. Defaults to latest.")
    parser.add_argument(
        "--author-id",
        default="v_vasanthi",
        help="Author ID for latest style profile.",
    )
    parser.add_argument("--model", help="Override OPENAI_MODEL for this bake-off.")
    parser.add_argument(
        "--relaxed-grounding",
        action="store_true",
        help="Allow neutral connective context without new factual claims.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print prompts and raw responses. Do not use with sensitive inputs.",
    )
    return parser.parse_args()


def _load_brief(
    repo: StyleScribeRepository,
    brief_id: str | None,
) -> GroundedBriefRecord:
    brief = (
        repo.fetch_grounded_brief(brief_id)
        if brief_id
        else repo.fetch_latest_grounded_brief()
    )
    if brief is None:
        raise RuntimeError("No grounded brief found in local repository.")
    return brief


def _load_profile(
    repo: StyleScribeRepository,
    author_id: str,
) -> AuthorStyleProfileRecord:
    profile = repo.fetch_latest_author_style_profile(author_id)
    if profile is None:
        raise RuntimeError(f"No author style profile found for author_id: {author_id}")
    return profile


def _run_baseline(
    client: OpenAI,
    model: str,
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    debug: bool,
) -> dict[str, object]:
    prompt = _plain_article_prompt(brief, profile, grounding_rule)
    response = _chat(client, model, _plain_system_prompt(), prompt)
    words = approximate_tamil_word_count(response.content)
    return _report(
        variant_name="current_baseline_plain_text",
        model=model,
        prompt_char_count=len(prompt),
        raw_response_word_count=words,
        runtime_seconds=response.runtime_seconds,
        debug=debug,
        prompt=prompt,
        response=response.content,
        assembled_word_count=words,
    )


def _run_hard_section_contract(
    client: OpenAI,
    model: str,
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    debug: bool,
) -> dict[str, object]:
    start = time.monotonic()
    prompt_char_count = 0
    raw_response_word_count = 0
    section_reports: list[dict[str, object]] = []
    selected_sections: list[str] = []

    for section_number in range(1, SECTION_COUNT + 1):
        prompt = _hard_section_prompt(brief, profile, grounding_rule, section_number)
        prompt_char_count += len(prompt)
        first_response = _chat(client, model, _section_system_prompt(), prompt)
        first_words = approximate_tamil_word_count(first_response.content)
        raw_response_word_count += first_words
        selected_text = first_response.content.strip()
        retry_attempted = first_words < SECTION_MIN_WORDS
        retry_words: int | None = None

        if retry_attempted:
            retry_prompt = _section_retry_prompt(
                brief=brief,
                profile=profile,
                grounding_rule=grounding_rule,
                section_number=section_number,
                previous_text=selected_text,
                previous_word_count=first_words,
            )
            prompt_char_count += len(retry_prompt)
            retry_response = _chat(
                client,
                model,
                _section_system_prompt(),
                retry_prompt,
            )
            retry_words = approximate_tamil_word_count(retry_response.content)
            raw_response_word_count += retry_words
            if retry_words >= first_words:
                selected_text = retry_response.content.strip()
            _debug_dump(
                debug,
                f"hard_section_{section_number}_retry",
                retry_prompt,
                retry_response.content,
            )

        selected_words = approximate_tamil_word_count(selected_text)
        selected_sections.append(selected_text)
        section_reports.append(
            {
                "section_number": section_number,
                "first_pass_section_words": first_words,
                "retry_attempted": retry_attempted,
                "retry_section_words": retry_words,
                "selected_section_words": selected_words,
            }
        )
        _debug_dump(
            debug,
            f"hard_section_{section_number}_first_pass",
            prompt,
            first_response.content,
        )

    assembled = "\n\n".join(selected_sections)
    assembled_words = approximate_tamil_word_count(assembled)
    return _report(
        variant_name="hard_section_contract",
        model=model,
        prompt_char_count=prompt_char_count,
        raw_response_word_count=raw_response_word_count,
        runtime_seconds=time.monotonic() - start,
        debug=False,
        prompt="",
        response="",
        assembled_word_count=assembled_words,
        extra={
            "section_count": len(section_reports),
            "sections": section_reports,
        },
    )


def _run_multi_part_article(
    client: OpenAI,
    model: str,
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    debug: bool,
) -> dict[str, object]:
    start = time.monotonic()
    prompt_char_count = 0
    raw_response_word_count = 0
    part_reports: list[dict[str, object]] = []
    selected_parts: list[str] = []

    for part_name, part_purpose, target_words, minimum_words in PARTS:
        prompt = _part_prompt(
            brief,
            profile,
            grounding_rule,
            part_name,
            part_purpose,
            target_words,
            minimum_words,
        )
        prompt_char_count += len(prompt)
        first_response = _chat(client, model, _plain_system_prompt(), prompt)
        first_words = approximate_tamil_word_count(first_response.content)
        raw_response_word_count += first_words
        selected_text = first_response.content.strip()
        retry_attempted = first_words < minimum_words
        retry_words: int | None = None

        if retry_attempted:
            retry_prompt = _part_retry_prompt(
                brief=brief,
                profile=profile,
                grounding_rule=grounding_rule,
                part_name=part_name,
                part_purpose=part_purpose,
                previous_text=selected_text,
                previous_word_count=first_words,
                minimum_words=minimum_words,
            )
            prompt_char_count += len(retry_prompt)
            retry_response = _chat(client, model, _plain_system_prompt(), retry_prompt)
            retry_words = approximate_tamil_word_count(retry_response.content)
            raw_response_word_count += retry_words
            if retry_words >= first_words:
                selected_text = retry_response.content.strip()
            _debug_dump(
                debug,
                f"{part_name}_retry",
                retry_prompt,
                retry_response.content,
            )

        selected_words = approximate_tamil_word_count(selected_text)
        selected_parts.append(selected_text)
        part_reports.append(
            {
                "part_name": part_name,
                "first_pass_words": first_words,
                "retry_attempted": retry_attempted,
                "retry_words": retry_words,
                "selected_words": selected_words,
            }
        )
        _debug_dump(debug, f"{part_name}_first_pass", prompt, first_response.content)

    assembled = "\n\n".join(selected_parts)
    assembled_words = approximate_tamil_word_count(assembled)
    return _report(
        variant_name="multi_part_article",
        model=model,
        prompt_char_count=prompt_char_count,
        raw_response_word_count=raw_response_word_count,
        runtime_seconds=time.monotonic() - start,
        debug=False,
        prompt="",
        response="",
        assembled_word_count=assembled_words,
        extra={
            "part_word_counts": part_reports,
            "retry_count": sum(1 for part in part_reports if part["retry_attempted"]),
        },
    )


def _chat(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> DirectResponse:
    start = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    runtime = time.monotonic() - start
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("OpenAI returned an empty response.")
    return DirectResponse(content=content, runtime_seconds=runtime)


def _brief_sufficiency_report(brief: GroundedBriefRecord) -> dict[str, object]:
    brief_json = StyleScribeRepository.decode_json_object(brief.brief_json)
    confirmed_facts = _list_value(brief_json.get("confirmed_facts"))
    expansion_items = _expansion_items(brief_json)
    sufficiency_score = (
        len(confirmed_facts)
        + len(_list_value(brief_json.get("key_entities")))
        + len(_list_value(brief_json.get("numbers_and_statistics")))
        + len(_list_value(brief_json.get("quotes")))
        + len(_list_value(brief_json.get("affected_groups")))
        + len(_list_value(brief_json.get("policy_or_legal_context")))
        + len(_list_value(brief_json.get("background_from_source")))
    )
    appears_thin = len(StyleScribeRepository.encode_json(brief_json)) < 2000 or (
        len(confirmed_facts) < 4 and sufficiency_score < 10
    )
    return {
        "report_type": "brief_sufficiency",
        "brief_char_count": len(StyleScribeRepository.encode_json(brief_json)),
        "brief_word_count_approx": approximate_tamil_word_count(
            StyleScribeRepository.encode_json(brief_json)
        ),
        "grounded_fact_count": len(confirmed_facts),
        "expansion_item_count": len(expansion_items),
        "expansion_item_preview": expansion_items[:5],
        "appears_thin_for_600_words": appears_thin,
        "sufficiency_assessment": "thin" if appears_thin else "likely_sufficient",
    }


def _expansion_items(brief_json: dict[str, object]) -> list[str]:
    keys = [
        "numbers_and_statistics",
        "affected_groups",
        "quotes",
        "policy_or_legal_context",
        "background_from_source",
        "dates_or_timeline",
    ]
    items: list[str] = []
    for key in keys:
        for item in _list_value(brief_json.get(key)):
            items.append(f"{key}: {_compact_preview(item)}")
    return items


def _list_value(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _compact_preview(value: object) -> str:
    text = str(value).replace("\n", " ").strip()
    return text[:120]


def _plain_system_prompt() -> str:
    return (
        "You write publication-ready Tamil news articles. Follow grounding, "
        "length, and continuation instructions exactly."
    )


def _section_system_prompt() -> str:
    return (
        "You write exactly one publication-ready Tamil article section. Return "
        "plain Tamil prose only, without headings or JSON."
    )


def _grounding_rule(relaxed_grounding: bool) -> str:
    if relaxed_grounding:
        return (
            "Use only the facts in the brief, but you may add neutral connective "
            "context that does not introduce new factual claims."
        )
    return (
        "Use only facts from grounded_brief_for_facts_only. "
        "Do not use outside knowledge."
    )


def _plain_article_prompt(
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
) -> str:
    payload: dict[str, object] = {
        "task": "Write one complete publication-ready Tamil article.",
        "target_language": "ta",
        "desired_word_count": DESIRED_WORD_COUNT,
        "article_type": "public_interest",
        "grounding_rule": grounding_rule,
        "length_rule": (
            "Write 500-700 Tamil words, aiming near 600. Do not return a short "
            "summary. Use multiple developed paragraphs."
        ),
        "grounded_brief_for_facts_only": _brief_payload(brief),
        "style_profile_for_voice_only": _profile_payload(profile),
        "style_rule": (
            "Use the style profile only for writing voice. Do not take facts from it."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def _hard_section_prompt(
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    section_number: int,
) -> str:
    payload = {
        "task": "Write one section of an 8-section Tamil article.",
        "section_number": section_number,
        "total_sections": SECTION_COUNT,
        "target_words": SECTION_TARGET_WORDS,
        "minimum_words": SECTION_MIN_WORDS,
        "instruction": (
            "Write this section as at least 70 Tamil words and close to 75 words. "
            "Do not summarize. Do not write headings. Use a distinct angle for "
            "this section based on section_number."
        ),
        "section_angles": [
            "opening news hook",
            "core development",
            "key people and entities",
            "numbers and statistics",
            "affected groups",
            "quotes and attribution",
            "policy or legal context",
            "reader relevance and cautious close",
        ],
        "grounding_rule": grounding_rule,
        "grounded_brief_for_facts_only": _brief_payload(brief),
        "style_profile_for_voice_only": _profile_payload(profile),
    }
    return json.dumps(payload, ensure_ascii=False)


def _section_retry_prompt(
    *,
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    section_number: int,
    previous_text: str,
    previous_word_count: int,
) -> str:
    payload = {
        "task": "Expand the same Tamil article section.",
        "section_number": section_number,
        "previous_section_text": previous_text,
        "instruction": (
            f"You wrote only {previous_word_count} words. Expand this same "
            "section to at least 70 Tamil words using only the provided brief. "
            "Do not summarize. Do not restart."
        ),
        "grounding_rule": grounding_rule,
        "grounded_brief_for_facts_only": _brief_payload(brief),
        "style_profile_for_voice_only": _profile_payload(profile),
    }
    return json.dumps(payload, ensure_ascii=False)


def _part_prompt(
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    part_name: str,
    part_purpose: str,
    target_words: int,
    minimum_words: int,
) -> str:
    payload = {
        "task": "Write one deterministic part of a three-part Tamil article.",
        "part_name": part_name,
        "part_purpose": part_purpose,
        "target_words": target_words,
        "minimum_words": minimum_words,
        "instruction": (
            f"Write the {part_purpose} part as publication-ready Tamil prose. "
            f"Write at least {minimum_words} Tamil words and aim for "
            f"{target_words}. Do not write a full article; write only this part. "
            "Do not use headings."
        ),
        "grounding_rule": grounding_rule,
        "grounded_brief_for_facts_only": _brief_payload(brief),
        "style_profile_for_voice_only": _profile_payload(profile),
    }
    return json.dumps(payload, ensure_ascii=False)


def _part_retry_prompt(
    *,
    brief: GroundedBriefRecord,
    profile: AuthorStyleProfileRecord,
    grounding_rule: str,
    part_name: str,
    part_purpose: str,
    previous_text: str,
    previous_word_count: int,
    minimum_words: int,
) -> str:
    payload = {
        "task": "Expand the same Tamil article part.",
        "part_name": part_name,
        "part_purpose": part_purpose,
        "previous_part_text": previous_text,
        "instruction": (
            f"You wrote only {previous_word_count} words. Expand this same "
            f"{part_purpose} part to at least {minimum_words} Tamil words using "
            "only the provided brief. Do not summarize. Do not restart."
        ),
        "grounding_rule": grounding_rule,
        "grounded_brief_for_facts_only": _brief_payload(brief),
        "style_profile_for_voice_only": _profile_payload(profile),
    }
    return json.dumps(payload, ensure_ascii=False)


def _brief_payload(brief: GroundedBriefRecord) -> dict[str, object]:
    return {
        "brief_id": brief.brief_id,
        "source_language": brief.source_language,
        "target_language": brief.target_language,
        "brief": StyleScribeRepository.decode_json_object(brief.brief_json),
    }


def _profile_payload(profile: AuthorStyleProfileRecord) -> dict[str, object]:
    return {
        "profile_id": profile.profile_id,
        "language": profile.language,
        "profile": StyleScribeRepository.decode_json_object(profile.profile_json),
    }


def _report(
    *,
    variant_name: str,
    model: str,
    prompt_char_count: int,
    raw_response_word_count: int,
    runtime_seconds: float,
    debug: bool,
    prompt: str,
    response: str,
    assembled_word_count: int | None = None,
    parsed_body_word_count: int | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    _debug_dump(debug, variant_name, prompt, response)
    success_word_count = (
        assembled_word_count or parsed_body_word_count or raw_response_word_count
    )
    report: dict[str, object] = {
        "variant_name": variant_name,
        "model": model,
        "prompt_char_count": prompt_char_count,
        "max_tokens_configured": False,
        "response_format": "text",
        "raw_response_word_count": raw_response_word_count,
        "runtime_seconds": round(runtime_seconds, 2),
        "success": SUCCESS_MIN_WORDS <= success_word_count <= SUCCESS_MAX_WORDS,
    }
    if parsed_body_word_count is not None:
        report["parsed_body_word_count"] = parsed_body_word_count
    if assembled_word_count is not None:
        report["assembled_word_count"] = assembled_word_count
    if extra:
        report.update(extra)
    return report


def _debug_dump(debug: bool, label: str, prompt: str, response: str) -> None:
    if not debug:
        return
    print(f"\n--- DEBUG {label} PROMPT ---")
    print(prompt)
    print(f"\n--- DEBUG {label} RESPONSE ---")
    print(response)


if __name__ == "__main__":
    main()
