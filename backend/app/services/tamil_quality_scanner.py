"""Lightweight Tamil quality red-flag scanner."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TamilQualityStatus = Literal["pass", "warning", "fail"]
LengthStatus = Literal["pass", "warning"]

ALLOWLISTED_ENGLISH_TERMS = {
    "H-1B",
    "SMS",
    "US",
    "FIIDS",
    "F-1",
    "L-1",
    "visa",
    "green",
    "card",
}

KNOWN_BAD_PHRASES = (
    "pertencிக்கிறார்கள்",
    "இந்த ruling",
    "ruling",
)
RISKY_CONTEXTUAL_PHRASES = (
    "இந்திய குடியுரிமை பெற்றவர்கள்",
)
WEAK_GENERIC_CLOSINGS = (
    "சமூகத்தில் தொடர்புடையது",
    "சமூகத்துடன் தொடர்புடையதாக பார்க்கப்படுகிறது",
)
LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*")
CORRUPTED_MIXED_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9-]*[\u0B80-\u0BFF]+")


@dataclass(frozen=True)
class TamilQualityScanResult:
    tamil_quality_status: TamilQualityStatus
    tamil_quality_issues_count: int
    tamil_quality_warnings: list[str]
    requested_word_count: int | None
    final_article_word_count: int
    length_status: LengthStatus
    length_warning_reason: str | None
    final_article_word_count_ratio: float | None


def scan_tamil_quality(
    draft: dict[str, object],
    desired_word_count: int | None,
) -> TamilQualityScanResult:
    """Scan final Tamil output for obvious publication-quality red flags."""

    article_body = _string_value(draft.get("article_body"))
    full_text = "\n".join(
        _string_value(draft.get(field))
        for field in (
            "headline",
            "subheadline",
            "article_body",
            "seo_title",
            "meta_description",
        )
    )
    failures: list[str] = []
    warnings: list[str] = []

    for phrase in KNOWN_BAD_PHRASES:
        if phrase in full_text:
            failures.append(f"Known bad Tamil/mixed-language phrase found: {phrase}")

    corrupted_tokens = sorted(set(CORRUPTED_MIXED_TOKEN_RE.findall(full_text)))
    if corrupted_tokens:
        failures.append(
            "Corrupted mixed-language token(s): " + ", ".join(corrupted_tokens)
        )

    for phrase in RISKY_CONTEXTUAL_PHRASES:
        if phrase in full_text:
            warnings.append(f"Risky contextual translation found: {phrase}")

    for phrase in WEAK_GENERIC_CLOSINGS:
        if phrase in full_text:
            warnings.append(f"Weak generic phrasing found: {phrase}")

    latin_tokens = sorted(
        {
            token
            for token in LATIN_TOKEN_RE.findall(full_text)
            if token not in ALLOWLISTED_ENGLISH_TERMS
        }
    )
    if latin_tokens:
        warnings.append("Unexpected Latin-script token(s): " + ", ".join(latin_tokens))

    allowed_tokens = sorted(
        {
            token
            for token in LATIN_TOKEN_RE.findall(full_text)
            if token in ALLOWLISTED_ENGLISH_TERMS
        }
    )
    if allowed_tokens:
        warnings.append("Allowed English term(s) present: " + ", ".join(allowed_tokens))

    final_word_count = approximate_tamil_word_count(article_body)
    length_status: LengthStatus = "pass"
    length_warning_reason = None
    final_article_word_count_ratio = (
        round(final_word_count / desired_word_count, 3)
        if desired_word_count
        else None
    )
    if desired_word_count and final_word_count < int(desired_word_count * 0.75):
        length_status = "warning"
        length_warning_reason = (
            "Final article body is materially shorter than requested "
            f"({final_word_count}/{desired_word_count} words)."
        )
        warnings.append(length_warning_reason)

    status: TamilQualityStatus = "pass"
    if warnings:
        status = "warning"
    if failures:
        status = "fail"

    return TamilQualityScanResult(
        tamil_quality_status=status,
        tamil_quality_issues_count=len(failures) + len(warnings),
        tamil_quality_warnings=[*failures, *warnings],
        requested_word_count=desired_word_count,
        final_article_word_count=final_word_count,
        length_status=length_status,
        length_warning_reason=length_warning_reason,
        final_article_word_count_ratio=final_article_word_count_ratio,
    )


def approximate_tamil_word_count(text: str) -> int:
    """Return a simple whitespace-delimited word count for Tamil article body."""

    punctuation = " \t\r\n.,;:!?()[]{}\"'“”‘’…|/\\-–—"
    return len(
        [
            token.strip(punctuation)
            for token in re.split(r"\s+", text.strip())
            if token.strip(punctuation)
        ]
    )


def _string_value(value: object) -> str:
    return value if isinstance(value, str) else ""
