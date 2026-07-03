"""Conservative cleanup for copy-pasted website article text."""

from __future__ import annotations

from dataclasses import dataclass

BOILERPLATE_LINES = {
    "advertisement",
    "read more",
    "subscribe",
    "sign in",
    "share",
    "follow us",
    "related articles",
    "comments",
    "cookie notice",
    "accept cookies",
    "newsletter",
    "trending",
}


@dataclass(frozen=True)
class PastedTextCleanupResult:
    cleaned_text: str
    removed_line_count: int
    original_char_count: int
    cleaned_char_count: int
    warnings: list[str]


def clean_pasted_website_text(text: str) -> PastedTextCleanupResult:
    """Remove obvious website noise while preserving article-like paragraphs."""

    original_char_count = len(text)
    seen_lines: set[str] = set()
    cleaned_lines: list[str] = []
    removed_line_count = 0

    for raw_line in text.splitlines():
        line = " ".join(raw_line.split()).strip()
        if not line:
            if cleaned_lines and cleaned_lines[-1]:
                cleaned_lines.append("")
            continue

        normalized = line.lower().strip(":|")
        if _is_boilerplate(normalized) or _is_navigation_like(line):
            removed_line_count += 1
            continue

        duplicate_key = normalized
        if duplicate_key in seen_lines:
            removed_line_count += 1
            continue

        seen_lines.add(duplicate_key)
        cleaned_lines.append(line)

    cleaned_text = _collapse_blank_lines(cleaned_lines)
    warnings = _build_cleanup_warnings(
        original_char_count=original_char_count,
        cleaned_char_count=len(cleaned_text),
        removed_line_count=removed_line_count,
    )
    return PastedTextCleanupResult(
        cleaned_text=cleaned_text,
        removed_line_count=removed_line_count,
        original_char_count=original_char_count,
        cleaned_char_count=len(cleaned_text),
        warnings=warnings,
    )


def _is_boilerplate(normalized_line: str) -> bool:
    return normalized_line in BOILERPLATE_LINES


def _is_navigation_like(line: str) -> bool:
    words = line.split()
    if len(words) > 3:
        return False
    has_sentence_signal = any(char in line for char in ".?!:;\"'")
    has_digit = any(char.isdigit() for char in line)
    return not has_sentence_signal and not has_digit and len(line) <= 24


def _collapse_blank_lines(lines: list[str]) -> str:
    collapsed: list[str] = []
    for line in lines:
        if not line:
            if collapsed and collapsed[-1]:
                collapsed.append("")
            continue
        collapsed.append(line)
    while collapsed and not collapsed[-1]:
        collapsed.pop()
    return "\n".join(collapsed).strip()


def _build_cleanup_warnings(
    original_char_count: int,
    cleaned_char_count: int,
    removed_line_count: int,
) -> list[str]:
    warnings: list[str] = []
    if original_char_count and cleaned_char_count < original_char_count * 0.5:
        warnings.append("Pasted text cleanup removed more than 50% of the input.")
    if removed_line_count >= 10:
        warnings.append("Pasted text cleanup removed many short or duplicate lines.")
    return warnings
