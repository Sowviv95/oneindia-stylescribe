"""Lightweight language detection for source material."""

import re

TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")
LATIN_PATTERN = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    """Detect Tamil, English, or unknown with simple character heuristics."""

    if TAMIL_PATTERN.search(text):
        return "ta"

    non_space_chars = [character for character in text if not character.isspace()]
    if not non_space_chars:
        return "unknown"

    latin_count = len(LATIN_PATTERN.findall(text))
    if latin_count / len(non_space_chars) >= 0.6:
        return "en"
    return "unknown"
