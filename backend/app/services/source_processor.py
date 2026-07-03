"""Source text and URL processing for grounded brief generation."""

from dataclasses import dataclass
from hashlib import sha256

import requests
from bs4 import BeautifulSoup

from backend.app.services.pasted_text_cleaner import clean_pasted_website_text

MIN_SOURCE_CHARS = 40
SOURCE_EXCERPT_CHARS = 700
REQUEST_TIMEOUT_SECONDS = 10


@dataclass(frozen=True)
class ProcessedSource:
    source_type: str
    source_url: str | None
    cleaned_text: str
    source_text_excerpt: str
    source_input_hash: str
    warnings: list[str]
    original_char_count: int
    cleaned_char_count: int
    removed_line_count: int


class SourceProcessingError(RuntimeError):
    """Raised when source input cannot be processed."""


def process_source(
    source_type: str,
    source_input: str,
    source_input_mode: str = "plain_text",
) -> ProcessedSource:
    """Process source input into cleaned text and metadata."""

    normalized_type = source_type.strip().lower()
    if normalized_type == "text":
        return _process_text_source(source_input, source_input_mode)
    if normalized_type == "url":
        return _process_url_source(source_input)
    raise SourceProcessingError("source_type must be 'text' or 'url'.")


def _process_text_source(
    source_input: str,
    source_input_mode: str,
) -> ProcessedSource:
    normalized_mode = source_input_mode.strip().lower()
    if normalized_mode not in {"plain_text", "pasted_web_text"}:
        raise SourceProcessingError(
            "source_input_mode must be 'plain_text' or 'pasted_web_text'."
        )

    warnings: list[str] = []
    removed_line_count = 0
    original_char_count = len(source_input)
    if normalized_mode == "pasted_web_text":
        cleanup = clean_pasted_website_text(source_input)
        cleaned_text = cleanup.cleaned_text
        removed_line_count = cleanup.removed_line_count
        warnings.extend(cleanup.warnings)
    else:
        cleaned_text = _clean_text(source_input)

    _validate_minimum_length(cleaned_text)
    if normalized_mode == "pasted_web_text" and len(cleaned_text) < 250:
        warnings.append("Cleaned pasted website text is short for article generation.")

    return ProcessedSource(
        source_type="text",
        source_url=None,
        cleaned_text=cleaned_text,
        source_text_excerpt=_excerpt(cleaned_text),
        source_input_hash=_hash_source(f"text:{normalized_mode}", source_input),
        warnings=warnings,
        original_char_count=original_char_count,
        cleaned_char_count=len(cleaned_text),
        removed_line_count=removed_line_count,
    )


def _process_url_source(source_input: str) -> ProcessedSource:
    url = source_input.strip()
    if not url.startswith(("http://", "https://")):
        raise SourceProcessingError("URL source_input must start with http:// or https://.")

    try:
        response = requests.get(
            url,
            timeout=REQUEST_TIMEOUT_SECONDS,
            headers={"User-Agent": "StyleScribe/0.1"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SourceProcessingError(f"Unable to fetch URL: {url}") from exc

    cleaned_text = _extract_readable_text(response.text)
    _validate_minimum_length(cleaned_text)
    warnings: list[str] = []
    if len(cleaned_text) < 500:
        warnings.append("URL extraction produced limited readable text.")

    return ProcessedSource(
        source_type="url",
        source_url=url,
        cleaned_text=cleaned_text,
        source_text_excerpt=_excerpt(cleaned_text),
        source_input_hash=_hash_source("url", url),
        warnings=warnings,
        original_char_count=len(response.text),
        cleaned_char_count=len(cleaned_text),
        removed_line_count=0,
    )


def _extract_readable_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()

    article = soup.find("article")
    root = article if article is not None else soup.body or soup
    parts = [
        text.strip()
        for text in root.stripped_strings
        if text.strip()
    ]
    return _clean_text("\n".join(parts))


def _clean_text(text: str) -> str:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _validate_minimum_length(cleaned_text: str) -> None:
    if len(cleaned_text) < MIN_SOURCE_CHARS:
        message = (
            "Source text is too short; minimum useful length is "
            f"{MIN_SOURCE_CHARS} characters."
        )
        raise SourceProcessingError(message)


def _excerpt(text: str) -> str:
    if len(text) <= SOURCE_EXCERPT_CHARS:
        return text
    return text[: SOURCE_EXCERPT_CHARS - 3].rstrip() + "..."


def _hash_source(source_type: str, source_input: str) -> str:
    return sha256(f"{source_type}:{source_input}".encode()).hexdigest()
