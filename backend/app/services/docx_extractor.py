"""DOCX text extraction for local author samples and corpus pipelines."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docx import Document

SHORT_HEADLINE_MAX_WORDS = 18
LONG_LEDE_MIN_WORDS = 28
SUBHEADLINE_MAX_WORDS = 35
SENTENCE_ENDINGS = (".", "?", "!", "।", "?", "!")


@dataclass(frozen=True)
class ExtractedParagraph:
    text: str
    style_name: str
    word_count: int
    is_bold_only: bool
    max_font_size_pt: float | None


@dataclass(frozen=True)
class ExtractedDocx:
    text: str
    char_count: int
    paragraphs: list[str]
    headline: str | None
    subheadline: str | None
    body_text: str
    word_count: int
    headline_status: str
    subheadline_status: str
    structure_confidence: str
    headline_candidate: str | None
    paragraph_metadata: list[dict[str, object]]


class DocxExtractionError(RuntimeError):
    """Raised when a DOCX file cannot be read."""


def extract_docx_text(path: Path) -> ExtractedDocx:
    """Extract non-empty paragraph text from a DOCX file."""

    try:
        document = Document(str(path))
    except Exception as exc:
        raise DocxExtractionError(f"Unable to read DOCX file: {path}") from exc

    extracted_paragraphs = [
        _extract_paragraph(paragraph)
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    paragraphs = [paragraph.text for paragraph in extracted_paragraphs]
    text = "\n".join(paragraphs)
    article_parts = split_article_paragraphs(extracted_paragraphs)
    return ExtractedDocx(
        text=text,
        char_count=len(text),
        paragraphs=paragraphs,
        headline=article_parts.headline,
        subheadline=article_parts.subheadline,
        body_text=article_parts.body_text,
        word_count=count_words(text),
        headline_status=article_parts.headline_status,
        subheadline_status=article_parts.subheadline_status,
        structure_confidence=article_parts.structure_confidence,
        headline_candidate=article_parts.headline_candidate,
        paragraph_metadata=[
            {
                "style_name": paragraph.style_name,
                "word_count": paragraph.word_count,
                "is_bold_only": paragraph.is_bold_only,
                "max_font_size_pt": paragraph.max_font_size_pt,
            }
            for paragraph in extracted_paragraphs
        ],
    )


@dataclass(frozen=True)
class ArticleParts:
    headline: str | None
    subheadline: str | None
    body_text: str
    headline_status: str
    subheadline_status: str
    structure_confidence: str
    headline_candidate: str | None


def split_article_paragraphs(
    paragraphs: list[ExtractedParagraph] | list[str],
) -> ArticleParts:
    """Infer article parts without treating long ledes as headlines."""

    if not paragraphs:
        return ArticleParts(None, None, "", "not_present", "not_present", "low", None)

    extracted = [
        paragraph
        if isinstance(paragraph, ExtractedParagraph)
        else ExtractedParagraph(
            text=paragraph,
            style_name="",
            word_count=count_words(paragraph),
            is_bold_only=False,
            max_font_size_pt=None,
        )
        for paragraph in paragraphs
    ]
    first = extracted[0]
    headline_candidate = first.text
    headline_status = _headline_status(first)
    if headline_status in {"explicit", "inferred"}:
        headline = first.text
        body_start = 1
        structure_confidence = "high" if headline_status == "explicit" else "medium"
    else:
        headline = None
        body_start = 0
        structure_confidence = "low" if headline_status == "uncertain" else "medium"

    subheadline = None
    subheadline_status = "not_present"
    if headline is not None and len(extracted) > body_start + 1:
        candidate = extracted[body_start]
        if 1 <= candidate.word_count <= SUBHEADLINE_MAX_WORDS:
            subheadline = candidate.text
            subheadline_status = "inferred"
            body_start += 1

    body_text = "\n".join(paragraph.text for paragraph in extracted[body_start:])
    return ArticleParts(
        headline=headline,
        subheadline=subheadline,
        body_text=body_text,
        headline_status=headline_status,
        subheadline_status=subheadline_status,
        structure_confidence=structure_confidence,
        headline_candidate=headline_candidate,
    )


def count_words(text: str) -> int:
    """Return a whitespace-delimited word count for Tamil and mixed text."""

    return len(text.split())


def _extract_paragraph(paragraph: Any) -> ExtractedParagraph:
    text = paragraph.text.strip()
    runs = [run for run in paragraph.runs if run.text.strip()]
    bold_values = [run.bold for run in runs]
    sizes = [run.font.size.pt for run in runs if run.font.size is not None]
    is_bold_only = bool(bold_values) and all(value is True for value in bold_values)
    style = paragraph.style.name if paragraph.style is not None else ""
    return ExtractedParagraph(
        text=text,
        style_name=style,
        word_count=count_words(text),
        is_bold_only=is_bold_only,
        max_font_size_pt=max(sizes) if sizes else None,
    )


def _headline_status(paragraph: ExtractedParagraph) -> str:
    style = paragraph.style_name.strip().lower()
    if "title" in style or style.startswith("heading"):
        return "explicit"
    if paragraph.word_count >= LONG_LEDE_MIN_WORDS:
        return "not_present"
    if paragraph.is_bold_only and paragraph.word_count <= SUBHEADLINE_MAX_WORDS:
        return "explicit"
    if paragraph.max_font_size_pt is not None and paragraph.max_font_size_pt >= 14:
        return "explicit"
    if paragraph.word_count <= SHORT_HEADLINE_MAX_WORDS and not _is_sentence_like(
        paragraph.text
    ):
        return "inferred"
    return "uncertain"


def _is_sentence_like(text: str) -> bool:
    stripped = text.strip()
    return stripped.endswith(SENTENCE_ENDINGS) or stripped.count(".") >= 1
