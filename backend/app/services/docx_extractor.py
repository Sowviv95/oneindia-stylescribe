"""DOCX text extraction for local author samples and corpus pipelines."""

from dataclasses import dataclass
from pathlib import Path

from docx import Document


@dataclass(frozen=True)
class ExtractedDocx:
    text: str
    char_count: int
    paragraphs: list[str]
    headline: str | None
    subheadline: str | None
    body_text: str
    word_count: int


class DocxExtractionError(RuntimeError):
    """Raised when a DOCX file cannot be read."""


def extract_docx_text(path: Path) -> ExtractedDocx:
    """Extract non-empty paragraph text from a DOCX file."""

    try:
        document = Document(str(path))
    except Exception as exc:
        raise DocxExtractionError(f"Unable to read DOCX file: {path}") from exc

    paragraphs = [
        paragraph.text.strip()
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    ]
    text = "\n".join(paragraphs)
    headline, subheadline, body_text = split_article_paragraphs(paragraphs)
    return ExtractedDocx(
        text=text,
        char_count=len(text),
        paragraphs=paragraphs,
        headline=headline,
        subheadline=subheadline,
        body_text=body_text,
        word_count=count_words(text),
    )


def split_article_paragraphs(
    paragraphs: list[str],
) -> tuple[str | None, str | None, str]:
    """Infer headline, subheadline and body text from paragraph order."""

    if not paragraphs:
        return None, None, ""
    if len(paragraphs) == 1:
        return paragraphs[0], None, ""
    if len(paragraphs) == 2:
        return paragraphs[0], None, paragraphs[1]
    return paragraphs[0], paragraphs[1], "\n".join(paragraphs[2:])


def count_words(text: str) -> int:
    """Return a whitespace-delimited word count for Tamil and mixed text."""

    return len(text.split())
