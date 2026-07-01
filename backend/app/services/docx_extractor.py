"""DOCX text extraction for local author samples."""

from dataclasses import dataclass
from pathlib import Path

from docx import Document


@dataclass(frozen=True)
class ExtractedDocx:
    text: str
    char_count: int


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
    return ExtractedDocx(text=text, char_count=len(text))
