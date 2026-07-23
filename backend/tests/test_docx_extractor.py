from pathlib import Path

from docx import Document

from backend.app.services.docx_extractor import extract_docx_text


def test_extract_docx_text_strips_empty_paragraphs(tmp_path: Path) -> None:
    docx_path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("First paragraph")
    document.add_paragraph("   ")
    document.add_paragraph("Second paragraph")
    document.save(docx_path)

    extracted = extract_docx_text(docx_path)

    assert extracted.text == "First paragraph\nSecond paragraph"
    assert extracted.char_count == len(extracted.text)
    assert extracted.paragraphs == ["First paragraph", "Second paragraph"]
    assert extracted.headline == "First paragraph"
    assert extracted.subheadline is None
    assert extracted.body_text == "Second paragraph"


def test_extract_docx_text_infers_headline_subheadline_and_body(
    tmp_path: Path,
) -> None:
    docx_path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Headline")
    document.add_paragraph("Subheadline")
    document.add_paragraph("Body paragraph one")
    document.add_paragraph("Body paragraph two")
    document.save(docx_path)

    extracted = extract_docx_text(docx_path)

    assert extracted.headline == "Headline"
    assert extracted.subheadline == "Subheadline"
    assert extracted.body_text == "Body paragraph one\nBody paragraph two"
    assert extracted.word_count == 8
