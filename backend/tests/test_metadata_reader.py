from pathlib import Path

from openpyxl import Workbook

from backend.app.services.metadata_reader import read_metadata_rows


def test_read_metadata_rows_normalizes_expected_columns(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.xlsx"
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.append(["URL", "Added_date", "Meta Description", "Content"])
    worksheet.append(["https://example.com/a", "2026-01-01", "Meta text", "Body"])
    workbook.save(metadata_path)

    rows = read_metadata_rows(metadata_path)

    assert rows == [
        {
            "url": "https://example.com/a",
            "added_date": "2026-01-01",
            "meta_description": "Meta text",
            "content": "Body",
        }
    ]
