import pytest
import requests

from backend.app.services.source_processor import (
    SourceProcessingError,
    process_source,
)


def test_process_text_source() -> None:
    source = process_source(
        "text",
        "This is a useful source text with enough detail for a factual brief.",
    )

    assert source.source_type == "text"
    assert source.source_url is None
    assert "useful source text" in source.cleaned_text
    assert source.source_input_hash
    assert source.warnings == []


def test_process_text_source_applies_pasted_web_text_cleaner() -> None:
    source = process_source(
        "text",
        """
        Advertisement
        Share
        Chennai officials said a pilot will begin next month with 18 sensors.
        Subscribe
        Residents will receive SMS alerts during heavy rain.
        """,
        source_input_mode="pasted_web_text",
    )

    assert "Advertisement" not in source.cleaned_text
    assert "Subscribe" not in source.cleaned_text
    assert "18 sensors" in source.cleaned_text
    assert source.removed_line_count == 3


def test_process_text_source_plain_text_does_not_apply_cleaner() -> None:
    source = process_source(
        "text",
        "Advertisement\nThis useful source text has enough factual detail for a brief.",
    )

    assert "Advertisement" in source.cleaned_text
    assert source.removed_line_count == 0


def test_process_url_source_with_mocked_http(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeResponse:
        text = """
        <html><body><article><h1>Headline</h1><p>This URL contains enough
        readable text for extraction and factual processing.</p></article></body></html>
        """

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(
        "backend.app.services.source_processor.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    source = process_source("url", "https://example.com/news")

    assert source.source_type == "url"
    assert source.source_url == "https://example.com/news"
    assert "Headline" in source.cleaned_text
    assert source.warnings == ["URL extraction produced limited readable text."]


def test_process_url_source_fetch_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_get(*args: object, **kwargs: object) -> object:
        raise requests.RequestException("network down")

    monkeypatch.setattr(
        "backend.app.services.source_processor.requests.get",
        fail_get,
    )

    with pytest.raises(SourceProcessingError):
        process_source("url", "https://example.com/news")
