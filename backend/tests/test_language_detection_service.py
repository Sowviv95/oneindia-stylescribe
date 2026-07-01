from backend.app.services.language_detection_service import detect_language


def test_detect_tamil_language() -> None:
    assert detect_language("தமிழ் செய்தி") == "ta"


def test_detect_english_language() -> None:
    assert detect_language("This is an English source story.") == "en"


def test_detect_unknown_language() -> None:
    assert detect_language("12345 !!!") == "unknown"
