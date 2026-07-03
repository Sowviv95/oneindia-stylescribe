from backend.app.services.tamil_quality_scanner import scan_tamil_quality


def test_scanner_flags_corrupted_mixed_token() -> None:
    result = scan_tamil_quality(
        {"article_body": "அவர்கள் pertencிக்கிறார்கள்."},
        desired_word_count=None,
    )

    assert result.tamil_quality_status == "fail"
    assert any("pertenc" in warning for warning in result.tamil_quality_warnings)


def test_scanner_flags_ruling_leftover() -> None:
    result = scan_tamil_quality(
        {"headline": "இந்த ruling குறித்து புதிய தகவல்"},
        desired_word_count=None,
    )

    assert result.tamil_quality_status == "fail"
    assert any("ruling" in warning for warning in result.tamil_quality_warnings)


def test_scanner_flags_risky_contextual_translation() -> None:
    result = scan_tamil_quality(
        {"article_body": "இந்திய குடியுரிமை பெற்றவர்கள் குறித்து செய்தி."},
        desired_word_count=None,
    )

    assert result.tamil_quality_status == "warning"
    assert any(
        "இந்திய குடியுரிமை பெற்றவர்கள்" in warning
        for warning in result.tamil_quality_warnings
    )


def test_scanner_flags_materially_short_article() -> None:
    result = scan_tamil_quality(
        {"article_body": "சிறிய செய்தி மட்டும்."},
        desired_word_count=600,
    )

    assert result.length_status == "warning"
    assert result.tamil_quality_status == "warning"
    assert result.final_article_word_count < 450
    assert result.length_warning_reason is not None
    assert "materially shorter" in result.length_warning_reason
    assert result.final_article_word_count_ratio is not None
    assert result.final_article_word_count_ratio < 0.75


def test_scanner_allows_selected_english_terms() -> None:
    result = scan_tamil_quality(
        {"article_body": "H-1B visa, SMS, US, FIIDS, green card குறித்து செய்தி."},
        desired_word_count=None,
    )

    assert result.tamil_quality_status == "warning"
    assert not any(
        "Unexpected Latin-script" in warning
        for warning in result.tamil_quality_warnings
    )
    assert any(
        "Allowed English term" in warning
        for warning in result.tamil_quality_warnings
    )
