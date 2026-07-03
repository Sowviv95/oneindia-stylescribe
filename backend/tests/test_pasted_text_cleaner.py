from backend.app.services.pasted_text_cleaner import clean_pasted_website_text


def test_pasted_text_cleaner_removes_boilerplate_and_duplicates() -> None:
    text = """
    Advertisement
    Share
    Chennai city officials said a flood-warning pilot will begin next month.
    Chennai city officials said a flood-warning pilot will begin next month.
    Read more
    The civic body said 18 sensors will be installed near low-lying streets.
    Subscribe
    """

    result = clean_pasted_website_text(text)

    assert "Advertisement" not in result.cleaned_text
    assert "Read more" not in result.cleaned_text
    assert result.cleaned_text.count("flood-warning pilot") == 1
    assert "18 sensors" in result.cleaned_text
    assert result.removed_line_count == 5


def test_pasted_text_cleaner_preserves_article_paragraphs() -> None:
    text = """
    Newsletter

    Chennai city officials said on Tuesday that a new flood-warning pilot will
    begin in three neighborhoods next month.

    Officials said residents will receive SMS alerts during heavy rain.
    """

    result = clean_pasted_website_text(text)

    assert "new flood-warning pilot" in result.cleaned_text
    assert "SMS alerts" in result.cleaned_text
    assert "\n\n" in result.cleaned_text
