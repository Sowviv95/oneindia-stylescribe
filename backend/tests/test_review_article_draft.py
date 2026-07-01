from backend.app.db.repository import (
    ArticleDraftRecord,
    AuthorStyleProfileRecord,
    GroundedBriefRecord,
    StyleScribeRepository,
)
from backend.app.models.article_draft_models import ArticleDraftResponse
from backend.app.scripts import review_article_draft


def test_review_article_draft_helper_output(
    capsys: object,
    monkeypatch: object,
) -> None:
    draft = ArticleDraftRecord(
        draft_id="draft-1",
        author_id="v_vasanthi",
        profile_id="profile-1",
        brief_id="brief-1",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        author_instruction=None,
        article_type="public_interest",
        desired_word_count=600,
        tone_override="measured public-interest",
        include_seo=True,
        draft_json=StyleScribeRepository.encode_json(
            {
                "headline": "தலைப்பு",
                "subheadline": "துணை தலைப்பு",
                "article_body": "கட்டுரை உடல்",
                "seo_title": "SEO",
                "meta_description": "Meta",
                "suggested_tags": ["tag"],
            }
        ),
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
    profile = AuthorStyleProfileRecord(
        profile_id="profile-1",
        author_id="v_vasanthi",
        snapshot_id="snapshot-1",
        language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        profile_json=StyleScribeRepository.encode_json(
            {
                "overall_tone": "Measured",
                "headline_style": "Direct",
                "intro_style": "Context first",
                "paragraph_style": "Compact",
                "tamil_register": "Conversational",
                "dos": ["Be factual"],
                "donts": ["Do not invent"],
            }
        ),
        source_excerpt_refs_json="[]",
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )
    brief = GroundedBriefRecord(
        brief_id="brief-1",
        source_type="text",
        source_input_hash="hash",
        source_url=None,
        source_text_excerpt="excerpt",
        source_language="en",
        target_language="ta",
        model_provider="openai",
        model_name="gpt-4o-mini",
        status="completed",
        brief_json=StyleScribeRepository.encode_json(
            {
                "topic": "Flood warning",
                "one_line_summary": "Summary",
                "confirmed_facts": ["Fact"],
                "claims_to_avoid": ["Avoid"],
            }
        ),
        warnings_json="[]",
        created_at="2026-01-01T00:00:00+00:00",
    )

    class FakeStyleScribeRepository:
        @staticmethod
        def decode_json_object(value: str) -> dict[str, object]:
            return StyleScribeRepository.decode_json_object(value)

        @staticmethod
        def decode_json_list(value: str) -> list[str]:
            return StyleScribeRepository.decode_json_list(value)

        def fetch_author_style_profile(
            self,
            profile_id: str,
        ) -> AuthorStyleProfileRecord:
            return profile

        def fetch_grounded_brief(self, brief_id: str) -> GroundedBriefRecord:
            return brief

    monkeypatch.setattr(
        review_article_draft,
        "get_article_draft",
        lambda draft_id, repo: ArticleDraftResponse(
            draft_id=draft.draft_id,
            author_id=draft.author_id,
            profile_id=draft.profile_id,
            brief_id=draft.brief_id,
            target_language=draft.target_language,
            model_provider=draft.model_provider,
            model_name=draft.model_name,
            status=draft.status,
            article_type=draft.article_type,
            desired_word_count=draft.desired_word_count,
            tone_override=draft.tone_override,
            include_seo=draft.include_seo,
            draft=StyleScribeRepository.decode_json_object(draft.draft_json),
            warnings=[],
            created_at=draft.created_at,
        ),
    )
    monkeypatch.setattr(
        review_article_draft,
        "StyleScribeRepository",
        FakeStyleScribeRepository,
    )
    monkeypatch.setattr("sys.argv", ["review_article_draft", "--draft-id", "draft-1"])

    review_article_draft.main()

    output = capsys.readouterr().out
    assert "Source brief highlights" in output
    assert "Style profile highlights" in output
    assert "Generated draft" in output
    assert "தலைப்பு" in output
