from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_generate_article_accepts_valid_payload() -> None:
    payload = {
        "author_id": "v_vasanthi",
        "target_language": "ta",
        "source_type": "text",
        "source_input": "news text or URL",
        "author_instruction": "write as a news article",
        "category": "Politics",
        "models": ["openai", "qwen", "gemma"],
    }

    response = client.post("/generate/article", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "stub"
    assert body["message"] == "Article generation pipeline is not implemented yet."
    assert body["selected_models"] == ["openai", "qwen", "gemma"]
    assert body["target_language"] == "ta"
    assert body["pipeline_steps"] == [
        "source_processing",
        "grounded_brief_generation",
        "author_style_retrieval",
        "multi_model_generation",
        "qc_evaluation",
    ]
    assert body["request_id"]


def test_generate_article_rejects_invalid_source_type() -> None:
    payload = {
        "author_id": "v_vasanthi",
        "source_type": "pdf",
        "source_input": "news text",
    }

    response = client.post("/generate/article", json=payload)

    assert response.status_code == 422


def test_generate_article_rejects_missing_source_input() -> None:
    payload = {
        "author_id": "v_vasanthi",
        "source_type": "text",
    }

    response = client.post("/generate/article", json=payload)

    assert response.status_code == 422
