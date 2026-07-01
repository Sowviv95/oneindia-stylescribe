# StyleScribe API Contract

Base service name: `stylescribe-api`

## GET /health

Returns API health.

Response:

```json
{
  "status": "ok",
  "service": "stylescribe-api"
}
```

## POST /generate/article

Creates a stub article generation request. This endpoint validates the request
shape but does not run the generation pipeline in Sprint 1.

Request body:

```json
{
  "author_id": "v_vasanthi",
  "target_language": "ta",
  "source_type": "text",
  "source_input": "news text or URL",
  "author_instruction": "write as a news article",
  "category": "Politics",
  "models": ["openai", "qwen", "gemma"]
}
```

Validation:

- `author_id` is required.
- `source_input` is required.
- `target_language` defaults to `ta`.
- `models` defaults to `["openai"]`.
- `source_type` must be `url` or `text`.
- `models` may contain `openai`, `qwen`, or `gemma`.

Response:

```json
{
  "request_id": "generated-uuid",
  "status": "stub",
  "message": "Article generation pipeline is not implemented yet.",
  "selected_models": ["openai", "qwen", "gemma"],
  "target_language": "ta",
  "pipeline_steps": [
    "source_processing",
    "grounded_brief_generation",
    "author_style_retrieval",
    "multi_model_generation",
    "qc_evaluation"
  ]
}
```
