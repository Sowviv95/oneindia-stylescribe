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

## POST /drafts/{draft_id}/revise-grounding

Creates a grounded revision from the selected or latest grounding evaluation.

Request body:

```json
{
  "evaluation_id": "optional-evaluation-id",
  "run_final_evaluation": true,
  "export_review": true,
  "export_format": "html"
}
```

Response includes the stored revision, initial evaluation summary payload,
optional final evaluation payload, and optional review export paths.

## GET /drafts/{draft_id}/revision/latest

Returns the latest stored grounded revision for a draft.

## POST /workflows/pasted-text-to-draft

Sprint 10 request fields:

```json
{
  "run_auto_revision": true,
  "run_final_evaluation": true
}
```

When `run_auto_revision` is true, the workflow returns `initial_evaluation_id`,
`revision_id`, optional `final_evaluation_id`, `initial_readiness`,
`final_readiness`, and `export_paths`.

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
