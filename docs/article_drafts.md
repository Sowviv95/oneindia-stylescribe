# Article Drafts

Sprint 6 adds controlled Tamil article draft generation using OpenAI only.

## Purpose

An article draft combines:

- the latest author style profile for writing style only
- a saved grounded brief for facts only
- an optional author/editor instruction

The output is a Tamil draft with headline, subheadline, body, SEO metadata,
tags, fact usage notes, and style usage notes.

## Style And Facts Stay Separate

The style profile is used only for tone, register, headline tendencies,
paragraph rhythm, and reusable writing guidance. The grounded brief is the only
factual source. The model is instructed not to use author sample facts, style
excerpt facts, or outside knowledge.

## Generate A Draft

```powershell
curl -X POST http://127.0.0.1:8000/drafts/article `
  -H "Content-Type: application/json" `
  -d '{
    "author_id": "v_vasanthi",
    "brief_id": "<saved_brief_id>",
    "author_instruction": "Write this as a 500-word Tamil news article in the author's style.",
    "target_language": "ta"
  }'
```

Fetch a saved draft:

```powershell
curl http://127.0.0.1:8000/drafts/<draft_id>
```

Review a draft:

```powershell
python -m backend.app.scripts.review_article_draft --draft-id <draft_id>
```

## Current Limitations

- OpenAI is the only model provider used.
- The MVP is Tamil-focused.
- QC scoring, Qwen/Gemma comparison, and multi-model selection are planned for a
  later sprint.
- No UI is included.
