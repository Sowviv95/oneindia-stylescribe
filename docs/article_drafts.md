# Article Drafts

Sprint 6 adds controlled Tamil article draft generation using OpenAI only.

## Purpose

An article draft combines:

- the latest author style profile for writing style only
- a saved grounded brief for facts only
- an optional author/editor instruction

The output is a Tamil draft with headline, subheadline, body, SEO metadata,
tags, fact usage notes, and style usage notes.

Sprint 7 adds generation controls:

- `article_type`: `news`, `analysis`, `explainer`, `public_interest`,
  `entertainment`, or another editor-defined label
- `desired_word_count`: optional target, validated between 250 and 1200 and
  defaulting to about 600
- `tone_override`: optional editor tone guidance such as `measured
  public-interest`
- `include_seo`: defaults to `true`

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
    "target_language": "ta",
    "article_type": "public_interest",
    "desired_word_count": 600,
    "tone_override": "measured public-interest",
    "include_seo": true
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

For Tamil-readable review on Windows, export UTF-8 Markdown or HTML:

```powershell
python -m backend.app.scripts.review_article_draft --draft-id <draft_id> --format markdown --output review_outputs/draft_review.md
python -m backend.app.scripts.review_article_draft --draft-id <draft_id> --format html --output review_outputs/draft_review.html
```

Windows terminal/font rendering can make Tamil look muddled even when Unicode is
stored correctly. Prefer the HTML export for review; it includes UTF-8 metadata
and a Tamil-friendly font stack.

## Current Limitations

- The main application workflow still uses OpenAI for non-generation stages.
- The standalone benchmark runner can use OpenAI, Gemini, or Grok for article
  generation only.
- The MVP is Tamil-focused.
- Multi-model benchmark comparison is saved-output based and does not add a new
  subjective LLM judge.
- Automated multi-model selection is still planned for a later sprint.
- No UI is included.

## Benchmark 10 Trace

Date: 2026-07-17

The generation benchmark uses one provider/model per run and reuses the same
source input, grounded brief, article plan, author profile, prompts, desired
word count, tone, article type, and workflow controls for each input.

Supported benchmark generation models:

- OpenAI: `gpt-5.5`
- Gemini: `gemini-3.5-flash`
- Grok: `grok-4.20-0309-non-reasoning`

Benchmark root:

- `comparison/benchmark_10`

Current three-model comparison HTML:

- `comparison/benchmark_10/comparisons/index.html`
