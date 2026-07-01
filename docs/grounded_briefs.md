# Grounded Briefs

Sprint 5 adds a source-input to grounded-brief pipeline. A grounded brief is a
structured factual source of truth for later Tamil article generation.

## Purpose

The brief extracts facts from a source news text or URL and stores them in
SQLite. It keeps factual grounding separate from author style. Later article
generation should use the brief for facts and the author style profile for
voice.

## Cross-Language Support

The source can be in any language. Lightweight language detection marks Tamil
when Tamil Unicode characters are present, English when the text is mostly
Latin, and `unknown` otherwise. The default target language is Tamil (`ta`).

## Text And URL Input

Text input is cleaned and validated directly.

URL input is fetched with `requests` and extracted with BeautifulSoup. This MVP
handles ordinary server-rendered HTML. It does not attempt paywall bypass,
browser rendering, or dynamic JavaScript extraction.

## OpenAI Input Bounds

Only bounded source text is sent to OpenAI. Very long inputs are truncated and a
warning is saved. The API response includes only a short source excerpt, not the
full source text.

## Generate A Brief

```powershell
curl -X POST http://127.0.0.1:8000/briefs/grounded `
  -H "Content-Type: application/json" `
  -d '{
    "source_type": "text",
    "source_input": "Short news source text here",
    "target_language": "ta"
  }'
```

Fetch a saved brief:

```powershell
curl http://127.0.0.1:8000/briefs/<brief_id>
```

Review a saved brief:

```powershell
python -m backend.app.scripts.review_grounded_brief --brief-id <brief_id>
```

## Limitations

- OpenAI is the only model provider used in this sprint.
- Qwen, Gemma, final article generation, and news URL article generation are
  out of scope.
- URL extraction is intentionally simple and may be weak on dynamic pages.
- The brief should not include facts not present in the supplied source.
