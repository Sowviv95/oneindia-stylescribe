# Deterministic Style Snapshots

Sprint 3 creates an LLM-ready input pack without calling any LLM. It analyzes
already-ingested SQLite article text and stores a deterministic style snapshot
for each author.

## Purpose

A style snapshot captures measurable writing-style signals and bounded excerpts
from an author's ingested articles. The next sprint can use this snapshot as
grounded input for LLM-based author style profile generation.

## Statistics Calculated

The snapshot includes deterministic metrics for:

- article counts and character counts
- paragraph counts, paragraph length, short paragraph ratio, long paragraph ratio
- approximate sentence counts and sentence length
- title and heading availability and length
- category distribution
- intro and closing paragraph length
- punctuation signals such as questions, exclamations, colons, semicolons,
  quotes, and ellipses
- approximate Tamil and Latin character ratios

Tamil sentence splitting is intentionally approximate. It uses punctuation such
as `.`, `?`, `!`, `।`, and `॥` rather than full NLP parsing.

## Excerpt Pack

The excerpt pack contains bounded examples, never full article dumps:

- `headline_examples`
- `intro_examples`
- `body_examples`
- `closing_examples`
- `short_article_examples`
- `long_article_examples`

Each excerpt includes the article ID, filename, title or heading, category,
excerpt text, excerpt type, and character count.

## No LLM Usage

No OpenAI, Qwen, Gemma, Ollama, or other model calls are made in this sprint.
The output is deterministic and comes only from ingested SQLite records.

## API Usage

Build and store a snapshot:

```powershell
curl -X POST http://127.0.0.1:8000/authors/v_vasanthi/style-snapshot
```

Fetch the latest snapshot:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/style-snapshot/latest
```

The author must already have ingested articles in SQLite.
