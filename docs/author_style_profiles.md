# Author Style Profiles

Sprint 4 generates an LLM-based author style profile from the latest
deterministic style snapshot. It uses OpenAI only and stores the structured
profile in SQLite.

## Purpose

The profile converts deterministic statistics and curated excerpts into
practical writing guidance for future Tamil article generation. It captures
writing style, not author opinions, personal details, or permanent facts from
the sample articles.

## Input Source

The service uses the latest `author_style_snapshots` row for the author:

- `stats_json`
- bounded excerpt examples from `excerpt_pack_json`

It does not send the full article corpus, raw sample files, or extracted article
database text beyond the curated excerpts already saved in the snapshot.

## Generate A Profile

Set `OPENAI_API_KEY` in `.env`. `OPENAI_MODEL` is optional and defaults to
`gpt-4o-mini`.

```powershell
curl -X POST http://127.0.0.1:8000/authors/v_vasanthi/style-profile
```

Fetch the latest profile:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/style-profile/latest
```

If no deterministic snapshot exists yet, run:

```powershell
curl -X POST http://127.0.0.1:8000/authors/v_vasanthi/style-snapshot
```

## Review Helper

Print a bounded human-review view:

```powershell
python -m backend.app.scripts.review_style_profile --author-id v_vasanthi --limit 2
```

The helper shows:

- profile ID, snapshot ID, model, and warnings
- 1-2 bounded source excerpts used for generation
- filename, title or heading, category, and excerpt type
- generated style profile sections

## Limitations

- Only OpenAI is used in this sprint.
- Qwen, Gemma, news URL processing, and article generation are out of scope.
- The profile quality depends on the deterministic snapshot and excerpt pack.
- The profile should be treated as reusable style guidance, not as a factual
  summary of the source articles.
