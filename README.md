# StyleScribe

StyleScribe is an author-style regional article generation module for WISE+.
It will generate Tamil articles from multilingual news URLs or source text while
retaining the selected author's writing style.

This repository is currently backend-first. Sprint 1 provides a FastAPI API
skeleton, configuration loading, request/response contracts, model provider
placeholders, and a stub generation endpoint. It does not perform real LLM
calls, process author samples, or include a full UI.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r backend/requirements.txt
```

## Environment

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=
OPENAI_MODEL_DEFAULT=gpt-4o-mini
OPENAI_MODEL_PLANNING=
OPENAI_MODEL_GENERATION=gpt-4o
OPENAI_MODEL_REVISION=
OPENAI_MODEL_EVALUATION=
OPENAI_MODEL_LENGTH_RECOVERY=
OPENAI_COST_INPUT_PER_1M_GPT_4O=
OPENAI_COST_OUTPUT_PER_1M_GPT_4O=
OPENAI_COST_INPUT_PER_1M_GPT_4O_MINI=
OPENAI_COST_OUTPUT_PER_1M_GPT_4O_MINI=
QWEN_PROVIDER=ollama
QWEN_BASE_URL=http://localhost:11434
QWEN_MODEL=
GEMMA_PROVIDER=ollama
GEMMA_BASE_URL=http://localhost:11434
GEMMA_MODEL=
DEFAULT_TARGET_LANGUAGE=ta
DEFAULT_SOURCE_LANGUAGE=
```

The API loads these values with `python-dotenv`. Missing optional model values
do not prevent the app from starting, and secret values are not exposed by the
model registry. Stage-specific OpenAI model values override `OPENAI_MODEL`;
otherwise the workflow falls back to `OPENAI_MODEL`, then `OPENAI_MODEL_DEFAULT`,
then `gpt-4o-mini`. Current validation uses `gpt-4o` for article/section
generation and cheaper defaults for planning, revision, evaluation, and length
recovery unless configured otherwise.

Cost telemetry uses actual API token usage when available. Pricing can be
overridden with the `OPENAI_COST_*_PER_1M_*` variables above; built-in fallback
prices are estimates and should be reviewed periodically.

## Run The API

```powershell
python -m uvicorn backend.app.main:app --reload
```

Health check:

```powershell
curl http://127.0.0.1:8000/health
```

## Run Tests And Checks

```powershell
python -m pytest
python -m ruff check .
python -m mypy backend/app
```

## Local Author Ingestion

Sprint 2 can ingest local DOCX author samples into SQLite. The default database
path is `data/stylescribe.db`; override it with `STYLESCRIBE_DB_PATH` if needed.

Run the API, then call:

```powershell
curl -X POST http://127.0.0.1:8000/authors/ingest-local `
  -H "Content-Type: application/json" `
  -d '{
    "author_id": "v_vasanthi",
    "display_name": "V Vasanthi",
    "language": "ta",
    "articles_dir": "sample_data/extracted/authors/v_vasanthi/articles_docx",
    "metadata_path": "sample_data/raw/authors/v_vasanthi/metadata/Author Context _ Author wise content.xlsx"
  }'
```

List lightweight article records:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/articles
```

See [docs/ingestion.md](docs/ingestion.md) for details and metadata matching
limitations.

## Deterministic Style Snapshots

After an author's articles are ingested, build an LLM-ready deterministic style
snapshot:

```powershell
curl -X POST http://127.0.0.1:8000/authors/v_vasanthi/style-snapshot
```

Fetch the latest stored snapshot:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/style-snapshot/latest
```

The snapshot contains measurable style statistics and bounded excerpts only. It
does not call any LLM or generate a final author style profile. See
[docs/style_snapshots.md](docs/style_snapshots.md).

## LLM Author Style Profiles

Sprint 4 generates a structured author style profile from the latest
deterministic snapshot using OpenAI only. Set `OPENAI_API_KEY` in `.env`.
`OPENAI_MODEL` is optional and defaults to `gpt-4o-mini`.

Generate and save a profile:

```powershell
curl -X POST http://127.0.0.1:8000/authors/v_vasanthi/style-profile
```

Fetch the latest profile:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/style-profile/latest
```

Review bounded excerpts and generated profile sections:

```powershell
python -m backend.app.scripts.review_style_profile --author-id v_vasanthi --limit 2
```

See [docs/author_style_profiles.md](docs/author_style_profiles.md).

## Grounded Briefs

Sprint 5 generates a structured factual brief from source text or a URL using
OpenAI only. The grounded brief is separate from author style and will be used
later as the factual source of truth for article generation.

```powershell
curl -X POST http://127.0.0.1:8000/briefs/grounded `
  -H "Content-Type: application/json" `
  -d '{
    "source_type": "text",
    "source_input": "Short news source text here",
    "target_language": "ta"
  }'
```

Fetch or review a saved brief:

```powershell
curl http://127.0.0.1:8000/briefs/<brief_id>
python -m backend.app.scripts.review_grounded_brief --brief-id <brief_id>
```

See [docs/grounded_briefs.md](docs/grounded_briefs.md).

## Article Drafts

Sprint 6 generates the first controlled Tamil article draft using OpenAI, the
latest author style profile, and a saved grounded brief.

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

Review a saved draft:

```powershell
python -m backend.app.scripts.review_article_draft --draft-id <draft_id>
python -m backend.app.scripts.review_article_draft --draft-id <draft_id> --format html --output review_outputs/draft_review.html
```

Use Markdown/HTML export when Windows terminal rendering makes Tamil hard to
read. See [docs/article_drafts.md](docs/article_drafts.md).

## Draft Grounding Evaluation

Sprint 8 evaluates generated drafts against the grounded brief to flag
unsupported claims, overclaims, invented facts, and claims-to-avoid violations.

```powershell
curl -X POST http://127.0.0.1:8000/drafts/<draft_id>/evaluate-grounding
curl http://127.0.0.1:8000/drafts/<draft_id>/evaluation/latest
python -m backend.app.scripts.review_draft_evaluation --draft-id <draft_id> --format html --output review_outputs/evaluation.html
```

Recommended workflow:

1. Generate grounded brief.
2. Generate article draft.
3. Evaluate grounding.
4. Revise automatically when evaluator feedback flags unsupported claims.

See [docs/draft_grounding_evaluations.md](docs/draft_grounding_evaluations.md).

## Grounded Auto Revision

Sprint 10 adds an OpenAI-only revision loop. It uses the saved grounding
evaluation to revise the headline, subheadline, body, SEO title, meta
description, and tags while preserving the grounded brief as the only factual
source.

```powershell
curl -X POST http://127.0.0.1:8000/drafts/<draft_id>/revise-grounding `
  -H "Content-Type: application/json" `
  -d '{
    "run_final_evaluation": true,
    "export_review": true,
    "export_format": "html"
  }'
curl http://127.0.0.1:8000/drafts/<draft_id>/revision/latest
```

## Pasted Website Text Workflow

Sprint 9 adds an OpenAI-only end-to-end workflow for copied website article
text. It cleans common website boilerplate, generates a grounded brief,
generates a Tamil author-style draft, runs grounding evaluation by default, and
can export a UTF-8 review file.

```powershell
curl -X POST http://127.0.0.1:8000/workflows/pasted-text-to-draft `
  -H "Content-Type: application/json" `
  -d '{
    "author_id": "v_vasanthi",
    "source_text": "Advertisement\nShare\nChennai city officials said...\nRead more\nSubscribe",
    "author_instruction": "Write this as a Tamil news article for Oneindia readers.",
    "target_language": "ta",
    "article_type": "news",
    "desired_word_count": 600,
    "tone_override": "clear, engaging and factual",
    "run_grounding_evaluation": true,
    "run_auto_revision": true,
    "run_final_evaluation": true,
    "export_review": true,
    "export_format": "html"
  }'
```

For direct grounded brief generation from pasted website text, use
`source_input_mode: "pasted_web_text"` with `POST /briefs/grounded`.

See [docs/pasted_text_workflow.md](docs/pasted_text_workflow.md). Qwen/Gemma
comparison is planned after this workflow is stable.

Manual Sprint 10 request file:

```powershell
python -c "import json, pathlib; source=pathlib.Path('manual_test_input.txt').read_text(encoding='utf-8'); payload={'author_id':'v_vasanthi','source_text':source,'author_instruction':'Write this as a Tamil news article for Oneindia readers.','target_language':'ta','article_type':'news','desired_word_count':600,'tone_override':'clear, engaging and factual','run_grounding_evaluation':True,'run_auto_revision':True,'run_final_evaluation':True,'export_review':True,'export_format':'html'}; pathlib.Path('manual_request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')"
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/workflows/pasted-text-to-draft" -ContentType "application/json; charset=utf-8" -InFile ".\manual_request.json"
start $response.export_paths[0]
```

## Sample Data

Sample author articles may exist locally under `sample_data`. The
`sample_data/raw/` and `sample_data/extracted/` directories are local-only and
ignored by Git. Do not commit raw article files, extracted sample text, `.env`,
or API keys.
