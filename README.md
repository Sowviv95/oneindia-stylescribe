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
model registry.

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

## Sample Data

Sample author articles may exist locally under `sample_data`. The
`sample_data/raw/` and `sample_data/extracted/` directories are local-only and
ignored by Git. Do not commit raw article files, extracted sample text, `.env`,
or API keys.
