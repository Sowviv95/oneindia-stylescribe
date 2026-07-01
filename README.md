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

## Sample Data

Sample author articles may exist locally under `sample_data`. The
`sample_data/raw/` and `sample_data/extracted/` directories are local-only and
ignored by Git. Do not commit raw article files, extracted sample text, `.env`,
or API keys.
