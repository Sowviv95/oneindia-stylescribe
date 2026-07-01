# Local Author Sample Ingestion

Sprint 2 adds deterministic ingestion for local author samples. It reads DOCX
article files, optionally attaches Excel metadata, and stores records in local
SQLite.

## Local Folder Structure

Real samples are expected locally under:

```text
sample_data/
  extracted/
    authors/
      v_vasanthi/
        articles_docx/
  raw/
    authors/
      v_vasanthi/
        metadata/
```

The raw and extracted sample directories are ignored by Git because they may
contain copyrighted article text, large local exports, and source metadata that
should not be committed.

## Database

The default SQLite path is:

```text
data/stylescribe.db
```

Override it with:

```powershell
$env:STYLESCRIBE_DB_PATH="D:\path\to\stylescribe.db"
```

The `data/` directory is ignored by Git.

## Run Ingestion For `v_vasanthi`

Start the API:

```powershell
python -m uvicorn backend.app.main:app --reload
```

Call the local ingestion endpoint:

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

List ingested articles without full text:

```powershell
curl http://127.0.0.1:8000/authors/v_vasanthi/articles
```

## Metadata Matching Limitations

Metadata matching is intentionally simple for this sprint:

- filename matching is preferred when a metadata filename field exists
- title/heading similarity is attempted as a best-effort fallback
- sequential fallback is used only when DOCX and metadata row counts match
- unmatched files are still ingested with null metadata fields

No LLM calls, author style profile generation, or sample text transformation is
performed during ingestion.
