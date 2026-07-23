# Oneindia Tamil Newsroom Corpus Extraction

Sprint 1 builds the repeatable inventory and DOCX extraction foundation for the
generic newsroom baseline corpus.

## Raw Corpus Rule

Treat this folder as immutable:

```text
data/newsroom_corpus/00_raw/
```

Do not rename, move, edit or delete source DOCX files. Generated outputs are
written only to later corpus folders and reports.

## Run Inventory Only

```powershell
python scripts/run_newsroom_corpus_extraction.py --inventory-only
```

This writes inventory and duplicate-file reports without opening DOCX content.

## Run Full Extraction

```powershell
python scripts/run_newsroom_corpus_extraction.py
```

Generated outputs:

- `data/newsroom_corpus/01_extracted/articles.jsonl`
- `data/newsroom_corpus/03_rejected/extraction_rejections.jsonl`
- `data/newsroom_corpus/reports/raw_corpus_inventory.csv`
- `data/newsroom_corpus/reports/exact_duplicate_files.csv`
- `data/newsroom_corpus/reports/exact_duplicate_text.csv`
- `data/newsroom_corpus/reports/extraction_summary.md`
- `data/newsroom_corpus/reports/author_counts.csv`

## Extracted Record Shape

Each article JSONL record preserves:

- `article_id`
- `author_id`
- `source_filename`
- `relative_source_path`
- `file_size_bytes`
- `file_sha256`
- `text_sha256`
- `extraction_status`
- `extraction_warnings`
- `headline`
- `subheadline`
- `body_text`
- `paragraph_sequence`
- `total_word_count`
- `char_count`

## DOCX Structure Assumptions

The extractor reads non-empty Word paragraphs in document order. It treats the
first paragraph as headline, the second paragraph as subheadline only when at
least three non-empty paragraphs exist, and the remaining paragraphs as body
text.

This foundation does not perform near-duplicate detection, semantic cleaning,
topic classification, benchmark selection, Gemini generation, or author-style
modelling.
