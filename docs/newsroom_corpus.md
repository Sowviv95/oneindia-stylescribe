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
python scripts/run_newsroom_corpus_extraction.py --mode inventory
```

This writes inventory and duplicate-file reports without opening DOCX content.

## Run Full Extraction

```powershell
python scripts/run_newsroom_corpus_extraction.py --mode extract
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

The extraction step does not perform near-duplicate detection, semantic
cleaning, topic classification, benchmark selection, Gemini generation, or
author-style modelling.

## Run Sprint 1 Preparation

After `01_extracted/articles.jsonl` exists, run deterministic preparation:

```powershell
python scripts/run_newsroom_corpus_extraction.py --mode prepare
```

The same preparation pipeline is used for these modes:

- `profile`
- `duplicates`
- `clean`
- `classify`
- `prepare`

Run extraction and preparation together:

```powershell
python scripts/run_newsroom_corpus_extraction.py --mode full-run
```

Preparation outputs:

- `data/newsroom_corpus/02_cleaned/cleaned_articles.jsonl`
- `data/newsroom_corpus/02_cleaned/review_required_articles.jsonl`
- `data/newsroom_corpus/03_rejected/rejected_articles.jsonl`
- `data/newsroom_corpus/04_classified/classified_articles.jsonl`
- `data/newsroom_corpus/reports/near_duplicate_clusters.jsonl`
- `data/newsroom_corpus/reports/cleaning_decisions.csv`
- `data/newsroom_corpus/reports/structural_anomalies.csv`
- `data/newsroom_corpus/reports/topic_distribution.csv`
- `data/newsroom_corpus/reports/author_topic_distribution.csv`
- `data/newsroom_corpus/reports/length_distribution.csv`
- `data/newsroom_corpus/reports/sprint1_corpus_report.md`

## Preparation Rules

Normalisation is for comparison only. It applies Unicode NFKC, zero-width
character removal, line-ending unification, whitespace collapse and Latin case
folding. Original extracted text is preserved unchanged in article records.

Rejection is intentionally narrow and deterministic:

- empty or missing usable body text
- non-canonical member of an explainable near-duplicate cluster

Review-required records include suspicious but potentially usable structure,
such as long inferred headline/subheadline fields, likely boilerplate, repeated
headline/body text, large single-paragraph files or low-confidence topic
classification.

First-pass topic classification uses deterministic Tamil/English keyword
matching only. It is not semantic classification and should be treated as a
triage aid for later corpus review.
