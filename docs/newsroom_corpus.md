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

## Run Sprint 2 Newsroom Profile

After Sprint 1 preparation exists, build the deterministic generic newsroom
profile:

```powershell
python scripts/run_newsroom_corpus_extraction.py --mode newsroom-profile
```

Generated Sprint 2 outputs:

- `data/newsroom_corpus/reports/oneindia_tamil_newsroom_style_guide.md`
- `data/newsroom_corpus/reports/oneindia_tamil_newsroom_profile.json`
- `data/newsroom_corpus/reports/preferred_phrase_bank.csv`
- `data/newsroom_corpus/reports/phrase_review_list.csv`
- `data/newsroom_corpus/reports/structural_pattern_report.csv`
- `data/newsroom_corpus/reports/author_commonality_report.csv`
- `data/newsroom_corpus/reports/newsroom_profile_evidence.jsonl`

The profile uses accepted articles only. Headline candidates are treated as
opening lede evidence, not as confirmed editorial headlines. Topic labels are
weak metadata and are not used to force style conclusions.

## Sprint 3 Retrieval Baseline Candidate

Sprint 3 added an explicit retrieval mode:

```text
newsroom_v1_retrieval
```

This mode remains opt-in. It does not change the default generation mode and it
does not alter legacy, newsroom v1.0, or newsroom v1.1 length-calibrated runs.

The preferred retrieval candidate uses:

- prompt-only baseline: `newsroom_v1` with `oneindia_newsroom_v1.0`
- retrieval prompt: `oneindia_newsroom_v1.0_retrieval_v1`
- embedding provider/model: `sentence_transformers` /
  `intfloat/multilingual-e5-small`
- article-level retrieval from accepted corpus records only
- top-k: 3
- source, exact duplicate and near-duplicate exclusion
- author diversity safeguards
- optional high-confidence, non-conflict topic soft boost
- no hard topic filtering

The earlier lexical hashing embedder remains preserved as an experimental
baseline, but E5 replaced it for controlled retrieval benchmarking because it
improved semantic relevance on the bounded comparison set. Hashing retrieval
must not be used as an automatic fallback.

The 10-input Sprint 3 benchmark found:

- editorial improvement on 7 of 10 inputs
- average grounding improved from 87.5 to 88.5
- unsupported claims reduced from 2 to 1
- overclaim warnings remained 2
- retrieval leakage findings: 0
- claims-to-avoid violations: 0

These results make E5 retrieval a generic newsroom baseline candidate, not a
fully production-approved default. Editorial review is still required.

## Sprint 4 Impact-Framing Watchpoints

Sprint 4 preserves the original retrieval prompt and adds a separate guarded
retrieval prompt:

```text
oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard
```

The guard addresses a narrow failure pattern observed in transport and visa
review outputs: source-stated aims, expectations, trials, reviews or future
plans can be framed too strongly as current implementation or confirmed benefit.

The new prompt version instructs the model to:

- attribute expected benefits, aims and official expectations to the relevant
  source
- preserve uncertainty and future tense for proposals, reviews, trials,
  consultations, inspections and plans
- avoid presenting a system, rule, service or outcome as operational unless the
  brief confirms it
- avoid generic benefit claims such as traffic reduction, easier procedures or
  improved service quality unless the brief explicitly supports them

The factual-isolation rules are unchanged: retrieved examples are structural
references only and are never factual evidence for the current story.

## Proposed Operational Defaults

These defaults are proposed for controlled promotion only; they are not enforced
as the application default yet.

- generation mode: `newsroom_v1_retrieval`
- retrieval prompt version:
  `oneindia_newsroom_v1.0_retrieval_v1_1_impact_guard` if regression validation
  confirms no quality regression
- fallback prompt-only mode: `newsroom_v1` with `oneindia_newsroom_v1.0`
- embedding provider/model: `sentence_transformers` /
  `intfloat/multilingual-e5-small`
- retrieval index:
  `data/newsroom_corpus/03_retrieval/sentence_transformers_multilingual_e5_small/newsroom_retrieval_index.json`
- retrieval records:
  `data/newsroom_corpus/03_retrieval/sentence_transformers_multilingual_e5_small/retrieval_records.jsonl`
- top-k: 3
- candidate pool size: 12
- maximum examples per author: 1
- retrieval context cap: 9000 characters
- topic boost: optional soft boost only for high-confidence, non-conflict
  matching topics
- leakage diagnostic: enabled for retrieval runs
- fallback behavior: if the E5 index or model cannot be loaded, fall back to
  prompt-only newsroom v1.0, record the fallback reason, and never silently use
  hashing retrieval

Topic labels remain provisional and noisy. Topic-wise reports are descriptive
only and must not be treated as statistically robust until topic classification
is improved.
