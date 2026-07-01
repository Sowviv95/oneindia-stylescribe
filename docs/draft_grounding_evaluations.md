# Draft Grounding Evaluations

Sprint 8 adds post-generation grounding and unsupported-claim checks for Tamil
article drafts.

## Purpose

The evaluator compares a generated draft against its grounded brief. The
grounded brief is treated as the only factual source of truth. This catches
unsupported claims before editorial review and before future multi-model
comparison.

## What It Checks

The evaluator flags unsupported claims, overclaims, invented facts,
contradictions, claims-to-avoid violations, unsupported impact or benefit
language, missing key facts, and number/date/name/place/timeline preservation
issues.

Examples of unsupported benefit language:

- `பாதுகாப்பு உறுதி`
- `பாதிப்புகளை குறைக்க உதவும்`
- `புதிய நம்பிக்கை`
- `மக்களின் பாதுகாப்பை உறுதி செய்யும்`

These should be flagged unless the grounded brief explicitly supports them.

## Run Evaluation

```powershell
curl -X POST http://127.0.0.1:8000/drafts/<draft_id>/evaluate-grounding
curl http://127.0.0.1:8000/drafts/<draft_id>/evaluation/latest
```

Review/export:

```powershell
python -m backend.app.scripts.review_draft_evaluation --draft-id <draft_id>
python -m backend.app.scripts.review_draft_evaluation --draft-id <draft_id> --format html --output review_outputs/evaluation.html
```

The article draft review helper also includes latest evaluation highlights when
an evaluation exists.

## Limitation

This evaluator is LLM-based. It supports human editorial review; it does not
replace editorial judgment.
