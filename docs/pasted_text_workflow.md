# Pasted Website Text Workflow

Sprint 9 adds a practical workflow for editors who copy article text from a
website instead of relying on URL extraction. Sprint 10 adds an optional
grounded auto-revision loop after the initial grounding evaluation.

## Purpose

URL scraping can fail on dynamic pages, paywalls, login walls, or pages with
heavy navigation content. The pasted text workflow lets an editor paste the
visible article text and run the OpenAI-only pipeline:

1. Clean pasted website noise.
2. Generate a grounded factual brief.
3. Generate a Tamil author-style draft.
4. Run grounding evaluation.
5. Optionally revise unsupported claims using evaluator feedback.
6. Optionally run a final grounding evaluation.
7. Optionally export a UTF-8 before/after review file.

## Cleanup

Cleanup is deterministic and conservative. It normalizes whitespace, removes
repeated blank lines, drops obvious boilerplate such as `Advertisement`,
`Read more`, `Subscribe`, `Share`, `Trending`, and removes duplicate lines. It
does not rewrite, summarize, translate, or alter article facts.

The cleaner is intentionally limited. Editors should still review the cleaned
source excerpt and grounded brief before using the draft.

## API Example

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

The response returns IDs for the brief, original draft, initial evaluation,
optional revision, optional final evaluation, cleanup counts, summaries,
readiness values, warnings, and optional review export paths.

## Direct Brief Cleanup

`POST /briefs/grounded` also supports:

```json
{
  "source_type": "text",
  "source_input_mode": "pasted_web_text",
  "source_input": "copied article text",
  "target_language": "ta"
}
```

## Review Export

When `export_review` is `true`, the workflow writes a UTF-8 Markdown or HTML
file under `review_outputs/`. With auto revision enabled, the export shows the
cleaned source excerpt, grounded brief, original draft, initial evaluation,
unsupported claims, overclaim phrases, revision summary, revised draft, and
final evaluation when available. HTML uses a Tamil-friendly font stack and is
the recommended format when Windows terminal rendering makes Tamil difficult to
read.

`review_outputs/` is ignored by Git.

## Recommended Editor Workflow

1. Paste article text from the website into the workflow request.
2. Review cleanup counts and warnings.
3. Review the grounded brief facts and claims to avoid.
4. Review the Tamil draft.
5. Review grounding evaluation before publication or revision.
6. Use `run_auto_revision: true` when the evaluator flags unsupported impact,
   safety, effectiveness, hope, or guarantee language.
7. Review the final evaluation before publication.

## Limitations

- OpenAI is the only model provider used in this workflow.
- Auto revision does not add new facts; it rewrites against the grounded brief
  and evaluator feedback only.
- Qwen/Gemma comparison is planned after this workflow is stable.
- Cleanup removes obvious website noise only; it is not a full article parser.
- The workflow supports human editorial review and does not replace it.

## Manual Sprint 10 Command

```powershell
python -c "import json, pathlib; source=pathlib.Path('manual_test_input.txt').read_text(encoding='utf-8'); payload={'author_id':'v_vasanthi','source_text':source,'author_instruction':'Write this as a Tamil news article for Oneindia readers.','target_language':'ta','article_type':'news','desired_word_count':600,'tone_override':'clear, engaging and factual','run_grounding_evaluation':True,'run_auto_revision':True,'run_final_evaluation':True,'export_review':True,'export_format':'html'}; pathlib.Path('manual_request.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')"
$response = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/workflows/pasted-text-to-draft" -ContentType "application/json; charset=utf-8" -InFile ".\manual_request.json"
start $response.export_paths[0]
```
