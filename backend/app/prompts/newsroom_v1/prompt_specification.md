# Oneindia Newsroom V1 Prompt Specification

Version: `oneindia_newsroom_v1.0`

This prompt is a separate generic newsroom-generation path. It does not replace
or edit the legacy Gemini prompt.

## Runtime Layers

1. `factual_source_brief`: the only factual source.
2. `generic_newsroom_editorial_rules`: compact conditional guidance derived
   from Sprint 2 newsroom profiling.
3. `optional_topic_guidance`: weak topic guidance, omitted when unavailable or
   low-confidence.
4. `output_schema`: the JSON shape expected by the existing draft workflow.
5. `prohibited_behaviours`: explicit factual and stylistic guardrails.

## Phrase Handling

Frequent corpus phrases are not mandatory. The prompt treats them as examples
of reusable newsroom construction types only when they fit the source facts.

Author-skewed, topic-specific, fact-bearing and repetitive phrases are excluded
from runtime guidance unless the source brief itself contains the fact-bearing
wording.

## Compatibility

`legacy` remains the default benchmark generation mode. `newsroom_v1` is selected
explicitly and does not load an author style profile.
