# StyleScribe Architecture

StyleScribe is a backend-first article generation module for WISE+. Sprint 1
only establishes the API skeleton and contracts; real model calls, sample
processing, retrieval, and persistence are intentionally deferred.

## Future Pipeline

1. Source input
   - Accept a multilingual news URL or source text.
   - Extract and normalize the factual source material.

2. Grounded brief generation
   - Convert the source into a concise factual brief.
   - Preserve entities, quotes, dates, locations, and attribution.

3. Author style retrieval
   - Retrieve the selected author's sample articles and style profile.
   - Provide stylistic constraints without copying sample text.

4. Multi-model generation
   - Generate candidate Tamil articles through configured providers such as
     OpenAI, Qwen, and Gemma.
   - Keep provider configuration isolated from request handling.

5. QC comparison
   - Compare candidates for factual grounding, Tamil quality, style fit, and
     policy/editorial constraints.
   - Return the best candidate and diagnostics to WISE+.

## Sprint 1 Scope

Sprint 1 includes configuration loading, safe model registry placeholders,
request/response models, a health endpoint, and a stub generation endpoint.
It does not call LLMs, ingest author samples, or build a UI.
