"""Google Signals evaluation for generated StyleScribe articles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from backend.app.models.google_signals_models import (
    GoogleSignalsComponentConfig,
    GoogleSignalsComponentScore,
    GoogleSignalsEvaluationResult,
    GoogleSignalsScore,
    GoogleSignalsScoringConfig,
)
from backend.app.services.model_clients.openai_client import (
    OpenAIClientError,
    OpenAIJsonClient,
)

PROMPT_PATH = (
    Path(__file__).parents[1] / "prompts" / "google_signals_evaluator_prompt.txt"
)

GOOGLE_SIGNALS_V1_COMPONENTS = [
    GoogleSignalsComponentConfig(name="search_intent_clarity", weight=20),
    GoogleSignalsComponentConfig(name="headline_search_clarity", weight=15),
    GoogleSignalsComponentConfig(name="freshness_timeliness", weight=15),
    GoogleSignalsComponentConfig(name="originality_angle", weight=15),
    GoogleSignalsComponentConfig(name="eeat_trust", weight=15),
    GoogleSignalsComponentConfig(name="snippet_meta_readiness", weight=10),
    GoogleSignalsComponentConfig(name="structured_data_readiness", weight=10),
]
GOOGLE_SIGNALS_V1_CONFIG = GoogleSignalsScoringConfig(
    version="google_signals_v1",
    components=GOOGLE_SIGNALS_V1_COMPONENTS,
)


class StructuredJsonClient(Protocol):
    provider: str
    model_name: str

    def generate_structured_json(
        self,
        system_prompt: str,
        user_payload: str,
        prompt_cache_key: str | None = None,
    ) -> dict[str, object]:
        """Generate a structured JSON object."""


def evaluate_google_signals(
    *,
    final_article: dict[str, object],
    grounded_brief: dict[str, object],
    author_id: str,
    article_type: str,
    target_language: str,
    desired_word_count: int | None = None,
    workflow_metadata: dict[str, object] | None = None,
    config: GoogleSignalsScoringConfig = GOOGLE_SIGNALS_V1_CONFIG,
    model_client: StructuredJsonClient | None = None,
) -> GoogleSignalsEvaluationResult:
    """Evaluate Google Search readiness without blocking article generation."""

    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    payload = build_google_signals_input(
        final_article=final_article,
        grounded_brief=grounded_brief,
        author_id=author_id,
        article_type=article_type,
        target_language=target_language,
        desired_word_count=desired_word_count,
        workflow_metadata=workflow_metadata,
        config=config,
    )

    try:
        client = model_client or OpenAIJsonClient(
            missing_key_message="OPENAI_API_KEY is required for Google Signals."
        )
        raw_evaluation = client.generate_structured_json(
            prompt,
            payload,
            prompt_cache_key=f"stylescribe:{config.version}",
        )
        score = build_google_signals_score(raw_evaluation, config)
    except (OpenAIClientError, ValidationError, ValueError, TypeError) as exc:
        return GoogleSignalsEvaluationResult(
            available=False,
            error=f"Google Signals evaluator failed: {exc}",
        )
    except Exception as exc:
        return GoogleSignalsEvaluationResult(
            available=False,
            error=f"Google Signals evaluator failed unexpectedly: {exc}",
        )

    return GoogleSignalsEvaluationResult(available=True, google_signals=score)


def build_google_signals_input(
    *,
    final_article: dict[str, object],
    grounded_brief: dict[str, object],
    author_id: str,
    article_type: str,
    target_language: str,
    desired_word_count: int | None,
    workflow_metadata: dict[str, object] | None,
    config: GoogleSignalsScoringConfig,
) -> str:
    payload = {
        "task": "Evaluate Google Search readiness for a generated Tamil article.",
        "scoring_version": config.version,
        "enabled_components": [
            component.model_dump() for component in config.enabled_components
        ],
        "generated_article_to_evaluate": {
            "headline": final_article.get("headline"),
            "subheadline": final_article.get("subheadline"),
            "article_body": final_article.get("article_body"),
            "seo_title": final_article.get("seo_title"),
            "meta_description": final_article.get("meta_description"),
            "suggested_tags": final_article.get("suggested_tags"),
        },
        "grounded_brief_only_factual_source": {
            "topic": grounded_brief.get("topic"),
            "one_line_summary": grounded_brief.get("one_line_summary"),
            "confirmed_facts": grounded_brief.get("confirmed_facts", []),
            "key_entities": grounded_brief.get("key_entities", []),
            "places": grounded_brief.get("places", []),
            "dates_or_timeline": grounded_brief.get("dates_or_timeline", []),
            "numbers_and_statistics": grounded_brief.get(
                "numbers_and_statistics",
                [],
            ),
            "quotes": grounded_brief.get("quotes", []),
            "claims_to_avoid": grounded_brief.get("claims_to_avoid", []),
            "editorial_risk_notes": grounded_brief.get("editorial_risk_notes", []),
        },
        "metadata": {
            "author_id": author_id,
            "article_type": article_type,
            "target_language": target_language,
            "desired_word_count": desired_word_count,
            **(workflow_metadata or {}),
        },
        "evaluation_rule": (
            "Use only grounded_brief_only_factual_source and the generated "
            "article. Do not use outside knowledge or recommend unverified facts."
        ),
    }
    return json.dumps(payload, ensure_ascii=False)


def build_google_signals_score(
    raw_evaluation: dict[str, object],
    config: GoogleSignalsScoringConfig = GOOGLE_SIGNALS_V1_CONFIG,
) -> GoogleSignalsScore:
    """Normalize evaluator output into the configured weighted score container."""

    configured_by_name = {component.name: component for component in config.components}
    raw_by_name = _raw_components_by_name(raw_evaluation.get("components"))
    component_scores: list[GoogleSignalsComponentScore] = []

    for component_config in config.components:
        raw_component = raw_by_name.get(component_config.name)
        score = _component_score(raw_component)
        rationale = _component_rationale(raw_component)
        risk_level = _component_risk_level(raw_component)
        if raw_component is None and component_config.enabled:
            rationale = "Evaluator did not return this enabled component."
            risk_level = "high"
        component = GoogleSignalsComponentScore(
            name=component_config.name,
            score=score,
            weight=component_config.weight,
            rationale=rationale,
            risk_level=risk_level,
            enabled=component_config.enabled,
        )
        component_scores.append(component)

    for name, raw_component in raw_by_name.items():
        if name not in configured_by_name:
            component_scores.append(
                GoogleSignalsComponentScore(
                    name=name,
                    score=_component_score(raw_component),
                    weight=0,
                    rationale=_component_rationale(raw_component),
                    risk_level=_component_risk_level(raw_component),
                    enabled=False,
                )
            )

    metadata = _dict_value(raw_evaluation.get("metadata"))
    metadata.setdefault("schema_type", "NewsArticle")
    risk_flags = _string_list(raw_evaluation.get("risk_flags"))
    apply_google_signals_score_caps(
        components=component_scores,
        risk_flags=risk_flags,
        metadata=metadata,
    )
    return GoogleSignalsScore(
        score=_weighted_google_signals_score(component_scores),
        version=config.version,
        components=component_scores,
        risk_flags=risk_flags,
        recommendations=_sanitize_recommendations(
            _string_list(raw_evaluation.get("recommendations"))
        ),
        metadata=metadata,
        overall_rationale=_optional_string(raw_evaluation.get("overall_rationale")),
    )


def apply_google_signals_score_caps(
    *,
    components: list[GoogleSignalsComponentScore],
    risk_flags: list[str],
    metadata: dict[str, Any],
) -> None:
    """Apply deterministic caps for obvious Google-readiness problems."""

    try:
        caps: list[dict[str, object]] = []
        risk_text = _normalized_risk_text(risk_flags)
        primary_search_intent = metadata.get("primary_search_intent")

        if _has_any_risk(risk_text, ["vague_headline", "vague headline"]):
            _cap_component(
                components,
                caps,
                "headline_search_clarity",
                55,
                "vague_headline",
            )
        if _has_any_risk(
            risk_text,
            ["weak_first_paragraph", "weak first paragraph"],
        ):
            _cap_component(
                components,
                caps,
                "search_intent_clarity",
                65,
                "weak_first_paragraph",
            )
            _cap_component(
                components,
                caps,
                "snippet_meta_readiness",
                65,
                "weak_first_paragraph",
            )
        if _is_vague_search_intent(primary_search_intent):
            _cap_component(
                components,
                caps,
                "search_intent_clarity",
                60,
                "missing_or_unclear_primary_search_intent",
            )
        if _has_any_risk(
            risk_text,
            [
                "generic_content",
                "generic rewritten content",
                "generic_rewritten_content",
            ],
        ):
            _cap_component(
                components,
                caps,
                "originality_angle",
                65,
                "generic_content",
            )
        if _has_any_risk(risk_text, ["thin_content", "thin content"]):
            _cap_component(
                components,
                caps,
                "snippet_meta_readiness",
                60,
                "thin_content",
            )
            _cap_component(
                components,
                caps,
                "structured_data_readiness",
                55,
                "thin_content",
            )
        if _has_any_risk(
            risk_text,
            [
                "unsupported_claims_remaining",
                "unsupported claims",
                "claims_to_avoid_violations_remaining",
                "claims to avoid violations",
            ],
        ):
            _cap_component(
                components,
                caps,
                "eeat_trust",
                65,
                "unsupported_claims",
            )
        if _lacks_clear_primary_entity_or_topic(metadata):
            _cap_component(
                components,
                caps,
                "structured_data_readiness",
                55,
                "missing_primary_entity_or_topic",
            )

        if caps:
            metadata["applied_score_caps"] = caps
    except Exception:
        return


def _raw_components_by_name(value: object) -> dict[str, dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ValueError("Google Signals evaluator returned no component list.")
    components: dict[str, dict[str, object]] = {}
    for item in value:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if isinstance(name, str) and name:
            components[name] = dict(item)
    return components


def _component_score(component: dict[str, object] | None) -> int:
    if component is None:
        return 0
    value = component.get("score")
    if isinstance(value, bool):
        return 0
    if isinstance(value, int | float):
        return max(0, min(100, round(value)))
    return 0


def _component_rationale(component: dict[str, object] | None) -> str:
    if component is None:
        return ""
    value = component.get("rationale")
    return value if isinstance(value, str) else ""


def _component_risk_level(component: dict[str, object] | None) -> str:
    if component is None:
        return "unknown"
    value = component.get("risk_level")
    if value in {"low", "medium", "high"}:
        return str(value)
    score = _component_score(component)
    if score >= 80:
        return "low"
    if score >= 60:
        return "medium"
    return "high"


def _dict_value(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _string_list(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return []
    return [item for item in value if isinstance(item, str)]


def _weighted_google_signals_score(
    components: list[GoogleSignalsComponentScore],
) -> int:
    weighted_total = 0.0
    enabled_weight_total = 0
    for component in components:
        if component.enabled:
            weighted_total += component.score * component.weight
            enabled_weight_total += component.weight
    if enabled_weight_total <= 0:
        raise ValueError("Google Signals scoring config has no enabled weight.")
    return round(weighted_total / enabled_weight_total)


def _normalized_risk_text(risk_flags: list[str]) -> str:
    return " | ".join(flag.lower().replace("-", "_") for flag in risk_flags)


def _has_any_risk(risk_text: str, patterns: list[str]) -> bool:
    normalized_patterns = [
        pattern.lower().replace("-", "_") for pattern in patterns
    ]
    return any(pattern in risk_text for pattern in normalized_patterns)


def _cap_component(
    components: list[GoogleSignalsComponentScore],
    caps: list[dict[str, object]],
    name: str,
    cap: int,
    reason: str,
) -> None:
    for component in components:
        if component.name != name:
            continue
        original_score = component.score
        component.score = min(component.score, cap)
        if component.score < original_score:
            caps.append(
                {
                    "component": name,
                    "original_score": original_score,
                    "capped_score": component.score,
                    "cap": cap,
                    "reason": reason,
                }
            )
        return


def _is_vague_search_intent(value: object) -> bool:
    if not isinstance(value, str):
        return True
    normalized = value.strip().lower()
    if not normalized:
        return True
    vague_values = {
        "unclear",
        "unknown",
        "n/a",
        "na",
        "none",
        "not available",
        "not clear",
        "vague",
        "general update",
        "general news",
        "news update",
    }
    if normalized in vague_values:
        return True
    return any(
        phrase in normalized
        for phrase in [
            "unclear intent",
            "vague intent",
            "not enough information",
            "cannot identify",
        ]
    )


def _lacks_clear_primary_entity_or_topic(metadata: dict[str, Any]) -> bool:
    primary_entities = metadata.get("primary_entities")
    has_primary_entity = (
        isinstance(primary_entities, Sequence)
        and not isinstance(primary_entities, str)
        and any(isinstance(item, str) and item.strip() for item in primary_entities)
    )
    if has_primary_entity:
        return False
    return _is_vague_search_intent(metadata.get("primary_search_intent"))


def _sanitize_recommendations(recommendations: list[str]) -> list[str]:
    sanitized: list[str] = []
    for recommendation in recommendations:
        if _is_source_conditioned(recommendation):
            sanitized.append(recommendation)
        elif _asks_for_unsupported_material(recommendation):
            sanitized.extend(
                _safe_recommendation_replacements(recommendation)
            )
        else:
            sanitized.append(recommendation)
    return _dedupe_strings(sanitized)


def _is_source_conditioned(value: str) -> bool:
    normalized = value.lower()
    return any(
        phrase in normalized
        for phrase in [
            "if present in the source",
            "if present in source",
            "if present in the grounded brief",
            "if present in grounded brief",
            "if present in the source brief",
            "if available in the source",
            "from the source",
            "from available workflow metadata",
            "from workflow metadata",
            "using available workflow metadata",
            "using source-grounded fields",
        ]
    )


def _asks_for_unsupported_material(value: str) -> bool:
    normalized = value.lower()
    unsafe_patterns = [
        "add additional context",
        "add author information",
        "add detail",
        "add details",
        "add more specific detail",
        "add more specific details",
        "add direct quote",
        "add direct quotes",
        "add expert quote",
        "add expert quotes",
        "add expert opinion",
        "add expert opinions",
        "add citations",
        "add citation",
        "add references",
        "add reference",
        "add specific dates",
        "add specific detail",
        "add specific details",
        "add dates",
        "add methodology",
        "add publication date",
        "add recent developments",
        "add recent development",
        "add stakeholder quote",
        "add stakeholder quotes",
        "add structured data field",
        "add structured data fields",
        "add updates",
        "add update",
        "add statistics",
        "add statistic",
        "add unique insights",
        "add unique insight",
        "add context",
        "clarify author information",
        "clarify detail",
        "clarify details",
        "clarify methodology",
        "clarify publication date",
        "clarify specific detail",
        "clarify specific details",
        "clarify structured data",
        "clarify structured data elements",
        "clarify structured data field",
        "clarify structured data fields",
        "consider adding additional context",
        "consider adding author information",
        "consider adding detail",
        "consider adding details",
        "consider adding more specific detail",
        "consider adding more specific details",
        "consider adding direct quote",
        "consider adding direct quotes",
        "consider adding expert quote",
        "consider adding expert quotes",
        "consider adding expert opinion",
        "consider adding expert opinions",
        "consider adding citations",
        "consider adding citation",
        "consider adding references",
        "consider adding reference",
        "consider adding specific dates",
        "consider adding specific detail",
        "consider adding specific details",
        "consider adding dates",
        "consider adding methodology",
        "consider adding publication date",
        "consider adding recent developments",
        "consider adding recent development",
        "consider adding stakeholder quote",
        "consider adding stakeholder quotes",
        "consider adding structured data field",
        "consider adding structured data fields",
        "consider adding updates",
        "consider adding update",
        "consider adding statistics",
        "consider adding statistic",
        "consider adding unique insights",
        "consider adding unique insight",
        "consider adding context",
        "include additional context",
        "include author information",
        "include detail",
        "include details",
        "include more specific detail",
        "include more specific details",
        "include direct quote",
        "include direct quotes",
        "include expert quote",
        "include expert quotes",
        "include expert opinion",
        "include expert opinions",
        "include citations",
        "include citation",
        "include specific dates",
        "include dates",
        "include references",
        "include reference",
        "include recent developments",
        "include recent development",
        "include methodology",
        "include publication date",
        "include specific detail",
        "include specific details",
        "include stakeholder quote",
        "include stakeholder quotes",
        "include structured data field",
        "include structured data fields",
        "include updates",
        "include update",
        "include statistics",
        "include statistic",
        "include unique insights",
        "include unique insight",
        "include context",
        "incorporate additional context",
        "incorporate author information",
        "incorporate detail",
        "incorporate details",
        "incorporate more specific detail",
        "incorporate more specific details",
        "incorporate expert quote",
        "incorporate expert quotes",
        "incorporate expert opinion",
        "incorporate expert opinions",
        "incorporate direct quote",
        "incorporate direct quotes",
        "incorporate citations",
        "incorporate references",
        "incorporate specific dates",
        "incorporate dates",
        "incorporate methodology",
        "incorporate publication date",
        "incorporate recent developments",
        "incorporate stakeholder quote",
        "incorporate stakeholder quotes",
        "incorporate structured data field",
        "incorporate structured data fields",
        "incorporate updates",
        "incorporate statistics",
        "incorporate unique insights",
        "incorporate unique insight",
        "incorporate context",
        "provide additional context",
        "provide author information",
        "provide detail",
        "provide details",
        "provide more specific detail",
        "provide more specific details",
        "provide methodology",
        "provide publication date",
        "provide specific detail",
        "provide specific details",
        "provide stakeholder quote",
        "provide stakeholder quotes",
        "provide structured data field",
        "provide structured data fields",
        "stakeholder quotes",
        "expert quotes",
        "expert opinions",
        "direct quotes",
        "citations",
        "references",
        "recent developments",
        "specific dates",
        "statistics",
        "unique insights",
        "additional context",
        "author information",
        "publication date",
        "survey methodology",
    ]
    return any(pattern in normalized for pattern in unsafe_patterns)


def _safe_recommendation_replacements(value: str) -> list[str]:
    normalized = value.lower()
    replacements: list[str] = []
    if _mentions_any(normalized, ["detail", "details", "specific details"]):
        replacements.append(
            "Use source-provided details more clearly if present; do not add new "
            "details unless they are in the source brief."
        )
    if _mentions_any(
        normalized,
        [
            "quote",
            "quotes",
            "stakeholder",
            "stakeholders",
            "expert opinion",
            "expert opinions",
            "citation",
            "citations",
            "reference",
            "references",
        ],
    ):
        replacements.append(
            "Surface source-provided expert attribution more clearly if present; "
            "do not add new quotes or citations unless present in the source brief."
        )
    if _mentions_any(normalized, ["stakeholder", "stakeholders"]):
        replacements.append(
            "Surface stakeholder views only if they are present in the source "
            "brief; do not add new quotes."
        )
    if _mentions_any(normalized, ["methodology"]):
        replacements.append(
            "Clarify methodology only using details already present in the source."
        )
    if _mentions_any(
        normalized,
        [
            "structured data",
            "publication date",
            "author information",
            "metadata",
        ],
    ):
        replacements.append(
            "Populate structured data only from available workflow metadata or "
            "source-grounded fields."
        )
    if _mentions_any(
        normalized,
        ["recent development", "recent developments", "update", "updates"],
    ):
        replacements.append(
            "Clarify the available freshness cues from the source; do not add "
            "recent developments unless present in the source brief."
        )
    if _mentions_any(
        normalized,
        ["unique insight", "unique insights", "context", "originality"],
    ):
        replacements.append(
            "Strengthen originality using the source-grounded angle; do not add "
            "unsupported context."
        )
    if _mentions_any(normalized, ["statistic", "statistics", "date", "dates"]):
        replacements.append(
            "Do not add new facts, quotes, citations, statistics, dates, "
            "methodology, metadata, or developments unless present in the source "
            "brief or workflow metadata."
        )
    if replacements:
        replacements.append(
            "Strengthen this section using available grounded details, not external "
            "additions."
        )
        return replacements
    return [
        "Use only source-grounded details to strengthen this section.",
        "Clarify what is confirmed versus expected based on the source.",
    ]


def _mentions_any(value: str, patterns: list[str]) -> bool:
    return any(pattern in value for pattern in patterns)


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped
