"""Deterministic retrieval-leakage diagnostics."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from backend.app.services.newsroom_retrieval_service import RetrievalRecord

NUMBER_RE = re.compile(r"(?<!\w)(?:₹|\$)?\d+(?:[,.:-]\d+)*(?:\s?%|\s?percent)?")
DATE_RE = re.compile(
    r"(?<!\w)(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Z][a-z]+\s+\d{4}|[A-Z][a-z]+\s+\d{1,2},\s+\d{4})(?!\w)"
)
QUOTE_RE = re.compile(r"[\"“”'‘’]([^\"“”'‘’]{8,160})[\"“”'‘’]")
TOKEN_RE = re.compile(r"[\w\u0b80-\u0bff]+", re.UNICODE)
PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z0-9.-]*(?:\s+[A-Z][A-Za-z0-9.-]*){0,4}\b")
COMMON_PHRASES = {
    "according to",
    "said that",
    "in this situation",
    "at the same time",
    "meanwhile",
    "however",
}
COMMON_TOKENS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "said",
    "news",
    "article",
    "சென்னை",
    "தமிழ்",
    "தமிழக",
    "மேலும்",
    "இதனால்",
}


@dataclass(frozen=True)
class LeakageFinding:
    finding_type: str
    generated_text_fragment: str
    retrieved_article_id: str
    retrieved_source_fragment: str
    exists_in_grounded_brief: bool
    severity: str
    confidence: float
    explanation: str


def run_retrieval_leakage_diagnostic(
    *,
    grounded_brief: dict[str, object],
    generated_article: str,
    retrieved_records: list[RetrievalRecord],
) -> dict[str, object]:
    brief_text = _brief_text(grounded_brief)
    findings: list[LeakageFinding] = []
    for record in retrieved_records:
        source = record.cleaned_article_text
        findings.extend(
            _literal_findings(
                finding_type="date",
                pattern=DATE_RE,
                generated_article=generated_article,
                source=source,
                brief_text=brief_text,
                article_id=record.article_id,
                severity="high",
            )
        )
        findings.extend(
            _literal_findings(
                finding_type="number",
                pattern=NUMBER_RE,
                generated_article=generated_article,
                source=source,
                brief_text=brief_text,
                article_id=record.article_id,
                severity="medium",
            )
        )
        findings.extend(
            _quote_findings(
                generated_article=generated_article,
                source=source,
                brief_text=brief_text,
                article_id=record.article_id,
            )
        )
        findings.extend(
            _proper_noun_findings(
                generated_article=generated_article,
                source=source,
                brief_text=brief_text,
                article_id=record.article_id,
            )
        )
        findings.extend(
            _distinctive_phrase_findings(
                generated_article=generated_article,
                source=source,
                brief_text=brief_text,
                article_id=record.article_id,
            )
        )
    deduped = _dedupe(findings)
    return {
        "diagnostic_version": "retrieval_leakage_heuristic_v1",
        "finding_count": len(deduped),
        "findings": [asdict(finding) for finding in deduped],
        "status": "review_required" if deduped else "clear",
    }


def _literal_findings(
    *,
    finding_type: str,
    pattern: re.Pattern[str],
    generated_article: str,
    source: str,
    brief_text: str,
    article_id: str,
    severity: str,
) -> list[LeakageFinding]:
    generated_values = _matches(pattern, generated_article)
    source_values = _matches(pattern, source)
    findings: list[LeakageFinding] = []
    for value in sorted(generated_values & source_values):
        if _is_ordinary_number(value):
            continue
        exists = _contains(brief_text, value)
        if exists:
            continue
        findings.append(
            LeakageFinding(
                finding_type=finding_type,
                generated_text_fragment=value,
                retrieved_article_id=article_id,
                retrieved_source_fragment=value,
                exists_in_grounded_brief=False,
                severity=severity,
                confidence=0.86 if finding_type == "date" else 0.76,
                explanation=(
                    f"{finding_type} appears in generated article and retrieved "
                    "example but not in the grounded brief."
                ),
            )
        )
    return findings


def _quote_findings(
    *,
    generated_article: str,
    source: str,
    brief_text: str,
    article_id: str,
) -> list[LeakageFinding]:
    generated = {item.strip() for item in QUOTE_RE.findall(generated_article)}
    source_quotes = {item.strip() for item in QUOTE_RE.findall(source)}
    findings: list[LeakageFinding] = []
    for quote in sorted(generated & source_quotes):
        if _contains(brief_text, quote):
            continue
        findings.append(
            LeakageFinding(
                finding_type="quotation",
                generated_text_fragment=quote,
                retrieved_article_id=article_id,
                retrieved_source_fragment=quote,
                exists_in_grounded_brief=False,
                severity="high",
                confidence=0.9,
                explanation=(
                    "Quoted text appears in generated article and retrieved "
                    "example but not in the grounded brief."
                ),
            )
        )
    return findings


def _proper_noun_findings(
    *,
    generated_article: str,
    source: str,
    brief_text: str,
    article_id: str,
) -> list[LeakageFinding]:
    generated = {
        item.strip()
        for item in PROPER_NOUN_RE.findall(generated_article)
        if _proper_noun_candidate(item)
    }
    source_values = {
        item.strip()
        for item in PROPER_NOUN_RE.findall(source)
        if _proper_noun_candidate(item)
    }
    findings: list[LeakageFinding] = []
    for value in sorted(generated & source_values):
        if _contains(brief_text, value):
            continue
        findings.append(
            LeakageFinding(
                finding_type="proper_noun",
                generated_text_fragment=value,
                retrieved_article_id=article_id,
                retrieved_source_fragment=value,
                exists_in_grounded_brief=False,
                severity="medium",
                confidence=0.72,
                explanation=(
                    "Proper noun appears in generated article and retrieved "
                    "example but not in the grounded brief."
                ),
            )
        )
    return findings


def _distinctive_phrase_findings(
    *,
    generated_article: str,
    source: str,
    brief_text: str,
    article_id: str,
) -> list[LeakageFinding]:
    generated_phrases = _distinctive_ngrams(generated_article)
    source_phrases = _distinctive_ngrams(source)
    findings: list[LeakageFinding] = []
    for phrase in sorted(generated_phrases & source_phrases):
        if _contains(brief_text, phrase):
            continue
        findings.append(
            LeakageFinding(
                finding_type="distinctive_phrase",
                generated_text_fragment=phrase,
                retrieved_article_id=article_id,
                retrieved_source_fragment=phrase,
                exists_in_grounded_brief=False,
                severity="low",
                confidence=0.62,
                explanation=(
                    "Distinctive multi-word phrase overlaps generated article "
                    "and retrieved example but not the grounded brief."
                ),
            )
        )
    return findings


def _brief_text(brief: dict[str, object]) -> str:
    parts: list[str] = []
    for value in brief.values():
        parts.extend(_flatten(value))
    return "\n".join(parts)


def _flatten(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        parts: list[str] = []
        for item in value.values():
            parts.extend(_flatten(item))
        return parts
    if isinstance(value, list):
        parts = []
        for item in value:
            parts.extend(_flatten(item))
        return parts
    return [str(value)]


def _matches(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).strip() for match in pattern.finditer(text)}


def _contains(text: str, fragment: str) -> bool:
    if fragment.casefold() in text.casefold():
        return True
    normalized_text = _normalized_token_text(text)
    normalized_fragment = _normalized_token_text(fragment)
    if bool(normalized_fragment) and normalized_fragment in normalized_text:
        return True
    return _ordered_tokens_present(
        haystack=[token.casefold() for token in TOKEN_RE.findall(text)],
        needle=[token.casefold() for token in TOKEN_RE.findall(fragment)],
    )


def _is_ordinary_number(value: str) -> bool:
    digits = re.sub(r"\D", "", value)
    return len(digits) <= 1


def _proper_noun_candidate(value: str) -> bool:
    tokens = value.split()
    if not tokens or len(value) < 4:
        return False
    return not all(token.casefold() in COMMON_TOKENS for token in tokens)


def _distinctive_ngrams(text: str) -> set[str]:
    tokens = [
        token
        for token in (raw.strip().casefold() for raw in TOKEN_RE.findall(text))
        if len(token) >= 3 and token not in COMMON_TOKENS
    ]
    phrases: set[str] = set()
    for size in (5, 6, 7):
        for index in range(0, max(0, len(tokens) - size + 1)):
            phrase = " ".join(tokens[index : index + size])
            if phrase in COMMON_PHRASES:
                continue
            if len(set(phrase.split())) < 3:
                continue
            phrases.add(phrase)
    return phrases


def _normalized_token_text(text: str) -> str:
    return " ".join(token.casefold() for token in TOKEN_RE.findall(text))


def _ordered_tokens_present(*, haystack: list[str], needle: list[str]) -> bool:
    if not needle:
        return False
    position = 0
    for token in haystack:
        if token == needle[position]:
            position += 1
            if position == len(needle):
                return True
    return False


def _dedupe(findings: list[LeakageFinding]) -> list[LeakageFinding]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[LeakageFinding] = []
    for finding in findings:
        key = (
            finding.finding_type,
            finding.generated_text_fragment.casefold(),
            finding.retrieved_article_id,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(finding)
    severity_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        deduped,
        key=lambda item: (
            severity_order.get(item.severity, 3),
            item.finding_type,
            item.generated_text_fragment,
        ),
    )
