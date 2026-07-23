"""Generic Oneindia Tamil newsroom profile from prepared corpus artifacts."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any

from backend.app.services.newsroom_corpus_preparation_service import (
    DEFAULT_CLASSIFIED_DIR,
    DEFAULT_CLEANED_DIR,
    normalize_for_comparison,
)
from backend.app.services.newsroom_corpus_service import DEFAULT_REPORTS_DIR

DEFAULT_CLEANED_ARTICLES_JSONL = DEFAULT_CLEANED_DIR / "cleaned_articles.jsonl"
DEFAULT_CLASSIFIED_ARTICLES_JSONL = DEFAULT_CLASSIFIED_DIR / "classified_articles.jsonl"

DATE_PATTERN = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b")
NUMBER_PATTERN = re.compile(r"\d+")
QUOTE_PATTERN = re.compile(r"[\"'“”‘’]")
PLACE_PREFIX_PATTERN = re.compile(r"^[^:\n]{2,40}:")
TOKEN_STRIP_CHARS = " \t\n\r.,;:!?()[]{}\"'“”‘’…|/\\<>"

ATTRIBUTION_MARKERS = [
    "said",
    "according",
    "told",
    "à®¤à¯†à®°à®¿à®µà®¿à®¤à¯à®¤",
    "à®•à¯‚à®±",
    "à®Žà®©à¯à®±à¯",
    "à®ªà¯‡à®Ÿà¯à®Ÿà®¿",
]
CAUTION_MARKERS = [
    "reportedly",
    "likely",
    "may",
    "à®•à¯‚à®±à®ªà¯à®ªà®Ÿà¯à®•à®¿à®±",
    "à®¤à®•à®µà®²à¯",
    "à®Žà®© à®¤à¯†à®°à®¿à®•à®¿à®±",
]
TRANSITION_MARKERS = [
    "meanwhile",
    "however",
    "also",
    "à®‡à®¤à®©à®¿à®Ÿà¯ˆà®¯à¯‡",
    "à®†à®©à®¾à®²à¯",
    "à®®à¯‡à®²à¯à®®à¯",
    "à®‡à®¨à¯à®¤ à®¨à®¿à®²à¯ˆà®¯à®¿à®²à¯",
]
BACKGROUND_MARKERS = [
    "background",
    "earlier",
    "already",
    "à®•à®Ÿà®¨à¯à®¤",
    "à®®à¯à®©à¯à®©à®¤à®¾à®•",
    "à®‡à®¤à®±à¯à®•à¯ à®®à¯à®©à¯",
]
UPDATE_MARKERS = [
    "update",
    "latest",
    "à®¤à®±à¯à®ªà¯‹à®¤à¯",
    "à®‡à®©à¯à®±à¯",
    "à®ªà¯à®¤à®¿à®¯",
]

# Keep Tamil markers as Unicode escapes so source encoding issues cannot weaken
# deterministic corpus statistics.
QUOTE_PATTERN = re.compile(r"[\"'\u201c\u201d\u2018\u2019]")
TOKEN_STRIP_CHARS = " \t\n\r.,;:!?()[]{}\"'\u201c\u201d\u2018\u2019\u2026|/\\<>"
ATTRIBUTION_MARKERS = [
    "said",
    "according",
    "told",
    "\u0ba4\u0bc6\u0bb0\u0bbf\u0bb5\u0bbf\u0ba4\u0bcd\u0ba4",
    "\u0b95\u0bc2\u0bb1",
    "\u0b8e\u0ba9\u0bcd\u0bb1\u0bc1",
    "\u0baa\u0bc7\u0b9f\u0bcd\u0b9f\u0bbf",
    "\u0b85\u0bb5\u0bb0\u0bcd",
]
CAUTION_MARKERS = [
    "reportedly",
    "likely",
    "may",
    "\u0b95\u0bc2\u0bb1\u0baa\u0bcd\u0baa\u0b9f\u0bc1\u0b95\u0bbf\u0bb1",
    "\u0ba4\u0b95\u0bb5\u0bb2\u0bcd",
    "\u0b8e\u0ba9 \u0ba4\u0bc6\u0bb0\u0bbf\u0b95\u0bbf\u0bb1",
    "\u0b8e\u0ba4\u0bbf\u0bb0\u0bcd\u0baa\u0bbe\u0bb0\u0bcd\u0b95\u0bcd\u0b95",
]
TRANSITION_MARKERS = [
    "meanwhile",
    "however",
    "also",
    "\u0b87\u0ba4\u0ba9\u0bbf\u0b9f\u0bc8\u0baf\u0bc7",
    "\u0b86\u0ba9\u0bbe\u0bb2\u0bcd",
    "\u0bae\u0bc7\u0bb2\u0bc1\u0bae\u0bcd",
    "\u0b87\u0ba8\u0bcd\u0ba4 \u0ba8\u0bbf\u0bb2\u0bc8\u0baf\u0bbf\u0bb2\u0bcd",
    "\u0b85\u0ba4\u0bc7 \u0ba8\u0bc7\u0bb0\u0ba4\u0bcd\u0ba4\u0bbf\u0bb2\u0bcd",
]
BACKGROUND_MARKERS = [
    "background",
    "earlier",
    "already",
    "\u0b95\u0b9f\u0ba8\u0bcd\u0ba4",
    "\u0bae\u0bc1\u0ba9\u0bcd\u0ba9\u0ba4\u0bbe\u0b95",
    "\u0b87\u0ba4\u0bb1\u0bcd\u0b95\u0bc1 \u0bae\u0bc1\u0ba9\u0bcd",
    "\u0b8f\u0bb1\u0bcd\u0b95\u0ba9\u0bb5\u0bc7",
]
UPDATE_MARKERS = [
    "update",
    "latest",
    "\u0ba4\u0bb1\u0bcd\u0baa\u0bcb\u0ba4\u0bc1",
    "\u0b87\u0ba9\u0bcd\u0bb1\u0bc1",
    "\u0baa\u0bc1\u0ba4\u0bbf\u0baf",
]


@dataclass(frozen=True)
class NewsroomProfilePathConfig:
    cleaned_articles_jsonl: Path = DEFAULT_CLEANED_ARTICLES_JSONL
    classified_articles_jsonl: Path = DEFAULT_CLASSIFIED_ARTICLES_JSONL
    reports_dir: Path = DEFAULT_REPORTS_DIR


@dataclass(frozen=True)
class EvidenceSample:
    article_id: str
    author_id: str
    relative_source_path: str
    excerpt: str


@dataclass(frozen=True)
class PatternConclusion:
    pattern_id: str
    category: str
    conclusion: str
    frequency: int
    prevalence: float
    authors_represented: list[str]
    confidence: str
    evidence_samples: list[EvidenceSample]


@dataclass(frozen=True)
class PhraseRecord:
    phrase: str
    phrase_type: str
    article_count: int
    occurrence_count: int
    prevalence: float
    authors_represented: list[str]
    author_distribution: dict[str, int]
    author_skew: float
    recommendation: str
    evidence_article_ids: list[str]


@dataclass(frozen=True)
class NewsroomProfileResult:
    profile: dict[str, Any]
    conclusions: list[PatternConclusion]
    phrase_bank: list[PhraseRecord]
    phrase_review_list: list[PhraseRecord]
    output_paths: dict[str, Path]
    summary: dict[str, int]


def run_newsroom_profile_analysis(
    paths: NewsroomProfilePathConfig | None = None,
) -> NewsroomProfileResult:
    paths = paths or NewsroomProfilePathConfig()
    accepted = _load_jsonl(paths.cleaned_articles_jsonl)
    classified = {
        str(article["article_id"]): article
        for article in _load_jsonl(paths.classified_articles_jsonl)
    }
    articles = [
        _merge_classification(article, classified.get(str(article["article_id"])))
        for article in accepted
    ]
    conclusions = build_pattern_conclusions(articles)
    phrase_bank, phrase_review_list = extract_phrase_records(articles)
    profile = build_machine_profile(
        articles=articles,
        conclusions=conclusions,
        phrase_bank=phrase_bank,
        phrase_review_list=phrase_review_list,
    )
    output_paths = write_newsroom_profile_outputs(
        paths=paths,
        profile=profile,
        conclusions=conclusions,
        phrase_bank=phrase_bank,
        phrase_review_list=phrase_review_list,
        articles=articles,
    )
    return NewsroomProfileResult(
        profile=profile,
        conclusions=conclusions,
        phrase_bank=phrase_bank,
        phrase_review_list=phrase_review_list,
        output_paths=output_paths,
        summary={
            "accepted_articles_analyzed": len(articles),
            "pattern_conclusions": len(conclusions),
            "preferred_phrases": len(phrase_bank),
            "review_phrases": len(phrase_review_list),
        },
    )


def build_pattern_conclusions(
    articles: list[dict[str, Any]],
) -> list[PatternConclusion]:
    metrics = _article_metrics(articles)
    conclusions = [
        _boolean_conclusion(
            pattern_id="opening_place_prefix",
            category="opening_lede",
            conclusion=(
                "Openings often begin with a place-style prefix before the main lede; "
                "treat this as a body lede pattern, not a confirmed headline."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"]
                for metric in metrics
                if metric["opening_has_place_prefix"]
            },
        ),
        _numeric_conclusion(
            pattern_id="paragraph_sequence",
            category="structure",
            conclusion=(
                "Articles commonly use many compact paragraphs with short "
                "section-label paragraphs interspersed."
            ),
            articles=articles,
            values=[int(metric["paragraph_count"]) for metric in metrics],
            frequency=sum(
                1 for metric in metrics if int(metric["paragraph_count"]) >= 8
            ),
        ),
        _numeric_conclusion(
            pattern_id="opening_lede_length",
            category="opening_lede",
            conclusion=(
                "Opening ledes are usually long, fact-dense paragraphs rather "
                "than standalone editorial headlines."
            ),
            articles=articles,
            values=[int(metric["opening_word_count"]) for metric in metrics],
            frequency=sum(
                1 for metric in metrics if int(metric["opening_word_count"]) >= 25
            ),
        ),
        _boolean_conclusion(
            pattern_id="attribution_presence",
            category="attribution",
            conclusion=(
                "Attribution and reported-speech markers are a recurring "
                "newsroom device."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_attribution"]
            },
        ),
        _boolean_conclusion(
            pattern_id="background_context",
            category="fact_ordering",
            conclusion=(
                "Background/context is commonly introduced after the initial "
                "lede rather than before it."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_background"]
            },
        ),
        _boolean_conclusion(
            pattern_id="numbers_dates_places",
            category="factual_detail",
            conclusion=(
                "Numbers, dates, places and institutional names are frequent "
                "factual anchors."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"]
                for metric in metrics
                if metric["has_number"] or metric["has_date"]
            },
        ),
        _boolean_conclusion(
            pattern_id="transition_markers",
            category="transitions",
            conclusion=(
                "Transition markers are used to move from lede to context, "
                "reactions and implications."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_transition"]
            },
        ),
        _boolean_conclusion(
            pattern_id="cautious_or_unconfirmed_phrasing",
            category="attribution",
            conclusion=(
                "Cautious phrasing appears where claims, expectations or "
                "reported information are not presented as fully settled facts."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_caution"]
            },
        ),
        _boolean_conclusion(
            pattern_id="update_context_phrasing",
            category="fact_ordering",
            conclusion=(
                "Update and context-setting markers help place the latest "
                "development inside an ongoing story."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_update"]
            },
        ),
        _boolean_conclusion(
            pattern_id="quote_presence",
            category="quotation",
            conclusion=(
                "Direct quotation appears, but not in every article; it should "
                "be optional, evidence-driven."
            ),
            articles=articles,
            matched_ids={
                metric["article_id"] for metric in metrics if metric["has_quote"]
            },
        ),
    ]
    return conclusions


def extract_phrase_records(
    articles: list[dict[str, Any]],
    *,
    min_articles: int = 8,
) -> tuple[list[PhraseRecord], list[PhraseRecord]]:
    phrase_article_ids: dict[str, set[str]] = defaultdict(set)
    phrase_counts: Counter[str] = Counter()
    phrase_authors: dict[str, Counter[str]] = defaultdict(Counter)
    for article in articles:
        article_id = str(article["article_id"])
        author_id = str(article["author_id"])
        tokens = _tokens(_article_text(article))
        seen_in_article: set[str] = set()
        for ngram_size in (2, 3, 4):
            for phrase in _ngrams(tokens, ngram_size):
                if _is_low_value_phrase(phrase):
                    continue
                phrase_counts[phrase] += 1
                seen_in_article.add(phrase)
        for phrase in seen_in_article:
            phrase_article_ids[phrase].add(article_id)
            phrase_authors[phrase][author_id] += 1

    records: list[PhraseRecord] = []
    article_count = len(articles)
    for phrase, ids in phrase_article_ids.items():
        if len(ids) < min_articles:
            continue
        author_distribution = dict(sorted(phrase_authors[phrase].items()))
        authors = sorted(author_distribution)
        skew = max(author_distribution.values()) / len(ids)
        phrase_type = _phrase_type(phrase)
        recommendation = "preferred" if len(authors) >= 2 and skew < 0.75 else "review"
        records.append(
            PhraseRecord(
                phrase=phrase,
                phrase_type=phrase_type,
                article_count=len(ids),
                occurrence_count=phrase_counts[phrase],
                prevalence=_ratio(len(ids), article_count),
                authors_represented=authors,
                author_distribution=author_distribution,
                author_skew=round(skew, 4),
                recommendation=recommendation,
                evidence_article_ids=sorted(ids)[:5],
            )
        )

    sorted_records = sorted(
        records,
        key=lambda item: (-item.article_count, item.phrase, item.recommendation),
    )
    return (
        [record for record in sorted_records if record.recommendation == "preferred"][
            :80
        ],
        [record for record in sorted_records if record.recommendation == "review"][
            :80
        ],
    )


def build_machine_profile(
    *,
    articles: list[dict[str, Any]],
    conclusions: list[PatternConclusion],
    phrase_bank: list[PhraseRecord],
    phrase_review_list: list[PhraseRecord],
) -> dict[str, Any]:
    authors = sorted({str(article["author_id"]) for article in articles})
    high_confidence_topics = Counter(
        str(article.get("topic"))
        for article in articles
        if not article.get("topic_low_confidence")
        and not article.get("topic_multi_category_conflict")
    )
    return {
        "profile_id": "oneindia_tamil_generic_newsroom_sprint2",
        "source_article_count": len(articles),
        "authors": authors,
        "headline_metadata_warning": (
            "Corpus DOCX files do not reliably contain standalone headlines; "
            "headline_candidate is treated as body lede evidence only."
        ),
        "topic_metadata_warning": (
            "Topic labels are provisional and used only as weak metadata."
        ),
        "high_confidence_topic_distribution": dict(
            sorted(high_confidence_topics.items())
        ),
        "style_conclusions": [asdict(conclusion) for conclusion in conclusions],
        "preferred_phrase_bank": [asdict(record) for record in phrase_bank[:40]],
        "phrase_review_list": [asdict(record) for record in phrase_review_list[:40]],
    }


def write_newsroom_profile_outputs(
    *,
    paths: NewsroomProfilePathConfig,
    profile: dict[str, Any],
    conclusions: list[PatternConclusion],
    phrase_bank: list[PhraseRecord],
    phrase_review_list: list[PhraseRecord],
    articles: list[dict[str, Any]],
) -> dict[str, Path]:
    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "style_guide_markdown": (
            paths.reports_dir / "oneindia_tamil_newsroom_style_guide.md"
        ),
        "profile_json": paths.reports_dir / "oneindia_tamil_newsroom_profile.json",
        "preferred_phrase_bank_csv": paths.reports_dir / "preferred_phrase_bank.csv",
        "phrase_review_list_csv": paths.reports_dir / "phrase_review_list.csv",
        "structural_pattern_report_csv": (
            paths.reports_dir / "structural_pattern_report.csv"
        ),
        "author_commonality_report_csv": (
            paths.reports_dir / "author_commonality_report.csv"
        ),
        "evidence_jsonl": paths.reports_dir / "newsroom_profile_evidence.jsonl",
    }
    output_paths["profile_json"].write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_style_guide(output_paths["style_guide_markdown"], profile, conclusions)
    _write_phrase_csv(output_paths["preferred_phrase_bank_csv"], phrase_bank)
    _write_phrase_csv(output_paths["phrase_review_list_csv"], phrase_review_list)
    _write_structural_csv(output_paths["structural_pattern_report_csv"], conclusions)
    _write_author_commonality_csv(
        output_paths["author_commonality_report_csv"],
        phrase_bank + phrase_review_list,
    )
    _write_evidence_jsonl(output_paths["evidence_jsonl"], conclusions, articles)
    return output_paths


def _article_metrics(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metrics = []
    for article in articles:
        paragraphs = _paragraphs(article)
        text = _article_text(article)
        opening = paragraphs[0] if paragraphs else ""
        closing = paragraphs[-1] if paragraphs else ""
        normalized_text = normalize_for_comparison(text)
        after_opening = "\n".join(paragraphs[1:])
        metrics.append(
            {
                "article_id": str(article["article_id"]),
                "paragraph_count": len(paragraphs),
                "opening_word_count": len(_tokens(opening)),
                "closing_word_count": len(_tokens(closing)),
                "opening_has_place_prefix": bool(PLACE_PREFIX_PATTERN.search(opening)),
                "has_attribution": _contains_any(normalized_text, ATTRIBUTION_MARKERS),
                "has_background": _contains_any(
                    normalize_for_comparison(after_opening),
                    BACKGROUND_MARKERS,
                ),
                "has_number": bool(NUMBER_PATTERN.search(text)),
                "has_date": bool(DATE_PATTERN.search(text)),
                "has_transition": _contains_any(normalized_text, TRANSITION_MARKERS),
                "has_quote": bool(QUOTE_PATTERN.search(text)),
                "has_caution": _contains_any(normalized_text, CAUTION_MARKERS),
                "has_update": _contains_any(normalized_text, UPDATE_MARKERS),
            }
        )
    return metrics


def _boolean_conclusion(
    *,
    pattern_id: str,
    category: str,
    conclusion: str,
    articles: list[dict[str, Any]],
    matched_ids: set[str],
) -> PatternConclusion:
    matched_articles = [
        article for article in articles if str(article["article_id"]) in matched_ids
    ]
    return PatternConclusion(
        pattern_id=pattern_id,
        category=category,
        conclusion=conclusion,
        frequency=len(matched_ids),
        prevalence=_ratio(len(matched_ids), len(articles)),
        authors_represented=sorted(
            {str(article["author_id"]) for article in matched_articles}
        ),
        confidence=_confidence(len(matched_ids), len(articles), matched_articles),
        evidence_samples=_evidence_samples(matched_articles),
    )


def _numeric_conclusion(
    *,
    pattern_id: str,
    category: str,
    conclusion: str,
    articles: list[dict[str, Any]],
    values: list[int],
    frequency: int,
) -> PatternConclusion:
    enriched = f"{conclusion} Median={median(values) if values else 0}."
    matched_articles = articles[:]
    return PatternConclusion(
        pattern_id=pattern_id,
        category=category,
        conclusion=enriched,
        frequency=frequency,
        prevalence=_ratio(frequency, len(articles)),
        authors_represented=sorted({str(article["author_id"]) for article in articles}),
        confidence=_confidence(frequency, len(articles), matched_articles),
        evidence_samples=_evidence_samples(matched_articles),
    )


def _confidence(
    frequency: int,
    total: int,
    matched_articles: list[dict[str, Any]],
) -> str:
    author_count = len({str(article["author_id"]) for article in matched_articles})
    prevalence = _ratio(frequency, total)
    if author_count == 3 and prevalence >= 0.45:
        return "high"
    if author_count >= 2 and prevalence >= 0.15:
        return "medium"
    return "low"


def _evidence_samples(
    articles: list[dict[str, Any]],
    limit: int = 5,
) -> list[EvidenceSample]:
    samples = []
    seen_authors: set[str] = set()
    for article in sorted(articles, key=lambda item: str(item["article_id"])):
        author_id = str(article["author_id"])
        if author_id in seen_authors and len(seen_authors) < 3:
            continue
        seen_authors.add(author_id)
        samples.append(_sample(article))
        if len(samples) >= limit:
            return samples
    for article in sorted(articles, key=lambda item: str(item["article_id"])):
        sample = _sample(article)
        if sample not in samples:
            samples.append(sample)
        if len(samples) >= limit:
            break
    return samples


def _sample(article: dict[str, Any]) -> EvidenceSample:
    return EvidenceSample(
        article_id=str(article["article_id"]),
        author_id=str(article["author_id"]),
        relative_source_path=str(article["relative_source_path"]),
        excerpt=_excerpt(_article_text(article)),
    )


def _merge_classification(
    article: dict[str, Any],
    classified: dict[str, Any] | None,
) -> dict[str, Any]:
    merged = dict(article)
    if classified:
        for key in (
            "topic",
            "topic_confidence",
            "topic_low_confidence",
            "topic_multi_category_conflict",
            "topic_evidence",
        ):
            merged[key] = classified.get(key)
    return merged


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _article_text(article: dict[str, Any]) -> str:
    return str(article.get("body_text") or "")


def _paragraphs(article: dict[str, Any]) -> list[str]:
    paragraphs = article.get("paragraph_sequence") or []
    if not isinstance(paragraphs, list):
        return []
    return [str(paragraph) for paragraph in paragraphs if str(paragraph).strip()]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (
            raw.strip(TOKEN_STRIP_CHARS)
            for raw in normalize_for_comparison(text).split()
        )
        if token
    ]


def _ngrams(tokens: list[str], size: int) -> list[str]:
    return [
        " ".join(tokens[index : index + size])
        for index in range(0, max(0, len(tokens) - size + 1))
    ]


def _is_low_value_phrase(phrase: str) -> bool:
    tokens = phrase.split()
    if len(set(tokens)) == 1:
        return True
    if any(len(token) <= 1 for token in tokens):
        return True
    return False


def _phrase_type(phrase: str) -> str:
    normalized = normalize_for_comparison(phrase)
    if _contains_any(normalized, ATTRIBUTION_MARKERS):
        return "attribution"
    if _contains_any(normalized, CAUTION_MARKERS):
        return "cautious_or_unconfirmed"
    if _contains_any(normalized, UPDATE_MARKERS):
        return "update_context"
    if _contains_any(normalized, TRANSITION_MARKERS):
        return "transition"
    return "common_newsroom"


def _contains_any(text: str, markers: list[str]) -> bool:
    return any(normalize_for_comparison(marker) in text for marker in markers)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _excerpt(text: str, limit: int = 240) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _write_style_guide(
    path: Path,
    profile: dict[str, Any],
    conclusions: list[PatternConclusion],
) -> None:
    lines = [
        "# Oneindia Tamil Generic Newsroom Style Guide",
        "",
        "This profile is deterministic and corpus-grounded. It does not use LLM calls.",
        "",
        "## Metadata Limits",
        "",
        f"- {profile['headline_metadata_warning']}",
        f"- {profile['topic_metadata_warning']}",
        "",
        "## Style Rules",
        "",
    ]
    for conclusion in conclusions:
        lines.extend(
            [
                f"### {conclusion.category}: {conclusion.pattern_id}",
                "",
                conclusion.conclusion,
                "",
                (
                    f"Evidence: {conclusion.frequency} articles "
                    f"({conclusion.prevalence:.1%}); authors="
                    f"{', '.join(conclusion.authors_represented)}; "
                    f"confidence={conclusion.confidence}."
                ),
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_phrase_csv(path: Path, records: list[PhraseRecord]) -> None:
    fieldnames = [
        "phrase",
        "phrase_type",
        "article_count",
        "occurrence_count",
        "prevalence",
        "authors_represented",
        "author_distribution",
        "author_skew",
        "recommendation",
        "evidence_article_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["authors_represented"] = "|".join(record.authors_represented)
            row["author_distribution"] = json.dumps(
                record.author_distribution,
                ensure_ascii=False,
                sort_keys=True,
            )
            row["evidence_article_ids"] = "|".join(record.evidence_article_ids)
            writer.writerow(row)


def _write_structural_csv(
    path: Path,
    conclusions: list[PatternConclusion],
) -> None:
    fieldnames = [
        "pattern_id",
        "category",
        "conclusion",
        "frequency",
        "prevalence",
        "authors_represented",
        "confidence",
        "evidence_article_ids",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for conclusion in conclusions:
            writer.writerow(
                {
                    "pattern_id": conclusion.pattern_id,
                    "category": conclusion.category,
                    "conclusion": conclusion.conclusion,
                    "frequency": conclusion.frequency,
                    "prevalence": conclusion.prevalence,
                    "authors_represented": "|".join(conclusion.authors_represented),
                    "confidence": conclusion.confidence,
                    "evidence_article_ids": "|".join(
                        sample.article_id for sample in conclusion.evidence_samples
                    ),
                }
            )


def _write_author_commonality_csv(
    path: Path,
    records: list[PhraseRecord],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "phrase",
                "authors_represented",
                "author_distribution",
                "author_skew",
                "recommendation",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "phrase": record.phrase,
                    "authors_represented": "|".join(record.authors_represented),
                    "author_distribution": json.dumps(
                        record.author_distribution,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "author_skew": record.author_skew,
                    "recommendation": record.recommendation,
                }
            )


def _write_evidence_jsonl(
    path: Path,
    conclusions: list[PatternConclusion],
    articles: list[dict[str, Any]],
) -> None:
    article_by_id = {str(article["article_id"]): article for article in articles}
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for conclusion in conclusions:
            for sample in conclusion.evidence_samples:
                article = article_by_id[sample.article_id]
                handle.write(
                    json.dumps(
                        {
                            "pattern_id": conclusion.pattern_id,
                            "article_id": sample.article_id,
                            "author_id": sample.author_id,
                            "relative_source_path": sample.relative_source_path,
                            "topic": article.get("topic"),
                            "topic_confidence": article.get("topic_confidence"),
                            "excerpt": sample.excerpt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
