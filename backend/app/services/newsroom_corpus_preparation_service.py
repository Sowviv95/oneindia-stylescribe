"""Deterministic preparation for extracted newsroom corpus articles."""

from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any

from backend.app.services.newsroom_corpus_service import (
    DEFAULT_EXTRACTED_DIR,
    DEFAULT_REJECTED_DIR,
    DEFAULT_REPORTS_DIR,
)

DEFAULT_CLEANED_DIR = Path("data/newsroom_corpus/02_cleaned")
DEFAULT_CLASSIFIED_DIR = Path("data/newsroom_corpus/04_classified")
DEFAULT_ARTICLES_JSONL = DEFAULT_EXTRACTED_DIR / "articles.jsonl"

ZERO_WIDTH_PATTERN = re.compile("[\u200b\u200c\u200d\u200e\u200f\ufeff]")
TOKEN_STRIP_CHARS = " \t\n\r.,;:!?()[]{}\"'“”‘’…|/\\<>"
LONG_HEADLINE_WORDS = 35
LONG_SUBHEADLINE_WORDS = 45
LARGE_SINGLE_PARAGRAPH_WORDS = 150
MISSING_BODY_MAX_WORDS = 25
LOW_TOPIC_CONFIDENCE = 0.45
MIN_TOPIC_EVIDENCE = 2

TOPIC_CATEGORIES = [
    "politics",
    "national",
    "state_and_local",
    "crime",
    "weather",
    "cinema",
    "sports",
    "business",
    "technology",
    "education",
    "health",
    "lifestyle",
    "spirituality_and_astrology",
    "agriculture",
    "jobs_and_careers",
    "international",
    "human_interest",
    "other",
]

TOPIC_KEYWORDS = {
    "politics": [
        "அரசியல்",
        "திமுக",
        "அதிமுக",
        "பாஜக",
        "காங்கிரஸ்",
        "முதல்வர்",
        "அமைச்சர்",
        "தேர்தல்",
        "வாக்கு",
        "கூட்டணி",
        "எம்எல்ஏ",
        "எம்பி",
        "government",
        "election",
        "minister",
    ],
    "national": [
        "இந்தியா",
        "மத்திய அரசு",
        "பிரதமர்",
        "டெல்லி",
        "நாடாளுமன்ற",
        "லோக்சபா",
        "உச்ச நீதிமன்ற",
        "supreme court",
    ],
    "state_and_local": [
        "தமிழகம்",
        "தமிழ்நாடு",
        "சென்னை",
        "மதுரை",
        "கோவை",
        "திருச்சி",
        "கிருஷ்ணகிரி",
        "மாவட்டம்",
        "கிராமம்",
        "நகராட்சி",
    ],
    "crime": [
        "கொலை",
        "கொள்ளை",
        "கைது",
        "போலீஸ்",
        "வழக்கு",
        "விசாரணை",
        "பாலியல்",
        "மோசடி",
        "தாக்குதல்",
        "crime",
    ],
    "weather": [
        "மழை",
        "வானிலை",
        "புயல்",
        "வெள்ளம்",
        "வெயில்",
        "வெப்ப",
        "காற்றழுத்த",
        "weather",
        "rain",
    ],
    "cinema": [
        "சினிமா",
        "நடிகர்",
        "நடிகை",
        "திரைப்பட",
        "விஜய்",
        "அஜித்",
        "ரஜினி",
        "கமல்",
        "பாலிவுட்",
        "movie",
        "film",
    ],
    "sports": [
        "கிரிக்கெட்",
        "கால்பந்து",
        "விளையாட்டு",
        "ஐபிஎல்",
        "போட்டி",
        "வீரர்",
        "cricket",
        "match",
    ],
    "business": [
        "பொருளாதாரம்",
        "பங்குச்சந்தை",
        "ரூபாய்",
        "வரி",
        "விலை",
        "நிதி",
        "வங்கி",
        "பட்ஜெட்",
        "business",
        "market",
    ],
    "technology": [
        "தொழில்நுட்ப",
        "செயற்கை நுண்ணறிவு",
        "ஏஐ",
        "மொபைல்",
        "இணைய",
        "யுபிஐ",
        "டிஜிட்டல்",
        "technology",
        "ai",
        "upi",
    ],
    "education": [
        "கல்வி",
        "பள்ளி",
        "கல்லூரி",
        "மாணவர்",
        "தேர்வு",
        "நீட்",
        "பல்கலை",
        "education",
    ],
    "health": [
        "மருத்துவ",
        "சுகாதார",
        "மருத்துவர்",
        "நோய்",
        "மருந்து",
        "மருத்துவமனை",
        "health",
    ],
    "lifestyle": [
        "லைஃப்ஸ்டைல்",
        "உணவு",
        "சமையல்",
        "அழகு",
        "பயணம்",
        "வீடு",
        "lifestyle",
    ],
    "spirituality_and_astrology": [
        "ராசி",
        "ஜோதிடம்",
        "ஆன்மீகம்",
        "கோவில்",
        "பூஜை",
        "குரு",
        "சனி",
        "horoscope",
    ],
    "agriculture": [
        "விவசாய",
        "விவசாயி",
        "பயிர்",
        "நெல்",
        "வேளாண்",
        "agriculture",
    ],
    "jobs_and_careers": [
        "வேலை",
        "வேலைவாய்ப்பு",
        "பணி",
        "சம்பளம்",
        "ஊதியம்",
        "தேர்வு அறிவிப்பு",
        "jobs",
        "career",
    ],
    "international": [
        "அமெரிக்கா",
        "சீனா",
        "ரஷ்யா",
        "பாகிஸ்தான்",
        "இலங்கை",
        "இஸ்ரேல்",
        "உலக",
        "international",
        "world",
    ],
    "human_interest": [
        "மக்கள்",
        "குடும்பம்",
        "குழந்தை",
        "மூதாட்டி",
        "உதவி",
        "வாழ்வாதாரம்",
        "நெகிழ்ச்சி",
    ],
}

BOILERPLATE_PATTERNS = [
    "read more",
    "advertisement",
    "story first published",
    "subscribe",
    "மேலும் படிக்க",
    "விளம்பரம்",
]


@dataclass(frozen=True)
class PreparationPathConfig:
    articles_jsonl: Path = DEFAULT_ARTICLES_JSONL
    cleaned_dir: Path = DEFAULT_CLEANED_DIR
    rejected_dir: Path = DEFAULT_REJECTED_DIR
    classified_dir: Path = DEFAULT_CLASSIFIED_DIR
    reports_dir: Path = DEFAULT_REPORTS_DIR


@dataclass(frozen=True)
class StructuralProfile:
    article_id: str
    author_id: str
    word_count: int
    char_count: int
    paragraph_count: int
    headline_present: bool
    subheadline_present: bool
    headline_word_count: int
    subheadline_word_count: int
    headline_char_count: int
    subheadline_char_count: int
    headline_status: str
    subheadline_status: str
    structure_confidence: str
    informational_flags: list[str]
    structure_warning_flags: list[str]
    content_review_flags: list[str]
    rejection_flags: list[str]
    anomaly_flags: list[str]
    anomaly_evidence: list[str]


@dataclass(frozen=True)
class TopicResult:
    topic: str
    confidence: float
    matched_evidence: list[str]
    review_flag: bool
    low_confidence: bool
    multi_category_conflict: bool


@dataclass(frozen=True)
class NearDuplicateCluster:
    cluster_id: str
    canonical_article_id: str
    article_ids: list[str]
    similarity_evidence: list[dict[str, object]]


@dataclass(frozen=True)
class PreparedArticle:
    article_id: str
    author_id: str
    source_filename: str
    relative_source_path: str
    file_sha256: str
    text_sha256: str
    normalized_text_sha256: str
    preparation_status: str
    article_usable: bool
    structure_confidence: str
    headline_status: str
    subheadline_status: str
    decision_reasons: list[str]
    review_flags: list[str]
    informational_flags: list[str]
    structure_warning_flags: list[str]
    content_review_flags: list[str]
    rejection_flags: list[str]
    duplicate_cluster_id: str | None
    canonical_article_id: str | None
    topic: str
    topic_confidence: float
    topic_evidence: list[str]
    topic_review_flag: bool
    topic_low_confidence: bool
    topic_multi_category_conflict: bool
    word_count: int
    char_count: int
    paragraph_count: int
    headline: str | None
    headline_candidate: str | None
    subheadline: str | None
    body_text: str
    paragraph_sequence: list[str]


@dataclass(frozen=True)
class CorpusPreparationResult:
    prepared_articles: list[PreparedArticle]
    structural_profiles: list[StructuralProfile]
    near_duplicate_clusters: list[NearDuplicateCluster]
    output_paths: dict[str, Path]
    summary: dict[str, int]


def run_newsroom_corpus_preparation(
    *,
    paths: PreparationPathConfig | None = None,
    mode: str = "prepare",
) -> CorpusPreparationResult:
    paths = paths or PreparationPathConfig()
    articles = load_articles(paths.articles_jsonl)
    profiles = [profile_article(article) for article in articles]
    clusters = detect_near_duplicate_clusters(articles)
    topics = {
        str(article["article_id"]): classify_topic(article)
        for article in articles
    }
    prepared = prepare_articles(articles, profiles, clusters, topics)
    output_paths = write_preparation_outputs(
        paths=paths,
        prepared_articles=prepared,
        structural_profiles=profiles,
        near_duplicate_clusters=clusters,
        mode=mode,
    )
    return CorpusPreparationResult(
        prepared_articles=prepared,
        structural_profiles=profiles,
        near_duplicate_clusters=clusters,
        output_paths=output_paths,
        summary=preparation_summary(prepared, clusters),
    )


def load_articles(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def normalize_for_comparison(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = ZERO_WIDTH_PATTERN.sub("", normalized)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(" ".join(line.split()) for line in normalized.split("\n"))
    normalized = "\n".join(line for line in normalized.split("\n") if line)
    return normalized.casefold().strip()


def profile_article(article: dict[str, object]) -> StructuralProfile:
    headline = str(article.get("headline") or "")
    subheadline = str(article.get("subheadline") or "")
    body_text = str(article.get("body_text") or "")
    paragraphs = _paragraphs(article)
    text = _article_text(article)
    headline_status = str(article.get("headline_status") or "uncertain")
    subheadline_status = str(article.get("subheadline_status") or "not_present")
    structure_confidence = str(article.get("structure_confidence") or "low")
    normalized_headline = normalize_for_comparison(headline)
    normalized_body = normalize_for_comparison(body_text)
    informational: list[str] = []
    structure_warnings: list[str] = []
    content_review: list[str] = []
    rejection: list[str] = []
    evidence: list[str] = []

    headline_words = _word_count(headline)
    subheadline_words = _word_count(subheadline)
    body_words = _word_count(body_text)
    text_words = _word_count(text)

    if headline_status != "explicit":
        informational.append("no_explicit_headline")
    if not headline.strip():
        informational.append("no_headline")
    if not subheadline.strip() or subheadline_status == "not_present":
        informational.append("no_subheadline")
    if text_words >= 700:
        informational.append("long_article")

    if not body_text.strip() or body_words <= MISSING_BODY_MAX_WORDS:
        if text_words <= MISSING_BODY_MAX_WORDS:
            rejection.append("no_meaningful_body")
        else:
            content_review.append("uncertain_body_boundary")

    if headline_status == "uncertain":
        structure_warnings.append("uncertain_headline_body_boundary")
    if headline_words > LONG_HEADLINE_WORDS:
        structure_warnings.append("long_headline_candidate")
        evidence.append(f"headline_words={headline_words}")
    if subheadline_words > LONG_SUBHEADLINE_WORDS:
        structure_warnings.append("long_subheadline_candidate")
        evidence.append(f"subheadline_words={subheadline_words}")
    if (
        normalized_headline
        and normalized_body
        and normalized_headline[:120] in normalized_body[:400]
    ):
        structure_warnings.append("possible_headline_repeated_in_body")
    if len(paragraphs) == 1 and _word_count(text) >= LARGE_SINGLE_PARAGRAPH_WORDS:
        informational.append("large_single_paragraph")
    boilerplate = _boilerplate_evidence(text)
    if boilerplate:
        informational.append("boilerplate_evidence")
        evidence.extend(boilerplate[:5])
    markup_ratio = _markup_paragraph_ratio(paragraphs)
    if markup_ratio >= 0.4:
        content_review.append("malformed_or_mixed_content")
        evidence.append(f"markup_paragraph_ratio={markup_ratio:.2f}")

    anomaly_flags = sorted(
        set(informational + structure_warnings + content_review + rejection)
    )

    return StructuralProfile(
        article_id=str(article["article_id"]),
        author_id=str(article["author_id"]),
        word_count=text_words,
        char_count=len(text),
        paragraph_count=len(paragraphs),
        headline_present=bool(headline.strip()),
        subheadline_present=bool(subheadline.strip()),
        headline_word_count=headline_words,
        subheadline_word_count=subheadline_words,
        headline_char_count=len(headline),
        subheadline_char_count=len(subheadline),
        headline_status=headline_status,
        subheadline_status=subheadline_status,
        structure_confidence=structure_confidence,
        informational_flags=sorted(set(informational)),
        structure_warning_flags=sorted(set(structure_warnings)),
        content_review_flags=sorted(set(content_review)),
        rejection_flags=sorted(set(rejection)),
        anomaly_flags=anomaly_flags,
        anomaly_evidence=evidence,
    )


def detect_near_duplicate_clusters(
    articles: list[dict[str, object]],
    *,
    token_jaccard_threshold: float = 0.82,
    headline_similarity_threshold: float = 0.9,
    headline_body_jaccard_threshold: float = 0.55,
) -> list[NearDuplicateCluster]:
    token_sets = {
        str(article["article_id"]): set(_tokens(_article_text(article)))
        for article in articles
    }
    headline_text = {
        str(article["article_id"]): normalize_for_comparison(
            str(article.get("headline") or "")
        )
        for article in articles
    }
    article_by_id = {str(article["article_id"]): article for article in articles}
    ids = sorted(article_by_id)
    parent = {article_id: article_id for article_id in ids}
    evidence_by_pair: dict[tuple[str, str], dict[str, object]] = {}
    candidates = _simhash_candidates(token_sets)
    candidates.update(_headline_candidates(headline_text))

    for left_id, right_id in sorted(candidates):
        left_tokens = token_sets[left_id]
        if not left_tokens:
            continue
        right_tokens = token_sets[right_id]
        if not right_tokens:
            continue
        token_jaccard = _jaccard(left_tokens, right_tokens)
        headline_similarity = SequenceMatcher(
            None,
            headline_text[left_id],
            headline_text[right_id],
        ).ratio()
        is_duplicate = token_jaccard >= token_jaccard_threshold or (
            headline_similarity >= headline_similarity_threshold
            and token_jaccard >= headline_body_jaccard_threshold
        )
        if not is_duplicate:
            continue
        _union(parent, left_id, right_id)
        evidence_by_pair[(left_id, right_id)] = {
            "article_id_a": left_id,
            "article_id_b": right_id,
            "token_jaccard": round(token_jaccard, 4),
            "headline_similarity": round(headline_similarity, 4),
        }

    grouped: dict[str, list[str]] = defaultdict(list)
    for article_id in ids:
        grouped[_find(parent, article_id)].append(article_id)

    clusters: list[NearDuplicateCluster] = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        sorted_members = sorted(members)
        canonical = recommend_canonical_article(
            [article_by_id[article_id] for article_id in sorted_members]
        )
        digest = sha256("|".join(sorted_members).encode()).hexdigest()[:12]
        cluster_id = f"near_duplicate:{digest}"
        pair_evidence = [
            evidence
            for pair, evidence in sorted(evidence_by_pair.items())
            if pair[0] in sorted_members and pair[1] in sorted_members
        ]
        clusters.append(
            NearDuplicateCluster(
                cluster_id=cluster_id,
                canonical_article_id=canonical,
                article_ids=sorted_members,
                similarity_evidence=pair_evidence,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.cluster_id)


def recommend_canonical_article(articles: list[dict[str, object]]) -> str:
    ranked = sorted(
        articles,
        key=lambda article: (
            -_word_count(_article_text(article)),
            -len(_paragraphs(article)),
            str(article["article_id"]),
        ),
    )
    return str(ranked[0]["article_id"])


def classify_topic(article: dict[str, object]) -> TopicResult:
    text = normalize_for_comparison(_article_text(article))
    scores: dict[str, int] = {}
    evidence: dict[str, list[str]] = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        matches = []
        for keyword in keywords:
            normalized_keyword = normalize_for_comparison(keyword)
            if normalized_keyword and normalized_keyword in text:
                matches.append(keyword)
        if matches:
            scores[topic] = len(matches)
            evidence[topic] = matches[:8]

    if not scores:
        return TopicResult(
            topic="other",
            confidence=0.0,
            matched_evidence=[],
            review_flag=True,
            low_confidence=True,
            multi_category_conflict=False,
        )

    best_topic, best_score = sorted(
        scores.items(),
        key=lambda item: (-item[1], TOPIC_CATEGORIES.index(item[0])),
    )[0]
    total_score = sum(scores.values())
    confidence = round(best_score / total_score, 3)
    tied = sum(1 for score in scores.values() if score == best_score) > 1
    low_confidence = confidence < LOW_TOPIC_CONFIDENCE
    if best_score < MIN_TOPIC_EVIDENCE:
        return TopicResult(
            topic="other",
            confidence=confidence,
            matched_evidence=evidence[best_topic],
            review_flag=True,
            low_confidence=True,
            multi_category_conflict=tied,
        )
    return TopicResult(
        topic=best_topic,
        confidence=confidence,
        matched_evidence=evidence[best_topic],
        review_flag=low_confidence or tied,
        low_confidence=low_confidence,
        multi_category_conflict=tied,
    )


def prepare_articles(
    articles: list[dict[str, object]],
    profiles: list[StructuralProfile],
    clusters: list[NearDuplicateCluster],
    topics: dict[str, TopicResult],
) -> list[PreparedArticle]:
    profile_by_id = {profile.article_id: profile for profile in profiles}
    cluster_by_article_id: dict[str, NearDuplicateCluster] = {}
    for duplicate_cluster in clusters:
        for article_id in duplicate_cluster.article_ids:
            cluster_by_article_id[article_id] = duplicate_cluster

    prepared: list[PreparedArticle] = []
    for article in sorted(articles, key=lambda item: str(item["article_id"])):
        article_id = str(article["article_id"])
        profile = profile_by_id[article_id]
        topic = topics[article_id]
        cluster: NearDuplicateCluster | None = cluster_by_article_id.get(article_id)
        reasons: list[str] = []
        review_flags: list[str] = []
        status = "accepted"
        article_usable = not profile.rejection_flags

        if profile.rejection_flags:
            status = "rejected"
            reasons.extend(profile.rejection_flags)
        elif cluster is not None and article_id != cluster.canonical_article_id:
            status = "rejected"
            article_usable = False
            reasons.append("near_duplicate_non_canonical")
        else:
            if profile.content_review_flags:
                status = "review_required"
                reasons.extend(profile.content_review_flags)
                review_flags.extend(profile.content_review_flags)

        text = _article_text(article)
        prepared.append(
            PreparedArticle(
                article_id=article_id,
                author_id=str(article["author_id"]),
                source_filename=str(article["source_filename"]),
                relative_source_path=str(article["relative_source_path"]),
                file_sha256=str(article["file_sha256"]),
                text_sha256=str(article["text_sha256"]),
                normalized_text_sha256=sha256(
                    normalize_for_comparison(text).encode()
                ).hexdigest(),
                preparation_status=status,
                article_usable=article_usable,
                structure_confidence=profile.structure_confidence,
                headline_status=profile.headline_status,
                subheadline_status=profile.subheadline_status,
                decision_reasons=sorted(set(reasons)),
                review_flags=sorted(set(review_flags)),
                informational_flags=profile.informational_flags,
                structure_warning_flags=profile.structure_warning_flags,
                content_review_flags=profile.content_review_flags,
                rejection_flags=profile.rejection_flags,
                duplicate_cluster_id=cluster.cluster_id if cluster else None,
                canonical_article_id=(
                    cluster.canonical_article_id if cluster else article_id
                ),
                topic=topic.topic,
                topic_confidence=topic.confidence,
                topic_evidence=topic.matched_evidence,
                topic_review_flag=topic.review_flag,
                topic_low_confidence=topic.low_confidence,
                topic_multi_category_conflict=topic.multi_category_conflict,
                word_count=profile.word_count,
                char_count=profile.char_count,
                paragraph_count=profile.paragraph_count,
                headline=(
                    str(article.get("headline")) if article.get("headline") else None
                ),
                headline_candidate=(
                    str(article.get("headline_candidate"))
                    if article.get("headline_candidate")
                    else None
                ),
                subheadline=(
                    str(article.get("subheadline"))
                    if article.get("subheadline")
                    else None
                ),
                body_text=str(article.get("body_text") or ""),
                paragraph_sequence=_paragraphs(article),
            )
        )
    return prepared


def write_preparation_outputs(
    *,
    paths: PreparationPathConfig,
    prepared_articles: list[PreparedArticle],
    structural_profiles: list[StructuralProfile],
    near_duplicate_clusters: list[NearDuplicateCluster],
    mode: str,
) -> dict[str, Path]:
    paths.cleaned_dir.mkdir(parents=True, exist_ok=True)
    paths.rejected_dir.mkdir(parents=True, exist_ok=True)
    paths.classified_dir.mkdir(parents=True, exist_ok=True)
    paths.reports_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "cleaned_articles_jsonl": paths.cleaned_dir / "cleaned_articles.jsonl",
        "review_required_jsonl": paths.cleaned_dir / "review_required_articles.jsonl",
        "rejected_articles_jsonl": paths.rejected_dir / "rejected_articles.jsonl",
        "classified_articles_jsonl": paths.classified_dir / "classified_articles.jsonl",
        "near_duplicate_clusters_jsonl": (
            paths.reports_dir / "near_duplicate_clusters.jsonl"
        ),
        "cleaning_decisions_csv": paths.reports_dir / "cleaning_decisions.csv",
        "structural_anomaly_csv": paths.reports_dir / "structural_anomalies.csv",
        "review_reason_distribution_csv": (
            paths.reports_dir / "review_reason_distribution.csv"
        ),
        "topic_distribution_csv": paths.reports_dir / "topic_distribution.csv",
        "author_topic_distribution_csv": (
            paths.reports_dir / "author_topic_distribution.csv"
        ),
        "length_distribution_csv": paths.reports_dir / "length_distribution.csv",
        "sprint1_report_markdown": paths.reports_dir / "sprint1_corpus_report.md",
    }

    _write_jsonl(
        output_paths["cleaned_articles_jsonl"],
        [
            article
            for article in prepared_articles
            if article.preparation_status == "accepted"
        ],
    )
    _write_jsonl(
        output_paths["review_required_jsonl"],
        [
            article
            for article in prepared_articles
            if article.preparation_status == "review_required"
        ],
    )
    _write_jsonl(
        output_paths["rejected_articles_jsonl"],
        [
            article
            for article in prepared_articles
            if article.preparation_status == "rejected"
        ],
    )
    _write_jsonl(output_paths["classified_articles_jsonl"], prepared_articles)
    _write_jsonl(output_paths["near_duplicate_clusters_jsonl"], near_duplicate_clusters)
    _write_cleaning_decisions_csv(
        output_paths["cleaning_decisions_csv"],
        prepared_articles,
    )
    _write_structural_anomalies_csv(
        output_paths["structural_anomaly_csv"],
        structural_profiles,
    )
    _write_review_reason_distribution_csv(
        output_paths["review_reason_distribution_csv"],
        prepared_articles,
    )
    _write_topic_distribution_csv(
        output_paths["topic_distribution_csv"],
        prepared_articles,
    )
    _write_author_topic_distribution_csv(
        output_paths["author_topic_distribution_csv"],
        prepared_articles,
    )
    _write_length_distribution_csv(
        output_paths["length_distribution_csv"],
        prepared_articles,
    )
    _write_sprint_report(
        output_paths["sprint1_report_markdown"],
        prepared_articles,
        near_duplicate_clusters,
        mode,
    )
    return output_paths


def preparation_summary(
    prepared_articles: list[PreparedArticle],
    clusters: list[NearDuplicateCluster],
) -> dict[str, int]:
    status_counts = Counter(article.preparation_status for article in prepared_articles)
    topic_counts = Counter(article.topic for article in prepared_articles)
    headline_counts = Counter(article.headline_status for article in prepared_articles)
    summary = {
        "articles_seen": len(prepared_articles),
        "accepted_count": status_counts["accepted"],
        "review_required_count": status_counts["review_required"],
        "rejected_count": status_counts["rejected"],
        "usable_article_count": sum(
            1 for article in prepared_articles if article.article_usable
        ),
        "explicit_headline_count": headline_counts["explicit"],
        "inferred_headline_count": headline_counts["inferred"],
        "no_headline_count": headline_counts["not_present"],
        "uncertain_headline_count": headline_counts["uncertain"],
        "low_confidence_topic_count": sum(
            1 for article in prepared_articles if article.topic_low_confidence
        ),
        "multi_category_conflict_count": sum(
            1 for article in prepared_articles if article.topic_multi_category_conflict
        ),
        "near_duplicate_cluster_count": len(clusters),
        "near_duplicate_article_count": sum(
            len(cluster.article_ids) for cluster in clusters
        ),
    }
    for topic in TOPIC_CATEGORIES:
        summary[f"topic_{topic}"] = topic_counts[topic]
    return summary


def _article_text(article: dict[str, object]) -> str:
    parts = [
        str(article.get("headline") or ""),
        str(article.get("subheadline") or ""),
        str(article.get("body_text") or ""),
    ]
    return "\n".join(part for part in parts if part.strip())


def _paragraphs(article: dict[str, object]) -> list[str]:
    paragraphs = article.get("paragraph_sequence") or []
    if not isinstance(paragraphs, list):
        return []
    return [str(paragraph) for paragraph in paragraphs if str(paragraph).strip()]


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in (
            raw_token.strip(TOKEN_STRIP_CHARS)
            for raw_token in normalize_for_comparison(text).split()
        )
        if token
    ]


def _word_count(text: str) -> int:
    return len(_tokens(text))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _simhash_candidates(token_sets: dict[str, set[str]]) -> set[tuple[str, str]]:
    buckets: dict[tuple[int, int], list[str]] = defaultdict(list)
    candidates: set[tuple[str, str]] = set()
    for article_id, tokens in token_sets.items():
        signature = _simhash(tokens)
        for band_index in range(4):
            band = (signature >> (band_index * 16)) & 0xFFFF
            buckets[(band_index, band)].append(article_id)
    for article_ids in buckets.values():
        if len(article_ids) < 2:
            continue
        sorted_ids = sorted(article_ids)
        for left_index, left_id in enumerate(sorted_ids):
            for right_id in sorted_ids[left_index + 1 :]:
                candidates.add((left_id, right_id))
    return candidates


def _headline_candidates(headline_text: dict[str, str]) -> set[tuple[str, str]]:
    buckets: dict[str, list[str]] = defaultdict(list)
    for article_id, headline in headline_text.items():
        if headline:
            buckets[headline].append(article_id)
    candidates: set[tuple[str, str]] = set()
    for article_ids in buckets.values():
        if len(article_ids) < 2:
            continue
        sorted_ids = sorted(article_ids)
        for left_index, left_id in enumerate(sorted_ids):
            for right_id in sorted_ids[left_index + 1 :]:
                candidates.add((left_id, right_id))
    return candidates


def _simhash(tokens: set[str]) -> int:
    vector = [0] * 64
    for token in tokens:
        digest = int(sha256(token.encode()).hexdigest()[:16], 16)
        for bit_index in range(64):
            if digest & (1 << bit_index):
                vector[bit_index] += 1
            else:
                vector[bit_index] -= 1
    signature = 0
    for bit_index, value in enumerate(vector):
        if value >= 0:
            signature |= 1 << bit_index
    return signature


def _boilerplate_evidence(text: str) -> list[str]:
    normalized = normalize_for_comparison(text)
    return [pattern for pattern in BOILERPLATE_PATTERNS if pattern in normalized]


def _markup_paragraph_ratio(paragraphs: list[str]) -> float:
    if not paragraphs:
        return 0.0
    markup_count = sum(1 for paragraph in paragraphs if _looks_like_markup(paragraph))
    return markup_count / len(paragraphs)


def _looks_like_markup(paragraph: str) -> bool:
    normalized = normalize_for_comparison(paragraph)
    return (
        normalized.startswith("<")
        or "gmail-" in normalized
        or "class=" in normalized
        or "style=" in normalized
        or normalized in {"&nbsp;", "<br>"}
    )


def _find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def _union(parent: dict[str, str], left: str, right: str) -> None:
    left_root = _find(parent, left)
    right_root = _find(parent, right)
    if left_root == right_root:
        return
    if left_root < right_root:
        parent[right_root] = left_root
    else:
        parent[left_root] = right_root


def _write_jsonl(path: Path, records: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def _write_cleaning_decisions_csv(
    path: Path,
    prepared_articles: list[PreparedArticle],
) -> None:
    fieldnames = [
        "article_id",
        "author_id",
        "preparation_status",
        "article_usable",
        "structure_confidence",
        "headline_status",
        "subheadline_status",
        "decision_reasons",
        "review_flags",
        "informational_flags",
        "structure_warning_flags",
        "content_review_flags",
        "rejection_flags",
        "duplicate_cluster_id",
        "canonical_article_id",
        "topic",
        "topic_confidence",
        "topic_low_confidence",
        "topic_multi_category_conflict",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for article in prepared_articles:
            writer.writerow(
                {
                    "article_id": article.article_id,
                    "author_id": article.author_id,
                    "preparation_status": article.preparation_status,
                    "article_usable": article.article_usable,
                    "structure_confidence": article.structure_confidence,
                    "headline_status": article.headline_status,
                    "subheadline_status": article.subheadline_status,
                    "decision_reasons": "|".join(article.decision_reasons),
                    "review_flags": "|".join(article.review_flags),
                    "informational_flags": "|".join(article.informational_flags),
                    "structure_warning_flags": "|".join(
                        article.structure_warning_flags
                    ),
                    "content_review_flags": "|".join(article.content_review_flags),
                    "rejection_flags": "|".join(article.rejection_flags),
                    "duplicate_cluster_id": article.duplicate_cluster_id or "",
                    "canonical_article_id": article.canonical_article_id or "",
                    "topic": article.topic,
                    "topic_confidence": article.topic_confidence,
                    "topic_low_confidence": article.topic_low_confidence,
                    "topic_multi_category_conflict": (
                        article.topic_multi_category_conflict
                    ),
                }
            )


def _write_structural_anomalies_csv(
    path: Path,
    profiles: list[StructuralProfile],
) -> None:
    fieldnames = [
        "article_id",
        "author_id",
        "word_count",
        "char_count",
        "paragraph_count",
        "headline_word_count",
        "subheadline_word_count",
        "headline_status",
        "subheadline_status",
        "structure_confidence",
        "informational_flags",
        "structure_warning_flags",
        "content_review_flags",
        "rejection_flags",
        "anomaly_flags",
        "anomaly_evidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for profile in profiles:
            if not profile.anomaly_flags:
                continue
            writer.writerow(
                {
                    "article_id": profile.article_id,
                    "author_id": profile.author_id,
                    "word_count": profile.word_count,
                    "char_count": profile.char_count,
                    "paragraph_count": profile.paragraph_count,
                    "headline_word_count": profile.headline_word_count,
                    "subheadline_word_count": profile.subheadline_word_count,
                    "headline_status": profile.headline_status,
                    "subheadline_status": profile.subheadline_status,
                    "structure_confidence": profile.structure_confidence,
                    "informational_flags": "|".join(profile.informational_flags),
                    "structure_warning_flags": "|".join(
                        profile.structure_warning_flags
                    ),
                    "content_review_flags": "|".join(profile.content_review_flags),
                    "rejection_flags": "|".join(profile.rejection_flags),
                    "anomaly_flags": "|".join(profile.anomaly_flags),
                    "anomaly_evidence": "|".join(profile.anomaly_evidence),
                }
            )


def _write_review_reason_distribution_csv(
    path: Path,
    prepared_articles: list[PreparedArticle],
) -> None:
    reason_counts: Counter[tuple[str, str]] = Counter()
    combination_counts: Counter[str] = Counter()
    accepted_without_headline_flags = 0
    for article in prepared_articles:
        for severity, flags in [
            ("informational", article.informational_flags),
            ("structure_warning", article.structure_warning_flags),
            ("content_review", article.content_review_flags),
            ("rejection", article.rejection_flags),
        ]:
            for flag in flags:
                reason_counts[(severity, flag)] += 1
        combination = "|".join(article.decision_reasons)
        combination_counts[combination] += 1
        non_headline_reasons = [
            reason
            for reason in article.decision_reasons
            if "headline" not in reason and "subheadline" not in reason
        ]
        if article.article_usable and not non_headline_reasons:
            accepted_without_headline_flags += 1

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["distribution_type", "severity", "reason", "article_count"],
        )
        writer.writeheader()
        for (severity, reason), count in sorted(reason_counts.items()):
            writer.writerow(
                {
                    "distribution_type": "single_reason",
                    "severity": severity,
                    "reason": reason,
                    "article_count": count,
                }
            )
        for reason, count in sorted(combination_counts.items()):
            writer.writerow(
                {
                    "distribution_type": "decision_reason_combination",
                    "severity": "decision",
                    "reason": reason or "none",
                    "article_count": count,
                }
            )
        writer.writerow(
            {
                "distribution_type": "calibration",
                "severity": "informational",
                "reason": "accepted_if_headline_related_flags_excluded",
                "article_count": accepted_without_headline_flags,
            }
        )


def _write_topic_distribution_csv(
    path: Path,
    prepared_articles: list[PreparedArticle],
) -> None:
    counts = Counter(article.topic for article in prepared_articles)
    low_counts = Counter(
        article.topic for article in prepared_articles if article.topic_low_confidence
    )
    conflict_counts = Counter(
        article.topic
        for article in prepared_articles
        if article.topic_multi_category_conflict
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "topic",
                "article_count",
                "low_confidence_count",
                "multi_category_conflict_count",
            ],
        )
        writer.writeheader()
        for topic in TOPIC_CATEGORIES:
            writer.writerow(
                {
                    "topic": topic,
                    "article_count": counts[topic],
                    "low_confidence_count": low_counts[topic],
                    "multi_category_conflict_count": conflict_counts[topic],
                }
            )


def _write_author_topic_distribution_csv(
    path: Path,
    prepared_articles: list[PreparedArticle],
) -> None:
    counts = Counter(
        (article.author_id, article.topic) for article in prepared_articles
    )
    authors = sorted({article.author_id for article in prepared_articles})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["author_id", "topic", "article_count"],
        )
        writer.writeheader()
        for author_id in authors:
            for topic in TOPIC_CATEGORIES:
                writer.writerow(
                    {
                        "author_id": author_id,
                        "topic": topic,
                        "article_count": counts[(author_id, topic)],
                    }
                )


def _write_length_distribution_csv(
    path: Path,
    prepared_articles: list[PreparedArticle],
) -> None:
    buckets = [
        (0, 199, "0-199"),
        (200, 299, "200-299"),
        (300, 399, "300-399"),
        (400, 499, "400-499"),
        (500, 699, "500-699"),
        (700, 999, "700-999"),
        (1000, 999999, "1000+"),
    ]
    counts: Counter[str] = Counter()
    for article in prepared_articles:
        for low, high, label in buckets:
            if low <= article.word_count <= high:
                counts[label] += 1
                break
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["length_bucket", "article_count"])
        writer.writeheader()
        for _, _, label in buckets:
            writer.writerow({"length_bucket": label, "article_count": counts[label]})


def _write_sprint_report(
    path: Path,
    prepared_articles: list[PreparedArticle],
    clusters: list[NearDuplicateCluster],
    mode: str,
) -> None:
    status_counts = Counter(article.preparation_status for article in prepared_articles)
    topic_counts = Counter(article.topic for article in prepared_articles)
    headline_counts = Counter(article.headline_status for article in prepared_articles)
    author_review_counts = Counter(
        article.author_id
        for article in prepared_articles
        if article.preparation_status == "review_required"
    )
    usable_count = sum(1 for article in prepared_articles if article.article_usable)
    low_topic_count = sum(
        1 for article in prepared_articles if article.topic_low_confidence
    )
    conflict_count = sum(
        1 for article in prepared_articles if article.topic_multi_category_conflict
    )
    shortest = sorted(prepared_articles, key=lambda article: article.word_count)[:5]
    longest = sorted(
        prepared_articles,
        key=lambda article: article.word_count,
        reverse=True,
    )[:5]
    lines = [
        "# Sprint 1 Corpus Report",
        "",
        f"- mode: {mode}",
        f"- articles_seen: {len(prepared_articles)}",
        f"- accepted: {status_counts['accepted']}",
        f"- review_required: {status_counts['review_required']}",
        f"- rejected: {status_counts['rejected']}",
        f"- usable_articles: {usable_count}",
        f"- explicit_headlines: {headline_counts['explicit']}",
        f"- inferred_headlines: {headline_counts['inferred']}",
        f"- no_headline: {headline_counts['not_present']}",
        f"- uncertain_headlines: {headline_counts['uncertain']}",
        f"- low_confidence_topics: {low_topic_count}",
        f"- multi_category_topic_conflicts: {conflict_count}",
        f"- near_duplicate_clusters: {len(clusters)}",
        (
            "- near_duplicate_articles: "
            f"{sum(len(cluster.article_ids) for cluster in clusters)}"
        ),
        "",
        "## Shortest Articles",
        "",
        "| article_id | author_id | words | status |",
        "| --- | --- | ---: | --- |",
    ]
    for article in shortest:
        lines.append(
            f"| {article.article_id} | {article.author_id} | "
            f"{article.word_count} | {article.preparation_status} |"
        )
    lines.extend(
        [
            "",
            "## Longest Articles",
            "",
            "| article_id | author_id | words | status |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for article in longest:
        lines.append(
            f"| {article.article_id} | {article.author_id} | "
            f"{article.word_count} | {article.preparation_status} |"
        )
    lines.extend(
        [
            "",
            "## Review Counts By Author",
            "",
            "| author_id | review_required |",
            "| --- | ---: |",
        ]
    )
    for author_id in sorted({article.author_id for article in prepared_articles}):
        lines.append(f"| {author_id} | {author_review_counts[author_id]} |")
    lines.extend(
        ["", "## Topic Distribution", "", "| topic | count |", "| --- | ---: |"]
    )
    for topic in TOPIC_CATEGORIES:
        lines.append(f"| {topic} | {topic_counts[topic]} |")
    lines.extend(
        [
            "",
            "## Representative Review Examples",
            "",
            "| reason | article_id | author_id | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for reason, article in _representative_review_examples(prepared_articles):
        evidence = _example_text(article)
        lines.append(
            f"| {reason} | {article.article_id} | {article.author_id} | "
            f"{evidence} |"
        )
    lines.extend(
        [
            "",
            "## Small Topic Samples",
            "",
            "| topic | article_id | author_id | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for article in _representative_topic_examples(prepared_articles, small_only=True):
        lines.append(
            f"| {article.topic} | {article.article_id} | {article.author_id} | "
            f"{_example_text(article)} |"
        )
    lines.extend(
        [
            "",
            "## Largest Topic Samples",
            "",
            "| topic | article_id | author_id | evidence |",
            "| --- | --- | --- | --- |",
        ]
    )
    for article in _representative_topic_examples(prepared_articles, small_only=False):
        lines.append(
            f"| {article.topic} | {article.article_id} | {article.author_id} | "
            f"{_example_text(article)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _representative_review_examples(
    prepared_articles: list[PreparedArticle],
) -> list[tuple[str, PreparedArticle]]:
    reason_counts: Counter[str] = Counter()
    for article in prepared_articles:
        for reason in article.decision_reasons:
            reason_counts[reason] += 1
    examples: list[tuple[str, PreparedArticle]] = []
    for reason, _ in reason_counts.most_common(5):
        for article in prepared_articles:
            if reason in article.decision_reasons:
                examples.append((reason, article))
                break
    return examples


def _representative_topic_examples(
    prepared_articles: list[PreparedArticle],
    *,
    small_only: bool,
) -> list[PreparedArticle]:
    topic_counts = Counter(article.topic for article in prepared_articles)
    if small_only:
        topics = [
            topic
            for topic, count in sorted(topic_counts.items(), key=lambda item: item[1])
            if count <= 10
        ]
    else:
        topics = [topic for topic, _ in topic_counts.most_common(5)]
    examples: list[PreparedArticle] = []
    for topic in topics:
        match = next(
            (article for article in prepared_articles if article.topic == topic),
            None,
        )
        if match is not None:
            examples.append(match)
    return examples


def _example_text(article: PreparedArticle) -> str:
    text = article.headline or article.headline_candidate or article.body_text
    return (
        normalize_for_comparison(text)
        .replace("|", " ")
        .replace("\n", " ")[:120]
    )
