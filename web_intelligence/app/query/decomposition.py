"""Deterministic search intent classification and query decomposition."""

from __future__ import annotations

import re

from web_intelligence.app.query.processor import extract_keywords, identify_topics, normalize_query
from web_intelligence.app.schemas.research import PlannedQuery, SearchIntent

_TEMPORAL_RE = re.compile(r"\b(when|release|released|date|timeline|202\d|latest|recent|history)\b", re.I)
_COMPARISON_RE = re.compile(r"\b(vs|versus|compare|comparison|difference|better|tradeoff)\b", re.I)
_VERIFICATION_RE = re.compile(r"\b(verify|true|false|fact|claim|benchmark|evidence|conflict)\b", re.I)
_FACTUAL_RE = re.compile(r"\b(who|what|how many|how much|size|cost|price|limit|latency)\b", re.I)


def classify_intent(query: str) -> SearchIntent:
    """Classify research intent deterministically using lexical patterns."""
    if _VERIFICATION_RE.search(query):
        return SearchIntent.VERIFICATION
    if _COMPARISON_RE.search(query):
        return SearchIntent.COMPARISON
    if _TEMPORAL_RE.search(query):
        return SearchIntent.TEMPORAL
    if _FACTUAL_RE.search(query):
        return SearchIntent.FACTUAL
    return SearchIntent.EXPLORATORY


def decompose_query(raw_query: str) -> tuple[str, SearchIntent, list[str], list[str], list[str], list[PlannedQuery]]:
    """Decompose a natural-language research query into subtopics and search subqueries."""
    norm = normalize_query(raw_query)
    intent = classify_intent(norm)
    keywords = extract_keywords(norm)
    topics = identify_topics(norm, keywords)

    # Build subtopics used later by ResearchCoverageAnalyzer.
    subtopics: list[str] = []
    for topic in topics:
        if topic not in subtopics:
            subtopics.append(topic)

    # Also detect explicit multi-question segments separated by '?' or ';'
    parts = [p.strip() for p in re.split(r"[?;]+", raw_query) if p.strip()]
    if len(parts) > 1:
        for part in parts:
            kw = extract_keywords(part, max_keywords=5)
            label = " ".join(kw)
            if label and label not in subtopics:
                subtopics.append(label)

    if not subtopics and norm:
        subtopics = [norm.lower()]

    planned: list[PlannedQuery] = []
    seen_queries: set[str] = set()

    def _add_query(q: str, q_intent: SearchIntent, subtopic: str, rationale: str) -> None:
        cleaned = " ".join(q.split()).strip()
        key = cleaned.lower()
        if cleaned and key not in seen_queries:
            seen_queries.add(key)
            planned.append(
                PlannedQuery(
                    query=cleaned,
                    intent=q_intent,
                    subtopic=subtopic,
                    rationale=rationale,
                )
            )

    primary_query = " ".join(keywords[:8]) if len(keywords) >= 2 else norm
    _add_query(
        primary_query or norm,
        intent,
        subtopics[0] if subtopics else norm,
        "Primary core-keyword search capturing main research objective.",
    )

    for topic in subtopics[1:3]:
        _add_query(
            topic,
            intent,
            topic,
            f"Targeted subtopic search for '{topic}'.",
        )

    if intent in {SearchIntent.VERIFICATION, SearchIntent.TEMPORAL, SearchIntent.COMPARISON} and keywords:
        facet_terms = " ".join(keywords[:5])
        suffix = "benchmarks evidence" if intent == SearchIntent.COMPARISON else "release date details"
        if intent == SearchIntent.VERIFICATION:
            suffix = "independent evidence documentation"
        _add_query(
            f"{facet_terms} {suffix}",
            intent,
            subtopics[0] if subtopics else facet_terms,
            "Secondary angle seeking independent corroboration or temporal specifics.",
        )

    return norm, intent, topics, keywords, subtopics, planned
