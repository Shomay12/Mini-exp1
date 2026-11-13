"""Deterministic query normalization, keyword extraction, and topic identification."""

from __future__ import annotations

import re

STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "between", "by", "can", "compare",
        "did", "do", "does", "for", "from", "had", "has", "have", "how", "in", "into",
        "is", "it", "its", "of", "on", "or", "research", "should", "that", "the",
        "their", "there", "these", "they", "this", "to", "vs", "versus", "was",
        "were", "what", "when", "where", "which", "who", "why", "with", "would",
        "about", "investigate", "explain", "analyze", "summary", "overview",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._\-+]{0,40}")
_SPACE_RE = re.compile(r"\s+")


def normalize_query(raw_query: str) -> str:
    """Normalize whitespace and strip trailing punctuation noise."""
    cleaned = _SPACE_RE.sub(" ", raw_query).strip()
    return cleaned.strip("?!.;: ")


def extract_keywords(text: str, *, max_keywords: int = 12) -> list[str]:
    """Extract ordered unique non-stopword terms from query text."""
    seen: set[str] = set()
    keywords: list[str] = []
    for match in _TOKEN_RE.findall(text.lower()):
        token = match.strip("._-+")
        if len(token) < 2 or token in STOPWORDS or token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= max_keywords:
            break
    return keywords


def identify_topics(normalized_query: str, keywords: list[str]) -> list[str]:
    """Split conjunctions/clauses to identify core query topics."""
    clause_splits = re.split(
        r"\s+(?:and|vs\.?|versus|compared to|with respect to|,|;)\s+",
        normalized_query,
        flags=re.IGNORECASE,
    )
    topics: list[str] = []
    seen: set[str] = set()
    for raw_clause in clause_splits:
        clause = raw_clause.strip(" ?!.,;:")
        clause_kw = extract_keywords(clause, max_keywords=5)
        if not clause_kw:
            continue
        topic_label = " ".join(clause_kw)
        if topic_label not in seen:
            seen.add(topic_label)
            topics.append(topic_label)

    if not topics and keywords:
        topics.append(" ".join(keywords[:5]))
    return topics
