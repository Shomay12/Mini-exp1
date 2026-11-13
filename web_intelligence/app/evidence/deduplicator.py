"""Multi-level document deduplication preserving original source references."""

from __future__ import annotations

import re
from dataclasses import dataclass

from web_intelligence.app.core.security import canonicalize_url
from web_intelligence.app.schemas.documents import ExtractionStatus, RetrievedDocument, SourceRecord

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize_words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _build_shingles(words: list[str], k: int = 5) -> set[tuple[str, ...]]:
    if len(words) < k:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + k]) for i in range(len(words) - k + 1)}


def compute_shingle_similarity(text_a: str, text_b: str, *, shingle_size: int = 5) -> float:
    """Compute Jaccard similarity of ordered word shingles between two documents."""
    words_a = _tokenize_words(text_a)
    words_b = _tokenize_words(text_b)
    if not words_a or not words_b:
        return 0.0

    # Guard against merging short distinct snippets that merely share a topic phrase
    if min(len(words_a), len(words_b)) < 20:
        return 1.0 if words_a == words_b else 0.0

    length_ratio = min(len(words_a), len(words_b)) / max(len(words_a), len(words_b))
    if length_ratio < 0.60:
        return 0.0

    shingles_a = _build_shingles(words_a, k=shingle_size)
    shingles_b = _build_shingles(words_b, k=shingle_size)
    if not shingles_a or not shingles_b:
        return 0.0

    intersection = len(shingles_a & shingles_b)
    union = len(shingles_a | shingles_b)
    return intersection / union if union else 0.0


@dataclass
class DeduplicationResult:
    """All processed source records separated into unique primaries and detected duplicates."""

    all_sources: list[SourceRecord]
    primary_sources: list[SourceRecord]
    duplicate_count: int


class MultiLevelDeduplicator:
    """Identifies exact URL, canonical URL, exact content-hash, and near-duplicate pages."""

    def __init__(self, *, near_duplicate_threshold: float = 0.72) -> None:
        self._threshold = near_duplicate_threshold

    def deduplicate(
        self,
        items: list[tuple[RetrievedDocument, str, str]],
    ) -> DeduplicationResult:
        """Deduplicate documents paired with (search_snippet, search_provider).

        Preserves every retrieved source record in `all_sources` while marking duplicates
        and attaching alias URLs to the surviving primary source record.
        """
        all_sources: list[SourceRecord] = []
        primary_sources: list[SourceRecord] = []

        by_exact_url: dict[str, SourceRecord] = {}
        by_canonical_url: dict[str, SourceRecord] = {}
        by_content_hash: dict[str, SourceRecord] = {}
        duplicate_count = 0

        for doc, snippet, provider in items:
            record = SourceRecord(
                **doc.model_dump(),
                search_snippet=snippet,
                search_provider=provider,
            )

            exact_key = record.original_url.strip()
            canon_key = canonicalize_url(record.canonical_url or record.final_url or record.original_url)
            record.canonical_url = canon_key

            matched_primary: SourceRecord | None = None
            dup_reason: str | None = None
            dup_similarity: float | None = None

            if exact_key in by_exact_url:
                matched_primary = by_exact_url[exact_key]
                dup_reason = "exact_url"
                dup_similarity = 1.0
            elif canon_key in by_canonical_url:
                matched_primary = by_canonical_url[canon_key]
                dup_reason = "canonical_url"
                dup_similarity = 1.0
            elif (
                record.content_hash
                and record.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
                and record.content_hash in by_content_hash
            ):
                matched_primary = by_content_hash[record.content_hash]
                dup_reason = "exact_content_hash"
                dup_similarity = 1.0
            elif (
                record.main_content
                and record.extraction_status in {ExtractionStatus.SUCCESS, ExtractionStatus.PARTIAL}
            ):
                best_sim = 0.0
                best_candidate: SourceRecord | None = None
                for candidate in primary_sources:
                    if not candidate.main_content:
                        continue
                    sim = compute_shingle_similarity(record.main_content, candidate.main_content)
                    if sim >= self._threshold and sim > best_sim:
                        best_sim = sim
                        best_candidate = candidate
                if best_candidate is not None:
                    matched_primary = best_candidate
                    dup_reason = "near_duplicate"
                    dup_similarity = round(best_sim, 4)

            if matched_primary is not None:
                record.is_duplicate = True
                record.duplicate_of_source_id = matched_primary.source_id
                record.duplicate_reason = dup_reason
                record.duplicate_similarity = dup_similarity
                if record.original_url not in matched_primary.alias_urls:
                    matched_primary.alias_urls.append(record.original_url)
                duplicate_count += 1
            else:
                by_exact_url[exact_key] = record
                by_canonical_url[canon_key] = record
                if record.content_hash:
                    by_content_hash[record.content_hash] = record
                primary_sources.append(record)

            all_sources.append(record)

        return DeduplicationResult(
            all_sources=all_sources,
            primary_sources=primary_sources,
            duplicate_count=duplicate_count,
        )
