"""Tests for multi-level document deduplication and edge cases."""

from __future__ import annotations

from datetime import datetime, timezone

from web_intelligence.app.evidence.deduplicator import MultiLevelDeduplicator
from web_intelligence.app.retrieval.normalizer import compute_content_hash
from web_intelligence.app.schemas.documents import ExtractionStatus, RetrievedDocument


def _doc(
    source_id: str,
    url: str,
    text: str,
    *,
    canonical_url: str | None = None,
    status: ExtractionStatus = ExtractionStatus.SUCCESS,
) -> RetrievedDocument:
    return RetrievedDocument(
        source_id=source_id,
        original_url=url,
        canonical_url=canonical_url or url,
        final_url=url,
        domain=url.split("/")[2],
        title=f"Title {source_id}",
        retrieved_at=datetime.now(timezone.utc),
        main_content=text,
        content_hash=compute_content_hash(text),
        extraction_status=status,
    )


def test_duplicate_urls_and_canonical_differences() -> None:
    dedup = MultiLevelDeduplicator()
    text_a = "Small language models execute locally on consumer hardware with quantized weights."
    d1 = _doc("s1", "https://example.com/guide?a=1&b=2", text_a, canonical_url="https://example.com/guide?a=1&b=2")
    d2 = _doc("s2", "https://example.com/guide?a=1&b=2", text_a, canonical_url="https://example.com/guide?a=1&b=2")
    d3 = _doc(
        "s3",
        "https://example.com/guide/?b=2&a=1&utm_source=feed#top",
        text_a,
        canonical_url="https://example.com/guide/?b=2&a=1&utm_source=feed#top",
    )

    result = dedup.deduplicate([(d1, "s1", "mock"), (d2, "s2", "mock"), (d3, "s3", "mock")])
    assert len(result.primary_sources) == 1
    assert result.duplicate_count == 2
    assert result.all_sources[1].duplicate_reason == "exact_url"
    assert result.all_sources[2].duplicate_reason == "canonical_url"
    assert d3.original_url in result.primary_sources[0].alias_urls


def test_identical_and_near_duplicate_content_different_urls() -> None:
    dedup = MultiLevelDeduplicator(near_duplicate_threshold=0.72)
    base = (
        "Small language models in the 4B parameter tier run efficiently on laptop CPUs when "
        "quantized to 4-bit GGUF precision. Local BM25 evidence retrieval supplies verifiable citations "
        "and prevents parametric hallucinations during multi-document synthesis. Adaptive context "
        "budgeting removes duplicate snippets before prompt assembly so compact models remain accurate."
    )
    near_copy = (
        "Small language models in the 4B parameter tier run efficiently on laptop CPUs when "
        "quantized to 4-bit GGUF precision. Local BM25 evidence retrieval supplies verifiable citations "
        "and prevents parametric hallucinations during multi-document synthesis. Adaptive context "
        "budgeting removes duplicate snippets before prompt assembly so compact models stay accurate."
    )

    d1 = _doc("s1", "https://origin.example.org/slm", base)
    d2 = _doc("s2", "https://mirror1.example.net/slm-copy", base)
    d3 = _doc("s3", "https://mirror2.example.io/slm-reprint", near_copy)

    result = dedup.deduplicate([(d1, "", "mock"), (d2, "", "mock"), (d3, "", "mock")])
    assert len(result.primary_sources) == 1
    assert result.duplicate_count == 2
    assert result.all_sources[1].duplicate_reason == "exact_content_hash"
    assert result.all_sources[2].duplicate_reason == "near_duplicate"
    assert result.all_sources[2].duplicate_of_source_id == "s1"


def test_different_articles_on_same_topic_not_merged() -> None:
    dedup = MultiLevelDeduplicator(near_duplicate_threshold=0.72)
    article_a = (
        "Small language models reduce inference latency on edge devices. Engineering benchmarks on "
        "Apple M2 laptops show quantized GGUF models sustain forty tokens per second without network calls. "
        "Careful memory profiling reveals embedding indexes consume negligible RAM."
    )
    article_b = (
        "Evaluating small language models requires inspecting conflict rates and citation accuracy. "
        "Independent academic audits found notable discrepancies in published context window measurements "
        "and vendor throughput charts across consumer laptops."
    )

    d1 = _doc("s1", "https://site-a.example.org/benchmarks", article_a)
    d2 = _doc("s2", "https://site-b.example.edu/audit", article_b)

    result = dedup.deduplicate([(d1, "", "mock"), (d2, "", "mock")])
    assert len(result.primary_sources) == 2
    assert result.duplicate_count == 0


def test_empty_and_missing_metadata_documents_handled_cleanly() -> None:
    dedup = MultiLevelDeduplicator()
    empty_a = _doc("e1", "https://empty-a.example.com/1", "", status=ExtractionStatus.FAILED)
    empty_b = _doc("e2", "https://empty-b.example.com/2", "", status=ExtractionStatus.FAILED)

    result = dedup.deduplicate([(empty_a, "", "mock"), (empty_b, "", "mock")])
    # Empty failed pages with distinct URLs should not falsely collide on empty content hash
    assert result.duplicate_count == 0
    assert result.all_sources[0].published_at is None
    assert result.all_sources[0].author is None
