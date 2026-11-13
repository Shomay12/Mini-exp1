"""Tests for passage-aware chunking and hybrid BM25/semantic/rerank relevance."""

from __future__ import annotations

from datetime import datetime, timezone

from web_intelligence.app.evidence.chunker import PassageChunker
from web_intelligence.app.evidence.relevance import HybridRelevanceEngine
from web_intelligence.app.evidence.reranker import (
    CosineSemanticRetriever,
    CrossEncoderReranker,
    EmbeddingProvider,
)
from web_intelligence.app.schemas.documents import ExtractionStatus, SourceRecord


def _source(source_id: str, title: str, content: str) -> SourceRecord:
    return SourceRecord(
        source_id=source_id,
        original_url=f"https://{source_id}.example.org/doc",
        canonical_url=f"https://{source_id}.example.org/doc",
        final_url=f"https://{source_id}.example.org/doc",
        domain=f"{source_id}.example.org",
        title=title,
        retrieved_at=datetime.now(timezone.utc),
        main_content=content,
        content_hash=source_id,
        extraction_status=ExtractionStatus.SUCCESS,
    )


def test_chunker_preserves_headings_sentences_and_offsets() -> None:
    text = (
        "## Architecture Overview\n\n"
        "Small language models require 4.5 GB of RAM in 4-bit GGUF format. "
        "They run locally without cloud APIs.\n\n"
        "## Benchmark Metrics\n\n"
        "Measured throughput is 48 tokens per second on Apple M2 hardware. "
        "Context windows scale up to 32k tokens during evaluation. "
        "Additional memory profiling tracks peak KV-cache allocation accurately."
    )
    src = _source("doc1", "SLM Guide", text)
    chunker = PassageChunker(target_tokens=45, overlap_sentences=1)
    chunks = chunker.chunk_document(src)

    assert len(chunks) >= 2
    assert chunks[0].section_heading == "Architecture Overview"
    assert any(c.section_heading == "Benchmark Metrics" for c in chunks)
    for chunk in chunks:
        assert chunk.source_id == "doc1"
        assert chunk.char_end > chunk.char_start
        # Verify exact substring traceability via offsets
        sliced = text[chunk.char_start : chunk.char_end]
        assert sliced.split()[0] == chunk.passage_text.split()[0]
        assert chunk.token_estimate > 0


async def test_hybrid_relevance_ranking_consistency_and_pluggable_hooks() -> None:
    relevant_src = _source(
        "rel",
        "Quantized Local Language Models",
        "## Local Inference\n\nQuantized small language models run locally with BM25 retrieval.",
    )
    offtopic_src = _source(
        "off",
        "Gardening Soil Compost Guide",
        "## Soil Preparation\n\nCompost improves drainage and earthworm activity in spring gardens.",
    )
    chunker = PassageChunker(target_tokens=180)
    chunks = chunker.chunk_document(relevant_src) + chunker.chunk_document(offtopic_src)

    # First test pure lexical BM25 independence
    lexical_engine = HybridRelevanceEngine()
    doc_scores, breakdowns = await lexical_engine.score_chunks(
        "quantized small language models",
        ["quantized", "small", "language", "models"],
        [relevant_src, offtopic_src],
        chunks,
    )
    assert doc_scores["rel"] > doc_scores["off"]
    rel_chunk = breakdowns[chunks[0].chunk_id]
    off_chunk = breakdowns[chunks[1].chunk_id]
    assert rel_chunk.combined_score > off_chunk.combined_score
    assert rel_chunk.bm25_score > 0.0
    assert off_chunk.bm25_score == 0.0
    assert rel_chunk.is_factual_confidence is False
    assert rel_chunk.semantic_score is None

    # Now plug in custom local EmbeddingProvider + CrossEncoderReranker
    class FakeEmbedder(EmbeddingProvider):
        async def embed_texts(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0] if "language" in t.lower() else [0.0, 1.0] for t in texts]

    class FakeReranker(CrossEncoderReranker):
        async def rerank(self, query: str, passages: list[str]) -> list[float]:
            return [0.95 if "language" in p.lower() else 0.05 for p in passages]

    hybrid_engine = HybridRelevanceEngine(
        semantic_retriever=CosineSemanticRetriever(FakeEmbedder()),
        reranker=FakeReranker(),
    )
    _, hybrid_breakdowns = await hybrid_engine.score_chunks(
        "quantized small language models",
        ["quantized", "small", "language", "models"],
        [relevant_src, offtopic_src],
        chunks,
    )
    assert hybrid_breakdowns[chunks[0].chunk_id].semantic_score == 1.0
    assert hybrid_breakdowns[chunks[0].chunk_id].rerank_score == 0.95
    assert hybrid_breakdowns[chunks[0].chunk_id].combined_score > hybrid_breakdowns[chunks[1].chunk_id].combined_score
