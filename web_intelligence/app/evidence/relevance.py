"""Hybrid relevance engine combining pure-Python BM25, lexical, heading, and pluggable semantic signals."""

from __future__ import annotations

import math
import re
from collections import Counter

from web_intelligence.app.evidence.chunker import TextChunk
from web_intelligence.app.evidence.reranker import CrossEncoderReranker, SemanticRetriever
from web_intelligence.app.query.processor import STOPWORDS
from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import RelevanceBreakdown

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize_relevant(text: str) -> list[str]:
    return [
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= 2 and t not in STOPWORDS
    ]


def _bm25_scores(
    query_tokens: list[str],
    docs_tokens: list[list[str]],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """Compute Okapi BM25 scores using smooth positive IDF safe for small corpora."""
    n_docs = len(docs_tokens)
    if n_docs == 0 or not query_tokens:
        return [0.0] * n_docs

    avg_dl = sum(len(d) for d in docs_tokens) / max(1, n_docs)
    if avg_dl == 0:
        return [0.0] * n_docs

    df: Counter[str] = Counter()
    for doc_tokens in docs_tokens:
        for token in set(doc_tokens):
            df[token] += 1

    unique_query = list(dict.fromkeys(query_tokens))
    scores: list[float] = []
    for doc_tokens in docs_tokens:
        tf = Counter(doc_tokens)
        dl = len(doc_tokens)
        score = 0.0
        for term in unique_query:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            term_df = df.get(term, 0)
            idf = math.log(1.0 + (n_docs - term_df + 0.5) / (term_df + 0.5))
            denom = freq + k1 * (1.0 - b + b * (dl / avg_dl))
            score += idf * ((freq * (k1 + 1.0)) / max(1e-9, denom))
        scores.append(score)
    return scores


def _saturate(raw_score: float, *, scale: float = 2.5) -> float:
    """Map non-negative raw BM25 into [0, 1] monotonically without losing 0.0 for zero matches."""
    if raw_score <= 0.0:
        return 0.0
    return round(raw_score / (raw_score + scale), 4)


class HybridRelevanceEngine:
    """Evaluates document and passage relevance without conflating relevance with factual confidence."""

    def __init__(
        self,
        *,
        semantic_retriever: SemanticRetriever | None = None,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self._semantic = semantic_retriever
        self._reranker = reranker

    async def score_chunks(
        self,
        query: str,
        keywords: list[str],
        sources: list[SourceRecord],
        chunks: list[TextChunk],
    ) -> tuple[dict[str, float], dict[str, RelevanceBreakdown]]:
        """Score document-level and passage-level relevance for all chunks."""
        query_tokens = _tokenize_relevant(query)
        keyword_set = {k.lower() for k in keywords if len(k) >= 2} or set(query_tokens)
        source_by_id = {s.source_id: s for s in sources}

        # 1. Document-level BM25 + query coverage
        doc_ids = [s.source_id for s in sources]
        doc_token_lists = [
            _tokenize_relevant(f"{s.title or ''} {s.main_content}") for s in sources
        ]
        raw_doc_bm25 = _bm25_scores(query_tokens, doc_token_lists)
        doc_scores: dict[str, float] = {}
        for sid, raw_score, d_tokens in zip(doc_ids, raw_doc_bm25, doc_token_lists, strict=True):
            token_set = set(d_tokens)
            cov = len(keyword_set & token_set) / max(1, len(keyword_set))
            doc_scores[sid] = round(0.70 * _saturate(raw_score) + 0.30 * cov, 4)

        if not chunks:
            return doc_scores, {}

        # 2. Passage-level BM25
        passage_token_lists = [_tokenize_relevant(c.passage_text) for c in chunks]
        raw_chunk_bm25 = _bm25_scores(query_tokens, passage_token_lists)

        # 3. Optional semantic & cross-encoder signals
        passage_texts = [c.passage_text for c in chunks]
        semantic_scores = (
            await self._semantic.score(query, passage_texts)
            if self._semantic is not None
            else [None] * len(chunks)
        )
        rerank_scores = (
            await self._reranker.rerank(query, passage_texts)
            if self._reranker is not None
            else [None] * len(chunks)
        )

        normalized_phrase = " ".join(query_tokens)
        breakdowns: dict[str, RelevanceBreakdown] = {}
        for idx, chunk in enumerate(chunks):
            p_tokens = set(passage_token_lists[idx])
            src = source_by_id.get(chunk.source_id)
            heading_text = f"{chunk.section_heading or ''} {src.title or '' if src else ''}"
            h_tokens = set(_tokenize_relevant(heading_text))

            bm25_norm = _saturate(raw_chunk_bm25[idx])
            coverage = round(len(keyword_set & p_tokens) / max(1, len(keyword_set)), 4)

            # Keyword match score includes exact multi-word phrase bonus when applicable
            phrase_bonus = (
                0.15
                if len(query_tokens) >= 2 and normalized_phrase in " ".join(passage_token_lists[idx])
                else 0.0
            )
            keyword_match = round(min(1.0, coverage + phrase_bonus), 4)
            heading_match = round(len(keyword_set & h_tokens) / max(1, len(keyword_set)), 4)
            doc_rel = doc_scores.get(chunk.source_id, 0.0)

            sem_val = semantic_scores[idx]
            rerank_val = rerank_scores[idx]

            # Weighted lexical baseline, augmented smoothly when semantic/rerank models exist
            lexical_score = (
                0.45 * bm25_norm
                + 0.20 * keyword_match
                + 0.15 * heading_match
                + 0.10 * coverage
                + 0.10 * doc_rel
            )
            if sem_val is not None and rerank_val is not None:
                combined = 0.45 * lexical_score + 0.30 * sem_val + 0.25 * rerank_val
            elif sem_val is not None:
                combined = 0.65 * lexical_score + 0.35 * sem_val
            elif rerank_val is not None:
                combined = 0.65 * lexical_score + 0.35 * rerank_val
            else:
                combined = lexical_score

            breakdowns[chunk.chunk_id] = RelevanceBreakdown(
                bm25_score=bm25_norm,
                keyword_match_score=keyword_match,
                heading_relevance_score=heading_match,
                query_term_coverage=coverage,
                document_relevance_score=doc_rel,
                semantic_score=round(sem_val, 4) if sem_val is not None else None,
                rerank_score=round(rerank_val, 4) if rerank_val is not None else None,
                combined_score=round(min(1.0, max(0.0, combined)), 4),
                is_factual_confidence=False,
            )

        return doc_scores, breakdowns
