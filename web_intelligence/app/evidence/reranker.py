"""Replaceable interfaces for local embeddings, semantic retrieval, and cross-encoder reranking."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Interface for future local embedding models (no cloud API required)."""

    @abstractmethod
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return dense vector embeddings for each input text."""


class SemanticRetriever(ABC):
    """Interface for scoring passage relevance semantically against a query."""

    @abstractmethod
    async def score(self, query: str, passages: list[str]) -> list[float]:
        """Return normalized [0, 1] semantic relevance scores per passage."""


class CosineSemanticRetriever(SemanticRetriever):
    """Default SemanticRetriever adapter powered by any pluggable EmbeddingProvider."""

    def __init__(self, embedding_provider: EmbeddingProvider) -> None:
        self._provider = embedding_provider

    async def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        vectors = await self._provider.embed_texts([query, *passages])
        query_vec = vectors[0]
        scores: list[float] = []
        for p_vec in vectors[1:]:
            cos = _cosine_similarity(query_vec, p_vec)
            # Map [-1, 1] smoothly into [0, 1]
            scores.append(round(max(0.0, min(1.0, (cos + 1.0) / 2.0)), 4))
        return scores


class CrossEncoderReranker(ABC):
    """Interface for optional local cross-encoder passage reranking."""

    @abstractmethod
    async def rerank(self, query: str, passages: list[str]) -> list[float]:
        """Return normalized [0, 1] reranking relevance scores per passage."""


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
