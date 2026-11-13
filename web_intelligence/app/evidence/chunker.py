"""Passage-aware chunking preserving headings, paragraph boundaries, sentences, and offsets."""

from __future__ import annotations

import re
from dataclasses import dataclass

from web_intelligence.app.research.future_interfaces import (
    HeuristicTokenEstimator,
    TokenEstimator,
)
from web_intelligence.app.schemas.documents import SourceRecord

_HEADING_LINE_RE = re.compile(r"^#{1,6}\s+(.+)$")
# Split sentences at punctuation followed by whitespace while avoiding decimal numbers like 4.5
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass(frozen=True)
class TextChunk:
    """Structured document chunk before relevance ranking."""

    chunk_id: str
    source_id: str
    chunk_index: int
    section_heading: str | None
    passage_text: str
    char_start: int
    char_end: int
    token_estimate: int


def split_sentences(paragraph: str) -> list[str]:
    """Split a paragraph into complete sentences without breaking decimals."""
    cleaned = paragraph.strip()
    if not cleaned:
        return []
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(cleaned) if p.strip()]
    return parts or [cleaned]


class PassageChunker:
    """Adaptive document chunker preserving structural headings, sentences, and char offsets."""

    def __init__(
        self,
        *,
        target_tokens: int = 180,
        overlap_sentences: int = 1,
        token_estimator: TokenEstimator | None = None,
    ) -> None:
        self._target_tokens = max(40, target_tokens)
        self._overlap_sentences = max(0, overlap_sentences)
        self._estimator = token_estimator or HeuristicTokenEstimator()

    def chunk_document(
        self,
        source: SourceRecord,
        *,
        target_tokens: int | None = None,
    ) -> list[TextChunk]:
        """Split a primary SourceRecord into traceable, heading-aware passages."""
        content = source.main_content
        if not content or not content.strip():
            return []

        budget = max(40, target_tokens or self._target_tokens)
        total_tokens = self._estimator.estimate_tokens(content)

        # Adaptive behavior: short single-topic documents stay coherent if under ~1.25x budget
        # unless multiple headings suggest distinct sections.
        blocks = self._parse_blocks(content)
        heading_count = sum(1 for is_h, _, _, _ in blocks if is_h)
        if total_tokens <= int(budget * 1.25) and heading_count <= 1:
            heading = next((txt for is_h, txt, _, _ in blocks if is_h), source.title)
            body_text = content.strip()
            return [
                TextChunk(
                    chunk_id=f"{source.source_id}::chunk-0",
                    source_id=source.source_id,
                    chunk_index=0,
                    section_heading=heading,
                    passage_text=body_text,
                    char_start=0,
                    char_end=len(content),
                    token_estimate=total_tokens,
                )
            ]

        chunks: list[TextChunk] = []
        current_heading: str | None = source.title
        buffer_sentences: list[tuple[str, int, int]] = []
        buffer_tokens = 0

        def _flush(keep_overlap: bool) -> None:
            nonlocal buffer_sentences, buffer_tokens
            if not buffer_sentences:
                return
            passage_text = " ".join(s for s, _, _ in buffer_sentences).strip()
            if passage_text:
                char_start = buffer_sentences[0][1]
                char_end = buffer_sentences[-1][2]
                idx = len(chunks)
                chunks.append(
                    TextChunk(
                        chunk_id=f"{source.source_id}::chunk-{idx}",
                        source_id=source.source_id,
                        chunk_index=idx,
                        section_heading=current_heading,
                        passage_text=passage_text,
                        char_start=char_start,
                        char_end=char_end,
                        token_estimate=self._estimator.estimate_tokens(passage_text),
                    )
                )
            if keep_overlap and self._overlap_sentences > 0:
                tail = buffer_sentences[-self._overlap_sentences :]
                buffer_sentences = list(tail)
                buffer_tokens = sum(self._estimator.estimate_tokens(s) for s, _, _ in buffer_sentences)
            else:
                buffer_sentences = []
                buffer_tokens = 0

        for is_heading, block_text, block_start, _ in blocks:
            if is_heading:
                _flush(keep_overlap=False)
                current_heading = block_text
                continue

            sentences = split_sentences(block_text)
            search_cursor = block_start
            for sent in sentences:
                found_at = content.find(sent, search_cursor)
                sent_start = found_at if found_at != -1 else search_cursor
                sent_end = sent_start + len(sent)
                search_cursor = sent_end

                sent_tokens = self._estimator.estimate_tokens(sent)
                if buffer_sentences and (buffer_tokens + sent_tokens > budget):
                    _flush(keep_overlap=True)
                    # Prevent an oversized overlap sentence from causing redundant single-sentence flush
                    if buffer_sentences and (buffer_tokens + sent_tokens > budget * 1.4):
                        buffer_sentences = []
                        buffer_tokens = 0

                buffer_sentences.append((sent, sent_start, sent_end))
                buffer_tokens += sent_tokens

        _flush(keep_overlap=False)
        return chunks

    def _parse_blocks(self, content: str) -> list[tuple[bool, str, int, int]]:
        """Parse normalized text into (is_heading, clean_text, char_start, char_end)."""
        blocks: list[tuple[bool, str, int, int]] = []
        cursor = 0
        for raw_block in content.split("\n\n"):
            block_len = len(raw_block)
            stripped = raw_block.strip()
            if stripped:
                match = _HEADING_LINE_RE.match(stripped)
                if match:
                    blocks.append((True, match.group(1).strip(), cursor, cursor + block_len))
                else:
                    blocks.append((False, stripped, cursor, cursor + block_len))
            cursor += block_len + 2
        return blocks
