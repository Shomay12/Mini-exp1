"""Document normalization converting raw fetch + extract output into RetrievedDocument."""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from datetime import datetime, timezone

from web_intelligence.app.core.security import canonicalize_url, extract_domain
from web_intelligence.app.retrieval.extractor import ExtractedPage
from web_intelligence.app.retrieval.fetcher import FetchOutcome
from web_intelligence.app.schemas.documents import (
    DocumentHeading,
    ExtractionStatus,
    RetrievedDocument,
    SourceMetadata,
)

_INLINE_SPACE_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+")


def normalize_document_text(text: str, headings: list[DocumentHeading] | None = None) -> tuple[str, list[DocumentHeading]]:
    """Normalize unicode and whitespace while preserving paragraph and heading structure."""
    if not text:
        return "", []

    norm = unicodedata.normalize("NFKC", text)
    heading_lookup = {
        h.text.strip().lower(): h.level
        for h in (headings or [])
        if h.text.strip()
    }

    normalized_blocks: list[str] = []
    for raw_block in re.split(r"\n\s*\n|\n", norm):
        cleaned = _INLINE_SPACE_RE.sub(" ", raw_block).strip()
        if not cleaned:
            continue
        plain_candidate = _HEADING_PREFIX_RE.sub("", cleaned).strip()
        # Promote known HTML headings to clean Markdown heading lines if not already prefixed
        if not _HEADING_PREFIX_RE.match(cleaned) and plain_candidate.lower() in heading_lookup:
            level = heading_lookup[plain_candidate.lower()]
            cleaned = f"{'#' * level} {plain_candidate}"
        normalized_blocks.append(cleaned)

    joined = _MULTI_NEWLINE_RE.sub("\n\n", "\n\n".join(normalized_blocks)).strip()

    # Compute exact character offsets of headings in normalized text
    located_headings: list[DocumentHeading] = []
    cursor = 0
    for block in joined.split("\n\n"):
        if block.startswith("#"):
            m = re.match(r"^(#{1,6})\s+(.+)$", block)
            if m:
                located_headings.append(
                    DocumentHeading(
                        level=len(m.group(1)),
                        text=m.group(2).strip(),
                        char_offset=cursor,
                    )
                )
        cursor += len(block) + 2

    if not located_headings and headings:
        located_headings = list(headings)

    return joined, located_headings


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized content for exact duplicate detection."""
    if not text:
        return ""
    canonical_text = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


class DocumentNormalizer:
    """Produces consistent RetrievedDocument objects preserving missing metadata honestly."""

    def normalize(
        self,
        outcome: FetchOutcome,
        extracted: ExtractedPage | None = None,
        *,
        source_id: str | None = None,
        fallback_title: str | None = None,
    ) -> RetrievedDocument:
        """Build a normalized RetrievedDocument from fetch and extraction results."""
        sid = source_id or str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        final_url = outcome.final_url or outcome.original_url
        canon = (
            extracted.canonical_url
            if extracted and extracted.canonical_url
            else canonicalize_url(final_url)
        )
        domain = extract_domain(final_url)

        if not outcome.success or extracted is None:
            return RetrievedDocument(
                source_id=sid,
                original_url=outcome.original_url,
                canonical_url=canon,
                final_url=final_url,
                domain=domain,
                title=fallback_title,
                author=None,
                published_at=None,
                retrieved_at=now,
                main_content="",
                content_hash="",
                extraction_status=outcome.status,
                extraction_error=outcome.error,
                source_metadata=SourceMetadata(
                    http_status=outcome.http_status,
                    content_type=outcome.content_type,
                    byte_length=len(outcome.body_bytes),
                    redirect_chain=outcome.redirect_chain,
                ),
            )

        normalized_text, located_headings = normalize_document_text(
            extracted.text, extracted.headings
        )
        content_hash = compute_content_hash(normalized_text)

        status = ExtractionStatus.SUCCESS
        error_msg = outcome.error
        if not normalized_text:
            status = ExtractionStatus.FAILED
            error_msg = "No extractable content found in document."
        elif len(normalized_text) < 80 or extracted.warnings:
            if len(normalized_text) < 80:
                status = ExtractionStatus.PARTIAL

        return RetrievedDocument(
            source_id=sid,
            original_url=outcome.original_url,
            canonical_url=canon,
            final_url=final_url,
            domain=domain,
            title=extracted.title or fallback_title,
            author=extracted.author,
            published_at=extracted.published_at,
            retrieved_at=now,
            main_content=normalized_text,
            content_hash=content_hash,
            extraction_status=status,
            extraction_error=error_msg,
            source_metadata=SourceMetadata(
                language=extracted.language,
                description=extracted.description,
                publisher=extracted.publisher,
                extractor_used=extracted.extractor_used,
                http_status=outcome.http_status,
                content_type=outcome.content_type,
                byte_length=len(outcome.body_bytes),
                headings=located_headings,
                warnings=extracted.warnings,
                redirect_chain=outcome.redirect_chain,
            ),
        )
