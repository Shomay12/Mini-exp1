"""Async repository for persisting and querying research runs, sources, and evidence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite

from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import CoverageStatus
from web_intelligence.app.schemas.research import (
    ResearchPackage,
    ResearchRunSummary,
    ResearchStatus,
)
from web_intelligence.app.storage.database import DatabaseManager


class ResearchRepository:
    """CRUD and privacy retention operations backed by local SQLite."""

    def __init__(self, db_manager: DatabaseManager, retention_days: int = 30) -> None:
        self._db = db_manager
        self._retention_days = max(1, retention_days)

    async def _open(self) -> aiosqlite.Connection:
        conn = await self._db.connect()
        await conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    async def save_research_package(self, package: ResearchPackage) -> None:
        """Persist a complete ResearchPackage and its normalized child records atomically."""
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self._retention_days)
        now_iso = now.isoformat()
        expires_iso = expires_at.isoformat()

        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO research_runs (
                    id, original_query, status, progress_json, package_json,
                    error_message, created_at, updated_at, expires_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status = excluded.status,
                    progress_json = excluded.progress_json,
                    package_json = excluded.package_json,
                    error_message = excluded.error_message,
                    updated_at = excluded.updated_at,
                    expires_at = excluded.expires_at
                """,
                (
                    package.research_id,
                    package.original_query,
                    package.research_status.value,
                    package.progress.model_dump_json(),
                    package.model_dump_json(),
                    package.error_message,
                    now_iso,
                    now_iso,
                    expires_iso,
                ),
            )

            await conn.execute("DELETE FROM search_queries WHERE research_id = ?", (package.research_id,))
            await conn.execute("DELETE FROM claim_conflicts WHERE research_id = ?", (package.research_id,))
            await conn.execute("DELETE FROM evidence_items WHERE research_id = ?", (package.research_id,))
            await conn.execute("DELETE FROM sources WHERE research_id = ?", (package.research_id,))

            for q in package.search_queries:
                await conn.execute(
                    """
                    INSERT INTO search_queries (
                        id, research_id, provider, original_subquery, sanitized_query,
                        redactions_json, intent, result_count, disclosed_at, is_external
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        package.research_id,
                        q.provider,
                        q.original_subquery,
                        q.sanitized_query,
                        json.dumps(q.redactions_applied),
                        q.intent,
                        q.result_count,
                        q.disclosed_at.isoformat(),
                        1 if q.is_external else 0,
                    ),
                )

            for src in package.sources:
                await conn.execute(
                    """
                    INSERT INTO sources (
                        id, research_id, original_url, canonical_url, final_url, domain,
                        title, author, publisher, published_at, retrieved_at, content_hash,
                        extraction_status, extraction_error, is_duplicate, duplicate_of_source_id,
                        duplicate_reason, main_content, source_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        src.source_id,
                        package.research_id,
                        src.original_url,
                        src.canonical_url,
                        src.final_url,
                        src.domain,
                        src.title,
                        src.author,
                        src.source_metadata.publisher,
                        src.published_at.isoformat() if src.published_at else None,
                        src.retrieved_at.isoformat(),
                        src.content_hash,
                        src.extraction_status.value,
                        src.extraction_error,
                        1 if src.is_duplicate else 0,
                        src.duplicate_of_source_id,
                        src.duplicate_reason,
                        src.main_content,
                        src.model_dump_json(),
                    ),
                )

            for ev in package.evidence:
                await conn.execute(
                    """
                    INSERT INTO evidence_items (
                        id, research_id, source_id, chunk_index, section_heading,
                        passage_text, char_start, char_end, token_estimate,
                        relevance_json, corroboration_json
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ev.evidence_id,
                        package.research_id,
                        ev.source_id,
                        ev.chunk_index,
                        ev.section_heading,
                        ev.passage_text,
                        ev.char_start,
                        ev.char_end,
                        ev.token_estimate,
                        ev.relevance.model_dump_json(),
                        ev.corroboration.model_dump_json(),
                    ),
                )

            for conflict in package.conflicts:
                await conn.execute(
                    """
                    INSERT INTO claim_conflicts (
                        id, research_id, topic, entity, attribute, conflict_type,
                        claim_a_json, claim_b_json, summary
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        conflict.conflict_id,
                        package.research_id,
                        conflict.topic,
                        conflict.entity,
                        conflict.attribute,
                        conflict.conflict_type,
                        conflict.claim_a.model_dump_json(),
                        conflict.claim_b.model_dump_json(),
                        conflict.summary,
                    ),
                )

            await conn.commit()
        finally:
            await conn.close()

    async def get_research_package(self, research_id: str) -> ResearchPackage | None:
        """Load a persisted ResearchPackage by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT package_json FROM research_runs WHERE id = ?",
                (research_id,),
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return None
            return ResearchPackage.model_validate_json(row[0])
        finally:
            await conn.close()

    async def get_source_by_id(self, source_id: str) -> SourceRecord | None:
        """Load a persisted SourceRecord by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT source_json FROM sources WHERE id = ?",
                (source_id,),
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return None
            return SourceRecord.model_validate_json(row[0])
        finally:
            await conn.close()

    async def list_research_runs(self, limit: int = 30) -> list[ResearchRunSummary]:
        """List recent research runs stored locally."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, original_query, status, package_json, created_at
                FROM research_runs
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()
            summaries: list[ResearchRunSummary] = []
            for row in rows:
                pkg = ResearchPackage.model_validate_json(row[3]) if row[3] else None
                summaries.append(
                    ResearchRunSummary(
                        research_id=row[0],
                        original_query=row[1],
                        research_status=ResearchStatus(row[2]),
                        created_at=datetime.fromisoformat(row[4]),
                        retrieved_sources=pkg.metadata.retrieved_sources if pkg else 0,
                        evidence_items=pkg.metadata.evidence_items if pkg else 0,
                        conflicts_detected=pkg.metadata.conflicts_detected if pkg else 0,
                        coverage_status=(
                            pkg.coverage.coverage_status if pkg else CoverageStatus.INSUFFICIENT
                        ),
                    )
                )
            return summaries
        finally:
            await conn.close()

    async def delete_research_run(self, research_id: str) -> bool:
        """Delete a research run and all cascaded artifacts."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM research_runs WHERE id = ?",
                (research_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def purge_expired_runs(self, *, reference_time: datetime | None = None) -> int:
        """Remove local research records whose retention window has expired."""
        now_iso = (reference_time or datetime.now(timezone.utc)).isoformat()
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM research_runs WHERE expires_at <= ?",
                (now_iso,),
            )
            await conn.commit()
            return cursor.rowcount
        finally:
            await conn.close()
