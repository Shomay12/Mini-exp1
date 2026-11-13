"""Async repository for persisting and querying research runs, sources, and evidence."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone

import aiosqlite
from pydantic import SecretStr

from web_intelligence.app.context.models import (
    ContextBudgetConfig,
    ConversationTurn,
    WorkspaceNote,
)
from web_intelligence.app.schemas.documents import SourceRecord
from web_intelligence.app.schemas.evidence import CoverageStatus
from web_intelligence.app.schemas.generation import GenerationResult, ModelRegistryEntry
from web_intelligence.app.schemas.llm import (
    ConnectionStatus,
    LLMConnectionConfig,
    ModelCapabilities,
)
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
            await conn.execute(
                "DELETE FROM conversation_turns WHERE research_id = ?",
                (research_id,),
            )
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

    @staticmethod
    def _row_to_llm_connection(row: tuple) -> LLMConnectionConfig:
        raw_key = row[6]
        caps = ModelCapabilities.model_validate_json(row[18]) if row[18] else None
        last_checked = datetime.fromisoformat(row[17]) if row[17] else None
        return LLMConnectionConfig(
            id=row[0],
            name=row[1],
            provider=row[2],
            preset=row[3],
            base_url=row[4],
            api_endpoint=row[5],
            api_key=SecretStr(raw_key) if raw_key else None,
            model=row[7],
            context_length=int(row[8]),
            max_output_tokens=int(row[9]),
            request_timeout_seconds=float(row[10]),
            streaming=bool(row[11]),
            allow_remote_endpoint=bool(row[12]),
            approved_by_user=bool(row[13]),
            is_active=bool(row[14]),
            status=ConnectionStatus(row[15]),
            last_error=row[16],
            last_checked_at=last_checked,
            capabilities=caps,
            created_at=datetime.fromisoformat(row[19]),
            updated_at=datetime.fromisoformat(row[20]),
        )

    async def save_llm_connection(self, config: LLMConnectionConfig) -> LLMConnectionConfig:
        """Persist or update an LLM endpoint configuration locally."""
        raw_key = config.api_key.get_secret_value() if config.api_key is not None else None
        caps_json = config.capabilities.model_dump_json() if config.capabilities else None
        last_checked_iso = (
            config.last_checked_at.isoformat() if config.last_checked_at else None
        )
        conn = await self._open()
        try:
            if config.is_active:
                await conn.execute("UPDATE llm_connections SET is_active = 0")
            await conn.execute(
                """
                INSERT INTO llm_connections (
                    id, name, provider, preset, base_url, api_endpoint, api_key,
                    model, context_length, max_output_tokens, request_timeout_seconds,
                    streaming, allow_remote_endpoint, approved_by_user, is_active,
                    status, last_error, last_checked_at, capabilities_json,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    provider = excluded.provider,
                    preset = excluded.preset,
                    base_url = excluded.base_url,
                    api_endpoint = excluded.api_endpoint,
                    api_key = excluded.api_key,
                    model = excluded.model,
                    context_length = excluded.context_length,
                    max_output_tokens = excluded.max_output_tokens,
                    request_timeout_seconds = excluded.request_timeout_seconds,
                    streaming = excluded.streaming,
                    allow_remote_endpoint = excluded.allow_remote_endpoint,
                    approved_by_user = excluded.approved_by_user,
                    is_active = excluded.is_active,
                    status = excluded.status,
                    last_error = excluded.last_error,
                    last_checked_at = excluded.last_checked_at,
                    capabilities_json = excluded.capabilities_json,
                    updated_at = excluded.updated_at
                """,
                (
                    config.id,
                    config.name,
                    config.provider,
                    config.preset,
                    config.base_url,
                    config.api_endpoint,
                    raw_key,
                    config.model,
                    config.context_length,
                    config.max_output_tokens,
                    config.request_timeout_seconds,
                    1 if config.streaming else 0,
                    1 if config.allow_remote_endpoint else 0,
                    1 if config.approved_by_user else 0,
                    1 if config.is_active else 0,
                    config.status.value,
                    config.last_error,
                    last_checked_iso,
                    caps_json,
                    config.created_at.isoformat(),
                    config.updated_at.isoformat(),
                ),
            )
            await conn.commit()
            return config
        finally:
            await conn.close()

    async def get_llm_connection(self, connection_id: str) -> LLMConnectionConfig | None:
        """Load a persisted LLMConnectionConfig by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, name, provider, preset, base_url, api_endpoint, api_key,
                       model, context_length, max_output_tokens, request_timeout_seconds,
                       streaming, allow_remote_endpoint, approved_by_user, is_active,
                       status, last_error, last_checked_at, capabilities_json,
                       created_at, updated_at
                FROM llm_connections
                WHERE id = ?
                """,
                (connection_id,),
            )
            row = await cursor.fetchone()
            return self._row_to_llm_connection(row) if row else None
        finally:
            await conn.close()

    async def list_llm_connections(self) -> list[LLMConnectionConfig]:
        """List all saved LLM endpoint connections (active first, then most recently updated)."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, name, provider, preset, base_url, api_endpoint, api_key,
                       model, context_length, max_output_tokens, request_timeout_seconds,
                       streaming, allow_remote_endpoint, approved_by_user, is_active,
                       status, last_error, last_checked_at, capabilities_json,
                       created_at, updated_at
                FROM llm_connections
                ORDER BY is_active DESC, updated_at DESC
                """
            )
            rows = await cursor.fetchall()
            return [self._row_to_llm_connection(row) for row in rows]
        finally:
            await conn.close()

    async def get_active_llm_connection(self) -> LLMConnectionConfig | None:
        """Return the currently active LLM connection, if any."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, name, provider, preset, base_url, api_endpoint, api_key,
                       model, context_length, max_output_tokens, request_timeout_seconds,
                       streaming, allow_remote_endpoint, approved_by_user, is_active,
                       status, last_error, last_checked_at, capabilities_json,
                       created_at, updated_at
                FROM llm_connections
                WHERE is_active = 1
                LIMIT 1
                """
            )
            row = await cursor.fetchone()
            return self._row_to_llm_connection(row) if row else None
        finally:
            await conn.close()

    async def set_active_llm_connection(
        self, connection_id: str | None
    ) -> LLMConnectionConfig | None:
        """Activate the specified connection or deactivate all connections when None."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await self._open()
        try:
            if connection_id is None:
                await conn.execute(
                    "UPDATE llm_connections SET is_active = 0, updated_at = ?",
                    (now_iso,),
                )
                await conn.commit()
                return None

            cursor = await conn.execute(
                "SELECT id FROM llm_connections WHERE id = ?",
                (connection_id,),
            )
            if await cursor.fetchone() is None:
                return None

            await conn.execute("UPDATE llm_connections SET is_active = 0")
            await conn.execute(
                "UPDATE llm_connections SET is_active = 1, updated_at = ? WHERE id = ?",
                (now_iso, connection_id),
            )
            await conn.commit()
        finally:
            await conn.close()
        return await self.get_llm_connection(connection_id)

    async def delete_llm_connection(self, connection_id: str) -> bool:
        """Delete a stored LLM connection by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM llm_connections WHERE id = ?",
                (connection_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def get_context_budget_config(self, config_id: str = "default") -> ContextBudgetConfig | None:
        """Load persisted context budget configuration, if saved."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT config_json FROM context_budget_configs WHERE id = ?",
                (config_id,),
            )
            row = await cursor.fetchone()
            if not row or not row[0]:
                return None
            return ContextBudgetConfig.model_validate_json(row[0])
        finally:
            await conn.close()

    async def save_context_budget_config(self, config: ContextBudgetConfig) -> ContextBudgetConfig:
        """Persist active context budget configuration locally."""
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO context_budget_configs (id, config_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (config.id, config.model_dump_json(), config.updated_at.isoformat()),
            )
            await conn.commit()
            return config
        finally:
            await conn.close()

    async def save_conversation_turn(self, turn: ConversationTurn) -> ConversationTurn:
        """Persist a single conversation turn scoped to a research session or global workspace."""
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO conversation_turns (
                    id, research_id, scope, role, content, decisions_json,
                    unresolved_questions_json, token_count, compacted, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content = excluded.content,
                    decisions_json = excluded.decisions_json,
                    unresolved_questions_json = excluded.unresolved_questions_json,
                    token_count = excluded.token_count,
                    compacted = excluded.compacted
                """,
                (
                    turn.turn_id,
                    turn.research_id,
                    turn.scope,
                    turn.role,
                    turn.content,
                    json.dumps(turn.decisions),
                    json.dumps(turn.unresolved_questions),
                    turn.token_count,
                    1 if turn.compacted else 0,
                    turn.created_at.isoformat(),
                ),
            )
            await conn.commit()
            return turn
        finally:
            await conn.close()

    async def list_conversation_turns(self, research_id: str = "global") -> list[ConversationTurn]:
        """Load chronological conversation turns for a research session."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, research_id, scope, role, content, decisions_json,
                       unresolved_questions_json, token_count, compacted, created_at
                FROM conversation_turns
                WHERE research_id = ?
                ORDER BY created_at ASC
                """,
                (research_id,),
            )
            rows = await cursor.fetchall()
            return [
                ConversationTurn(
                    turn_id=row[0],
                    research_id=row[1],
                    scope=row[2],
                    role=row[3],
                    content=row[4],
                    decisions=json.loads(row[5] or "[]"),
                    unresolved_questions=json.loads(row[6] or "[]"),
                    token_count=int(row[7]),
                    compacted=bool(row[8]),
                    created_at=datetime.fromisoformat(row[9]),
                )
                for row in rows
            ]
        finally:
            await conn.close()

    async def save_workspace_note(self, note: WorkspaceNote) -> WorkspaceNote:
        """Persist or update a user-controlled workspace research note."""
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO workspace_notes (
                    id, workspace_id, title, content, enabled, token_count, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    content = excluded.content,
                    enabled = excluded.enabled,
                    token_count = excluded.token_count,
                    updated_at = excluded.updated_at
                """,
                (
                    note.note_id,
                    note.workspace_id,
                    note.title,
                    note.content,
                    1 if note.enabled else 0,
                    note.token_count,
                    note.created_at.isoformat(),
                    note.updated_at.isoformat(),
                ),
            )
            await conn.commit()
            return note
        finally:
            await conn.close()

    async def get_workspace_note(self, note_id: str) -> WorkspaceNote | None:
        """Load a single workspace note by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, workspace_id, title, content, enabled, token_count, created_at, updated_at
                FROM workspace_notes
                WHERE id = ?
                """,
                (note_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return None
            return WorkspaceNote(
                note_id=row[0],
                workspace_id=row[1],
                title=row[2],
                content=row[3],
                enabled=bool(row[4]),
                user_controlled=True,
                token_count=int(row[5]),
                created_at=datetime.fromisoformat(row[6]),
                updated_at=datetime.fromisoformat(row[7]),
            )
        finally:
            await conn.close()

    async def list_workspace_notes(self, workspace_id: str = "default") -> list[WorkspaceNote]:
        """List all persisted user-controlled workspace notes."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT id, workspace_id, title, content, enabled, token_count, created_at, updated_at
                FROM workspace_notes
                WHERE workspace_id = ?
                ORDER BY updated_at DESC
                """,
                (workspace_id,),
            )
            rows = await cursor.fetchall()
            return [
                WorkspaceNote(
                    note_id=row[0],
                    workspace_id=row[1],
                    title=row[2],
                    content=row[3],
                    enabled=bool(row[4]),
                    user_controlled=True,
                    token_count=int(row[5]),
                    created_at=datetime.fromisoformat(row[6]),
                    updated_at=datetime.fromisoformat(row[7]),
                )
                for row in rows
            ]
        finally:
            await conn.close()

    async def delete_workspace_note(self, note_id: str) -> bool:
        """Delete a workspace note by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM workspace_notes WHERE id = ?",
                (note_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def save_model_registry_entry(
        self, entry: ModelRegistryEntry
    ) -> ModelRegistryEntry:
        """Persist a registered local LLM model record (never stores API credentials)."""
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO model_registry_entries (
                    entry_id, connection_id, model_id, entry_json, updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(entry_id) DO UPDATE SET
                    connection_id = excluded.connection_id,
                    model_id = excluded.model_id,
                    entry_json = excluded.entry_json,
                    updated_at = excluded.updated_at
                """,
                (
                    entry.entry_id,
                    entry.connection_id,
                    entry.model_id,
                    entry.model_dump_json(),
                    entry.updated_at.isoformat(),
                ),
            )
            await conn.commit()
            return entry
        finally:
            await conn.close()

    async def get_model_registry_entry(
        self, entry_id: str
    ) -> ModelRegistryEntry | None:
        """Load one persisted ModelRegistryEntry by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT entry_json FROM model_registry_entries WHERE entry_id = ?",
                (entry_id,),
            )
            row = await cursor.fetchone()
            return ModelRegistryEntry.model_validate_json(row[0]) if row and row[0] else None
        finally:
            await conn.close()

    async def list_model_registry_entries(
        self, connection_id: str | None = None
    ) -> list[ModelRegistryEntry]:
        """List all persisted model registry entries, optionally filtered by connection."""
        conn = await self._open()
        try:
            if connection_id is not None:
                cursor = await conn.execute(
                    """
                    SELECT entry_json
                    FROM model_registry_entries
                    WHERE connection_id = ?
                    ORDER BY updated_at DESC
                    """,
                    (connection_id,),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT entry_json
                    FROM model_registry_entries
                    ORDER BY updated_at DESC
                    """
                )
            rows = await cursor.fetchall()
            return [ModelRegistryEntry.model_validate_json(r[0]) for r in rows if r and r[0]]
        finally:
            await conn.close()

    async def delete_model_registry_entry(self, entry_id: str) -> bool:
        """Delete one model registry entry."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM model_registry_entries WHERE entry_id = ?",
                (entry_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()

    async def delete_model_registry_entries_for_connection(
        self, connection_id: str
    ) -> int:
        """Delete all model registry entries associated with a connection."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM model_registry_entries WHERE connection_id = ?",
                (connection_id,),
            )
            await conn.commit()
            return cursor.rowcount
        finally:
            await conn.close()

    async def save_generation_result(self, result: GenerationResult) -> GenerationResult:
        """Persist or update a local LLM generation result (including partial/cancelled states)."""
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO generation_runs (
                    generation_id, research_id, status, result_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(generation_id) DO UPDATE SET
                    status = excluded.status,
                    result_json = excluded.result_json,
                    updated_at = excluded.updated_at
                """,
                (
                    result.generation_id,
                    result.research_id,
                    result.status.value,
                    result.model_dump_json(),
                    result.created_at.isoformat(),
                    now_iso,
                ),
            )
            await conn.commit()
            return result
        finally:
            await conn.close()

    async def get_generation_result(
        self, generation_id: str
    ) -> GenerationResult | None:
        """Retrieve a persisted GenerationResult by ID."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT result_json FROM generation_runs WHERE generation_id = ?",
                (generation_id,),
            )
            row = await cursor.fetchone()
            return GenerationResult.model_validate_json(row[0]) if row and row[0] else None
        finally:
            await conn.close()

    async def list_generation_results(
        self, research_id: str | None = None, limit: int = 20
    ) -> list[GenerationResult]:
        """List generation runs ordered newest first."""
        conn = await self._open()
        try:
            if research_id is not None:
                cursor = await conn.execute(
                    """
                    SELECT result_json
                    FROM generation_runs
                    WHERE research_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (research_id, max(1, limit)),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT result_json
                    FROM generation_runs
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (max(1, limit),),
                )
            rows = await cursor.fetchall()
            return [GenerationResult.model_validate_json(r[0]) for r in rows if r and r[0]]
        finally:
            await conn.close()

    # --- Stage 7 Agent Research Sessions & Operational Trace Operations ---

    async def save_agent_session(self, session: object) -> None:
        """Persist or checkpoint a Stage 7 ResearchSession."""
        from web_intelligence.app.agent.models import ResearchSession

        obj = session if isinstance(session, ResearchSession) else ResearchSession.model_validate(session)
        conn = await self._open()
        try:
            await conn.execute(
                """
                INSERT INTO agent_sessions (
                    research_id, original_query, research_objective, status,
                    current_iteration, session_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(research_id) DO UPDATE SET
                    original_query = excluded.original_query,
                    research_objective = excluded.research_objective,
                    status = excluded.status,
                    current_iteration = excluded.current_iteration,
                    session_json = excluded.session_json,
                    updated_at = excluded.updated_at
                """,
                (
                    obj.research_id,
                    obj.original_query,
                    obj.research_objective,
                    obj.status.value,
                    obj.current_iteration,
                    obj.model_dump_json(),
                    obj.created_at.isoformat(),
                    obj.updated_at.isoformat(),
                ),
            )
            await conn.commit()
        finally:
            await conn.close()

    async def get_agent_session(self, research_id: str):
        """Load a persisted Stage 7 ResearchSession by ID."""
        from web_intelligence.app.agent.models import ResearchSession

        conn = await self._open()
        try:
            cursor = await conn.execute(
                "SELECT session_json FROM agent_sessions WHERE research_id = ?",
                (research_id,),
            )
            row = await cursor.fetchone()
            return ResearchSession.model_validate_json(row[0]) if row and row[0] else None
        finally:
            await conn.close()

    async def list_agent_sessions(self, limit: int = 30):
        """List Stage 7 agent sessions newest first as compact summaries."""
        from web_intelligence.app.agent.models import AgentSessionSummary, ResearchSession

        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT session_json
                FROM agent_sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (max(1, limit),),
            )
            rows = await cursor.fetchall()
            summaries: list[AgentSessionSummary] = []
            for row in rows:
                if not row or not row[0]:
                    continue
                s = ResearchSession.model_validate_json(row[0])
                summaries.append(
                    AgentSessionSummary(
                        research_id=s.research_id,
                        original_query=s.original_query,
                        research_objective=s.research_objective,
                        research_depth=s.research_depth,
                        status=s.status,
                        current_iteration=s.current_iteration,
                        current_phase=s.current_phase,
                        tasks_completed=s.research_metrics.tasks_completed,
                        tasks_total=s.research_metrics.tasks_total,
                        sources_retained=s.research_metrics.sources_retained,
                        conflicts_detected=s.research_metrics.conflicts_detected,
                        gaps_unresolved=s.research_metrics.gaps_unresolved,
                        active_model_id=s.active_model_id,
                        created_at=s.created_at,
                        updated_at=s.updated_at,
                    )
                )
            return summaries
        finally:
            await conn.close()

    async def append_agent_trace_entries(self, entries: list[object]) -> None:
        """Append chronological ResearchTraceEntry records idempotently."""
        if not entries:
            return
        from web_intelligence.app.agent.models import ResearchTraceEntry

        conn = await self._open()
        try:
            for item in entries:
                entry = (
                    item
                    if isinstance(item, ResearchTraceEntry)
                    else ResearchTraceEntry.model_validate(item)
                )
                await conn.execute(
                    """
                    INSERT INTO agent_trace_entries (
                        trace_id, research_id, sequence, iteration,
                        phase, event_type, entry_json, timestamp
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(trace_id) DO NOTHING
                    """,
                    (
                        entry.trace_id,
                        entry.research_id,
                        entry.sequence,
                        entry.iteration,
                        entry.phase,
                        entry.event_type.value,
                        entry.model_dump_json(),
                        entry.timestamp.isoformat(),
                    ),
                )
            await conn.commit()
        finally:
            await conn.close()

    async def list_agent_trace_entries(
        self, research_id: str, *, limit: int = 500
    ):
        """Return ordered operational trace entries for `research_id`."""
        from web_intelligence.app.agent.models import ResearchTraceEntry

        conn = await self._open()
        try:
            cursor = await conn.execute(
                """
                SELECT entry_json
                FROM agent_trace_entries
                WHERE research_id = ?
                ORDER BY sequence ASC
                LIMIT ?
                """,
                (research_id, max(1, limit)),
            )
            rows = await cursor.fetchall()
            return [ResearchTraceEntry.model_validate_json(r[0]) for r in rows if r and r[0]]
        finally:
            await conn.close()

    async def delete_agent_session(self, research_id: str) -> bool:
        """Delete an agent session and its trace rows cleanly."""
        conn = await self._open()
        try:
            cursor = await conn.execute(
                "DELETE FROM agent_sessions WHERE research_id = ?",
                (research_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0
        finally:
            await conn.close()
