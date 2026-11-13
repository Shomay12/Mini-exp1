"""Async SQLite connection and schema initialization."""

from __future__ import annotations

from pathlib import Path

import aiosqlite

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS research_runs (
    id TEXT PRIMARY KEY,
    original_query TEXT NOT NULL,
    status TEXT NOT NULL,
    progress_json TEXT NOT NULL,
    package_json TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS search_queries (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    original_subquery TEXT NOT NULL,
    sanitized_query TEXT NOT NULL,
    redactions_json TEXT NOT NULL,
    intent TEXT NOT NULL,
    result_count INTEGER NOT NULL,
    disclosed_at TEXT NOT NULL,
    is_external INTEGER NOT NULL,
    FOREIGN KEY (research_id) REFERENCES research_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    original_url TEXT NOT NULL,
    canonical_url TEXT NOT NULL,
    final_url TEXT NOT NULL,
    domain TEXT NOT NULL,
    title TEXT,
    author TEXT,
    publisher TEXT,
    published_at TEXT,
    retrieved_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    extraction_error TEXT,
    is_duplicate INTEGER NOT NULL DEFAULT 0,
    duplicate_of_source_id TEXT,
    duplicate_reason TEXT,
    main_content TEXT NOT NULL,
    source_json TEXT NOT NULL,
    FOREIGN KEY (research_id) REFERENCES research_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS evidence_items (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    section_heading TEXT,
    passage_text TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL,
    token_estimate INTEGER NOT NULL,
    relevance_json TEXT NOT NULL,
    corroboration_json TEXT NOT NULL,
    FOREIGN KEY (research_id) REFERENCES research_runs(id) ON DELETE CASCADE,
    FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS claim_conflicts (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    topic TEXT NOT NULL,
    entity TEXT NOT NULL,
    attribute TEXT NOT NULL,
    conflict_type TEXT NOT NULL,
    claim_a_json TEXT NOT NULL,
    claim_b_json TEXT NOT NULL,
    summary TEXT NOT NULL,
    FOREIGN KEY (research_id) REFERENCES research_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS llm_connections (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL,
    preset TEXT,
    base_url TEXT NOT NULL,
    api_endpoint TEXT,
    api_key TEXT,
    model TEXT NOT NULL,
    context_length INTEGER NOT NULL,
    max_output_tokens INTEGER NOT NULL,
    request_timeout_seconds REAL NOT NULL,
    streaming INTEGER NOT NULL,
    allow_remote_endpoint INTEGER NOT NULL,
    approved_by_user INTEGER NOT NULL,
    is_active INTEGER NOT NULL,
    status TEXT NOT NULL,
    last_error TEXT,
    last_checked_at TEXT,
    capabilities_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS context_budget_configs (
    id TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_turns (
    id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL DEFAULT 'global',
    scope TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    decisions_json TEXT NOT NULL DEFAULT '[]',
    unresolved_questions_json TEXT NOT NULL DEFAULT '[]',
    token_count INTEGER NOT NULL DEFAULT 0,
    compacted INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS workspace_notes (
    id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'default',
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    token_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry_entries (
    entry_id TEXT PRIMARY KEY,
    connection_id TEXT NOT NULL,
    model_id TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (connection_id) REFERENCES llm_connections(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS generation_runs (
    generation_id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    status TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (research_id) REFERENCES research_runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    research_id TEXT PRIMARY KEY,
    original_query TEXT NOT NULL,
    research_objective TEXT NOT NULL,
    status TEXT NOT NULL,
    current_iteration INTEGER NOT NULL DEFAULT 0,
    session_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_trace_entries (
    trace_id TEXT PRIMARY KEY,
    research_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    iteration INTEGER NOT NULL,
    phase TEXT NOT NULL,
    event_type TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (research_id) REFERENCES agent_sessions(research_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sources_research ON sources(research_id);
CREATE INDEX IF NOT EXISTS idx_evidence_research ON evidence_items(research_id);
CREATE INDEX IF NOT EXISTS idx_runs_expires ON research_runs(expires_at);
CREATE INDEX IF NOT EXISTS idx_llm_active ON llm_connections(is_active);
CREATE INDEX IF NOT EXISTS idx_conv_turns_research ON conversation_turns(research_id, created_at);
CREATE INDEX IF NOT EXISTS idx_workspace_notes_ws ON workspace_notes(workspace_id, updated_at);
CREATE INDEX IF NOT EXISTS idx_model_registry_conn ON model_registry_entries(connection_id, model_id);
CREATE INDEX IF NOT EXISTS idx_generation_research ON generation_runs(research_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_status ON agent_sessions(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_agent_trace_seq ON agent_trace_entries(research_id, sequence);
"""


class DatabaseManager:
    """Manages local SQLite initialization and connections."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    async def initialize(self) -> None:
        """Create database directory if needed and ensure all tables exist."""
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.database_path) as conn:
            await conn.executescript(SCHEMA_SQL)
            await conn.commit()

    def connect(self) -> aiosqlite.Connection:
        """Return an async SQLite connection context manager."""
        return aiosqlite.connect(self.database_path)
