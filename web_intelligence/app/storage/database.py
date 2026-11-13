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

CREATE INDEX IF NOT EXISTS idx_sources_research ON sources(research_id);
CREATE INDEX IF NOT EXISTS idx_evidence_research ON evidence_items(research_id);
CREATE INDEX IF NOT EXISTS idx_runs_expires ON research_runs(expires_at);
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
