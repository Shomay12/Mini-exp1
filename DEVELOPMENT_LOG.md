# Development Log — Local-First Web Intelligence Engine

## 2026-11-13 — Stage 1 & Stage 2 Initialization

### Completed
- Inspected workspace `/Users/quaxiom/Documents/DEVLOPMENT/Projects/mini-exp1` (preserved existing `.qodo/` directory).
- Initialized Git repository and created `.gitignore`, `pyproject.toml`, `requirements.txt`, and `.env.example`.
- Created isolated Python 3.11.18 virtual environment at `.venv/` via `uv` (`Python 3.15.2` system default avoided for C-extension wheel compatibility).
- Drafted initial `ARCHITECTURE.md` and `README.md`.

### Key Architectural Decisions
- **Python 3.11 runtime**: Selected locally installed Python `3.11.18` over `3.15.2` to guarantee binary wheel support for `lxml`, `pydantic-core`, and `aiosqlite`.
- **Zero-build UI**: Serving clean static HTML/CSS/vanilla JS directly from FastAPI to avoid unnecessary Node build complexity while keeping the UI fast and information-dense.
- **Offline-first testability**: Prioritizing `MockSearchProvider` and `httpx.MockTransport` fixtures so the entire pipeline is deterministically testable without external API credentials.

### Next Steps
- Stage 3 Increment 2: Security layer (`core/config.py`, `core/logging.py`, `core/security.py`) and privacy layer (`privacy/sanitizer.py`, `privacy/policy.py`, `privacy/gateway.py`) + tests.
- Stage 3 Increment 3: Schemas, `MockSearchProvider`, Brave/Tavily/Exa providers, and query processing + tests.
- Stage 3 Increment 4: SSRF-safe fetcher, `robots.txt`, Trafilatura/BS4 extraction, and SQLite storage + tests.
- Stage 4: Evidence normalization, multi-level deduplication, passage chunking, hybrid BM25 relevance, provenance evaluation, claim/conflict detection, research coverage analysis, full orchestrator, API, UI, and verification.
