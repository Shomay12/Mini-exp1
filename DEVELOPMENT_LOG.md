# Development Log — Local-First Web Intelligence Engine

## 2026-11-13 — Stages 2, 3, and 4 Implementation & Verification

### Stage 2 — Environment, Packaging, and Architecture Baseline
- Preserved existing `.qodo/` directory untouched and initialized Git on branch `main`.
- Built isolated Python 3.11.18 virtual environment via `uv` (`.venv/`), avoiding Python 3.15 default wheel compatibility risks.
- Installed minimal dependencies (`fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `trafilatura`, `beautifulsoup4`, `lxml`, `aiosqlite`, `python-dateutil`, `pytest`, `pytest-asyncio`).
- Created `.gitignore`, `pyproject.toml`, `requirements.txt`, `.env.example`, `ARCHITECTURE.md`, `DEVELOPMENT_LOG.md`, and `README.md`.

### Stage 3 — Core Security, Privacy, Providers, Retrieval, Storage, API & UI
- **Security & Privacy**: Implemented `Settings` (with `SecretStr`, `OUTBOUND_HTTP_PROXY`/`OUTBOUND_HTTPS_PROXY` isolated from ambient env proxies, and forced `enable_telemetry=False`), `SecretRedactingFilter` JSON logging, SSRF & DNS-rebinding guards (`validate_url_for_fetch`, private/loopback/CGNAT/metadata IPv4+IPv6 blocking), `canonicalize_url`, `DomainRateLimiter`, `QuerySanitizer`, `PrivacyPolicy`, and `PrivacyGateway`.
- **Search Providers & Query Engine**: Implemented `SearchProvider` ABC, retry-with-backoff helper, `MockSearchProvider` + `DEFAULT_FIXTURES`, official `BraveSearchProvider`, `TavilySearchProvider`, `ExaSearchProvider`, failover `SearchProviderRouter`, and deterministic `DeterministicQueryPlanner`.
- **Web Retrieval & Storage**: Implemented `RobotsTxtManager`, hop-by-hop SSRF-guarded `WebFetcher`, `ContentExtractor` (`trafilatura` 2.0 + `BeautifulSoup` fallback + strict `htmldate(..., extensive_search=False)` + JSON-LD date parsing), `DocumentNormalizer`, and async SQLite `DatabaseManager`/`ResearchRepository` (WAL mode, foreign key cascade, TTL retention purge).
- **API & UI**: Implemented FastAPI routes (`/api/v1/health`, `/api/v1/config`, `/api/v1/search`, `/api/v1/research`, `/api/v1/research/{id}`, `/api/v1/sources/{id}`) and zero-build static research workbench UI under `web_intelligence/app/ui/`.

### Stage 4 — Intelligent Evidence Processing & Research Quality Engine
- Implemented `MultiLevelDeduplicator` (exact URL, canonical URL, SHA-256 content hash, 5-word shingle near-duplicate with length-ratio guard).
- Implemented `PassageChunker` (heading-aware, sentence-preserving, exact char offsets, adaptive short-document handling, replaceable `TokenEstimator`).
- Implemented `HybridRelevanceEngine` (pure-Python Okapi BM25 + keyword + heading + coverage + document relevance, with replaceable `EmbeddingProvider`, `CosineSemanticRetriever`, `CrossEncoderReranker`).
- Implemented `SourceProvenanceEvaluator`, `RuleBasedClaimExtractor`, `DeterministicClaimComparator` (with replaceable `ClaimComparator`), `ResearchCoverageAnalyzer`, `ContextBuilder` (`ResearchPackage`), `ResearchOrchestrator`, and Phase 2 protocol interfaces (`future_interfaces.py`).

### Defects Found & Resolved During Verification
1. `trafilatura 2.0` `bare_extraction(..., with_metadata=True)` returns a `Document` object (not a `dict`) and its default `htmldate` heuristic inferred `2026-01-01` from prose mentions of `"January 2026"` on undated mirror fixtures. Resolved by switching publication-date extraction to explicit `<meta>`/`<time>`/JSON-LD plus `htmldate.find_date(..., extensive_search=False)`.
2. Ambient sandbox `HTTP_PROXY`/`HTTPS_PROXY` env vars were picked up by `Settings` (`case_sensitive=False`). Resolved by naming explicit proxy settings `OUTBOUND_HTTP_PROXY` and `OUTBOUND_HTTPS_PROXY`.
3. Updated `HTTP_422_UNPROCESSABLE_ENTITY` to `HTTP_422_UNPROCESSABLE_CONTENT` in API integration tests for Starlette 1.10 compatibility (eliminated deprecation warning).

### Next Steps (Phase 2)
- Implement `AdaptiveContextBudgetManager` for token-constrained local SLMs.
- Implement `LocalModelAdapter` backends (e.g., `llama.cpp` / Ollama / MLX local adapters) and LLM-backed `QueryPlanner`, `ClaimComparator`, `AnswerGenerator`, and `EvidenceVerifier`.

---

## 2026-11-13 — Universal Local LLM Connection System Extension

### Architecture & Implementation
- **Schemas & Secret Protection**: Added `web_intelligence/app/schemas/llm.py` (`ModelCapabilities`, `CapabilityProvenance` distinguishing `verified`/`configured`/`unknown`, `DiscoveredModel`, `LLMConnectionConfig`/`Public`/`Summary`, chat/streaming DTOs) and updated `core/logging.py` (`register_secret`, `register_secrets`, `redact_text`) and `future_interfaces.py`.
- **Separate Connection Security Boundary**: Created `web_intelligence/app/llm/security.py` enforcing syntax validation, explicit user approval, loopback-by-default rules with explicit `allow_remote_endpoint` opt-in, unconditional cloud metadata/link-local/unspecified/multicast/redirect blocking, DNS rebinding validation, safe `api_endpoint` relative-path checking, and secret-safe error reporting.
- **Adapters, Registry & Persistence**: Extended SQLite (`llm_connections` table in `storage/database.py` and CRUD in `storage/repositories.py`) and built `web_intelligence/app/llm/` (`base.py`, `openai_compatible.py`, `ollama.py`, `registry.py`, `manager.py`) supporting connection testing, `/v1/models` and `/api/tags`+`/api/show` discovery, manual model fallback, SSE & NDJSON streaming, tool calling metadata, and JSON extraction.
- **Research Workflow & API/UI**: Implemented `AdaptiveContextBudgetManager` (`research/context_budget.py`), `LocalQueryPlanner`, `LocalAnswerGenerator`, and `LocalEvidenceVerifier` (`research/llm_hooks.py`), integrated into `ResearchOrchestrator`, `/api/v1/health`, `/api/v1/llm/*` routes (`api/routes/llm.py`), and the Workbench UI (`ui/js/components/llm_pane.js`, `api.js`, `state.js`, `workspace.js`, `app.js`, `styles/components.css`).
- **Verification**: Verified 91/91 automated pytest tests passing with 0 warnings across security, SQLite storage, adapter mock transports, streaming, adaptive context budgeting, verification, and end-to-end FastAPI routes, alongside `node --check` on UI modules and CLI demo run (`demo_research.py`).
