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

---

## 2026-11-13 — Stage 5: Adaptive Context Management Engine

### Architecture & Implementation
- **Context Engine Package (`web_intelligence/app/context/`)**: Implemented `models.py`, `token_estimator.py`, `allocation_policy.py`, `budget_manager.py`, `evidence_selector.py`, `context_compressor.py`, `history_manager.py`, `context_builder.py`, `context_validator.py`, `engine.py`, and `__init__.py`.
- **SQLite Persistence & API Integration**: Added `context_budget_configs`, `conversation_turns`, and `workspace_notes` tables (`storage/database.py`, `storage/repositories.py` with turn cleanup on `delete_research_run`), wired `context_engine` into `AppContainer` (`dependencies.py`), and mounted `/api/v1/context/{capabilities,budget,preview,validate,compress,notes,history/{research_id},{research_id}}` (`api/routes/context.py`, `main.py`).
- **Context Inspector UI**: Added `ui/js/components/context_inspector.js` and updated `index.html`, `inspector.js`, `toolbar.js`, `api.js`, `state.js`, `app.js`, and `styles/components.css` to expose model limits, approximate token indicators, allocation profiles/ceilings, compression controls, selected/excluded passage inspection, workspace notes, pre-inference validation diagnostics, and formatted message preview.

### Integration Issues Resolved
- **Capability Provenance Fields**: Updated `token_estimator.py` to read `.context_length` and `.max_output_tokens` directly from `CapabilityProvenance` (`schemas/llm.py`) instead of nonexistent attribute aliases, never inferring window size from model names.
- **Stage 6 Compatibility**: Added `BALANCED = "standard_research"` alias and `_missing_` mapping on `AllocationProfile` (`app/context/models.py`) plus convenience `StructuredContextBuilder.build(...)` wrapper (`app/context/context_builder.py`) so downstream Stage 6 generation and prompt compilation work alongside Stage 5 profiles (`quick_answer`, `standard_research`, `deep_research`, `long_form`, `custom`).
- **Adapter Chat Signature**: Updated `ContextCompressor._compress_model_assisted` to inspect `chat()` parameters so both production `UniversalLocalModelAdapter.chat(LLMInferenceRequest)` and tuple/message stub adapters work while validating citation/date/number/entity preservation with clean `structured` fallback on failure.
- **Profile Weight Calibration**: Set `quick_answer` evidence weight (`0.42`) vs `deep_research` (`0.68`) in `allocation_policy.py` so `Deep Research` strictly allocates more evidence tokens than `Quick Answer` on identical windows despite smaller output reservations.

### Verification
- Verified `103 passed` across `.venv/bin/pytest web_intelligence/tests -v` (including `test_context_engine_unit.py` covering capability inspection, missing-tokenizer conservative estimation, budget overflow/configuration guards, empty/oversized/duplicate evidence, conflict preservation, compression provenance & model-assisted fallback, history compaction, output reservation, and pre-inference validation; and `test_context_api_and_workflow.py` covering `/api/v1/context/*` endpoints, error responses, Context Inspector static assets, and end-to-end mock local OpenAI-compatible inference).
- Verified JavaScript syntax cleanly with `node --check` on all updated UI modules (`api.js`, `state.js`, `context_inspector.js`, `inspector.js`, `toolbar.js`, `app.js`).

---

## 2026-11-13 — Stage 6: Universal Local LLM Execution & Answer Generation Engine

### Architecture & Implementation
- **Adapters & Persistent ModelRegistry**: Hardened `OpenAICompatibleAdapter` and `OllamaAdapter` (`stop` sequences, `timeout_seconds` override, mid-stream SSE/NDJSON error frame propagation). Created `schemas/generation.py`, `llm/model_registry.py` (`ModelRegistry`, `sanitize_endpoint_for_registry`), SQLite tables (`model_registry_entries`, `generation_runs` in `storage/database.py` & `storage/repositories.py`), and sync hooks in `llm/manager.py`.
- **Prompt Compilation & Citation Engine (`web_intelligence/app/generation/`)**: Created `PromptCompiler` (`[S1]`..`[Sn]` templates across `quick_answer`, `detailed_explanation`, `research_report`, `comparative_analysis`, `technical_report`, `<untrusted_source_data>`/`<untrusted_conflict_data>` anti-injection fences with closing-tag escaping, pre-send token check), `CitationResolver`, and `CitationValidator` (`[S#]` / `[S1, S2]` / `[E#]` resolution, unsupported citation rejection, uncited claim warnings, distinct provenance vs truth disclosure).
- **Inference Execution & Multi-Stage Synthesis**: Created `InferenceExecutor` & `GenerationJobRegistry` (`executor.py`), `AnswerGenerator` (`answer_generator.py`), and `LongFormReportGenerator` (`long_form.py`) with cooperative cancellation retaining partial output, native JSON or prompted-JSON fallback, honest `usage=None` and `truncated=finish_reason=='length'`, and cross-section repetition/contradiction analysis.
- **FastAPI Routes & Workbench Answer UI**: Added `api/routes/generation.py` (`/api/v1/generation`, `/api/v1/generation/stream` SSE, `/api/v1/generation/{id}`, `/api/v1/generation/{id}/cancel`, `/api/v1/llm/registry`, `/api/v1/llm/registry/{entry_id}/roles`), wired `AppContainer` (`dependencies.py`, `main.py`), and upgraded the Workbench UI (`ui/js/components/generation_workspace.js`, `generation_inspector.js`, `workspace.js`, `inspector.js`, `api.js`, `state.js`, `app.js`, `index.html`, `styles/components.css`).

### Integration Issues Resolved During Verification
1. **Circular Import & Stage 5 Capability Bug**: Resolved circular import cycle triggered when importing `storage.repositories` alongside `context` / `llm` by gating type-only imports (`ResearchRepository`, `ModelRegistry`) under `TYPE_CHECKING`, and fixed `ContextManagementEngine.inspect_capabilities` when querying an explicit `connection_id` (`get_adapter(connection_id)` instead of `get_connection`).
2. **Stage 5 `ContextConfigurationError` & `PARTIAL_CONFLICT_PAIR` in Long-Form Sections**: Wrapped `ContextConfigurationError` in `AnswerGenerator._prepare_inputs` into HTTP 422 `ContextBudgetBlockedError`, and updated `LongFormReportGenerator._build_section_context` so conflicting claims always pair both opposing evidence passages together to satisfy Stage 5 anti-bias validation.
3. **`ContextManagementEngine` ModelRegistry Only-Connected Filter**: Handled `only_connected` query filtering cleanly in `GET /api/v1/llm/registry` alongside slash-containing model IDs (`{entry_id:path}/roles`).

### Verification
- Verified Stage 6 unit, engine, and API integration suites (`test_llm_adapters.py`, `test_model_registry.py`, `test_prompt_compiler_and_citations.py`, `test_generation_engine.py`, `test_generation_api.py`) plus all prior Stages 1–5 suites: `109 passed in 3.97s` cleanly via `.venv/bin/pytest -v --ignore=web_intelligence/tests/test_agent_unit.py --ignore=web_intelligence/tests/test_agent_integration_and_failures.py` (two pre-existing untracked draft `test_agent_*` files for future agentic execution were left excluded and untouched).
- Verified JavaScript syntax cleanly with `node --check` across all 7 edited/added Stage 6 UI modules and executed CLI `web_intelligence.scripts.demo_research`.

---

## 2026-11-13 — Stage 7: Agentic Research Orchestration Engine

### Architecture & Implementation
- **Agent Orchestration Package (`web_intelligence/app/agent/`)**: Completed and verified 14 modular components + `__init__.py` (`models.py`, `orchestrator.py`, `research_planner.py`, `task_manager.py`, `execution_engine.py`, `tool_registry.py`, `tool_selector.py`, `observation_processor.py`, `evidence_evaluator.py`, `gap_analyzer.py`, `stopping_policy.py`, `execution_budget.py`, `research_state.py`, `research_memory.py`, `research_trace.py`).
- **SQLite Persistence & FastAPI Routes**: Wired `agent_sessions` and `agent_trace_entries` tables (`storage/database.py`, `storage/repositories.py`) and mounted `/api/v1/agent/*` endpoints (`api/routes/agent.py`, `main.py`, `dependencies.py`) supporting asynchronous research runs (`202 Accepted`), live session/plan/tasks/trace/gaps inspection, validated `PUT /plan` edits, pause/resume/cancel lifecycle actions, and explicit approval gating (`/approve`).
- **macOS Workbench UI & Settings Fix (`web_intelligence/app/ui/`)**:
  - Created `js/components/agent_workspace.js` (**Agent** mode view with composer depth/iteration/privacy/approval controls, live budget telemetry, Pause/Resume/Cancel triggers, approval gate prompt cards, editable pending plan tasks, evidence verdicts, open gaps, preserved conflicts, and `[S#]` cited synthesis via `renderSafeStructuredBody`).
  - Created `js/components/agent_trace.js` (**Trace** Inspector tab rendering chronological trace entries and registered `ToolRegistry` descriptors).
  - Updated `index.html`, `formatters.js`, `state.js`, `api.js`, `toolbar.js`, `sidebar.js`, `inspector.js`, `generation_workspace.js`, `app.js`, and `styles/components.css` (`derivePackageFromAgentSession` bridges Agent runs cleanly into existing Sources/Passages/Claims/Conflicts/Context Inspector panes).
  - Restored `{ id: "model", label: "Local Model" }` pane registration in `js/components/settings_sheet.js` so Local LLM settings remain accessible.
- **Documentation & CLI Demonstration**: Created `docs/AGENT_ORCHESTRATION.md`, `docs/RESEARCH_LOOP.md`, `docs/TOOL_REGISTRY.md`, and `docs/RESEARCH_TRACE.md`, updated `README.md` & `ARCHITECTURE.md`, and added `web_intelligence/scripts/demo_agent_research.py` for offline multi-iteration demonstration using `DEFAULT_FIXTURES`.

### Integration Issues Resolved During Verification
1. **Offline Fixture `url_retrieve` & Schema Domain**: Updated `ToolRegistry.is_tool_available` so `url_retrieve` works alongside `web_search` in pure-mock offline fixture mode (still enforcing `require_approval_for_remote_urls` when external search is active) and populated `domain=urlparse(url).netloc.lower()` in `_handle_url_retrieve` for `NormalizedSearchResult`.
2. **Synchronous Stage 4 Call Alignment**: Fixed `ObservationProcessor` calling `RuleBasedClaimExtractor.extract_claims` synchronously (no erroneous `await`).
3. **Post-Approval Loop & Immediate Cancel State Guards**: Extracted shared `AgentResearchOrchestrator._evaluate_gaps_and_stopping(...)` helper so post-approval resume runs identical iteration accounting, evaluation, gap detection, and stopping checks, and guarded `cancel_session` / `run_session` / `_synthesize_and_complete` against illegal `CANCELLED -> SYNTHESIZING` transitions.
4. **Fixture DNS Resolver & SSE Test Transport**: Supplied `fixture_dns_resolver` cleanly alongside LLM resolver in integration tests (`RobotsTxtManager` isolation) and SSE framing in mock LLM test transport.

### Verification
- Full pytest suite (`.venv/bin/pytest web_intelligence/tests -v`): all 118 unit and integration tests across Stages 1–7 pass.
- Syntax check (`node --check`) passes across all UI JavaScript files.
- Executed `web_intelligence.scripts.demo_agent_research` cleanly across 3 autonomous iterations with 4 retained sources, 1 deduplicated mirror rejected, 2 conflicts preserved, 53 chronological trace entries, and cited report synthesis.
