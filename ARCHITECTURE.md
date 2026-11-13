# Architecture — Local-First Web Intelligence Engine

## 1. System Overview

The **Local-First Web Intelligence Engine** is Phase 1 (Stages 2–4) of the Open-Source Adaptive Research Intelligence system. Its single responsibility is to convert a natural-language research query into a verified, traceable, model-independent structured evidence package without depending on any cloud LLM.

**Core principle**: *The internet supplies external documents; the local engine retrieves, validates, extracts, deduplicates, scores, compares, and structures that evidence locally.*

---

## 2. Layered Architecture & Component Responsibilities

| Layer | Package (`web_intelligence/app/...`) | Responsibilities |
| :--- | :--- | :--- |
| **API & UI** | `api/routes/`, `ui/`, `main.py`, `dependencies.py` | FastAPI HTTP endpoints (`/api/v1/*`), Pydantic validation, consistent JSON error envelopes, static zero-build research workspace UI. |
| **Core & Security** | `core/` | Environment settings (`pydantic-settings`), secret-redacting JSON logging, URL validation & canonicalization, SSRF & DNS-rebinding defense, per-host rate limiter. |
| **Privacy Layer** | `privacy/` | Outbound query sanitizer (PII/secret/path stripping), explicit privacy policy enforcement, proxy-ready `PrivacyGateway`, outbound query disclosure records. |
| **Search Abstraction** | `providers/` | `SearchProvider` ABC, `MockSearchProvider` + `fixtures.py` (offline controlled HTML), `BraveSearchProvider`, `TavilySearchProvider`, `ExaSearchProvider`, and failover `SearchProviderRouter`. |
| **Query Engine** | `query/` | Deterministic query normalization, keyword/topic extraction, intent classification, multi-angle query decomposition, and replaceable `QueryPlanner` ABC. |
| **Web Retrieval** | `retrieval/` | `WebRetrievalEngine`, `RobotsTxtManager`, SSRF-safe async `WebFetcher` (hop-by-hop redirect checks, byte/timeout caps), `trafilatura` + `BeautifulSoup` + strict `htmldate` extractor, `DocumentNormalizer`. |
| **Evidence Engine (Stage 4)** | `evidence/` | `MultiLevelDeduplicator`, `PassageChunker`, `HybridRelevanceEngine`, `EmbeddingProvider`/`SemanticRetriever`/`CrossEncoderReranker` ABCs, `SourceProvenanceEvaluator`, `RuleBasedClaimExtractor`, `DeterministicClaimComparator`, `ResearchCoverageAnalyzer`. |
| **Research & Future LLM** | `research/` | End-to-end `ResearchOrchestrator`, `ContextBuilder` (`ResearchPackage`), `ResearchState`, and future Phase 2 protocols (`LocalModelAdapter`, `ModelCapabilities`, `ContextBudgetManager`, `TokenEstimator`, `ResearchPlanner`, `AnswerGenerator`, `EvidenceVerifier`). |
| **Local Storage** | `storage/` | Async SQLite persistence (`aiosqlite`, WAL mode, foreign keys) for runs, disclosed search queries, sources, evidence items, conflicts, and retention/deletion controls. |

---

## 3. Data-Flow Diagram

```text
  User Query
      │
      ▼
  [QueryProcessor + QueryPlanner] ──► subtopics + search queries + intent
      │
      ▼
  [QuerySanitizer + PrivacyGateway] ──► strips PII/secrets, records disclosure
      │
      ▼
  [SearchProviderRouter] ──► Mock / Brave / Tavily / Exa (retries + failover)
      │
      ▼
  [WebFetcher] ──► SSRF & DNS check (initial + each redirect) + robots.txt + rate limit
      │
      ▼
  [ContentExtractor + DocumentNormalizer] ──► Trafilatura + BS4 + strict date validation
      │
      ▼
  [MultiLevelDeduplicator] ──► exact URL -> canonical URL -> SHA-256 -> shingle near-dup
      │
      ▼
  [PassageChunker] ──► heading-preserving, sentence-aware passages + char offsets + token est
      │
      ▼
  [HybridRelevanceEngine] ──► BM25 + keyword + heading + coverage (+ optional embedding/reranker)
      │
      ▼
  [SourceProvenanceEvaluator] ──► separate provenance / relevance / corroboration signals
      │
      ▼
  [RuleBasedClaimExtractor + ClaimComparator] ──► date/quantity/polarity conflicts & corroboration
      │
      ▼
  [ResearchCoverageAnalyzer] ──► covered vs partial vs missing subtopics & unanswered gaps
      │
      ▼
  [ContextBuilder] ──► model-independent ResearchPackage ──► [SQLite Storage] + [API/UI]
```

---

## 4. API Contracts

Base prefix: `/api/v1`

- `POST /api/v1/search` — Sanitize query, execute search across configured providers, return `SearchResponse` with normalized results, provider diagnostics, and disclosed outbound queries.
- `POST /api/v1/research` — Execute full end-to-end Stage 3 + Stage 4 research pipeline (`ResearchRequest`), persist results locally, and return `ResearchPackage`.
- `GET /api/v1/research` — List locally persisted research runs (`ResearchRunSummary[]`).
- `GET /api/v1/research/{research_id}` — Fetch full persisted `ResearchPackage` and execution status (`404` if missing).
- `DELETE /api/v1/research/{research_id}` — Delete a research run and cascaded sources, evidence, queries, and conflicts (`204`, or `404` if missing).
- `GET /api/v1/sources/{source_id}` — Inspect full `SourceRecord` (title, author, publisher, publication date, provenance, duplicate status/aliases, extraction status, warnings, redirect chain, normalized content).
- `GET /api/v1/health` — Service health, SQLite readiness, `cloud_llm_required: false`, and per-provider availability status.
- `GET /api/v1/config` — Safe configuration snapshot (secrets masked/excluded; explicit privacy boundary flags).

---

## 5. SQLite Database Schema

Stored locally in SQLite (`PRAGMA journal_mode = WAL`, `PRAGMA foreign_keys = ON`):

1. `research_runs`: `id TEXT PK`, `original_query TEXT`, `status TEXT`, `progress_json TEXT`, `package_json TEXT`, `error_message TEXT`, `created_at TEXT`, `updated_at TEXT`, `expires_at TEXT`
2. `search_queries`: `id TEXT PK`, `research_id TEXT FK`, `provider TEXT`, `original_subquery TEXT`, `sanitized_query TEXT`, `redactions_json TEXT`, `intent TEXT`, `result_count INTEGER`, `disclosed_at TEXT`, `is_external INTEGER`
3. `sources`: `id TEXT PK`, `research_id TEXT FK`, `original_url TEXT`, `canonical_url TEXT`, `final_url TEXT`, `domain TEXT`, `title TEXT`, `author TEXT`, `publisher TEXT`, `published_at TEXT`, `retrieved_at TEXT`, `content_hash TEXT`, `extraction_status TEXT`, `extraction_error TEXT`, `is_duplicate INTEGER`, `duplicate_of_source_id TEXT`, `duplicate_reason TEXT`, `main_content TEXT`, `source_json TEXT`
4. `evidence_items`: `id TEXT PK`, `research_id TEXT FK`, `source_id TEXT FK`, `chunk_index INTEGER`, `section_heading TEXT`, `passage_text TEXT`, `char_start INTEGER`, `char_end INTEGER`, `token_estimate INTEGER`, `relevance_json TEXT`, `corroboration_json TEXT`
5. `claim_conflicts`: `id TEXT PK`, `research_id TEXT FK`, `topic TEXT`, `entity TEXT`, `attribute TEXT`, `conflict_type TEXT`, `claim_a_json TEXT`, `claim_b_json TEXT`, `summary TEXT`

---

## 6. Privacy Boundaries

- **Local-only processing & history**: Downloaded HTML, normalized documents, passages, claims, and SQLite state remain local with configurable TTL retention (`RETENTION_DAYS`) and explicit deletion (`DELETE /api/v1/research/{research_id}`).
- **No cloud LLM dependency & zero telemetry**: Core intelligence relies exclusively on local deterministic/lexical algorithms; `enable_telemetry` is structurally forced to `False`.
- **Explicit proxy isolation**: Outbound proxy configuration uses `OUTBOUND_HTTP_PROXY` / `OUTBOUND_HTTPS_PROXY` (`trust_env=False`), avoiding silent inheritance of ambient shell proxy variables.
- **Explicit outbound footprint**: Search queries sent to external APIs and webpage GET requests sent to websites are inherently observable by those endpoints and network intermediaries. All outbound search queries flow through `QuerySanitizer` and `PrivacyGateway` and are recorded in `search_queries` for user inspection.
- **SSRF isolation**: Outbound fetches reject non-HTTP(S) schemes, embedded credentials, non-allowlisted ports, localhost/internal hostnames, and private/loopback/CGNAT/link-local/metadata (`169.254.169.254`) IP addresses across initial URLs, DNS resolution, `robots.txt`, and every redirect hop.

---

## 7. Stage 4 Evidence Processing Methodologies & Limitations

### Deduplication Strategy (`MultiLevelDeduplicator`)
Four ordered tiers preserve duplicate traceability on `SourceRecord` while keeping unique primaries for chunking:
1. Exact `original_url` match (`exact_url`).
2. Normalized `canonical_url` match (`canonical_url`, stripping `utm_*`/tracking params, default ports, fragments).
3. Exact SHA-256 `content_hash` over whitespace-normalized text (`exact_content_hash`).
4. Near-duplicate detection (`near_duplicate`) using 5-word ordered shingle Jaccard similarity (`>= 0.72` threshold) plus a `>= 0.60` length-ratio guard and minimum 20-word guard so distinct articles covering the same topic are **not** falsely merged.

### Chunking Strategy (`PassageChunker`)
- Preserves Markdown-normalized headings (`#`–`####`) as `section_heading` metadata on child chunks.
- Preserves paragraph and sentence boundaries (`split_sentences` avoids splitting decimals such as `4.5`).
- Adapts to short single-topic documents (`<= 1.25 * target_tokens` and `<= 1` heading) by keeping them whole instead of fragmenting short evidence.
- Tracks exact `[char_start, char_end]` offsets in `main_content` and estimates tokens via replaceable `TokenEstimator` (`HeuristicTokenEstimator`).

### Hybrid Ranking Methodology (`HybridRelevanceEngine`)
- Works 100% locally via pure-Python Okapi BM25 (`k1=1.5`, `b=0.75`, smooth positive IDF) at both passage and document levels, combined with keyword match, heading/title relevance, and query-term coverage (`combined_score` in `[0, 1]`).
- Replaceable `EmbeddingProvider`, `CosineSemanticRetriever`, and `CrossEncoderReranker` protocols integrate cleanly when local models are supplied in Phase 2.
- Relevance breakdowns explicitly set `is_factual_confidence = False` so query match is never presented as truth.

### Source Provenance (`SourceProvenanceEvaluator`)
- Separates `SourceProvenance` (`domain`, `publisher`, `source_classification` `primary`/`secondary`/`reference`/`unknown`, `author`, `published_at`, `canonical_url`), `RelevanceBreakdown`, and `CorroborationSignals`.
- Uses structural cues (e.g., `docs.*`, `.gov`, `.edu`, `arxiv`, reference wikis, editorial paths/authors) without equating domain type or popularity with factual accuracy.
- Strict date extraction uses explicit `<meta>`, `<time>`, JSON-LD `datePublished`, and `htmldate(..., extensive_search=False)` while rejecting unparseable, pre-1990, or future dates—never inventing missing publication dates from body text years.

### Claim Extraction & Conflict Detection Limitations (`RuleBasedClaimExtractor` / `DeterministicClaimComparator`)
- Extracts structured assertions for **dates** (`release_date`, `founded_date`, etc.), **numerical quantities** with normalized units (`tokens/sec`, `gb`, `tokens`, `B`, `ms`, `usd`, `%`), and **polarity** (`affirmative` vs `negative` capability predicates) across independent domains.
- Preserves both verbatim statements and normalized values inside `ClaimConflict` without selecting a winner.
- **Explicit limitations**: Deterministic rule-based matching does not perform full semantic paraphrase equivalence, coreference resolution across pronouns in separate paragraphs, or causal inference. The replaceable `ClaimComparator` interface allows Phase 2 local LLMs to perform semantic comparison over ambiguous prose.

### Research Coverage Methodology (`ResearchCoverageAnalyzer`)
- Evaluates each decomposed subtopic from `QueryPlan.subtopics` against ranked passages meeting term-overlap and relevance thresholds (`min_relevance_threshold = 0.12`):
  - `covered`: `>= 2` independent domains with supporting passages and no detected conflict.
  - `partial`: only 1 independent domain retrieved OR active conflict detected.
  - `missing`: no retrieved passage meets relevance/overlap criteria.
- Surfaces `well_corroborated_claims`, `single_source_claims`, `conflicted_areas`, and explicit `unanswered_questions` so gaps are always visible.

---

## Universal Local LLM Connection System

Extends the core Web Intelligence Layer to connect any user-hosted HTTP inference server without bundling or requiring a cloud LLM or specific runtime binary.

### 1. Provider Abstraction (`web_intelligence/app/llm/`)
- `UniversalLocalModelAdapter` (extends `LocalModelAdapter`): unified interface for connection tests, model discovery, capability inspection, non-streaming/streaming chat, structured JSON generation, and single-prompt execution.
- `OpenAICompatibleAdapter`: supports vLLM, LM Studio, llama.cpp, LocalAI, and custom `/v1` OpenAI-compatible servers (`GET /v1/models`, `POST /v1/chat/completions`, SSE streaming, tool calling, JSON fallback) without blindly assuming all OpenAI capabilities are supported.
- `OllamaAdapter`: native REST adapter (`/api/tags`, `/api/show`, `/api/chat`, NDJSON streaming, `format="json"`), extracting verified architectural context limits and capability flags directly from `/api/show` and Modelfile `num_ctx` parameters.
- `AdapterRegistry`: extensible registry mapping provider identifiers to adapter classes alongside server presets (`PROVIDER_PRESETS`).
- `LLMConnectionManager`: manages persistent endpoint CRUD, active connection switching/disconnection, secret registration with `SecretRedactingFilter`, testing, discovery, model selection, and verification prompts.

### 2. Capability Provenance (`CapabilitySource`)
- `ModelCapabilities` pairs each capability (`max_context_tokens`, `recommended_output_tokens`, `supports_streaming`, `supports_tool_calling`, `supports_json_mode`, `supports_vision`) with `CapabilityProvenance`:
  - `verified`: directly reported by server metadata (e.g., Ollama `/api/show` or OpenAI `max_model_len`/`supported_parameters`).
  - `configured`: user-supplied assumptions (`context_length`, `max_output_tokens`, `streaming`).
  - `unknown`: not exposed by the endpoint metadata.

### 3. Local Inference Trust Boundary (`web_intelligence/app/llm/security.py`)
- Separate from the external web search boundary (`core/security.py`):
  - Loopback hosts (`localhost`, `127.0.0.0/8`, `::1`) are permitted when explicitly approved by the user (`approved_by_user=True`).
  - Non-loopback LAN or remote hosts require explicit `allow_remote_endpoint=True` and `approved_by_user=True`.
  - Cloud metadata (`169.254.169.254`, `metadata.google.internal`, `metadata.goog`), link-local (`169.254.0.0/16`, `fe80::/10`), unspecified (`0.0.0.0/8`, `::/128`), multicast, embedded URL credentials, non-HTTP(S) schemes, non-HTTP infrastructure daemon ports, `api_endpoint` host escapes, and HTTP redirects (`follow_redirects=False`) are unconditionally blocked, with pre-dial DNS rebinding verification (`validate_llm_endpoint_async`).
  - Raw API keys are stored only in local SQLite (`llm_connections`), injected via `SecretStr`, registered into log redaction, masked via `redact_llm_error`, and excluded from all frontend API DTOs (`LLMConnectionPublic`).

### 4. Research Engine Integration (`context_budget.py` & `llm_hooks.py`)
- Optional workflow when `use_local_llm=True` (or `llm_connection_id` is supplied) while leaving zero-LLM Phase 1 mode intact by default:
  1. `LocalQueryPlanner` decomposes queries into JSON subqueries and falls back automatically to `DeterministicQueryPlanner` on inference errors.
  2. Stages 3–4 retrieve, normalize, deduplicate, chunk, rank, extract claims, detect conflicts, and analyze coverage independently.
  3. `AdaptiveContextBudgetManager` dynamically packs evidence within `max_context_tokens - reserved_system_tokens - reserved_output_tokens`, prioritizing detected source conflicts and domain diversity.
  4. `LocalAnswerGenerator` produces grounded synthesis citing `[E#]` passages; `LocalEvidenceVerifier` verifies citations against allocated evidence, flags ungrounded numeric claims, and checks whether conflicts were acknowledged.

---

## Stage 5 — Adaptive Context Management Engine (`web_intelligence/app/context/`)

Prepares model-independent, validated, evidence-grounded context packages for locally hosted language models without silently exceeding context limits or dropping contradictory claims and citations.

### 1. Capabilities & Token Estimation (`token_estimator.py`)
- `inspect_adapter_context_capabilities` / `normalize_model_context_capabilities` read verified/configured limits from the active `LocalModelAdapter` (strictly honoring `CapabilityProvenance.context_length` and `.max_output_tokens` rather than guessing limits from model names).
- `resolve_token_estimator` prefers `TokenizerBackedEstimator` when the active adapter exposes a real tokenizer and otherwise falls back to `ConservativeHeuristicEstimator`, explicitly reporting `is_approximate=True` across `TokenEstimate`, `ModelContextCapabilities`, and `ContextPackage`.

### 2. Task-Aware Budget Management (`allocation_policy.py`, `budget_manager.py`)
- `DynamicContextBudgetManager` and `AllocationPolicy` partition the effective context ceiling (never exceeding verified/configured model limits) across 7 buckets: `system`, `query`, `history`, `evidence`, `metadata`, `output_reservation`, and `safety_margin`, dynamically reallocating unused history budget to evidence.
- Supports `AllocationProfile` presets (`quick_answer`, `standard_research`, `deep_research`, `long_form`, `custom`, plus compatibility alias `balanced` -> `standard_research`), optional user context ceilings, custom normalized bucket weights, `ExpectedOutputLength` hints, and strict `ContextLimitExceededError` / `ContextConfigurationError` guards.

### 3. Evidence Selection & Compression (`evidence_selector.py`, `context_compressor.py`)
- `EvidenceSelector` uses a swappable `EvidenceScoringStrategy` (default: `DeterministicEvidenceScoringStrategy`) balancing relevance, subtopic coverage, source independence, provenance classification, claim support, conflict pairing (paired `ClaimConflict` items stay together rather than discarding dissenting sources), recency, and token cost while recording excluded passages and reasons.
- `ContextCompressor` supports `none`, `extractive` (including sentence/chunk segmentation for long unpunctuated blocks), `deduplication` (minimizes redundant overlap while merging independent source references and keeping contradictory positions intact), `structured` (`StructuredFactEntry` facts, dates, numbers, entities, and `[E#]` refs), and `model_assisted` (validates citations, dates, numbers, and named entities against source evidence and falls back cleanly to `structured` compression if omitted/corrupted or if inference fails), always retaining original evidence for inspection.

### 4. History, Assembly & Pre-Inference Validation (`history_manager.py`, `context_builder.py`, `context_validator.py`, `engine.py`)
- `ConversationHistoryManager` manages `current_turn`, `session`, and persistent user-controlled `WorkspaceNote` entries in SQLite (`conversation_turns`, `workspace_notes`, plus `context_budget_configs` in `storage/database.py`), compacting older turns while preserving `decisions` and `unresolved_questions` without deleting active research evidence.
- `StructuredContextBuilder` assembles the model-independent `ContextPackage`, and `ContextAdapterConverter` formats `LLMInferenceRequest` chat messages for any adapter.
- `ContextValidator` executes 7 pre-inference checks (`token_usage_estimation`, `active_context_limit_compliance`, `output_token_reservation`, `source_reference_integrity`, `context_section_well_formedness`, `configured_budget_compliance`, `non_truncation_evidence_guard`) without ever silently truncating evidence.

### 5. Context API & Context Inspector UI
- FastAPI router `web_intelligence/app/api/routes/context.py` (orchestrated via `ContextManagementEngine`) exposes:
  - `GET /api/v1/context/capabilities`
  - `GET /api/v1/context/budget` & `PUT /api/v1/context/budget`
  - `POST /api/v1/context/preview` & `GET /api/v1/context/{research_id}`
  - `POST /api/v1/context/validate` & `POST /api/v1/context/compress`
  - Workspace note and turn helpers: `GET`/`POST /api/v1/context/notes`, `PATCH`/`DELETE /api/v1/context/notes/{note_id}`, `POST /api/v1/context/history/{research_id}`
- Workbench UI adds a **Context Inspector** tab (`ui/js/components/context_inspector.js` wired into `inspector.js`, `toolbar.js`, `state.js`, `api.js`, `app.js`, `index.html`, and `styles/components.css`) showing active model limits, adaptive allocation controls, compression status, selected vs excluded evidence passages, workspace notes, validation diagnostics, and pre-submission prompt preview.

---

## Stage 6 — Universal Local LLM Execution & Answer Generation Engine

Connects the Stage 4 Web Intelligence Engine and Stage 5 Adaptive Context Management Engine cleanly to user-hosted inference servers (`Ollama`, `LM Studio`, `llama.cpp`, `vLLM`, `LocalAI`, or custom OpenAI-compatible servers) with zero mandatory cloud dependencies.

### 1. Persistent ModelRegistry (`web_intelligence/app/llm/model_registry.py`)
- Synchronizes automatically with `LLMConnectionManager` across create/update/test/discover/select/activate/disconnect operations (`model_registry_entries` SQLite table).
- Tracks provider ID, sanitized display-safe endpoint URL (`sanitize_endpoint_for_registry`, zero secrets or query tokens), model identifier, capability provenance (`CapabilityProvenance`), connection status, and optional per-task roles (`ModelTaskRole`: `default`, `synthesis`, `outline`, `report_section`, `extraction`).
- Supports manual model registration (`POST /api/v1/llm/registry`) when an inference endpoint does not expose `/v1/models` or `/api/tags`.

### 2. Hardened Adapter Execution, Streaming & Cancellation (`web_intelligence/app/generation/executor.py`)
- Both `OpenAICompatibleAdapter` and `OllamaAdapter` support configurable `stop` sequences, per-request `timeout_seconds`, and mid-stream SSE/NDJSON error frame propagation without synthesizing fake streaming tokens when `streaming=False`.
- `InferenceExecutor` and `GenerationJobRegistry` handle cooperative mid-stream cancellation and client disconnect awareness, closing upstream HTTP streams and returning partial responses cleanly (`GenerationStatus.CANCELLED`, `partial=True`).
- All upstream auth, connectivity, timeout, and malformed-response errors normalize to secret-scrubbed `GenerationError` subclasses (`ModelNotConfiguredError`, `ContextBudgetBlockedError`, `ProviderAuthFailedError`, `ProviderTimeoutError`, `ProviderUnavailableError`, `ProviderResponseInvalidError`).

### 3. Prompt Compilation & Citation Integrity (`web_intelligence/app/generation/prompt_compiler.py`, `citations.py`)
- `PromptCompiler` supports five templates (`quick_answer`, `detailed_explanation`, `research_report`, `comparative_analysis`, `technical_report`), four formats (`prose`, `markdown`, `bullets`, `json`), and three output lengths (`short`, `medium`, `long`).
- Maps Stage 5 evidence passages and sources to concise `[S1]`..`[Sn]` citation tags, wraps untrusted webpage content in `<untrusted_source_data>` / `<untrusted_conflict_data>` boundaries with closing-tag escaping, and enforces token limits before inference via `BaseContextTokenEstimator.estimate_messages(...)`.
- `CitationResolver` and `CitationValidator` resolve `[S1]` / `[S1, S2]` (plus fallback Stage-5 `[E#]` refs) into `CitationEntry` objects with exact character spans, flag unsupported `[S#]` tags, and warn on uncited declarative claims while explicitly stating that valid citation provenance is distinct from independent factual proof.

### 4. Standard & Multi-Stage Long-Form Synthesis (`answer_generator.py`, `long_form.py`)
- `AnswerGenerator` coordinates Stage 5 `ContextManagementEngine` validation, prompt compilation, native JSON or prompted-JSON fallback parsing (`response_format`), truncation metadata (`finish_reason="length"`), omitted server usage (`usage=None`), and SQLite persistence (`generation_runs`).
- `LongFormReportGenerator` constructs section outlines from research subtopics and conflicts, enforces per-section Stage 5 context budgets (pairing conflicting sides together so dissent is never suppressed), streams section events (`outline`, `section_start`, `section_complete`, `delta`), and runs cross-section repetition and contradiction checks.

### 5. Generation API & Workbench Answer UI
- FastAPI router `web_intelligence/app/api/routes/generation.py` adds:
  - `POST /api/v1/generation` and `POST /api/v1/generation/stream` (true SSE frames: `start`, `outline`, `section_start`, `section_complete`, `delta`, `warning`, `citation`, `cancelled`, `error`, `complete`)
  - `GET /api/v1/generation`, `GET /api/v1/generation/{generation_id}`, `POST /api/v1/generation/{generation_id}/cancel`
  - `GET`/`POST /api/v1/llm/registry` and `PATCH /api/v1/llm/registry/{entry_id}/roles`
- Workbench UI adds `ui/js/components/generation_workspace.js` (multi-server model selector, template/format/length/mode controls, live streaming markdown/JSON output, **Stop Generating**, interactive clickable `[S#]` citation chips, expandable sources drawer, AI grounding notice) and `ui/js/components/generation_inspector.js` (**Gen** Inspector tab showing telemetry, server usage, citation validation diagnostics, model registry, and run history).

---

## Stage 7 — Agentic Research Orchestration Engine (`web_intelligence/app/agent/`)

Coordinates Stages 2–6 inside a bounded, persistent, user-controllable multi-iteration research loop (`AgentResearchOrchestrator`) rather than duplicating retrieval, extraction, context management, or generation subsystems.

### 1. Autonomous Research Loop & Planning (`orchestrator.py`, `research_planner.py`, `task_manager.py`, `execution_engine.py`)
- **Explicit Lifecycle (`AgentSessionStatus`)**: `pending` -> `planning` -> `executing` -> `evaluating` -> `synthesizing` -> `completed`, plus interactive `paused`, `waiting_approval`, `cancelled`, and `failed` states governed by `ALLOWED_STATUS_TRANSITIONS` (`research_state.py`).
- **DAG Planning & User Plan Edits**: `AgentResearchPlanner` builds an initial `ResearchPlanSpec` of `ResearchTask` nodes (local LLM JSON planning with deterministic multi-facet fallback) and gap-driven follow-up tasks (`origin="gap_followup"`). `TaskManager` enforces DAG acyclicity (`PlanValidationError`), priority/dependency scheduling, `max_research_depth` bounds, duplicate-task fingerprinting (`compute_task_fingerprint`), and safe `PUT /api/v1/agent/research/{id}/plan` edits for pending tasks.
- **Independent Bounded Execution (`execution_budget.py`)**: `ExecutionBudgetTracker` enforces hard limits on iterations, searches, page retrievals, concurrency, wall-clock seconds, local inference calls, estimated tokens, task depth, tool timeouts, and follow-ups per iteration across `quick` (`2`), `standard` (`3`), and `deep` (`5` iterations) presets in `default_budget_for_depth`.

### 2. Controlled Tool Registry & Privacy/Approval Gates (`tool_registry.py`, `tool_selector.py`)
- Strictly rejects arbitrary code/shell/browser execution (`FORBIDDEN_TOOL_NAMES`: `python`, `exec`, `eval`, `shell`, `bash`, `sh`, `zsh`, `subprocess`, `browser`, `playwright`, `javascript`).
- Registers eight `_StrictToolInput` (`extra="forbid"`) wrappers over existing Stage 2–6 services: `web_search`, `url_retrieve`, `page_extract`, `local_doc_retrieve`, `evidence_search`, `source_inspect`, `context_inspect`, and `local_llm_inference`.
- Supports mock-only offline execution when configured (`Settings(search_providers=["mock"], enable_mock_provider=True)`) and explicit user approval gates (`require_user_approval_for_external`, raising `ToolApprovalRequiredError` -> `PendingApprovalRequest`) resumed via `POST /api/v1/agent/research/{id}/approve`.

### 3. Observation Processing, Evidence Quality, Gap Analysis & Stopping Policy
- `ObservationProcessor` feeds raw tool results through Stage 3/4 deduplication, chunking, `HybridRelevanceEngine`, `SourceProvenanceEvaluator`, `RuleBasedClaimExtractor`, and `DeterministicClaimComparator` while tracking explicit `SourceDecisionRecord` retained/rejected audits.
- `EvidenceQualityEvaluator` classifies task and session support as `confirmed`, `corroborated`, `weak`, `conflicting`, or `unverified`, never equating lexical relevance with truth and keeping contradictory evidence visible.
- `GapAnalyzer` detects unresolved gaps across seven `GapCategory` classes (`unanswered_question`, `missing_category`, `single_source_claim`, `conflicting_results`, `outdated_information`, `weak_support`, `missing_primary_source`) while deduplicating previously tracked gaps.
- `StoppingPolicy` applies prioritized halting rules (`user_cancelled`, `waiting_user_approval`, `paused_by_user`, `model_unavailable`, hard budget limits, `tools_unavailable`, `diminishing_returns`, `objectives_covered`) cleanly through `_evaluate_gaps_and_stopping(...)` across both standard and post-approval paths.

### 4. SQLite Persistence, Operational Trace, Agent API & Workbench UI
- Adds `agent_sessions` and `agent_trace_entries` tables (`web_intelligence/app/storage/database.py`, `repositories.py`) storing checkpointed sessions and secret-scrubbed chronological `ResearchTraceEntry` records (`ResearchTraceRecorder`, 20 typed `TraceEventType` classes).
- FastAPI router `web_intelligence/app/api/routes/agent.py` exposes `/api/v1/agent/tools`, `POST`/`GET /api/v1/agent/research`, `GET /api/v1/agent/research/{id}`, `GET`/`PUT /plan`, `/tasks`, `/trace`, `/gaps`, `/pause`, `/resume`, `/cancel`, and `/approve`.
- Workbench UI adds **Agent** workspace mode (`ui/js/components/agent_workspace.js`) with live progress, budget consumption meters, Pause/Resume/Cancel controls, approval prompt cards, editable pending plan tasks, evidence verdicts, open gaps, conflicts, and cited Stage 6 synthesis, alongside the **Trace** (`agent_trace`) Inspector tab (`ui/js/components/agent_trace.js`) and restored Settings **Local Model** tab (`ui/js/components/settings_sheet.js`).
- Detailed documentation: [docs/AGENT_ORCHESTRATION.md](docs/AGENT_ORCHESTRATION.md), [docs/RESEARCH_LOOP.md](docs/RESEARCH_LOOP.md), [docs/TOOL_REGISTRY.md](docs/TOOL_REGISTRY.md), and [docs/RESEARCH_TRACE.md](docs/RESEARCH_TRACE.md).
