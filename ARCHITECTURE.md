# Architecture — Local-First Web Intelligence Engine

## 1. System Overview

The **Local-First Web Intelligence Engine** is Phase 1 of the Open-Source Adaptive Research Intelligence system. Its single responsibility is to convert a natural-language research query into a verified, traceable, model-independent structured evidence package without depending on any cloud LLM.

**Core principle**: *The internet supplies external documents; the local engine retrieves, validates, extracts, deduplicates, scores, compares, and structures that evidence locally.*

---

## 2. Layered Architecture & Component Responsibilities

| Layer | Package | Responsibilities |
| :--- | :--- | :--- |
| **API & UI** | `app/api/`, `app/ui/` | FastAPI HTTP endpoints (`/api/v1/*`), Pydantic validation, consistent error contracts, minimal information-dense research workspace UI. |
| **Core & Security** | `app/core/` | Environment settings (`pydantic-settings`), secret-redacting structured logging, URL validation & canonicalization, SSRF & DNS-rebinding defense, per-host rate limiter. |
| **Privacy Layer** | `app/privacy/` | Outbound query sanitizer (PII/secret/path stripping), explicit privacy policy enforcement, proxy-ready `PrivacyGateway`, outbound query disclosure ledger. |
| **Search Abstraction** | `app/providers/` | `SearchProvider` interface, `MockSearchProvider` (offline fixtures), `BraveSearchProvider`, `TavilySearchProvider`, `ExaSearchProvider`, and failover `SearchProviderRouter`. |
| **Query Engine** | `app/query/` | Deterministic query normalization, keyword/topic extraction, intent classification, multi-angle query decomposition, and replaceable `QueryPlanner` interface. |
| **Web Retrieval** | `app/retrieval/` | Sanitized search coordinator, `robots.txt` manager, SSRF-safe async `WebFetcher` (hop-by-hop redirect checks, byte/timeout caps), `trafilatura` + `BeautifulSoup` extractor, document normalizer. |
| **Evidence Engine (Stage 4)** | `app/evidence/` | Multi-level deduplicator, passage-aware adaptive chunker, hybrid lexical/pluggable-semantic relevance scorer, source provenance & corroboration evaluator, deterministic claim/conflict detector, research coverage evaluator. |
| **Research & Future LLM** | `app/research/` | End-to-end `ResearchOrchestrator`, `ContextBuilder` (`ResearchPackage`), `ResearchState`, and future Phase 2 protocols (`LocalModelAdapter`, `ContextBudgetManager`, `TokenEstimator`, etc.). |
| **Local Storage** | `app/storage/` | Async SQLite persistence (`aiosqlite`, WAL mode) for runs, disclosed search queries, sources, evidence chunks, conflicts, and retention/deletion controls. |

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
  [ContentExtractor + DocumentNormalizer] ──► Trafilatura + BS4 + metadata/date validation
      │
      ▼
  [MultiLevelDeduplicator] ──► exact URL -> canonical URL -> SHA-256 -> shingle near-dup
      │
      ▼
  [PassageChunker] ──► heading-preserving, sentence-aware passages + char offsets + token est
      │
      ▼
  [HybridRelevanceEngine] ──► BM25 + keyword + heading + query coverage (+ optional embedding/reranker)
      │
      ▼
  [SourceProvenanceEvaluator] ──► separate provenance / relevance / corroboration signals
      │
      ▼
  [ClaimExtractor + ConflictDetector] ──► structured date/numeric/polarity comparison
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

- `POST /api/v1/search` — Sanitize query, execute search across configured providers, return `SearchResponse` with normalized results and disclosed outbound queries.
- `POST /api/v1/research` — Execute full research pipeline (`ResearchRequest`) and return `ResearchPackage` (supports synchronous completion or background polling).
- `GET /api/v1/research` — List local research runs (`ResearchRunSummary[]`).
- `GET /api/v1/research/{research_id}` — Fetch full persisted `ResearchPackage` and status.
- `DELETE /api/v1/research/{research_id}` — Delete a research run and cascaded local artifacts (privacy retention control).
- `GET /api/v1/sources/{source_id}` — Inspect full `SourceRecord`, metadata, provenance, duplicate mapping, and extracted normalized content.
- `GET /api/v1/health` — Service health, database status, and per-provider availability/configuration status.
- `GET /api/v1/config` — Safe configuration snapshot (secrets excluded/masked, privacy policy summary).

---

## 5. SQLite Database Schema

Stored locally in SQLite (`WAL` mode, `PRAGMA foreign_keys = ON`):

1. `research_runs`: `id TEXT PK`, `original_query TEXT`, `status TEXT`, `progress_json TEXT`, `package_json TEXT`, `error_message TEXT`, `created_at TEXT`, `updated_at TEXT`, `expires_at TEXT`
2. `search_queries`: `id TEXT PK`, `research_id TEXT FK`, `provider TEXT`, `original_subquery TEXT`, `sanitized_query TEXT`, `redactions_json TEXT`, `intent TEXT`, `result_count INTEGER`, `disclosed_at TEXT`
3. `sources`: `id TEXT PK`, `research_id TEXT FK`, `original_url TEXT`, `canonical_url TEXT`, `final_url TEXT`, `domain TEXT`, `title TEXT`, `author TEXT`, `publisher TEXT`, `published_at TEXT`, `retrieved_at TEXT`, `content_hash TEXT`, `extraction_status TEXT`, `extraction_error TEXT`, `is_duplicate INTEGER`, `duplicate_of_source_id TEXT`, `duplicate_reason TEXT`, `main_content TEXT`, `metadata_json TEXT`, `provenance_json TEXT`
4. `evidence_items`: `id TEXT PK`, `research_id TEXT FK`, `source_id TEXT FK`, `chunk_index INTEGER`, `section_heading TEXT`, `passage_text TEXT`, `char_start INTEGER`, `char_end INTEGER`, `token_estimate INTEGER`, `relevance_json TEXT`, `corroboration_json TEXT`
5. `claim_conflicts`: `id TEXT PK`, `research_id TEXT FK`, `topic TEXT`, `entity TEXT`, `attribute TEXT`, `conflict_type TEXT`, `claim_a_json TEXT`, `claim_b_json TEXT`, `summary TEXT`

---

## 6. Privacy Boundaries

- **Local-only processing & history**: Downloaded HTML, extracted text, chunks, claims, and SQLite state remain local.
- **No cloud LLM dependency & zero telemetry**: Core intelligence relies on deterministic/local algorithms; outbound telemetry is structurally disabled.
- **Explicit outbound footprint**: Search queries (sent to configured search APIs) and webpage GET requests (sent to target domains) are inherently observable by those remote endpoints and network intermediaries. All outbound requests flow exclusively through `PrivacyGateway` after `QuerySanitizer` redaction and are logged in `search_queries` for user inspection.
- **SSRF isolation**: Outbound fetches never permit loopback, RFC1918 private, CGNAT, link-local, cloud metadata (`169.254.169.254`), or non-HTTP(S) schemes—including across redirects and DNS resolution.

---

## 7. Stage 4 Methodology Notes

- **Deduplication**: Four-tier pipeline (`exact_url` → `canonical_url` → `exact_content_hash` → `near_duplicate` via 5-word shingle Jaccard + length-ratio guard so distinct articles on the same topic are not falsely merged).
- **Chunking**: Heading-aware, paragraph/sentence-preserving adaptive segmentation with exact character offsets and replaceable `TokenEstimator`.
- **Hybrid Ranking**: Deterministic pure-Python BM25 + keyword overlap + heading match + query-term coverage, with replaceable `EmbeddingProvider`, `SemanticRetriever`, and `CrossEncoderReranker` interfaces. Relevance scores are explicitly isolated from factual confidence.
- **Conflict Detection**: Rule-based normalized extraction of dates, numbers+units, and polarity around shared entity/attribute contexts across independent sources, backed by a replaceable `ClaimComparator` interface for future local LLMs. Preserves both claims verbatim without picking a winner.
- **Research Coverage**: Evaluates subtopics from query decomposition against ranked evidence, corroboration count, and conflicts to classify `covered` / `partial` / `missing` topics and explicit unanswered gaps.
