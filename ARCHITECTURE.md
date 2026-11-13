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
