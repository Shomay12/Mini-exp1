# Local-First Web Intelligence Engine

**Phase 1 (Stages 2–4) of Open-Source Adaptive Research Intelligence** — a standalone, modular, local-first web research infrastructure that searches external sources, safely retrieves and extracts webpages, deduplicates documents, ranks evidence passages, evaluates provenance, detects conflicting claims, analyzes research coverage gaps, and outputs a model-independent `ResearchPackage` without mandatory cloud LLM dependencies.

## Capabilities Implemented

- **Search Provider Abstraction**: Official REST adapters for **Brave**, **Tavily**, and **Exa** with exponential backoff retries, automatic multi-provider failover, and a zero-credential **`MockSearchProvider`** + controlled HTML fixtures (`fixtures.py`) for deterministic offline runs and testing.
- **Privacy & SSRF Security**: Outbound query sanitization (PII/secret/path redaction), explicit outbound query disclosure, `OUTBOUND_HTTP_PROXY`/`OUTBOUND_HTTPS_PROXY`-ready `PrivacyGateway`, hop-by-hop redirect & DNS-rebinding SSRF blocking, `robots.txt` enforcement, per-host rate limiting, and local TTL retention controls.
- **Web Extraction & Normalization**: Primary extraction via `trafilatura` (`BeautifulSoup` fallback), strict metadata-only publication-date validation (never inventing missing dates from body text), and structured heading/paragraph preservation.
- **Stage 4 Evidence Intelligence**:
  - **Multi-level deduplication**: exact URL, canonical URL, SHA-256 content hash, and 5-word shingle Jaccard near-duplicate detection (`alias_urls` preserved).
  - **Passage-aware chunking**: heading- and sentence-preserving adaptive chunks with exact `[char_start, char_end]` offsets and replaceable `TokenEstimator`.
  - **Hybrid relevance ranking**: standalone pure-Python Okapi BM25 + keyword + heading + query-term coverage, with replaceable `EmbeddingProvider`, `SemanticRetriever`, and `CrossEncoderReranker` interfaces (strictly separated from factual confidence).
  - **Source provenance & corroboration**: separate structural classification (`primary`/`secondary`/`reference`/`unknown`), publication metadata, independent domain count, and mirror clustering.
  - **Claim & conflict detection**: deterministic extraction and cross-source comparison of dates, numerical quantities + normalized units, and polarity, preserving conflicting statements side-by-side without picking a winner, plus replaceable `ClaimComparator` interface.
  - **Research coverage evaluation**: classifies decomposed query subtopics into `covered`, `partial`, and `missing`, surfacing single-source claims, conflicts, and unanswered questions.
- **Universal Local LLM Connections & Model Registry** *(optional, no cloud LLM required)*: Connect user-hosted **Ollama**, **LM Studio**, **llama.cpp**, **vLLM**, **LocalAI**, or any OpenAI-compatible HTTP endpoint via the UI (`Local Model` sheet) or `/api/v1/llm/*` API, with capability verification (`verified` vs `configured` vs `unknown`), persistent multi-server `ModelRegistry` (`/api/v1/llm/registry`), manual model registration, and task roles. Zero-LLM mode stays fully functional when disconnected.
- **Stage 5 Adaptive Context Management Engine**: Token estimation, 7-bucket dynamic context allocation, conflict-pairing evidence selection, extractive/dedup/structured/model-assisted compression, conversation history & workspace notes, pre-inference validation, and the **Context Inspector** UI tab.
- **Stage 6 Universal Local LLM Execution & Answer Generation Engine**: Security-hardened prompt compiler (`[S1]`..`[Sn]` citations, untrusted webpage data isolation), native or prompted JSON mode, live SSE streaming with mid-stream cooperative cancellation preserving partial output, multi-stage long-form report synthesis with cross-section repetition/contradiction checks, separate `CitationValidator`, **Research Answer Workspace**, and **Gen** telemetry Inspector tab.
- **API, Local SQLite Persistence & Research UI**: FastAPI `/api/v1/*` endpoints and a minimal, information-dense browser workbench.

## Installation & Setup

```bash
# 1. Create an isolated Python 3.11 virtual environment via uv
uv venv --python 3.11 .venv
source .venv/bin/activate

# 2. Install pinned runtime and test dependencies
uv pip install -r requirements.txt

# 3. Configure environment (defaults to offline MockSearchProvider)
cp .env.example .env
```

### Optional Live Search Providers
To query live internet search APIs instead of `mock`, edit `.env`:
```env
SEARCH_PROVIDERS="brave,tavily,exa"
BRAVE_API_KEY="your-brave-key"
TAVILY_API_KEY="your-tavily-key"
EXA_API_KEY="your-exa-key"
```
Never commit `.env` or hardcode credentials.

## Running Tests & Demo

```bash
# Run full automated Stages 1–6 unit & integration test suite (109 tests)
.venv/bin/pytest -v --ignore=web_intelligence/tests/test_agent_unit.py --ignore=web_intelligence/tests/test_agent_integration_and_failures.py

# Run end-to-end CLI research demonstration using MockSearchProvider
.venv/bin/python -m web_intelligence.scripts.demo_research
```

## Starting the API & Research Interface

```bash
.venv/bin/uvicorn web_intelligence.app.main:app --host 127.0.0.1 --port 8000
```

- Research Workspace UI: `http://127.0.0.1:8000/`
- Interactive OpenAPI Docs: `http://127.0.0.1:8000/docs`

## API Endpoints

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `POST` | `/api/v1/search` | Sanitize query and return normalized search hits + outbound query disclosures |
| `POST` | `/api/v1/research` | Execute full Stage 3–4 research pipeline and return `ResearchPackage` |
| `GET` | `/api/v1/research` | List locally stored research runs |
| `GET` | `/api/v1/research/{research_id}` | Retrieve persisted `ResearchPackage` and status |
| `DELETE` | `/api/v1/research/{research_id}` | Delete local research run and all cascaded artifacts |
| `GET` | `/api/v1/sources/{source_id}` | Inspect full `SourceRecord`, provenance, duplicate links, and extracted text |
| `GET` | `/api/v1/health` | Service health, SQLite readiness, search providers, and active local LLM summary |
| `GET` | `/api/v1/config` | Safe non-secret configuration and privacy policy snapshot |
| `GET` | `/api/v1/llm/presets` | List built-in local/self-hosted LLM server presets |
| `GET`/`POST` | `/api/v1/llm/connections` | List saved connections (keys redacted) or save a new LLM endpoint |
| `POST` | `/api/v1/llm/connections/{id}/test` | Probe endpoint health, discover models, and inspect capabilities |
| `GET`/`POST` | `/api/v1/llm/connections/{id}/models` / `model` | Discover available models or apply a manual model identifier |
| `POST` | `/api/v1/llm/connections/{id}/activate` / `/api/v1/llm/disconnect` | Attach an active local model or revert to Zero-LLM mode |
| `POST` | `/api/v1/llm/connections/{id}/prompt` | Run non-streaming JSON or live SSE streaming verification prompt |
| `GET`/`POST`/`PATCH` | `/api/v1/llm/registry` (`/{entry_id}/roles`) | List multi-server registered models, register manual models, or assign task roles |
| `GET`/`POST`/`PUT` | `/api/v1/context/*` | Capabilities, adaptive budget rules, context preview, compression, validation & workspace notes |
| `POST` | `/api/v1/generation` / `/api/v1/generation/stream` | Execute grounded local LLM synthesis or SSE streaming answer/long-form report |
| `GET`/`POST` | `/api/v1/generation/{id}` / `/api/v1/generation/{id}/cancel` | Retrieve persisted generation runs (including partial) or cooperatively cancel in-flight generation |

See [ARCHITECTURE.md](ARCHITECTURE.md) for component design, data-flow diagram, SQLite schema, privacy boundaries, and Stage 4 methodology details, and [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for implementation milestones and decisions.
