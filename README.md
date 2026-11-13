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
# Run full automated unit & integration test suite (63 tests)
.venv/bin/pytest -v

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
| `GET` | `/api/v1/health` | Service health, SQLite readiness, and search provider availability |
| `GET` | `/api/v1/config` | Safe non-secret configuration and privacy policy snapshot |

See [ARCHITECTURE.md](ARCHITECTURE.md) for component design, data-flow diagram, SQLite schema, privacy boundaries, and Stage 4 methodology details, and [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md) for implementation milestones and decisions.
