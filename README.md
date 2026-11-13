# Local-First Web Intelligence Engine

**Phase 1 of Open-Source Adaptive Research Intelligence** — a standalone, modular, local-first web research infrastructure that acquires, validates, extracts, deduplicates, ranks, and structures internet evidence for downstream local small language models without mandatory cloud LLM dependencies.

## Features (Phase 1 — Stages 2–4)

- **Search Provider Abstraction**: Official API adapters for **Brave**, **Tavily**, and **Exa** with automatic multi-provider failover, retries, rate-limit handling, and a zero-credential **`MockSearchProvider`** for deterministic offline execution and testing.
- **Privacy & SSRF Security**: Outbound query sanitization (PII/secret redaction), explicit outbound query disclosure, proxy-ready `PrivacyGateway`, strict URL validation, hop-by-hop redirect SSRF protection, DNS-rebinding defense, `robots.txt` compliance, and per-host throttling.
- **Content Extraction & Normalization**: Primary extraction via `trafilatura` with `BeautifulSoup` fallback, validated publication-date parsing, and strict preservation of missing metadata (`None` instead of invented dates/authors).
- **Stage 4 Evidence Intelligence**:
  - Four-level deduplication (exact URL, canonical URL, SHA-256 content hash, shingle near-duplicate).
  - Heading- and sentence-aware passage chunking with character offsets and token estimation.
  - Pure-Python hybrid BM25 + keyword + heading relevance ranking with replaceable `EmbeddingProvider` / `CrossEncoderReranker` interfaces.
  - Separated Source Provenance, Relevance, and Corroboration signals.
  - Deterministic structured claim extraction & conflict detection (dates, quantities, polarity) across independent sources.
  - Subtopic Research Coverage evaluation highlighting covered topics, partial single-source areas, conflicts, and unanswered gaps.
- **Local SQLite Persistence, FastAPI API & Research UI**: Inspectable sources, evidence passages, conflicting claims, and downloadable model-independent JSON research packages.

## Quick Start

```bash
# 1. Create virtual environment (Python 3.11)
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env

# 3. Run the automated test suite
pytest

# 4. Start the API & Research UI
uvicorn web_intelligence.app.main:app --reload
```

Open `http://127.0.0.1:8000` in your browser for the research workspace UI, or `http://127.0.0.1:8000/docs` for OpenAPI documentation.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system design, data-flow diagrams, API contracts, database schema, and privacy boundaries.
