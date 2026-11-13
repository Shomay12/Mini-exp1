"""Deterministic offline fixture catalog for MockSearchProvider and E2E testing."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class FixturePage:
    """Controlled webpage fixture representing realistic retrieval and edge cases."""

    url: str
    title: str
    snippet: str
    published_at: str | None
    html: str
    status_code: int = 200
    headers: dict[str, str] | None = None
    keywords: tuple[str, ...] = ()


DEFAULT_FIXTURES: list[FixturePage] = [
    FixturePage(
        url="https://docs.openmodels.org/slm/overview",
        title="Small Language Models Architecture and Local Deployment Guide",
        snippet="Technical documentation covering 3B-8B quantized models, memory footprints, and GGUF inference.",
        published_at="2026-02-10T00:00:00Z",
        keywords=("small", "language", "models", "local", "quantized", "architecture", "deployment"),
        html="""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Small Language Models Architecture and Local Deployment Guide</title>
  <meta name="author" content="OpenModels Engineering Team" />
  <meta property="og:site_name" content="OpenModels Official Docs" />
  <meta property="article:published_time" content="2026-02-10T00:00:00Z" />
  <link rel="canonical" href="https://docs.openmodels.org/slm/overview" />
</head>
<body>
  <article>
    <h1>Small Language Models Architecture and Local Deployment Guide</h1>
    <h2>Memory and Quantization Characteristics</h2>
    <p>Small language models (SLMs) in the 3B to 8B parameter range run locally on consumer hardware when quantized to 4-bit GGUF format. A 4-bit quantized 7B model typically requires 4.5 GB of RAM during inference.</p>
    <p>The flagship Aurora-SLM-4B model was released in January 2026 under an Apache 2.0 license with a 32k context window.</p>
    <h2>Retrieval-Augmented Grounding</h2>
    <p>Pairing small language models with local BM25 and hybrid evidence retrieval reduces factual hallucination rates by supplying verifiable external citations rather than relying solely on parametric weights.</p>
    <h2>Benchmark Latency</h2>
    <p>On Apple M2 hardware, Aurora-SLM-4B achieves 48 tokens per second using llama.cpp without cloud API dependencies.</p>
  </article>
</body>
</html>""",
    ),
    FixturePage(
        url="https://research-digest.example.com/articles/slm-2026-review?utm_source=newsletter",
        title="2026 Field Review of Compact Local Language Models",
        snippet="Independent benchmark review examining Aurora-SLM-4B release timeline, latency, and grounding.",
        published_at="2026-03-18T00:00:00Z",
        keywords=("small", "language", "models", "local", "review", "benchmarks", "aurora"),
        html="""<!DOCTYPE html>
<html lang="en">
<head>
  <title>2026 Field Review of Compact Local Language Models</title>
  <meta name="author" content="Dr. Maya Lin" />
  <meta property="og:site_name" content="Research Digest" />
  <meta name="date" content="2026-03-18" />
  <link rel="canonical" href="https://research-digest.example.com/articles/slm-2026-review" />
</head>
<body>
  <article>
    <h1>2026 Field Review of Compact Local Language Models</h1>
    <h2>Release Timeline and Specifications</h2>
    <p>According to vendor distribution records, Aurora-SLM-4B was released in March 2026 with an expanded 32k context window and Apache 2.0 licensing.</p>
    <p>Independent hardware testing confirms that 4-bit quantized 7B models consume roughly 4.5 GB of RAM on consumer laptops.</p>
    <h2>Context Allocation Challenges</h2>
    <p>Smaller models degrade quickly when flooded with redundant scraped snippets. Adaptive context budgeting and passage deduplication consistently improve multi-source synthesis quality.</p>
  </article>
</body>
</html>""",
    ),
    FixturePage(
        url="https://mirror.news-syndicate.example.net/slm-overview-copy",
        title="Small Language Models Architecture and Local Deployment Guide (Syndicated)",
        snippet="Syndicated mirror of OpenModels technical guide on quantized 3B-8B local models.",
        published_at="2026-02-11T00:00:00Z",
        keywords=("small", "language", "models", "local", "quantized"),
        html="""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Small Language Models Architecture and Local Deployment Guide (Syndicated)</title>
  <meta property="og:site_name" content="News Syndicate Mirror" />
</head>
<body>
  <article>
    <h1>Small Language Models Architecture and Local Deployment Guide</h1>
    <h2>Memory and Quantization Characteristics</h2>
    <p>Small language models (SLMs) in the 3B to 8B parameter range run locally on consumer hardware when quantized to 4-bit GGUF format. A 4-bit quantized 7B model typically requires 4.5 GB of RAM during inference.</p>
    <p>The flagship Aurora-SLM-4B model was released in January 2026 under an Apache 2.0 license with a 32k context window.</p>
    <h2>Retrieval-Augmented Grounding</h2>
    <p>Pairing small language models with local BM25 and hybrid evidence retrieval reduces factual hallucination rates by supplying verifiable external citations rather than relying solely on parametric weights.</p>
    <h2>Benchmark Latency</h2>
    <p>On Apple M2 hardware, Aurora-SLM-4B achieves 48 tokens per second using llama.cpp without cloud API dependencies.</p>
  </article>
</body>
</html>""",
    ),
    FixturePage(
        url="https://arxiv-notes.example.edu/papers/evidence-chunking-2026",
        title="Passage-Aware Chunking and Hybrid Retrieval for Local Research Engines",
        snippet="Academic note without publication timestamp analyzing BM25 + semantic retrieval and evidence conflicts.",
        published_at=None,
        keywords=("evidence", "retrieval", "chunking", "bm25", "small", "language", "models", "research"),
        html="""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Passage-Aware Chunking and Hybrid Retrieval for Local Research Engines</title>
  <meta name="author" content="J. Patel, K. Okafor" />
  <meta property="og:site_name" content="Example University Technical Reports" />
</head>
<body>
  <article>
    <h1>Passage-Aware Chunking and Hybrid Retrieval for Local Research Engines</h1>
    <h2>Why Passage Boundaries Matter</h2>
    <p>Fixed-token sliding windows frequently split factual claims from their qualifying units or entity subjects. Preserving headings and sentence boundaries protects citation attribution.</p>
    <h2>Conflicting Benchmarks in Compact Models</h2>
    <p>In contrast to vendor figures, laboratory measurements reported that Aurora-SLM-4B reached 34 tokens per second on Apple M2 hardware due to thermal throttling during sustained runs.</p>
    <p>Hybrid BM25 retrieval paired with query-term coverage outperforms pure dense embeddings on exact entity and date lookup tasks.</p>
  </article>
</body>
</html>""",
    ),
    FixturePage(
        url="https://privacy-engineering.example.org/local-first-search",
        title="Privacy Boundaries in Web-Connected Local AI Assistants",
        snippet="Design patterns for query sanitization, SSRF prevention, and proxy isolation.",
        published_at="2026-04-02T00:00:00Z",
        keywords=("privacy", "local", "search", "ssrf", "sanitization", "web", "intelligence"),
        html="""<!DOCTYPE html>
<html lang="en">
<head>
  <title>Privacy Boundaries in Web-Connected Local AI Assistants</title>
  <meta property="og:site_name" content="Privacy Engineering Collective" />
  <meta property="article:published_time" content="2026-04-02T00:00:00Z" />
</head>
<body>
  <article>
    <h1>Privacy Boundaries in Web-Connected Local AI Assistants</h1>
    <h2>Network Egress Reality</h2>
    <p>A local-first research application cannot honestly claim zero online footprint whenever it queries remote search APIs or fetches webpages over HTTPS. Remote servers observe query strings and network metadata.</p>
    <h2>Mitigations</h2>
    <p>Query sanitization strips personal identifiers before search submission, while strict SSRF guards block access to localhost, private IP ranges, and cloud metadata endpoints across redirects.</p>
  </article>
</body>
</html>""",
    ),
]


def build_fixture_transport(pages: list[FixturePage] | None = None) -> httpx.MockTransport:
    """Create an httpx.MockTransport serving robots.txt and controlled fixture HTML."""
    catalog = pages if pages is not None else DEFAULT_FIXTURES
    by_url: dict[str, FixturePage] = {page.url: page for page in catalog}

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if request.url.path == "/robots.txt":
            return httpx.Response(
                status_code=200,
                text="User-agent: *\nAllow: /\n",
                headers={"Content-Type": "text/plain; charset=utf-8"},
            )
        page = by_url.get(url_str)
        if page is not None:
            headers = {"Content-Type": "text/html; charset=utf-8"}
            if page.headers:
                headers.update(page.headers)
            return httpx.Response(
                status_code=page.status_code,
                text=page.html,
                headers=headers,
            )
        return httpx.Response(status_code=404, text="<html><body>Not found</body></html>")

    return httpx.MockTransport(handler)


async def fixture_dns_resolver(hostname: str, port: int) -> list[str]:
    """Deterministic public-IP DNS resolver for offline fixture execution."""
    return ["93.184.216.34"]
