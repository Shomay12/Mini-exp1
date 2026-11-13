"""Integration tests for AdaptiveContextBudgetManager, LocalQueryPlanner, Generator, Verifier, and Orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from web_intelligence.app.core.config import Settings
from web_intelligence.app.dependencies import build_container
from web_intelligence.app.llm.registry import AdapterRegistry
from web_intelligence.app.query.planner import DeterministicQueryPlanner
from web_intelligence.app.research.context_budget import AdaptiveContextBudgetManager
from web_intelligence.app.research.llm_hooks import (
    LocalEvidenceVerifier,
    LocalQueryPlanner,
)
from web_intelligence.app.schemas.documents import SourceProvenance, SourceRecord
from web_intelligence.app.schemas.evidence import (
    ClaimConflict,
    CorroborationSignals,
    EvidenceItem,
    ExtractedClaim,
    RelevanceBreakdown,
)
from web_intelligence.app.schemas.llm import (
    LLMConnectionConfig,
    LLMConnectionCreate,
    ModelCapabilities,
)
from web_intelligence.app.schemas.research import ResearchRequest, ResearchStatus


def _make_evidence(
    eid: str,
    sid: str,
    domain: str,
    text: str,
    *,
    score: float,
    tokens: int,
    conflict_ids: list[str] | None = None,
) -> EvidenceItem:
    url = f"https://{domain}/article"
    return EvidenceItem(
        evidence_id=eid,
        source_id=sid,
        source_url=url,
        source_domain=domain,
        source_title=f"Article on {domain}",
        chunk_index=0,
        passage_text=text,
        char_start=0,
        char_end=len(text),
        token_estimate=tokens,
        relevance=RelevanceBreakdown(combined_score=score, bm25_score=score),
        provenance=SourceProvenance(
            domain=domain,
            original_url=url,
            canonical_url=url,
        ),
        corroboration=CorroborationSignals(
            conflicting_claim_ids=conflict_ids or [],
        ),
    )


def _make_claim(
    cid: str,
    eid: str,
    sid: str,
    domain: str,
    text: str,
    value: str,
    value_type: str = "date",
) -> ExtractedClaim:
    return ExtractedClaim(
        claim_id=cid,
        source_id=sid,
        evidence_id=eid,
        source_url=f"https://{domain}/article",
        source_domain=domain,
        entity="Engine",
        attribute="year",
        statement_text=text,
        value_type=value_type,
        normalized_value=value,
    )


def test_adaptive_context_budget_balances_domains_conflicts_and_limits() -> None:
    mgr = AdaptiveContextBudgetManager()
    caps = ModelCapabilities(
        model_name="small-3b",
        backend="ollama",
        max_context_tokens=800,
        recommended_output_tokens=200,
    )

    items = [
        _make_evidence("e1", "s1", "docs.a.org", "Primary alpha statement 2024.", score=0.95, tokens=180),
        _make_evidence("e2", "s1", "docs.a.org", "Second alpha statement from same domain.", score=0.90, tokens=180),
        _make_evidence("e3", "s2", "blog.b.org", "Independent beta statement agreeing.", score=0.80, tokens=150),
        _make_evidence("e4", "s3", "news.c.org", "Conflicting statement claiming 2025 instead.", score=0.70, tokens=140, conflict_ids=["cb"]),
        _make_evidence("e5", "s4", "extra.d.org", "Excess passage exceeding remaining window.", score=0.50, tokens=300),
    ]
    conflict = ClaimConflict(
        conflict_id="conf-1",
        topic="release year",
        entity="Engine",
        attribute="year",
        conflict_type="date_mismatch",
        claim_a=_make_claim("ca", "e1", "s1", "docs.a.org", "Released in 2024.", "2024"),
        claim_b=_make_claim("cb", "e4", "s3", "news.c.org", "Released in 2025.", "2025"),
        summary="Year conflict 2024 vs 2025",
    )

    selected, plan = mgr.allocate(
        items,
        caps,
        system_prompt="Answer accurately.",
        query_prompt="When was Engine released?",
        conflicts=[conflict],
    )

    assert plan.allocated_tokens <= plan.available_evidence_tokens
    selected_ids = {item.evidence_id for item in selected}
    assert "e4" in selected_ids  # Conflict item prioritized
    assert len(plan.covered_domains) >= 2  # Diversity across domains preserved
    assert "e5" in plan.dropped_evidence_ids


@pytest.mark.asyncio
async def test_local_query_planner_uses_llm_and_falls_back_on_error() -> None:
    def ok_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200,
                json={
                    "model": "qwen3-4b",
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "subqueries": [
                                            {"query": "sqlite wal concurrency benchmarks", "subtopic": "benchmarks", "rationale": "perf"},
                                            {"query": "sqlite locking mode documentation", "subtopic": "docs", "rationale": "spec"},
                                        ]
                                    }
                                )
                            },
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        return httpx.Response(404)

    registry = AdapterRegistry()
    cfg = LLMConnectionConfig(
        id="local-q",
        name="Planner LLM",
        provider="openai-compatible",
        base_url="http://localhost:8000/v1",
        model="qwen3-4b",
    )
    adapter = registry.create(cfg, transport=httpx.MockTransport(ok_handler))
    planner = LocalQueryPlanner(DeterministicQueryPlanner(), adapter)
    plan = await planner.plan("How does SQLite WAL mode improve concurrency?")
    assert plan.planner_name.startswith("local_llm:")
    assert [q.query for q in plan.planned_queries] == [
        "sqlite wal concurrency benchmarks",
        "sqlite locking mode documentation",
    ]

    def fail_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="backend error")

    broken_adapter = registry.create(cfg, transport=httpx.MockTransport(fail_handler))
    fallback_planner = LocalQueryPlanner(DeterministicQueryPlanner(), broken_adapter)
    fallback_plan = await fallback_planner.plan("How does SQLite WAL mode improve concurrency?")
    assert fallback_plan.planner_name == "deterministic_baseline_v1"
    assert len(fallback_plan.planned_queries) >= 1


def test_local_evidence_verifier_detects_hallucinations_and_conflicts() -> None:
    verifier = LocalEvidenceVerifier()
    ev1 = _make_evidence("ev-1", "src-1", "alpha.org", "Release date confirmed as 2024 with 99% uptime.", score=0.9, tokens=20)
    ev2 = _make_evidence("ev-2", "src-2", "beta.org", "Alternative benchmark showed 2025 launch.", score=0.8, tokens=20)
    ref_map = {"E1": ev1, "E2": ev2}
    now = datetime.now(timezone.utc)
    sources = [
        SourceRecord(
            source_id="src-1",
            original_url="https://alpha.org/a",
            canonical_url="https://alpha.org/a",
            final_url="https://alpha.org/a",
            domain="alpha.org",
            retrieved_at=now,
        ),
        SourceRecord(
            source_id="src-2",
            original_url="https://beta.org/b",
            canonical_url="https://beta.org/b",
            final_url="https://beta.org/b",
            domain="beta.org",
            retrieved_at=now,
        ),
    ]
    conflict = ClaimConflict(
        conflict_id="c-1",
        topic="date",
        entity="Project",
        attribute="year",
        conflict_type="date_mismatch",
        claim_a=_make_claim("ca", "ev-1", "src-1", "alpha.org", "2024 release.", "2024"),
        claim_b=_make_claim("cb", "ev-2", "src-2", "beta.org", "2025 release.", "2025"),
        summary="2024 vs 2025",
    )

    good = verifier.verify(
        "Sources confirm 99% uptime in 2024 [E1], though conflicting reporting cites 2025 [E2].",
        ref_map,
        sources,
        [conflict],
    )
    assert good.verified is True
    assert good.invalid_citations == []
    assert good.unacknowledged_conflicts == []
    assert good.cited_source_ids == ["src-1", "src-2"]

    bad = verifier.verify(
        "Everything reached 876% in 2099 [E99].",
        ref_map,
        sources,
        [conflict],
    )
    assert bad.verified is False
    assert "E99" in bad.invalid_citations
    assert len(bad.unsupported_claims) >= 1
    assert "c-1" in bad.unacknowledged_conflicts


@pytest.mark.asyncio
async def test_end_to_end_research_orchestrator_with_optional_local_llm(tmp_path: Path) -> None:
    llm_calls: list[dict] = []

    def mock_llm_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/chat":
            body = json.loads(request.content.decode())
            llm_calls.append(body)
            if body.get("format") == "json" or "subqueries" in body["messages"][0]["content"]:
                return httpx.Response(
                    200,
                    json={
                        "model": "qwen3:4b",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "subqueries": [
                                        {"query": "Aurora-SLM-4B release date benchmarks", "subtopic": "Aurora-SLM-4B", "rationale": "benchmarks"},
                                        {"query": "small language models local inference", "subtopic": "small language models", "rationale": "architecture"},
                                    ]
                                }
                            ),
                        },
                        "done": True,
                    },
                )
            return httpx.Response(
                200,
                json={
                    "model": "qwen3:4b",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "Retrieved sources document Aurora-SLM-4B benchmarks [E1], while conflicting reports "
                            "disagree across sources on exact figures [E2]."
                        ),
                    },
                    "done": True,
                    "prompt_eval_count": 120,
                    "eval_count": 28,
                },
            )
        return httpx.Response(404)

    settings = Settings(
        database_path=tmp_path / "orch_llm.db",
        search_providers=["mock"],
        enable_mock_provider=True,
        enable_live_network=False,
        per_host_min_interval_seconds=0.0,
    )
    container = build_container(
        settings,
        llm_transport=httpx.MockTransport(mock_llm_handler),
    )
    await container.database.initialize()
    await container.llm_manager.initialize()

    pkg_zero = await container.orchestrator.run_research(
        ResearchRequest(
            query="Aurora-SLM-4B release date and small language models benchmarks",
            use_local_llm=False,
        )
    )
    assert pkg_zero.research_status == ResearchStatus.COMPLETED
    assert pkg_zero.synthesis is None
    assert pkg_zero.context_allocation is None
    assert len(pkg_zero.evidence) > 0
    assert llm_calls == []

    await container.llm_manager.create_connection(
        LLMConnectionCreate(
            name="Local Ollama",
            provider="ollama",
            preset="ollama",
            base_url="http://localhost:11434",
            model="qwen3:4b",
            context_length=8192,
            max_output_tokens=1024,
            activate=True,
        )
    )

    pkg_llm = await container.orchestrator.run_research(
        ResearchRequest(
            query="Aurora-SLM-4B release date and small language models benchmarks",
            use_local_llm=True,
        )
    )
    assert pkg_llm.research_status == ResearchStatus.COMPLETED
    assert pkg_llm.query_plan is not None
    assert pkg_llm.query_plan.planner_name.startswith("local_llm:")
    assert pkg_llm.context_allocation is not None
    assert pkg_llm.context_allocation.allocated_tokens > 0
    assert pkg_llm.synthesis is not None
    assert "[E1]" in pkg_llm.synthesis.answer
    assert len(pkg_llm.synthesis.cited_evidence_ids) >= 1
    assert pkg_llm.synthesis.verification.invalid_citations == []
    assert "synthesizing" in pkg_llm.progress.completed_stages

    reloaded = await container.repository.get_research_package(pkg_llm.research_id)
    assert reloaded is not None
    assert reloaded.synthesis is not None
    assert reloaded.synthesis.answer == pkg_llm.synthesis.answer
    assert reloaded.context_allocation is not None
