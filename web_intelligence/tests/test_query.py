"""Unit tests for deterministic query normalization, decomposition, and planning."""

from __future__ import annotations

from web_intelligence.app.query.planner import DeterministicQueryPlanner
from web_intelligence.app.schemas.research import SearchIntent


async def test_query_planner_decomposes_comparison_and_topics() -> None:
    planner = DeterministicQueryPlanner()
    plan = await planner.plan("Compare Aurora-SLM-4B vs quantized 7B models and privacy boundaries?")

    assert plan.intent == SearchIntent.COMPARISON
    assert len(plan.topics) >= 2
    assert len(plan.subtopics) >= 2
    assert len(plan.planned_queries) >= 2
    assert "aurora-slm-4b" in plan.keywords
    assert "privacy" in plan.keywords


async def test_query_planner_classifies_temporal_and_verification() -> None:
    planner = DeterministicQueryPlanner()
    temporal = await planner.plan("When was Aurora-SLM-4B released in 2026?")
    verify = await planner.plan("Verify benchmark claims and conflicting evidence for Aurora-SLM-4B")

    assert temporal.intent == SearchIntent.TEMPORAL
    assert verify.intent == SearchIntent.VERIFICATION
