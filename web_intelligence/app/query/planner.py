"""Replaceable QueryPlanner interface and deterministic Phase 1 baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from web_intelligence.app.query.decomposition import decompose_query
from web_intelligence.app.schemas.research import QueryPlan


class QueryPlanner(ABC):
    """Interface allowing a future local LLM to replace the deterministic planner."""

    name: str

    @abstractmethod
    async def plan(self, raw_query: str) -> QueryPlan:
        """Convert a natural-language research request into a structured QueryPlan."""


class DeterministicQueryPlanner(QueryPlanner):
    """Rule-based query planner requiring zero LLM calls."""

    name = "deterministic_baseline_v1"

    async def plan(self, raw_query: str) -> QueryPlan:
        norm, intent, topics, keywords, subtopics, planned_queries = decompose_query(raw_query)
        return QueryPlan(
            original_query=raw_query,
            normalized_query=norm,
            intent=intent,
            topics=topics,
            keywords=keywords,
            subtopics=subtopics,
            planned_queries=planned_queries,
            planner_name=self.name,
        )
