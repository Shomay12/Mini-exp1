"""End-to-end offline multi-iteration CLI demonstration of Stage 7 Agentic Research Orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from web_intelligence.app.agent.models import (
    AgentResearchRequest,
    ExecutionBudgetConfig,
    ResearchDepth,
)
from web_intelligence.app.core.config import Settings
from web_intelligence.app.dependencies import build_container
from web_intelligence.app.providers.fixtures import (
    build_fixture_transport,
    fixture_dns_resolver,
)


async def main() -> None:
    query = (
        "Evaluate small language models for local research assistants covering "
        "Aurora-SLM-4B release date, Apple M2 benchmark latency, privacy boundaries, "
        "and neuromorphic hardware support"
    )
    with TemporaryDirectory() as tmpdir:
        settings = Settings(
            database_path=Path(tmpdir) / "agent_demo.db",
            search_providers=["mock"],
            enable_mock_provider=True,
            per_host_min_interval_seconds=0.0,
        )
        container = build_container(
            settings,
            transport_factory=build_fixture_transport,
            dns_resolver=fixture_dns_resolver,
        )
        await container.database.initialize()
        assert container.agent_orchestrator is not None

        request = AgentResearchRequest(
            query=query,
            research_objective=(
                "Determine Aurora-SLM-4B specifications and release conflicts, verify M2 "
                "inference benchmarks, document privacy boundaries, and check hardware support."
            ),
            research_depth=ResearchDepth.STANDARD,
            use_local_llm=False,
            allow_external_search=False,
            execution_budget=ExecutionBudgetConfig(
                max_iterations=3,
                max_search_calls=8,
                max_page_retrievals=10,
                max_concurrent_tasks=2,
                max_execution_seconds=60.0,
                max_inference_calls=0,
                max_research_depth=2,
                max_followups_per_iteration=3,
            ),
            max_results_per_query=6,
            temporal_focus=True,
        )

        session = await container.agent_orchestrator.start_research(request, background=False)
        trace_entries = await container.repository.list_agent_trace_entries(session.research_id)

        print(f"Agent Research ID : {session.research_id}")
        print(f"Final Status      : {session.status.value}")
        print(f"Iterations Run    : {session.budget_usage.iterations_completed}")
        print(
            f"Budget Usage      : searches={session.budget_usage.search_calls}/"
            f"{session.execution_budget.max_search_calls}, "
            f"fetches={session.budget_usage.page_retrievals}/"
            f"{session.execution_budget.max_page_retrievals}, "
            f"tokens~={session.budget_usage.estimated_tokens}, "
            f"elapsed={session.budget_usage.elapsed_seconds:.2f}s"
        )

        print("\n--- Research Plan & Iteration Tasks ---")
        tasks = session.research_plan.tasks if session.research_plan else []
        for t in tasks:
            print(
                f"  [iter={t.iteration_created} depth={t.depth} {t.origin}] "
                f"{t.task_id} ({t.status.value}) :: {t.question}"
            )
            if t.result_summary:
                print(f"    summary: {t.result_summary[:140]}")

        print("\n--- Retained vs Rejected Sources ---")
        for dec in session.source_decisions:
            tag = "RETAINED" if dec.retained else "REJECTED"
            print(f"  [{tag}] {dec.domain} | {dec.reason} | {dec.url}")

        print("\n--- Preserved Claim Conflicts ---")
        if not session.conflicts:
            print("  (none)")
        for conf in session.conflicts:
            print(f"  [{conf.conflict_type}] {conf.summary}")

        print("\n--- Evidence Evaluation & Gaps Addressed ---")
        if session.evidence_evaluation:
            print(f"  {session.evidence_evaluation.summary}")
        for gap in session.identified_gaps:
            flag = f"RESOLVED by {gap.addressed_by_task_id}" if gap.resolved else "OPEN"
            print(f"  [{gap.severity.upper()} {gap.category.value}] ({flag}) {gap.description[:120]}")

        print("\n--- Stopping Decision ---")
        if session.stopping_decision:
            reason_val = (
                session.stopping_decision.reason.value
                if session.stopping_decision.reason
                else "none"
            )
            print(
                f"  should_stop={session.stopping_decision.should_stop} | "
                f"reason={reason_val} | {session.stopping_decision.explanation}"
            )

        print(f"\n--- Operational Trace ({len(trace_entries)} entries) ---")
        for entry in trace_entries:
            task_tag = f" task={entry.task_id}" if entry.task_id else ""
            tool_tag = f" tool={entry.tool_name}" if entry.tool_name else ""
            print(
                f"  #{entry.sequence:02d} [iter={entry.iteration} {entry.phase}] "
                f"{entry.event_type.value}{task_tag}{tool_tag} :: {entry.reason[:100]}"
            )

        print("\n--- Grounded Synthesis Preview ---")
        if session.generation_result:
            print(
                f"  citations={len(session.generation_result.citations)}, "
                f"model={session.generation_result.model_id}"
            )
            print(session.generation_result.content[:900])


if __name__ == "__main__":
    asyncio.run(main())
