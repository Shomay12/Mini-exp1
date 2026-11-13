# Stage 7 — Autonomous Research Loop, Gap Analysis & Stopping Policy

`AgentResearchOrchestrator` executes a structured, iterative loop where each phase checkpoints state to SQLite and appends chronological trace entries to `agent_trace_entries`.

---

## 1. End-to-End Execution Cycle

```
1. PLAN        AgentResearchPlanner produces ResearchPlanSpec (DAG of ResearchTask nodes)
                 └─ TaskManager.validate_dag verifies dependencies and zero cycles
2. SELECT      TaskManager.select_next_batch picks READY tasks within max_concurrent_tasks
                 └─ Enforces max_research_depth & dependency failure/skip propagation
3. EXECUTE     ToolSelector picks validated ToolProposal(s) -> ExecutionBudgetTracker checks quotas
                 └─ ToolRegistry runs Stage 2-6 tools (or raises ToolApprovalRequiredError -> waiting_approval)
4. OBSERVE     ObservationProcessor runs Stage 3-4 dedup, chunking, HybridRelevanceEngine,
               SourceProvenanceEvaluator, RuleBasedClaimExtractor, DeterministicClaimComparator
5. EVALUATE    EvidenceQualityEvaluator scores per-task and overall support strength
6. GAPS        GapAnalyzer detects unresolved GapCategory issues across tasks/claims/conflicts
7. STOP?       StoppingPolicy checks user signals, hard budgets, diminishing returns, coverage
   ├─ Continue -> AgentResearchPlanner.create_followup_tasks appends deduplicated child tasks -> step 2
   └─ Stop     -> SYNTHESIZING via Stage 5 ContextManagementEngine + Stage 6 LongFormReportGenerator /
                  AnswerGenerator (or deterministic grounded synthesis if no local LLM is configured)
```

- **Post-Approval Resume Path**: After `POST /api/v1/agent/research/{research_id}/approve` executes or rejects a gated action via `AgentExecutionEngine.execute_approved_action`, the orchestrator calls the shared `_evaluate_gaps_and_stopping(...)` helper so iteration accounting, evidence evaluation, gap analysis, and stopping policy rules behave identically to the main loop.

---

## 2. Evidence Quality Classification (`EvidenceQualityEvaluator`)

`web_intelligence/app/agent/evidence_evaluator.py` explicitly separates **lexical relevance** from **evidential support**, classifying each task and session into `EvidenceSupportStatus`:

| Status | Classification Rule |
| :--- | :--- |
| `conflicting` | One or more `ClaimConflict` entries tie to task evidence (never suppressed in favor of a single high-relevance page). |
| `confirmed` | At least 1 `PRIMARY` domain + `>= 2` independent domains with direct overlap `>= 0.22`, OR `PRIMARY` domain + Stage 4 corroborated cluster (`>= 2` independent supporting sources). |
| `corroborated` | At least `2` independent domains with direct overlap `>= 0.20` (or Stage 4 corroborated cluster) without requiring a `PRIMARY` source. |
| `weak` | Evidence exists, but lacks multi-domain corroboration (`< 2` independent domains), lacks primary sources, has low direct overlap (`< 0.20`), or includes snippet-only results. |
| `unverified` | No accepted evidence passages match the task yet. |

---

## 3. Seven-Category Gap Detection (`GapAnalyzer`)

`web_intelligence/app/agent/gap_analyzer.py` inspects the latest `EvidenceEvaluationReport`, `ClaimConflict` records, extracted `ExtractedClaim` assertions, and source timestamps:

1. `unanswered_question` (`high`, priority 5) — completed or failed task remains `unverified`.
2. `missing_primary_source` (`high`, priority 4) — task has `weak` support backed only by secondary/reference domains.
3. `weak_support` (`medium`, priority 3) — task has `weak` support with low corroboration or directness.
4. `missing_category` (`high`, priority 4) — `CoverageAnalysis.missing_topics` identifies an uncovered subtopic facet.
5. `conflicting_results` (`high`, priority 5) — active `ClaimConflict` between independent sources needs methodological/primary comparison.
6. `single_source_claim` (`medium`, priority 3) — up to 2 notable unconflicted claims reported by only one domain.
7. `outdated_information` (`medium` when temporal focus, else `low`) — all dated sources exceed `stale_days` (default `400`) or sources lack dates on a temporal query.

Duplicate gaps are suppressed via `_gap_key` (`category::target`). Existing gaps transition to `resolved=True` once their related task reaches `confirmed`/`corroborated` or their `addressed_by_task_id` completes with retained evidence. Follow-up generation deduplicates candidate tasks via `compute_task_fingerprint` and `is_duplicate_task`.

---

## 4. Multi-Factor `StoppingPolicy` Priority Order

`web_intelligence/app/agent/stopping_policy.py` evaluates stopping conditions strictly in priority order:

1. **Interactive / Execution Control**
   - `user_cancelled`, `waiting_user_approval`, `paused_by_user`, `model_unavailable`
2. **Hard Budget Ceilings** (`ExecutionBudgetTracker.primary_exhaustion_reason()`)
   - `time_budget_exhausted` -> `max_iterations_reached` -> `token_budget_reached` -> `search_budget_reached` -> `fetch_budget_reached` -> `inference_budget_reached`
3. **Tool Availability**
   - `tools_unavailable` when remaining `pending`/`ready` tasks cannot run under permitted tools
4. **Diminishing Returns**
   - `diminishing_returns` when `>= 2` iterations completed, latest iteration yielded `0` new accepted evidence items, coverage delta is `< 0.02`, and no pending runnable tasks remain
5. **Objective Completion**
   - `objectives_covered` when `high_priority_gaps == 0`, `objective_coverage_ratio >= 0.72`, `total_accepted_evidence > 0`, and no runnable tasks remain (or when all planned tasks completed and zero unresolved gaps remain)

If none of these trigger, `StoppingDecision(should_stop=False)` keeps the loop running cleanly.
