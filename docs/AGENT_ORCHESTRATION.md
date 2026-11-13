# Stage 7 — Agentic Research Orchestration Architecture

`web_intelligence/app/agent/` coordinates the existing Local-First Web Intelligence Engine (Stages 2–6) inside a bounded, inspectable, persistent autonomous research loop rather than implementing parallel retrieval, extraction, context, or inference stacks.

---

## 1. Component Overview

| Module | Class / Symbol | Responsibility |
| :--- | :--- | :--- |
| `orchestrator.py` | `AgentResearchOrchestrator`, `AgentSessionNotFoundError` | Drives `ResearchSession` lifecycle, checkpoints state after every phase, coordinates pause/resume/cancel/approval gates, and invokes Stage 6 synthesis. |
| `research_planner.py` | `AgentResearchPlanner` | Converts `AgentResearchRequest` objectives into validated `ResearchPlanSpec` DAGs (`create_initial_plan`) and targeted gap follow-ups (`create_followup_tasks`), using connected local LLMs with automatic deterministic fallback. |
| `task_manager.py` | `TaskManager`, `PlanValidationError` | Validates DAG acyclicity (`validate_dag`), selects runnable priority batches (`select_next_batch`), enforces `max_research_depth` & dependency propagation, and validates user `PUT /plan` edits. |
| `execution_engine.py` | `AgentExecutionEngine` | Executes ready task batches bounded by `ExecutionBudgetTracker`, tracks cancellation/pause hooks, handles `ToolApprovalRequiredError`, and resumes approved actions via `execute_approved_action`. |
| `tool_registry.py` | `ToolRegistry`, `ToolApprovalRequiredError`, `FORBIDDEN_TOOL_NAMES` | Exposes typed, Pydantic-strict (`extra="forbid"`) wrappers over existing Stage 2–6 services while rejecting prohibited execution tools (`python`, `shell`, `bash`, `exec`, `eval`, etc.). |
| `tool_selector.py` | `ToolSelector` | Selects safe tool calls per task (optional local LLM proposal verified against `ToolRegistry` permissions and schemas, plus deterministic fallback). |
| `observation_processor.py` | `ObservationProcessor` | Normalizes raw tool outputs through Stage 3/4 deduplication, chunking, `HybridRelevanceEngine`, `SourceProvenanceEvaluator`, `RuleBasedClaimExtractor`, and `DeterministicClaimComparator`. |
| `evidence_evaluator.py` | `EvidenceQualityEvaluator` | Evaluates accumulated sources and passages per task and session-wide across support status (`confirmed`, `corroborated`, `weak`, `conflicting`, `unverified`) without equating lexical relevance with factual truth. |
| `gap_analyzer.py` | `GapAnalyzer` | Identifies unresolved knowledge gaps across seven `GapCategory` classes while deduplicating previously tracked gaps and marking resolved gaps as corroboration lands. |
| `stopping_policy.py` | `StoppingPolicy` | Evaluates prioritized stopping rules (user control -> hard execution budgets -> tool availability -> diminishing returns -> sufficient objective coverage). |
| `execution_budget.py` | `ExecutionBudgetTracker`, `default_budget_for_depth` | Independently enforces iteration, search, fetch, inference, token, depth, and wall-clock limits regardless of LLM proposals. |
| `research_state.py` | `ResearchSessionStateManager`, `ALLOWED_STATUS_TRANSITIONS`, `compute_task_fingerprint`, `is_duplicate_task` | Validates explicit `AgentSessionStatus` transitions, synchronizes `completed_tasks` / `pending_tasks` / `failed_tasks`, computes `ResearchMetrics`, and prevents redundant task re-creation. |
| `research_memory.py` | `ResearchWorkingMemory` | Sanitizes outbound external queries (`sanitize_external_query`), tracks canonical URLs and executed queries, and maintains working session state. |
| `research_trace.py` | `ResearchTraceRecorder` | Appends sequential, secret-redacted `ResearchTraceEntry` records explaining phase decisions without private chain-of-thought. |

---

## 2. Explicit Lifecycle State Machine (`AgentSessionStatus`)

Transitions are enforced by `ResearchSessionStateManager.transition(...)` using `ALLOWED_STATUS_TRANSITIONS` (`web_intelligence/app/agent/research_state.py`):

```
pending ─────────► planning ◄────────► evaluating ────────► synthesizing ──► completed
   │                  │  ▲                 │  ▲                   │             │
   │                  ▼  │                 ▼  │                   │             │
   │               executing ◄────────► paused                    │             │
   │                  │  ▲                 ▲                      │             │
   │                  ▼  │                 │                      │             │
   │           waiting_approval ───────────┘                      │             │
   ▼                  ▼                    ▼                      ▼             │
cancelled ◄───────────────────────────────────────────────────────┘             │
   ▲                                                                            │
failed  (reachable from any non-terminal state; both cancelled/failed/completed  │
         may reopen cleanly to planning/executing on explicit resume) ◄─────────┘
```

- **Pause / Cancel Semantics**: Immediate `POST /api/v1/agent/research/{research_id}/cancel` or `/pause` transitions non-running sessions directly (including `pending`, `paused`, `waiting_approval`) cleanly without triggering invalid `cancelled -> synthesizing` transitions.
- **SQLite Persistence**: Both `agent_sessions` (serialized `ResearchSession` JSON + indexed `status`, `updated_at`) and `agent_trace_entries` (cascade FK on `research_id`, indexed `(research_id, sequence)`) live in `web_intelligence/app/storage/database.py`.

---

## 3. Bounded Execution Presets (`default_budget_for_depth`)

| Preset (`ResearchDepth`) | `max_iterations` | `max_search_calls` | `max_page_retrievals` | `max_concurrent_tasks` | `max_execution_seconds` | `max_inference_calls` | `max_estimated_tokens` | `max_research_depth` | `tool_timeout_seconds` | `max_followups_per_iteration` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `quick` | 2 | 4 | 6 | 2 | 60.0 | 8 | 32,000 | 1 | 20.0 | 2 |
| `standard` | 3 | 8 | 12 | 2 | 120.0 | 16 | 64,000 | 2 | 25.0 | 3 |
| `deep` | 5 | 14 | 24 | 3 | 240.0 | 30 | 120,000 | 3 | 30.0 | 4 |

All bounds live on `ExecutionBudgetConfig` and can be customized per request via `POST /api/v1/agent/research` (`execution_budget` field).

---

## 4. API & Workbench Integration

- **Backend Router (`web_intelligence/app/api/routes/agent.py`)**:
  - `GET /api/v1/agent/tools`
  - `POST /api/v1/agent/research` (`202 Accepted`, starts background asyncio task tracked by `AgentResearchOrchestrator`)
  - `GET /api/v1/agent/research`
  - `GET /api/v1/agent/research/{research_id}`
  - `GET /api/v1/agent/research/{research_id}/plan` and `PUT /api/v1/agent/research/{research_id}/plan`
  - `GET /api/v1/agent/research/{research_id}/tasks`
  - `GET /api/v1/agent/research/{research_id}/trace`
  - `GET /api/v1/agent/research/{research_id}/gaps`
  - `POST /api/v1/agent/research/{research_id}/pause`
  - `POST /api/v1/agent/research/{research_id}/resume`
  - `POST /api/v1/agent/research/{research_id}/cancel`
  - `POST /api/v1/agent/research/{research_id}/approve`
- **macOS Workbench UI (`web_intelligence/app/ui/`)**:
  - `Agent` workspace mode (`#mode-agent` in `index.html`, rendered by `js/components/agent_workspace.js`) provides depth/iteration/privacy/approval controls, live budget telemetry, Pause/Resume/Cancel triggers, inline external-action approval cards, editable pending plan tasks, evidence verdicts, open gaps, preserved conflicts, and interactive `[S#]` cited synthesis.
  - **Inspector**: `derivePackageFromAgentSession` populates the Evidence/Coverage/Conflicts/Context panes directly from agent sessions, alongside a dedicated **Trace** (`agent_trace`) inspector tab in `js/components/agent_trace.js`.
