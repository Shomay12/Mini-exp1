# Stage 7 — Operational Research Trace & Auditability

Stage 7 records an explicit, secret-redacted chronological log of agent orchestration decisions in SQLite (`agent_trace_entries`) rather than exposing private model chain-of-thought.

---

## 1. Storage & Schema (`ResearchTraceEntry`)

Defined in `web_intelligence/app/agent/models.py` and persisted via `agent_trace_entries` (`web_intelligence/app/storage/database.py`):

| Field | Type | Description |
| :--- | :--- | :--- |
| `trace_id` | `str` | Unique trace identifier (`tr-...`). |
| `research_id` | `str` | Foreign key to `agent_sessions(research_id)` with `ON DELETE CASCADE`. |
| `sequence` | `int` (`>= 1`) | Monotonic per-session chronological sequence number (indexed `(research_id, sequence)`). |
| `iteration` | `int` (`>= 0`) | Autonomous loop iteration index (`0` during initial planning/setup). |
| `phase` | `str` | Active orchestration phase (`planning`, `executing`, `evaluating`, `synthesizing`, etc.). |
| `event_type` | `TraceEventType` | Typed event classification listed below. |
| `task_id` | `str \| None` | Related `ResearchTask.task_id` when task-scoped. |
| `tool_name` | `str \| None` | Invoked or rejected tool name when tool-scoped. |
| `sanitized_inputs` | `dict[str, Any]` | Secret-scrubbed inputs (`_redact_mapping` strips/masks sensitive keys and bearer tokens). |
| `outcome` | `str` | Concise outcome summary (truncated to 500 chars). |
| `reason` | `str` | Auditable rationale explaining why the transition, selection, retention, rejection, or stop occurred. |
| `details` | `dict[str, Any]` | Secret-scrubbed structured metadata (counts, support statuses, gap IDs, budget exhaustion reasons). |
| `timestamp` | `datetime` | Timezone-aware UTC timestamp. |

---

## 2. Typed Trace Events (`TraceEventType`)

| Category | Event Types | What Is Recorded |
| :--- | :--- | :--- |
| **Session & Lifecycle** | `session_started`, `iteration_started`, `session_paused`, `session_resumed`, `error` | Session launch mode, iteration start, user pause/resume checkpoints, and recoverable/fatal execution errors. |
| **Planning & Tasks** | `plan_created`, `plan_updated`, `task_created`, `task_skipped` | Initial or updated `ResearchPlanSpec` summaries, gap-driven follow-up tasks (`origin="gap_followup"`), user `PUT /plan` edits (`origin="user_edit"`), and skipped tasks (depth cap or upstream dependency failure). |
| **Tool Selection & Execution** | `tool_selected`, `tool_executed`, `tool_rejected` | Tool choices (`deterministic_selector` or validated LLM proposals), durations/tokens/outcome counts, and rejected prohibited/invalid/unapproved tool attempts. |
| **Source Triage** | `source_retained`, `source_rejected` | `SourceDecisionRecord` audit trail explaining duplicate detection (`exact_url`, `canonical_url`, `exact_content_hash`, `near_duplicate`), retrieval failures, or low-quality rejections. |
| **Quality, Gaps & Termination** | `evidence_evaluated`, `gap_identified`, `approval_requested`, `approval_resolved`, `stopping_decision`, `synthesis_completed` | Per-iteration support verdicts, coverage ratio, detected `ResearchGap` entries, approval gate lifecycle, multi-factor `StoppingPolicy` verdict, and final Stage 6 synthesis citation summary. |

---

## 3. Retrieval & UI Inspection

- **API Endpoint**: `GET /api/v1/agent/research/{research_id}/trace` returns `AgentTraceResponse` (`research_id`, `status`, sorted `entries: list[ResearchTraceEntry]`).
- **Workbench UI**: The Inspector's **Trace** (`agent_trace`) tab (`web_intelligence/app/ui/js/components/agent_trace.js`) renders the reverse/chronological timeline (`sequence`, `iteration`, `phase`, `event_type`, `reason`, `outcome`, task/tool badges, and sanitized metadata) alongside the registered tools table from `GET /api/v1/agent/tools`.
