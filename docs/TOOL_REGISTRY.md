# Stage 7 — Controlled Tool Registry & Security Boundaries

`ToolRegistry` (`web_intelligence/app/agent/tool_registry.py`) and `ToolSelector` (`web_intelligence/app/agent/tool_selector.py`) wrap existing Stage 2–6 research services behind strict Pydantic argument schemas, permission policies, timeouts, and approval gates.

---

## 1. Security & Prohibition Guarantees

- **Prohibited Execution Classes (`FORBIDDEN_TOOL_NAMES`)**:
  `ToolRegistry` and `ToolSelector` explicitly forbid arbitrary execution tools:
  `python`, `exec`, `eval`, `shell`, `bash`, `sh`, `zsh`, `subprocess`, `browser`, `playwright`, `javascript`.
- **Strict Input Models**: Every registered tool validates arguments against a `_StrictToolInput` Pydantic subclass (`ConfigDict(extra="forbid")`), rejecting unknown fields or invalid HTTP(S) URLs before any handler runs.
- **No Raw Credential Exposure**: Sanitized inputs recorded into `ResearchTraceEntry` or `PendingApprovalRequest` pass through `ResearchTraceRecorder` / `_redact_mapping`, stripping or redacting `api_key`, `authorization`, `token`, `secret`, `password`, and `Bearer ...` strings.

---

## 2. Registered Research Tools

| Tool Name | Input Schema Class | Permission (`ToolPermission`) | Network / Approval Behavior | Wrapped Local Service |
| :--- | :--- | :--- | :--- | :--- |
| `web_search` | `WebSearchToolInput` (`query`, `max_results`, `providers`, `intent`) | `external_search` | External network unless mock-only search mode (`allow_external_search` / `require_approval_for_external_search`). | `SearchProviderRouter` |
| `url_retrieve` | `UrlRetrieveToolInput` (`urls`, `snippets_by_url`, `titles_by_url`, `providers_by_url`, `max_urls`) | `external_fetch` | SSRF-guarded HTTP(S) retrieval (`WebRetrievalEngine`). Permitted in mock-only offline fixture mode; gated by `require_approval_for_remote_urls` / `require_user_approval_for_external` when external network is enabled. | `WebRetrievalEngine` |
| `page_extract` | `PageExtractToolInput` (`source_id`, `target_tokens`) | `read_local` | 100% local (`requires_external_network=False`), re-chunks already fetched `SourceRecord` objects. | `PassageChunker` |
| `local_doc_retrieve` | `LocalDocRetrieveToolInput` (`query`, `limit`) | `read_local` | 100% local search over previously stored workspace documents/sources. | `ResearchRepository` |
| `evidence_search` | `EvidenceSearchToolInput` (`query`, `top_k`) | `read_local` | 100% local lexical/hybrid retrieval over accumulated session evidence items. | `HybridRelevanceEngine` |
| `source_inspect` | `SourceInspectToolInput` (`source_id`) | `read_local` | 100% local inspection of stored `SourceRecord` metadata, duplicate status, and extracted text. | Session / `ResearchRepository` |
| `context_inspect` | `ContextInspectToolInput` (`question`, `profile`, `compression_strategy`) | `read_local` | 100% local Stage 5 budget/compression preview over accumulated evidence. | `ContextManagementEngine` |
| `local_llm_inference` | `LocalInferenceToolInput` (`instruction`, `evidence_excerpts`, `max_tokens`) | `local_inference` | 100% local LLM execution bounded by `max_inference_calls` and endpoint trust boundaries. | `LLMConnectionManager` |

---

## 3. Approval Gate & Privacy Interaction

- **`allow_external_search=False`**: External live providers are blocked; only strictly offline mock fixture providers (when configured via `Settings(search_providers=["mock"], enable_mock_provider=True)`) and `read_local` / `local_inference` tools remain available.
- **`require_user_approval_for_external=True`** (or `ExecutionBudgetConfig.require_approval_for_external_search` / `require_approval_for_remote_urls`):
  1. `ToolRegistry.execute(...)` raises `ToolApprovalRequiredError` with sanitized tool inputs and rationale when an external action lacks prior approval for that `(tool_name, args)` decision.
  2. `AgentExecutionEngine` records a `PendingApprovalRequest` on the session, logs `TraceEventType.APPROVAL_REQUESTED`, and transitions `ResearchSession.status` to `waiting_approval`.
  3. The user responds via `POST /api/v1/agent/research/{research_id}/approve` (`approved: true/false`, optional `decision_note`, `resume_execution: true`).
  4. Approved actions execute immediately via `AgentExecutionEngine.execute_approved_action(...)` with approval bypass tokens; rejected actions mark the task/action cleanly without failing the whole session and record `TraceEventType.APPROVAL_RESOLVED`.
