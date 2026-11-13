/**
 * Stage 7 Autonomous Agent Research Workspace — Bounded execution status,
 * approval gates, live DAG task plan & editor, gaps/conflicts, and cited report.
 */

import {
  escapeHtml,
  formatDateTime,
  formatNumber,
  formatScore,
  ICONS,
  renderStatusPill,
  safeHttpUrl,
  truncateText,
} from "../formatters.js";
import { renderSafeStructuredBody } from "./generation_workspace.js";

const TERMINAL_AGENT_STATUSES = new Set(["completed", "failed", "cancelled"]);

export function derivePackageFromAgentSession(session) {
  if (!session) return null;
  const sources = session.collected_sources || [];
  const processedDocs = sources.filter((s) => !s.is_duplicate).length;
  const duplicatesRemoved = sources.length - processedDocs;
  const evidence = session.evidence_records || [];
  const totalTok = evidence.reduce((acc, ev) => acc + (Number(ev.token_estimate) || 0), 0);

  return {
    research_id: session.research_package_id || session.research_id,
    original_query: session.original_query,
    research_status: session.status,
    retrieved_at: session.completed_at || session.updated_at || session.created_at,
    metadata: {
      processed_documents: processedDocs,
      duplicate_documents: duplicatesRemoved,
      evidence_items: evidence.length,
      conflicts_detected: (session.conflicts || []).length,
      total_token_estimate: totalTok,
      retrieved_at: session.completed_at || session.updated_at || session.created_at,
    },
    sources,
    evidence,
    claims: session.claims || [],
    conflicts: session.conflicts || [],
    search_queries: session.disclosed_queries || [],
    provider_diagnostics: session.provider_diagnostics || [],
    coverage: session.evidence_evaluation?.coverage_analysis || {
      coverage_status: session.evidence_evaluation?.overall_support_status || "unverified",
      summary: session.evidence_evaluation?.summary || "",
      subtopic_details: [],
    },
  };
}

export function renderAgentComposerControls(state) {
  const p = state.prefs;
  const disabledAttr = state.isRunning ? "disabled" : "";
  const depths = [
    { id: "quick", label: "Quick" },
    { id: "standard", label: "Standard" },
    { id: "deep", label: "Deep" },
  ];

  return `
    <label class="field-inline" title="Research depth controls initial decomposition and task count">
      <span>Depth</span>
      <select id="agent-depth-select" class="select-control" ${disabledAttr}>
        ${depths
          .map(
            (d) =>
              `<option value="${d.id}" ${p.agentDepth === d.id ? "selected" : ""}>${d.label}</option>`
          )
          .join("")}
      </select>
    </label>

    <label class="field-inline" title="Hard cap on autonomous research iterations">
      <span>Iters</span>
      <input id="agent-max-iterations" class="number-control number-control--narrow" type="number" min="1" max="8" value="${Number(p.agentMaxIterations) || 3}" ${disabledAttr} />
    </label>

    <label class="field-inline" title="Permit outbound web search and remote page fetches (mock works offline)">
      <input id="agent-allow-external" type="checkbox" ${p.agentAllowExternal ? "checked" : ""} ${disabledAttr} />
      <span>External Net</span>
    </label>

    <label class="field-inline" title="Pause before executing external search or remote URL fetches">
      <input id="agent-require-approval" type="checkbox" ${p.agentRequireApproval ? "checked" : ""} ${disabledAttr} />
      <span>Approve External</span>
    </label>
  `;
}

export function renderAgentStatusSection(state) {
  const session = state.agentSession;
  if (!session && !state.agentRunning) return "";

  const status = String(session?.status || "pending").toLowerCase();
  const isTerminal = TERMINAL_AGENT_STATUSES.has(status);
  const isPaused = status === "paused";
  const isWaitingApproval = status === "waiting_approval";

  const iterNow = session?.current_iteration ?? 0;
  const iterMax = session?.execution_budget?.max_iterations ?? (Number(state.prefs.agentMaxIterations) || 3);
  const usage = session?.budget_usage || {};
  const budget = session?.execution_budget || {};
  const metrics = session?.research_metrics || {};

  const pct = isTerminal
    ? 100
    : Math.min(92, Math.max(12, Math.round(((Math.max(1, iterNow)) / Math.max(1, iterMax)) * 100)));

  const actionButtons = [];
  if (!isTerminal && session?.research_id) {
    if (isPaused) {
      actionButtons.push(`<button id="btn-agent-resume" class="btn btn--prominent" type="button">Resume</button>`);
    } else if (!isWaitingApproval) {
      actionButtons.push(`<button id="btn-agent-pause" class="btn" type="button">Pause</button>`);
    }
    actionButtons.push(`<button id="btn-agent-cancel" class="btn" type="button">Cancel</button>`);
  }

  return `
    <div class="progress-panel agent-status-panel" role="status" aria-live="polite">
      <div class="progress-panel__head">
        <div class="progress-panel__title">
          ${!isTerminal && !isPaused && !isWaitingApproval ? ICONS.running : ""}
          <span>Iteration ${iterNow} / ${iterMax}</span>
          <span class="doc-section__sub mono">· ${escapeHtml(session?.current_phase || status)}</span>
          ${session?.current_task_id ? `<span class="doc-section__sub mono">· Task ${escapeHtml(session.current_task_id)}</span>` : ""}
          ${renderStatusPill(status)}
        </div>
        <div class="agent-action-group">${actionButtons.join("")}</div>
      </div>

      ${
        !isTerminal
          ? `<div class="progress-track" aria-hidden="true"><div class="progress-track__bar" style="width: ${pct}%;"></div></div>`
          : ""
      }

      <div class="progress-telemetry agent-budget-strip">
        <span>Tasks <strong>${metrics.tasks_completed ?? 0}/${metrics.tasks_total ?? 0}</strong></span>
        <span>· Search <strong>${usage.search_calls ?? 0}/${budget.max_search_calls ?? "—"}</strong></span>
        <span>· Fetches <strong>${usage.page_retrievals ?? 0}/${budget.max_page_retrievals ?? "—"}</strong></span>
        <span>· LLM <strong>${usage.inference_calls ?? 0}/${budget.max_inference_calls ?? "—"}</strong></span>
        <span>· Retained <strong>${metrics.sources_retained ?? 0}</strong> src / <strong>${metrics.evidence_accepted ?? 0}</strong> pass</span>
        <span>· Gaps <strong>${metrics.gaps_unresolved ?? 0}</strong> open</span>
      </div>
    </div>
  `;
}

function renderApprovalGates(session) {
  const approvals = (session?.pending_approvals || []).filter((a) => a.status === "pending");
  if (!approvals.length) return "";

  return `
    <section class="doc-section agent-approval-box" aria-labelledby="heading-agent-approvals">
      <div class="doc-section__heading">
        <h2 id="heading-agent-approvals" style="margin:0; font-size:inherit;">User Approval Required</h2>
        ${renderStatusPill("waiting_approval", `${approvals.length} pending`)}
      </div>
      <div class="agent-approval-list">
        ${approvals
          .map((apr) => {
            const inputsPreview = apr.proposed_inputs?.query
              ? `Query: "${apr.proposed_inputs.query}"`
              : Array.isArray(apr.proposed_inputs?.urls)
                ? `URLs (${apr.proposed_inputs.urls.length}): ${apr.proposed_inputs.urls.slice(0, 3).join(", ")}`
                : JSON.stringify(apr.proposed_inputs || {});
            return `
              <div class="agent-approval-row">
                <div class="agent-approval-row__meta">
                  <div>
                    <strong class="mono">${escapeHtml(apr.tool_name)}</strong>
                    <span class="doc-section__sub"> · Task ${escapeHtml(apr.task_id)}</span>
                    ${apr.external_network ? `<span class="doc-section__sub"> · External network</span>` : ""}
                  </div>
                  <p class="coverage-summary-text" style="margin:2px 0;">${escapeHtml(apr.reason)}</p>
                  <div class="doc-section__sub mono">${escapeHtml(truncateText(inputsPreview, 180))}</div>
                </div>
                <div class="agent-approval-row__actions">
                  <button class="btn btn--prominent" type="button" data-agent-approve="${escapeHtml(apr.approval_id)}" data-decision="approve">Approve</button>
                  <button class="btn" type="button" data-agent-approve="${escapeHtml(apr.approval_id)}" data-decision="reject">Reject</button>
                </div>
              </div>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function renderTaskPlanSection(state, session) {
  const plan = session?.research_plan;
  if (!plan) return "";
  const tasks = plan.tasks || [];
  const isTerminal = TERMINAL_AGENT_STATUSES.has(String(session.status || "").toLowerCase());
  const editableTasks = tasks.filter((t) => t.status === "pending" || t.status === "ready");
  const isEditing = Boolean(state.agentEditingPlan) && editableTasks.length > 0 && !isTerminal;

  return `
    <section class="doc-section" aria-labelledby="heading-agent-plan">
      <div class="doc-section__heading">
        <div>
          <h2 id="heading-agent-plan" style="margin:0; font-size:inherit;">Research Plan DAG</h2>
          <span class="doc-section__sub mono">v${plan.version ?? 1} · ${escapeHtml(plan.planned_by || "planner")}${plan.model_id ? ` · ${escapeHtml(plan.model_id)}` : ""}</span>
        </div>
        <div class="agent-action-group">
          ${
            !isTerminal && editableTasks.length > 0
              ? isEditing
                ? `<button id="btn-agent-save-plan" class="btn btn--prominent" type="button">Save Plan</button>
                   <button id="btn-agent-cancel-plan-edit" class="btn" type="button">Cancel</button>`
                : `<button id="btn-agent-edit-plan" class="btn" type="button">Edit Pending Tasks</button>`
              : ""
          }
        </div>
      </div>

      ${plan.planning_notes ? `<p class="coverage-summary-text">${escapeHtml(plan.planning_notes)}</p>` : ""}

      <div class="agent-task-list">
        ${tasks
          .map((t) => {
            const depsText = (t.dependencies || []).length ? `deps: ${t.dependencies.join(", ")}` : "root";
            const canEditRow = isEditing && (t.status === "pending" || t.status === "ready");
            return `
              <div class="agent-task-row ${session.current_task_id === t.task_id ? "is-current" : ""}">
                <div class="agent-task-row__head">
                  <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
                    <span class="mono" style="font-weight:600;">${escapeHtml(t.task_id)}</span>
                    <span class="doc-section__sub mono">P${t.priority} · depth ${t.depth} · ${escapeHtml(depsText)}</span>
                    ${t.parent_gap_id ? `<span class="doc-section__sub mono">· gap ${escapeHtml(t.parent_gap_id)}</span>` : ""}
                  </div>
                  ${renderStatusPill(t.status)}
                </div>

                ${
                  canEditRow
                    ? `<div class="agent-task-edit-fields">
                        <input
                          class="text-control agent-plan-question-input"
                          type="text"
                          data-edit-task-question="${escapeHtml(t.task_id)}"
                          value="${escapeHtml(t.question)}"
                          aria-label="Edit question for ${escapeHtml(t.task_id)}"
                        />
                        <input
                          class="text-control agent-plan-objective-input"
                          type="text"
                          data-edit-task-objective="${escapeHtml(t.task_id)}"
                          value="${escapeHtml(t.objective)}"
                          aria-label="Edit objective for ${escapeHtml(t.task_id)}"
                        />
                      </div>`
                    : `<div class="agent-task-row__question">${escapeHtml(t.question)}</div>
                       <div class="doc-section__sub">${escapeHtml(t.objective)}</div>`
                }

                <div class="agent-task-row__meta mono">
                  <span>tools: ${(t.suggested_tools || []).join(", ") || "auto"}</span>
                  ${t.source_ids?.length ? `<span>· ${t.source_ids.length} src</span>` : ""}
                  ${t.evidence_ids?.length ? `<span>· ${t.evidence_ids.length} pass</span>` : ""}
                </div>
                ${t.result_summary ? `<div class="doc-section__sub">${escapeHtml(t.result_summary)}</div>` : ""}
                ${t.error ? `<div class="doc-section__sub" style="color:var(--tone-negative);">${escapeHtml(t.error)}</div>` : ""}
              </div>
            `;
          })
          .join("")}
      </div>
    </section>
  `;
}

function renderEvaluationAndGapsSection(session) {
  const evalRep = session?.evidence_evaluation;
  const gaps = session?.identified_gaps || [];
  const stop = session?.stopping_decision;
  const unresolved = gaps.filter((g) => !g.resolved);

  if (!evalRep && !gaps.length && !stop) return "";

  const covPct = evalRep ? Math.round((Number(evalRep.objective_coverage_ratio) || 0) * 100) : 0;

  return `
    <section class="doc-section" aria-labelledby="heading-agent-eval">
      <div class="doc-section__heading">
        <h2 id="heading-agent-eval" style="margin:0; font-size:inherit;">Evidence Support &amp; Gaps</h2>
        ${evalRep ? renderStatusPill(evalRep.overall_support_status) : ""}
      </div>

      ${
        evalRep
          ? `<p class="coverage-summary-text">
              ${escapeHtml(evalRep.summary)}
              <span class="mono">(${covPct}% coverage · ${evalRep.unique_primary_domains} primary domain${evalRep.unique_primary_domains === 1 ? "" : "s"} · ${evalRep.conflict_count} conflict${evalRep.conflict_count === 1 ? "" : "s"})</span>
            </p>`
          : ""
      }

      ${
        stop
          ? `<div class="agent-verdict-row">
              <strong>Stopping Decision:</strong>
              <span class="mono">${escapeHtml(stop.reason || (stop.should_stop ? "stop" : "continue"))}</span>
              <span>— ${escapeHtml(stop.explanation)}</span>
            </div>`
          : ""
      }

      ${
        gaps.length
          ? `<div class="agent-gaps-list">
              ${gaps
                .map(
                  (g) => `
                <div class="agent-gap-row ${g.resolved ? "is-resolved" : ""}">
                  <div class="agent-task-row__head">
                    <span class="mono"><strong>[${escapeHtml(g.severity.toUpperCase())}]</strong> ${escapeHtml(g.category)}</span>
                    ${renderStatusPill(g.resolved ? "completed" : "partial", g.resolved ? "Addressed" : "Open")}
                  </div>
                  <div style="font-size:var(--text-body);">${escapeHtml(g.description)}</div>
                  ${
                    !g.resolved && g.suggested_followup_question
                      ? `<div class="doc-section__sub">Follow-up: ${escapeHtml(g.suggested_followup_question)}</div>`
                      : ""
                  }
                </div>
              `
                )
                .join("")}
            </div>`
          : `<p class="coverage-summary-text">${unresolved.length === 0 ? "No unresolved research gaps recorded." : ""}</p>`
      }
    </section>
  `;
}

function renderAgentReportSection(session) {
  const gen = session?.generation_result;
  if (!gen) return "";

  const sourceMap = gen.source_map || {};
  const pkgSources = session?.collected_sources || [];
  const citations = gen.citations || [];
  const val = gen.citation_validation || {};

  return `
    <section class="doc-section synthesis-card" aria-labelledby="heading-agent-report">
      <div class="doc-section__heading">
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
          <h2 id="heading-agent-report" style="margin:0; font-size:inherit;">Cited Research Synthesis</h2>
          <span class="doc-section__sub mono">${escapeHtml(gen.model_id || "synthesis")} · ${escapeHtml(gen.mode || "standard")} · ${citations.length} source ref${citations.length === 1 ? "" : "s"}</span>
        </div>
        ${renderStatusPill(val.passed ? "sufficient" : "partial", val.passed ? "Citations Grounded" : "Review Citations")}
      </div>

      <div class="gen-output-body">
        ${renderSafeStructuredBody(gen.content || "", gen.output_format || "markdown", sourceMap, pkgSources)}
      </div>

      ${
        (gen.warnings || []).length
          ? `<div class="synthesis-card__warnings">
              ${gen.warnings.map((w) => `<div>• ${escapeHtml(w)}</div>`).join("")}
            </div>`
          : ""
      }

      ${
        citations.length
          ? `<div class="gen-sources-drawer" style="margin-top:8px;">
              ${citations
                .map(
                  (c) => `
                <div class="gen-source-item">
                  <div class="gen-source-item__head">
                    <span class="citation-chip">${escapeHtml(c.citation_id)}</span>
                    <button class="inline-action-link" type="button" data-cite-source="${escapeHtml(c.source_id)}">
                      <strong>${escapeHtml(c.title || c.domain)}</strong>
                    </button>
                    <span class="doc-section__sub mono">${escapeHtml(c.domain)}</span>
                  </div>
                  <div class="doc-section__sub mono">${escapeHtml(safeHttpUrl(c.url))}</div>
                </div>
              `
                )
                .join("")}
            </div>`
          : ""
      }
    </section>
  `;
}

export function renderAgentWorkspaceDocument(state) {
  const session = state.agentSession;
  if (!session) {
    return `
      <div class="empty-blank">
        <h2 class="empty-blank__title">Autonomous Research Agent</h2>
        <p class="empty-blank__hint">Enter a complex multi-part objective above or select a session from History to plan tasks, gather independent evidence, identify gaps, and produce a cited synthesis.</p>
      </div>
    `;
  }

  const conflicts = session.conflicts || [];
  const metrics = session.research_metrics || {};

  return `
    <article class="research-doc">
      <header class="doc-header">
        <h2 class="doc-header__title">${escapeHtml(session.original_query)}</h2>
        <div class="doc-header__meta">
          <span>Objective: ${escapeHtml(truncateText(session.research_objective, 110))}</span>
          <span>· ${session.research_depth} depth</span>
          <span>· ${metrics.sources_retained ?? 0} sources retained</span>
          <span>· ${metrics.evidence_accepted ?? 0} passages</span>
          <span>· ${metrics.conflicts_detected ?? 0} conflicts</span>
          <span>· ${escapeHtml(formatDateTime(session.updated_at || session.created_at))}</span>
        </div>
      </header>

      ${renderApprovalGates(session)}

      ${renderAgentReportSection(session)}

      ${renderTaskPlanSection(state, session)}

      ${renderEvaluationAndGapsSection(session)}

      ${
        conflicts.length
          ? `<section class="doc-section" aria-labelledby="heading-agent-conflicts">
              <div class="doc-section__heading">
                <h2 id="heading-agent-conflicts" style="margin:0; font-size:inherit;">Conflicting Claims Preserved</h2>
                <span class="doc-section__sub">${conflicts.length} detected</span>
              </div>
              <div class="conflict-list">
                ${conflicts
                  .map(
                    (c) => `
                  <div class="conflict-card">
                    <div class="conflict-card__head">
                      <strong>${escapeHtml(c.topic || "Contested metric")}</strong>
                      <span class="doc-section__sub mono">${escapeHtml(c.conflict_type || "numeric_divergence")}</span>
                    </div>
                    <p class="conflict-card__desc">${escapeHtml(c.description || "")}</p>
                    <div class="conflict-side-row">
                      <span class="doc-section__sub mono">${escapeHtml(c.claim_a?.source_domain || "")}: ${escapeHtml(c.claim_a?.normalized_value || "")}</span>
                      <span class="doc-section__sub">vs</span>
                      <span class="doc-section__sub mono">${escapeHtml(c.claim_b?.source_domain || "")}: ${escapeHtml(c.claim_b?.normalized_value || "")}</span>
                    </div>
                  </div>
                `
                  )
                  .join("")}
              </div>
            </section>`
          : ""
      }
    </article>
  `;
}
