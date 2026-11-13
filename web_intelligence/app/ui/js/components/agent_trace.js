/**
 * Stage 7 Agent Trace & Tool Registry Inspector Tab.
 */

import {
  escapeHtml,
  formatDateTime,
  renderStatusPill,
  truncateText,
} from "../formatters.js";

export function renderAgentTraceInspectorTab(state) {
  const session = state.agentSession;
  const entries = state.agentTrace || [];
  const tools = state.agentTools || [];

  const toolsSection = `
    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Registered Agent Tools</h3>
        <span class="doc-section__sub mono">${tools.length} active</span>
      </div>
      ${
        tools.length
          ? `<div class="inspector-list">
              ${tools
                .map(
                  (t) => `
                <div class="inspector-row" style="cursor:default;">
                  <div class="inspector-head__top">
                    <strong class="mono">${escapeHtml(t.name)}</strong>
                    <span class="doc-section__sub mono">${t.requires_external_network ? "External" : "Local"}${t.requires_approval ? " · Gated" : ""}</span>
                  </div>
                  <div class="doc-section__sub">${escapeHtml(t.description)}</div>
                </div>
              `
                )
                .join("")}
            </div>`
          : `<p class="coverage-summary-text">Tool registry unavailable.</p>`
      }
    </div>
  `;

  if (!session) {
    return `
      <div class="context-inspector-stack">
        ${toolsSection}
        <p class="coverage-summary-text">Run or select an Agent session to inspect chronological trace decisions.</p>
      </div>
    `;
  }

  return `
    <div class="context-inspector-stack">
      <div class="settings-group">
        <div class="inspector-head__top">
          <h3 class="settings-group__title">Execution Trace</h3>
          <span class="doc-section__sub mono">${entries.length} event${entries.length === 1 ? "" : "s"}</span>
        </div>
        ${
          entries.length
            ? `<div class="agent-trace-list">
                ${entries
                  .map((e) => {
                    const inputsText =
                      e.sanitized_inputs && Object.keys(e.sanitized_inputs).length
                        ? truncateText(JSON.stringify(e.sanitized_inputs), 120)
                        : "";
                    return `
                      <div class="agent-trace-row">
                        <div class="inspector-head__top">
                          <span class="mono"><strong>#${e.sequence}</strong> · iter ${e.iteration} · ${escapeHtml(e.event_type)}</span>
                          ${e.outcome ? renderStatusPill(e.outcome, e.outcome) : `<span class="doc-section__sub mono">${escapeHtml(e.phase)}</span>`}
                        </div>
                        <div class="doc-section__sub mono">
                          ${e.task_id ? `task=${escapeHtml(e.task_id)} ` : ""}${e.tool_name ? `tool=${escapeHtml(e.tool_name)}` : ""}
                        </div>
                        <div style="font-size:var(--text-caption); color:var(--text-secondary); line-height:1.4;">${escapeHtml(e.reason)}</div>
                        ${inputsText ? `<div class="doc-section__sub mono">${escapeHtml(inputsText)}</div>` : ""}
                        <div class="doc-section__sub mono">${escapeHtml(formatDateTime(e.timestamp))}</div>
                      </div>
                    `;
                  })
                  .join("")}
              </div>`
            : `<p class="coverage-summary-text">No operational trace entries recorded yet.</p>`
        }
      </div>

      ${toolsSection}
    </div>
  `;
}
