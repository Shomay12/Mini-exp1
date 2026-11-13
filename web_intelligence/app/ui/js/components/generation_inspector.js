/**
 * Stage 6 Generation & ModelRegistry Inspector Tab —
 * Displays model target, context allocation stats, token telemetry,
 * citation validation diagnostics, registered models, and generation run history.
 */

import {
  escapeHtml,
  formatDateTime,
  formatNumber,
  renderStatusPill,
  truncateText,
} from "../formatters.js";
import { resolveSelectedModelEntry } from "./generation_workspace.js";

function renderCapabilitiesSummary(entry) {
  if (!entry) {
    return `<p class="coverage-summary-text">No registered model selected. Open Settings → Local LLM to configure a local server.</p>`;
  }
  return `
    <div class="ctx-controls-box">
      <div class="inspector-head__top">
        <strong class="mono">${escapeHtml(entry.model_id)}</strong>
        ${renderStatusPill(
          entry.connection_status === "connected" ? "completed" : "partial",
          entry.connection_status
        )}
      </div>
      <div class="kv-grid">
        <span class="kv-grid__k">Server</span>
        <span class="kv-grid__v">${escapeHtml(entry.connection_name)} (${escapeHtml(entry.provider_id)})</span>
        <span class="kv-grid__k">Endpoint</span>
        <span class="kv-grid__v mono">${escapeHtml(truncateText(entry.server_endpoint, 42))}</span>
        <span class="kv-grid__k">Context</span>
        <span class="kv-grid__v mono">${formatNumber(entry.context_limit)} ctx / ${formatNumber(entry.max_output_limit)} out</span>
      </div>
      <div class="gen-cap-row">
        <span class="cap-badge ${entry.supports_streaming ? "cap-badge--verified" : "cap-badge--unknown"}">Stream ${entry.supports_streaming ? "✓" : "×"}</span>
        <span class="cap-badge ${entry.supports_json_mode ? "cap-badge--verified" : "cap-badge--configured"}">JSON ${entry.supports_json_mode ? "native" : "prompt"}</span>
        <span class="cap-badge ${entry.supports_tool_calling ? "cap-badge--verified" : "cap-badge--unknown"}">Tools ${entry.supports_tool_calling ? "✓" : "×"}</span>
      </div>
    </div>
  `;
}

export function renderGenerationInspectorTab(state, pkg) {
  const entries = state.modelRegistry || [];
  const selectedEntry = resolveSelectedModelEntry(state);
  const res =
    state.generationResult &&
    (!pkg || state.generationResult.research_id === pkg.research_id)
      ? state.generationResult
      : null;
  const runs = (state.generationRuns || []).filter(
    (r) => !pkg || r.research_id === pkg.research_id
  );

  const telemetryHtml = res
    ? (() => {
        const u = res.usage;
        const cStats = res.context_stats;
        const val = res.citation_validation;
        return `
          <div class="settings-group">
            <h3 class="settings-group__title">Run Telemetry</h3>
            <div class="kv-grid">
              <span class="kv-grid__k">Run ID</span>
              <span class="kv-grid__v mono">${escapeHtml(res.generation_id)}</span>
              <span class="kv-grid__k">Status</span>
              <span class="kv-grid__v">${escapeHtml(res.status)}${res.partial ? " (partial)" : ""}${res.truncated ? " · truncated" : ""}</span>
              <span class="kv-grid__k">Model</span>
              <span class="kv-grid__v mono">${escapeHtml(res.model_id)}</span>
              <span class="kv-grid__k">Latency</span>
              <span class="kv-grid__v mono">${formatNumber(res.duration_ms || 0)} ms</span>
              <span class="kv-grid__k">Est. Prompt</span>
              <span class="kv-grid__v mono">${formatNumber(res.estimated_prompt_tokens || 0)} tok</span>
              <span class="kv-grid__k">Server Usage</span>
              <span class="kv-grid__v mono">${
                u && (u.prompt_tokens != null || u.completion_tokens != null || u.total_tokens != null)
                  ? `${u.prompt_tokens ?? "?"} in · ${u.completion_tokens ?? "?"} out · ${u.total_tokens ?? "?"} total`
                  : "Not reported by server"
              }</span>
              ${
                cStats
                  ? `
                <span class="kv-grid__k">Context Packed</span>
                <span class="kv-grid__v mono">${formatNumber(cStats.allocated_evidence_tokens)}/${formatNumber(cStats.effective_context_limit)} (${cStats.evidence_items_included} incl · ${cStats.evidence_items_dropped} drop)</span>
              `
                  : ""
              }
            </div>
          </div>

          ${
            val
              ? `
            <div class="settings-group">
              <div class="inspector-head__top">
                <h3 class="settings-group__title" style="margin:0;">Citation Validation</h3>
                ${renderStatusPill(val.passed ? "sufficient" : "partial", val.passed ? "Passed" : "Review")}
              </div>
              <div class="kv-grid" style="margin-top:6px;">
                <span class="kv-grid__k">Used Tags</span>
                <span class="kv-grid__v mono">${val.used_citation_ids?.length ? escapeHtml(val.used_citation_ids.join(", ")) : "None"}</span>
                <span class="kv-grid__k">Available</span>
                <span class="kv-grid__v mono">${val.known_citation_ids?.length ? escapeHtml(val.known_citation_ids.join(", ")) : "None"}</span>
              </div>
              ${
                val.unsupported_citations?.length
                  ? `<div class="ctx-issue ctx-issue--error">
                      <strong>Unsupported Citation Tags</strong>
                      <span>${escapeHtml(val.unsupported_citations.map((e) => e.raw_tag).join(", "))}</span>
                    </div>`
                  : ""
              }
              ${
                val.uncited_statements?.length
                  ? `<div class="ctx-issue ctx-issue--warning">
                      <strong>${val.uncited_statements.length} Uncited Claim(s)</strong>
                      <span>${escapeHtml(truncateText(val.uncited_statements[0].text, 120))}</span>
                    </div>`
                  : ""
              }
              <p class="doc-section__sub" style="margin:6px 0 0;">${escapeHtml(val.grounding_note || "")}</p>
            </div>
          `
              : ""
          }
        `;
      })()
    : `
      <div class="settings-group">
        <h3 class="settings-group__title">Run Telemetry</h3>
        <p class="coverage-summary-text">No answer has been generated for this research run yet.</p>
      </div>
    `;

  const registryHtml = `
    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title" style="margin:0;">Model Registry (${entries.length})</h3>
        <button id="btn-refresh-model-registry" class="btn btn--quiet btn--sm" type="button">Refresh</button>
      </div>
      ${
        entries.length
          ? `<div class="inspector-list" style="margin-top:6px;">
              ${entries
                .map((e) => {
                  const isSelected = selectedEntry && selectedEntry.entry_id === e.entry_id;
                  return `
                    <button
                      type="button"
                      class="inspector-row ${isSelected ? "is-selected" : ""}"
                      data-select-registry-entry="${escapeHtml(e.entry_id)}"
                      style="width:100%; text-align:left;"
                    >
                      <div class="inspector-head__top">
                        <strong class="mono">${escapeHtml(e.model_id)}</strong>
                        ${e.is_active ? renderStatusPill("sufficient", "Active") : ""}
                      </div>
                      <div class="inspector-row__meta">
                        ${escapeHtml(e.connection_name)} · ${escapeHtml(e.provider_id)} · ${formatNumber(e.context_limit)} ctx
                      </div>
                    </button>
                  `;
                })
                .join("")}
            </div>`
          : `<p class="coverage-summary-text">No local models registered yet.</p>`
      }
    </div>
  `;

  const historyHtml =
    runs.length > 0
      ? `
        <div class="settings-group">
          <h3 class="settings-group__title">Generated Answers (${runs.length})</h3>
          <div class="inspector-list">
            ${runs
              .map(
                (r) => `
              <button
                type="button"
                class="inspector-row ${res?.generation_id === r.generation_id ? "is-selected" : ""}"
                data-select-generation-run="${escapeHtml(r.generation_id)}"
                style="width:100%; text-align:left;"
              >
                <div class="inspector-head__top">
                  <span class="mono">${escapeHtml(r.template_type)} · ${escapeHtml(r.output_format)}</span>
                  ${renderStatusPill(r.status === "completed" ? "sufficient" : "partial", r.status)}
                </div>
                <div class="inspector-row__meta">
                  ${escapeHtml(r.model_id)} · ${formatNumber(r.duration_ms || 0)} ms · ${escapeHtml(formatDateTime(r.created_at))}
                </div>
              </button>
            `
              )
              .join("")}
          </div>
        </div>
      `
      : "";

  return `
    <div class="settings-group">
      <h3 class="settings-group__title">Target Model</h3>
      ${renderCapabilitiesSummary(selectedEntry)}
    </div>
    ${telemetryHtml}
    ${registryHtml}
    ${historyHtml}
  `;
}
