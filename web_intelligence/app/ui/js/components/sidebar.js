/**
 * Leading Sidebar — Local History list, filter, and deletion.
 */

import { escapeHtml, formatRelativeTime, truncateText } from "../formatters.js";

export function createSidebar({ store, onNewResearch, onSelectRun, onSelectAgentSession, onRequestDelete }) {
  const btnNew = document.getElementById("btn-new-research");
  const filterInput = document.getElementById("sidebar-search");
  const clearBtn = document.getElementById("sidebar-search-clear");
  const listEl = document.getElementById("sidebar-history-list");

  btnNew.addEventListener("click", () => {
    onNewResearch();
  });

  filterInput.addEventListener("input", () => {
    const val = filterInput.value;
    clearBtn.hidden = !val;
    store.setState({ historyFilter: val });
  });

  clearBtn.addEventListener("click", () => {
    filterInput.value = "";
    clearBtn.hidden = true;
    store.setState({ historyFilter: "" });
    filterInput.focus();
  });

  listEl.addEventListener("click", (event) => {
    const agentBtn = event.target.closest("[data-select-agent]");
    if (agentBtn) {
      if (onSelectAgentSession) onSelectAgentSession(agentBtn.dataset.selectAgent);
      return;
    }

    const deleteBtn = event.target.closest("[data-delete-run]");
    if (deleteBtn) {
      event.stopPropagation();
      const runId = deleteBtn.dataset.deleteRun;
      const run = store.getState().runs.find((r) => r.research_id === runId);
      if (run) onRequestDelete(run);
      return;
    }

    const selectBtn = event.target.closest("[data-select-run]");
    if (selectBtn) {
      onSelectRun(selectBtn.dataset.selectRun);
    }
  });

  function render(state) {
    const term = state.historyFilter.trim().toLowerCase();
    const agentList = state.agentSessions || [];
    const filteredAgents = term
      ? agentList.filter(
        (ag) =>
          String(ag.original_query || "").toLowerCase().includes(term) ||
          String(ag.research_id || "").toLowerCase().includes(term)
      )
      : agentList;

    const filtered = term
      ? state.runs.filter(
        (run) =>
          String(run.original_query || "").toLowerCase().includes(term) ||
          String(run.research_id || "").toLowerCase().includes(term)
      )
      : state.runs;

    if (state.runs.length === 0 && agentList.length === 0) {
      listEl.innerHTML = `<div class="sidebar-empty">No saved runs.</div>`;
      return;
    }

    if (filtered.length === 0 && filteredAgents.length === 0) {
      listEl.innerHTML = `<div class="sidebar-empty">No matching runs.</div>`;
      return;
    }

    const agentRowsHtml = filteredAgents
      .map((ag) => {
        const selected = state.mode === "agent" && ag.research_id === state.agentActiveId;
        const conflicts =
          ag.conflicts_detected > 0
            ? `<span style="color: var(--tone-negative);">· ${ag.conflicts_detected} conflict${ag.conflicts_detected === 1 ? "" : "s"}</span>`
            : "";
        return `
          <div class="history-row ${selected ? "is-selected" : ""}" role="listitem">
            <button
              class="history-row__select"
              type="button"
              data-select-agent="${escapeHtml(ag.research_id)}"
              aria-current="${selected ? "true" : "false"}"
              title="${escapeHtml(ag.original_query)}"
            >
              <div class="history-row__title"><span class="mono" style="color:var(--accent);">[Agent]</span> ${escapeHtml(truncateText(ag.original_query, 76))}</div>
              <div class="history-row__meta">
                <span>${escapeHtml(formatRelativeTime(ag.updated_at || ag.created_at))}</span>
                <span>· iter ${ag.current_iteration ?? 0}</span>
                <span>· ${ag.sources_retained ?? 0} src</span>
                <span>· ${escapeHtml(ag.status)}</span>
                ${conflicts}
              </div>
            </button>
          </div>
        `;
      })
      .join("");

    const runRowsHtml = filtered
      .map((run) => {
        const selected = run.research_id === state.activeResearchId;
        const conflicts =
          run.conflicts_detected > 0
            ? `<span style="color: var(--tone-negative);">· ${run.conflicts_detected} conflict${run.conflicts_detected === 1 ? "" : "s"}</span>`
            : "";
        const failed =
          String(run.research_status || "").toLowerCase() === "failed"
            ? `<span style="color: var(--tone-negative);">· Failed</span>`
            : "";

        return `
          <div class="history-row ${selected ? "is-selected" : ""}" role="listitem">
            <button
              class="history-row__select"
              type="button"
              data-select-run="${escapeHtml(run.research_id)}"
              aria-current="${selected ? "true" : "false"}"
              title="${escapeHtml(run.original_query)}"
            >
              <div class="history-row__title">${escapeHtml(truncateText(run.original_query, 86))}</div>
              <div class="history-row__meta">
                <span>${escapeHtml(formatRelativeTime(run.created_at))}</span>
                <span>· ${run.retrieved_sources ?? 0} src</span>
                <span>· ${run.evidence_items ?? 0} pass</span>
                ${conflicts}
                ${failed}
              </div>
            </button>
            <button
              class="btn btn--quiet btn--icon history-row__delete"
              type="button"
              data-delete-run="${escapeHtml(run.research_id)}"
              aria-label="Delete: ${escapeHtml(truncateText(run.original_query, 40))}"
              title="Delete run"
            >
              <svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true">
                <path d="M3 4.5h10"></path>
                <path d="M6 4.5V3h4v1.5"></path>
                <path d="M4.5 4.5l.7 8.2a1 1 0 0 0 1 .9h3.6a1 1 0 0 0 1-.9l.7-8.2"></path>
              </svg>
            </button>
          </div>
        `;
      })
      .join("");

    listEl.innerHTML = agentRowsHtml + runRowsHtml;
  }

  return { render };
}
