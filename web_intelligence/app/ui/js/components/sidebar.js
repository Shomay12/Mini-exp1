/**
 * Leading Sidebar — Local Research History list, instant filtering, and confirmed deletion.
 */

import { escapeHtml, formatRelativeTime, ICONS, truncateText } from "../formatters.js";

export function createSidebar({ store, onNewResearch, onSelectRun, onRequestDelete }) {
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

  function statusGlyph(status) {
    const s = String(status || "").toLowerCase();
    if (s === "completed") return { icon: ICONS.check, label: "Completed" };
    if (s === "failed") return { icon: ICONS.conflict, label: "Failed" };
    return { icon: ICONS.running, label: s || "Active" };
  }

  function render(state) {
    const term = state.historyFilter.trim().toLowerCase();
    const filtered = term
      ? state.runs.filter(
          (run) =>
            String(run.original_query || "").toLowerCase().includes(term) ||
            String(run.research_id || "").toLowerCase().includes(term)
        )
      : state.runs;

    if (state.runs.length === 0) {
      listEl.innerHTML = `
        <div class="empty-state" style="padding: 16px;">
          <p class="empty-state__desc">No saved runs yet. Executed research queries persist locally in SQLite.</p>
        </div>
      `;
      return;
    }

    if (filtered.length === 0) {
      listEl.innerHTML = `
        <div class="empty-state" style="padding: 16px;">
          <p class="empty-state__desc">No runs match “${escapeHtml(state.historyFilter)}”.</p>
        </div>
      `;
      return;
    }

    listEl.innerHTML = filtered
      .map((run) => {
        const selected = run.research_id === state.activeResearchId;
        const st = statusGlyph(run.research_status);
        const conflicts =
          run.conflicts_detected > 0
            ? `<span style="color: var(--signal-conflict); font-weight: 600;">· ${run.conflicts_detected} conflict${run.conflicts_detected === 1 ? "" : "s"}</span>`
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
                <span title="Status: ${escapeHtml(st.label)}">${st.icon}</span>
                <span>${escapeHtml(formatRelativeTime(run.created_at))}</span>
                <span>· ${run.retrieved_sources ?? 0} src</span>
                <span>· ${run.evidence_items ?? 0} ev</span>
                ${conflicts}
              </div>
            </button>
            <button
              class="btn btn--quiet btn--icon history-row__delete"
              type="button"
              data-delete-run="${escapeHtml(run.research_id)}"
              aria-label="Delete research run: ${escapeHtml(truncateText(run.original_query, 40))}"
              title="Delete local run"
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
  }

  return { render };
}
