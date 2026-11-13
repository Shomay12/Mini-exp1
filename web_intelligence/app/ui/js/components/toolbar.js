/**
 * Top Toolbar controller — mode toggle, concise status text, budget readout, export & pane toggles.
 */

import { escapeHtml, formatNumber, truncateText } from "../formatters.js";

export function createToolbar({ store, onModeChange, onExportJson, onOpenSettings, onCompactPaneChange }) {
  const appShell = document.getElementById("app-shell");
  const titleEl = document.getElementById("toolbar-run-title");
  const statusEl = document.getElementById("toolbar-engine-status");
  const budgetLabelEl = document.getElementById("toolbar-budget-label");
  const btnSidebar = document.getElementById("btn-toggle-sidebar");
  const btnInspector = document.getElementById("btn-toggle-inspector");
  const btnExport = document.getElementById("btn-export-json");
  const btnSettings = document.getElementById("btn-open-settings");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const compactTabButtons = Array.from(document.querySelectorAll("[data-compact-tab]"));

  btnSidebar.addEventListener("click", () => {
    store.updatePrefs({ sidebarCollapsed: !store.getState().sidebarCollapsed });
  });

  btnInspector.addEventListener("click", () => {
    store.updatePrefs({ inspectorCollapsed: !store.getState().inspectorCollapsed });
  });

  btnExport.addEventListener("click", () => {
    onExportJson();
  });

  btnSettings.addEventListener("click", () => {
    onOpenSettings();
  });

  for (const btn of modeButtons) {
    btn.addEventListener("click", () => {
      onModeChange(btn.dataset.mode);
    });
  }

  for (const btn of compactTabButtons) {
    btn.addEventListener("click", () => {
      onCompactPaneChange(btn.dataset.compactTab);
    });
  }

  function render(state) {
    appShell.classList.toggle("is-sidebar-collapsed", state.sidebarCollapsed);
    appShell.classList.toggle("is-inspector-collapsed", state.inspectorCollapsed);
    appShell.dataset.compactPane = state.compactPane;

    appShell.style.setProperty("--sidebar-width", `${state.sidebarWidth}px`);
    appShell.style.setProperty("--inspector-width", `${state.inspectorWidth}px`);

    btnSidebar.setAttribute("aria-pressed", String(!state.sidebarCollapsed));
    btnInspector.setAttribute("aria-pressed", String(!state.inspectorCollapsed));

    for (const btn of modeButtons) {
      btn.setAttribute("aria-selected", String(btn.dataset.mode === state.mode));
    }

    for (const btn of compactTabButtons) {
      btn.setAttribute("aria-selected", String(btn.dataset.compactTab === state.compactPane));
    }

    let title = "New Research";
    if (state.mode === "agent" && state.agentSession?.original_query) {
      title = truncateText(state.agentSession.original_query, 44);
    } else if (state.mode === "agent") {
      title = "Autonomous Agent";
    } else if (state.mode === "research" && state.researchPackage?.original_query) {
      title = truncateText(state.researchPackage.original_query, 44);
    } else if (state.mode === "search" && state.searchResponse?.original_query) {
      title = truncateText(state.searchResponse.original_query, 44);
    }
    titleEl.textContent = title;
    titleEl.title =
      (state.mode === "agent" && state.agentSession?.original_query) ||
      state.researchPackage?.original_query ||
      state.searchResponse?.original_query ||
      title;

    // Quiet single-dot engine status
    statusEl.className = "status-label toolbar-hide-compact";
    let statusText = "Connecting…";
    if (state.isRunning) {
      statusEl.classList.add("status-label--active");
      statusText = "Running…";
    } else if (state.backendError) {
      statusEl.classList.add("status-label--negative");
      statusText = "Offline";
    } else if (state.health) {
      statusEl.classList.add("status-label--positive");
      const enabled = (state.health.providers || []).filter((p) => p.enabled);
      const mockOnly = enabled.length > 0 && enabled.every((p) => p.is_mock);
      statusText = mockOnly ? "Local · Mock" : "Local · Ready";
    }
    statusEl.innerHTML = `<span class="status-dot" aria-hidden="true"></span><span>${escapeHtml(statusText)}</span>`;

    // Concise context budget text grounded in Stage 5 ContextBudget / Preview when available
    const previewPkg = state.contextPreview?.context_package;
    const alloc = previewPkg?.budget_allocation || state.contextBudget?.effective_allocation;
    if (state.mode === "agent" && state.agentSession) {
      const usedTok = state.agentSession.budget_usage?.estimated_tokens ?? 0;
      const capTok = state.agentSession.execution_budget?.max_estimated_tokens ?? 120000;
      budgetLabelEl.textContent = `~${formatNumber(usedTok)} / ${formatNumber(capTok)} tok`;
    } else if (previewPkg) {
      const approx = previewPkg.estimation_is_approximate ? "~" : "";
      budgetLabelEl.textContent = `${approx}${formatNumber(previewPkg.total_with_output_tokens)} / ${formatNumber(alloc?.effective_context_limit || 8192)} tok`;
    } else if (alloc) {
      budgetLabelEl.textContent = `0 / ${formatNumber(alloc.effective_context_limit)} tok`;
    } else {
      const totalBudget = Math.max(512, Number(state.prefs.contextBudgetTokens) || 8192);
      const reserved = Math.min(totalBudget - 256, Math.max(0, Number(state.prefs.reservedOutputTokens) || 1024));
      const availableEvidenceBudget = Math.max(256, totalBudget - reserved);
      const usedTokens = state.researchPackage?.metadata?.total_token_estimate ?? 0;
      budgetLabelEl.textContent = `${formatNumber(usedTokens)} / ${formatNumber(availableEvidenceBudget)} tok`;
    }

    const hasExportable = Boolean(
      (state.mode === "agent" && state.agentSession) ||
      (state.mode === "research" && state.researchPackage) ||
      (state.mode === "search" && state.searchResponse)
    );
    btnExport.disabled = !hasExportable;
  }

  return { render };
}
