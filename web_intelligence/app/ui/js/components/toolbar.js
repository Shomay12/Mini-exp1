/**
 * Top Window Toolbar controller — mode switcher, status indicator, context budget, export & pane toggles.
 */

import { formatNumber, ICONS, truncateText } from "../formatters.js";

export function createToolbar({ store, onModeChange, onExportJson, onOpenSettings, onCompactPaneChange }) {
  const appShell = document.getElementById("app-shell");
  const titleEl = document.getElementById("toolbar-run-title");
  const statusEl = document.getElementById("toolbar-engine-status");
  const budgetLabelEl = document.getElementById("toolbar-budget-label");
  const budgetFillEl = document.getElementById("toolbar-budget-fill");
  const btnSidebar = document.getElementById("btn-toggle-sidebar");
  const btnInspector = document.getElementById("btn-toggle-inspector");
  const btnExport = document.getElementById("btn-export-json");
  const btnSettings = document.getElementById("btn-open-settings");
  const modeButtons = Array.from(document.querySelectorAll("[data-mode]"));
  const compactTabButtons = Array.from(document.querySelectorAll("[data-compact-tab]"));

  btnSidebar.addEventListener("click", () => {
    const { sidebarCollapsed } = store.getState();
    store.updatePrefs({ sidebarCollapsed: !sidebarCollapsed });
  });

  btnInspector.addEventListener("click", () => {
    const { inspectorCollapsed } = store.getState();
    store.updatePrefs({ inspectorCollapsed: !inspectorCollapsed });
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

    // Concise window title (Toolbars HIG: never repeat app name)
    let title = "New Research";
    if (state.mode === "research" && state.researchPackage?.original_query) {
      title = truncateText(state.researchPackage.original_query, 44);
    } else if (state.mode === "search" && state.searchResponse?.original_query) {
      title = truncateText(state.searchResponse.original_query, 44);
    }
    titleEl.textContent = title;
    titleEl.title =
      state.researchPackage?.original_query ||
      state.searchResponse?.original_query ||
      "New Research";

    // Engine & Local Model Status Pill
    statusEl.className = "status-pill";
    if (state.isRunning) {
      statusEl.classList.add("status-pill--active");
      const stage = state.activeProgress?.current_stage || "running";
      statusEl.innerHTML = `${ICONS.running}<span>Running: ${stage}</span>`;
    } else if (state.backendError) {
      statusEl.classList.add("status-pill--conflict");
      statusEl.innerHTML = `${ICONS.conflict}<span>API Offline</span>`;
    } else if (state.health) {
      const enabled = (state.health.providers || []).filter((p) => p.enabled);
      const mockOnly = enabled.length > 0 && enabled.every((p) => p.is_mock);
      statusEl.classList.add(mockOnly ? "status-pill--privacy" : "status-pill--verified");
      const label = mockOnly ? "Local Engine · Offline Mock" : "Local Engine · Ready (No Cloud LLM)";
      statusEl.innerHTML = `${ICONS.shield}<span>${label}</span>`;
    } else {
      statusEl.innerHTML = `${ICONS.running}<span>Connecting…</span>`;
    }

    // Context Budget Meter
    const totalBudget = Math.max(512, Number(state.prefs.contextBudgetTokens) || 8192);
    const reserved = Math.min(totalBudget - 256, Math.max(0, Number(state.prefs.reservedOutputTokens) || 1024));
    const availableEvidenceBudget = Math.max(256, totalBudget - reserved);
    const usedTokens = state.researchPackage?.metadata?.total_token_estimate ?? 0;
    const pct = Math.min(100, Math.round((usedTokens / availableEvidenceBudget) * 100));

    budgetLabelEl.textContent = `${formatNumber(usedTokens)} / ${formatNumber(availableEvidenceBudget)} tok`;
    budgetFillEl.style.width = `${pct}%`;
    budgetFillEl.classList.toggle("is-near-limit", usedTokens > availableEvidenceBudget * 0.8 && usedTokens <= availableEvidenceBudget);
    budgetFillEl.classList.toggle("is-over-limit", usedTokens > availableEvidenceBudget);

    const hasExportable = Boolean(
      (state.mode === "research" && state.researchPackage) ||
      (state.mode === "search" && state.searchResponse)
    );
    btnExport.disabled = !hasExportable;
  }

  return { render };
}
