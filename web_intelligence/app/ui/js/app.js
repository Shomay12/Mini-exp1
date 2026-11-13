/**
 * Application Entrypoint — Connects the Workbench UI to FastAPI /api/v1 endpoints,
 * coordinates split-pane resizing, keyboard navigation, dialogs, and live announcements.
 */

import { ApiError, api, pollResearchUntilComplete } from "./api.js";
import { createInspector } from "./components/inspector.js";
import { createSettingsSheet } from "./components/settings_sheet.js";
import { createSidebar } from "./components/sidebar.js";
import { createToolbar } from "./components/toolbar.js";
import { createWorkspace } from "./components/workspace.js";
import { truncateText } from "./formatters.js";
import { createStore } from "./state.js";

const store = createStore();

const announcerEl = document.getElementById("live-announcer");
const confirmDialogEl = document.getElementById("confirm-delete-dialog");
const confirmMessageEl = document.getElementById("confirm-delete-message");
const btnCancelDelete = document.getElementById("btn-cancel-delete");
const btnConfirmDelete = document.getElementById("btn-confirm-delete");
const settingsDialogEl = document.getElementById("settings-dialog");

let activeRunAbortController = null;
let sourceDetailAbortController = null;

function announce(message) {
  if (!announcerEl) return;
  announcerEl.textContent = "";
  window.requestAnimationFrame(() => {
    announcerEl.textContent = message;
  });
}

async function refreshSidebarRuns() {
  try {
    const runs = await api.listResearchRuns();
    store.setState({ runs: Array.isArray(runs) ? runs : [], backendError: null });
  } catch (err) {
    if (err.name === "AbortError") return;
    store.setState({ backendError: err.message });
  }
}

async function loadSourceDetail(sourceId) {
  if (!sourceId) return;
  if (sourceDetailAbortController) {
    sourceDetailAbortController.abort();
  }
  sourceDetailAbortController = new AbortController();
  const { signal } = sourceDetailAbortController;

  store.setState({
    selectedSourceId: sourceId,
    sourceDetailLoading: true,
    sourceDetailError: null,
  });

  try {
    const detail = await api.getSourceDetail(sourceId, signal);
    if (store.getState().selectedSourceId === sourceId) {
      store.setState({
        selectedSourceDetail: detail,
        sourceDetailLoading: false,
        sourceDetailError: null,
      });
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    if (store.getState().selectedSourceId === sourceId) {
      store.setState({
        sourceDetailLoading: false,
        sourceDetailError: err.message,
      });
    }
  }
}

async function handleSelectRun(researchId) {
  if (activeRunAbortController) {
    activeRunAbortController.abort();
    activeRunAbortController = null;
  }

  store.setState({
    mode: "research",
    compactPane: "workspace",
    activeResearchId: researchId,
    isRunning: false,
    activeProgress: null,
    workspaceError: null,
    selectedSourceId: null,
    selectedSourceDetail: null,
    sourceDetailError: null,
    highlightedEvidenceId: null,
  });

  try {
    const pkg = await api.getResearchRun(researchId);
    const status = String(pkg.research_status || "").toLowerCase();
    store.setState({
      researchPackage: pkg,
      queryDraft: pkg.original_query || "",
      workspaceError: status === "failed" ? pkg.error_message || "Research execution failed." : null,
    });
    workspace.setInputValue(pkg.original_query || "");

    if (status && status !== "completed" && status !== "failed") {
      // Resume polling if an in-progress run was selected
      activeRunAbortController = new AbortController();
      store.setState({ isRunning: true, activeProgress: pkg.progress || { current_stage: status, completed_stages: [] } });
      const finalPkg = await pollResearchUntilComplete(researchId, {
        signal: activeRunAbortController.signal,
        onUpdate: (update) => {
          store.setState({
            researchPackage: update,
            activeProgress: update.progress || { current_stage: update.research_status, completed_stages: [] },
          });
        },
      });
      store.setState({
        isRunning: false,
        activeProgress: null,
        researchPackage: finalPkg,
        workspaceError: finalPkg.research_status === "failed" ? finalPkg.error_message || "Research execution failed." : null,
      });
      await refreshSidebarRuns();
    } else {
      announce(`Loaded research run with ${pkg.sources?.length ?? 0} sources.`);
    }
  } catch (err) {
    if (err.name === "AbortError") return;
    store.setState({
      isRunning: false,
      activeProgress: null,
      workspaceError: err instanceof ApiError ? err.message : `Unable to load run: ${err.message}`,
    });
  }
}

async function handleSubmitQuery(rawQuery) {
  const query = rawQuery.trim();
  if (query.length < 2) return;

  if (activeRunAbortController) {
    activeRunAbortController.abort();
  }
  activeRunAbortController = new AbortController();
  const { signal } = activeRunAbortController;

  const state = store.getState();
  const prefs = state.prefs;
  const providers = prefs.selectedProviders?.length ? prefs.selectedProviders : null;

  store.setState({
    queryDraft: query,
    isRunning: true,
    workspaceError: null,
    backendError: null,
    compactPane: "workspace",
    selectedSourceId: null,
    selectedSourceDetail: null,
    sourceDetailError: null,
    highlightedEvidenceId: null,
    activeProgress: {
      current_stage: state.mode === "search" ? "searching" : "planning",
      completed_stages: [],
      queries_planned: 0,
      urls_found: 0,
      documents_fetched: 0,
      duplicates_removed: 0,
      evidence_items_count: 0,
    },
  });

  try {
    if (state.mode === "search") {
      const resp = await api.runSearch(
        {
          query,
          max_results: Math.min(25, Math.max(1, Number(prefs.maxResultsPerQuery) || 6)),
          providers,
        },
        signal
      );
      store.setState({
        isRunning: false,
        activeProgress: null,
        searchResponse: resp,
        workspaceError: null,
      });
      announce(`Quick search completed with ${resp.total_results ?? resp.results?.length ?? 0} results.`);
      return;
    }

    const initialPkg = await api.runResearch(
      {
        query,
        max_results_per_query: Math.min(20, Math.max(1, Number(prefs.maxResultsPerQuery) || 6)),
        max_fetch_urls: Math.min(25, Math.max(1, Number(prefs.maxFetchUrls) || 8)),
        providers,
        chunk_target_tokens: Math.min(1000, Math.max(40, Number(prefs.chunkTargetTokens) || 180)),
      },
      signal
    );

    let finalPkg = initialPkg;
    const status = String(initialPkg?.research_status || "").toLowerCase();
    store.setState({
      activeResearchId: initialPkg.research_id,
      researchPackage: initialPkg,
      activeProgress: initialPkg.progress || { current_stage: status || "planning", completed_stages: [] },
    });

    if (status !== "completed" && status !== "failed") {
      finalPkg = await pollResearchUntilComplete(initialPkg.research_id, {
        signal,
        onUpdate: (update) => {
          store.setState({
            researchPackage: update,
            activeProgress: update.progress || {
              current_stage: update.research_status || "processing",
              completed_stages: [],
            },
          });
        },
      });
    }

    const finalStatus = String(finalPkg?.research_status || "").toLowerCase();
    store.setState({
      isRunning: false,
      activeProgress: null,
      activeResearchId: finalPkg.research_id,
      researchPackage: finalPkg,
      workspaceError:
        finalStatus === "failed" ? finalPkg.error_message || "Research execution failed." : null,
    });

    await refreshSidebarRuns();
    announce(
      finalStatus === "failed"
        ? "Research task failed."
        : `Research finished with ${finalPkg.sources?.length ?? 0} sources and ${finalPkg.evidence?.length ?? 0} evidence passages.`
    );
  } catch (err) {
    if (err.name === "AbortError") {
      store.setState({
        isRunning: false,
        activeProgress: null,
      });
      announce("Waiting canceled.");
      return;
    }
    store.setState({
      isRunning: false,
      activeProgress: null,
      workspaceError: err.message || "Request failed.",
    });
    announce(`Error: ${err.message}`);
  } finally {
    if (activeRunAbortController?.signal === signal) {
      activeRunAbortController = null;
    }
  }
}

function handleCancelRun() {
  if (activeRunAbortController) {
    activeRunAbortController.abort();
    activeRunAbortController = null;
  }
  store.setState({ isRunning: false, activeProgress: null });
}

function handleNewResearch() {
  handleCancelRun();
  store.setState({
    mode: "research",
    compactPane: "workspace",
    activeResearchId: null,
    researchPackage: null,
    searchResponse: null,
    queryDraft: "",
    workspaceError: null,
    selectedSourceId: null,
    selectedSourceDetail: null,
    sourceDetailError: null,
    highlightedEvidenceId: null,
  });
  workspace.setInputValue("");
  workspace.focusInput();
}

function handleExportJson() {
  const state = store.getState();
  const payload = state.mode === "research" ? state.researchPackage : state.searchResponse;
  if (!payload) return;

  const filename =
    state.mode === "research" && payload.research_id
      ? `research-${payload.research_id}.json`
      : `search-${Date.now()}.json`;

  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  announce(`Exported ${filename}.`);
}

function handleRequestDelete(run) {
  store.setState({ pendingDeleteRun: run });
  confirmMessageEl.textContent = `This permanently removes “${truncateText(run.original_query, 100)}” along with its persisted sources, extracted passages, and claim conflicts from local SQLite storage.`;
  if (!confirmDialogEl.open) {
    confirmDialogEl.showModal();
  }
}

btnCancelDelete.addEventListener("click", () => {
  store.setState({ pendingDeleteRun: null });
  if (confirmDialogEl.open) confirmDialogEl.close();
});

btnConfirmDelete.addEventListener("click", async () => {
  const run = store.getState().pendingDeleteRun;
  if (!run) {
    if (confirmDialogEl.open) confirmDialogEl.close();
    return;
  }

  btnConfirmDelete.disabled = true;
  try {
    await api.deleteResearchRun(run.research_id);
    const state = store.getState();
    const isDeletedActive = state.activeResearchId === run.research_id;
    store.setState({
      pendingDeleteRun: null,
      runs: state.runs.filter((r) => r.research_id !== run.research_id),
      ...(isDeletedActive
        ? {
            activeResearchId: null,
            researchPackage: null,
            selectedSourceId: null,
            selectedSourceDetail: null,
            highlightedEvidenceId: null,
          }
        : {}),
    });
    if (confirmDialogEl.open) confirmDialogEl.close();
    announce("Research run deleted.");
  } catch (err) {
    store.setState({ workspaceError: `Failed to delete run: ${err.message}` });
    if (confirmDialogEl.open) confirmDialogEl.close();
  } finally {
    btnConfirmDelete.disabled = false;
  }
});

// Close modal sheets when clicking backdrop
for (const dialog of [settingsDialogEl, confirmDialogEl]) {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) {
      dialog.close();
    }
  });
}

// Split View Resizing (Pointer drag + Accessible Keyboard ArrowLeft/ArrowRight)
function initSplitResizers() {
  const resizers = Array.from(document.querySelectorAll(".split-resizer"));
  for (const resizer of resizers) {
    const target = resizer.dataset.resizer;

    resizer.addEventListener("mousedown", (startEvent) => {
      startEvent.preventDefault();
      resizer.classList.add("is-dragging");
      const startX = startEvent.clientX;
      const startWidth =
        target === "sidebar" ? store.getState().sidebarWidth : store.getState().inspectorWidth;

      function onMove(moveEvent) {
        const delta = moveEvent.clientX - startX;
        if (target === "sidebar") {
          const next = Math.min(380, Math.max(220, startWidth + delta));
          store.updatePrefs({ sidebarWidth: Math.round(next) });
        } else {
          const next = Math.min(560, Math.max(300, startWidth - delta));
          store.updatePrefs({ inspectorWidth: Math.round(next) });
        }
      }

      function onUp() {
        resizer.classList.remove("is-dragging");
        window.removeEventListener("mousemove", onMove);
        window.removeEventListener("mouseup", onUp);
      }

      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);
    });

    resizer.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const step = event.shiftKey ? 32 : 16;
      const dir = event.key === "ArrowRight" ? 1 : -1;
      if (target === "sidebar") {
        const next = Math.min(380, Math.max(220, store.getState().sidebarWidth + dir * step));
        store.updatePrefs({ sidebarWidth: next });
      } else {
        const next = Math.min(560, Math.max(300, store.getState().inspectorWidth - dir * step));
        store.updatePrefs({ inspectorWidth: next });
      }
    });
  }
}

// Global Keyboard Shortcuts aligned with macOS Keyboard HIG
document.addEventListener("keydown", (event) => {
  const mod = event.metaKey || event.ctrlKey;
  if (!mod) return;

  const key = event.key.toLowerCase();
  if (key === "\\" && !event.altKey && !event.shiftKey) {
    event.preventDefault();
    store.updatePrefs({ sidebarCollapsed: !store.getState().sidebarCollapsed });
    return;
  }

  if (key === "i" && event.altKey && !event.shiftKey) {
    event.preventDefault();
    store.updatePrefs({ inspectorCollapsed: !store.getState().inspectorCollapsed });
    return;
  }

  if (event.key === "," && !event.altKey && !event.shiftKey) {
    event.preventDefault();
    settingsSheet.open();
    return;
  }

  if (key === "e" && !event.altKey && !event.shiftKey) {
    event.preventDefault();
    handleExportJson();
    return;
  }

  if (key === "n" && !event.altKey && !event.shiftKey) {
    event.preventDefault();
    handleNewResearch();
  }
});

// Initialize UI Components
const settingsSheet = createSettingsSheet({ store });

const toolbar = createToolbar({
  store,
  onModeChange: (mode) => {
    if (mode !== "research" && mode !== "search") return;
    store.setState({ mode, workspaceError: null });
    workspace.focusInput();
  },
  onExportJson: handleExportJson,
  onOpenSettings: () => settingsSheet.open(),
  onCompactPaneChange: (pane) => {
    store.setState({ compactPane: pane });
  },
});

const sidebar = createSidebar({
  store,
  onNewResearch: handleNewResearch,
  onSelectRun: handleSelectRun,
  onRequestDelete: handleRequestDelete,
});

const workspace = createWorkspace({
  store,
  onSubmitQuery: handleSubmitQuery,
  onCancelRun: handleCancelRun,
  onInspectSource: (sourceId, evidenceId) => {
    store.updatePrefs({ inspectorCollapsed: false });
    store.setState({
      inspectorTab: "sources",
      compactPane: "inspector",
      highlightedEvidenceId: evidenceId || null,
    });
    loadSourceDetail(sourceId);
  },
  onInspectEvidence: (evidenceId) => {
    store.updatePrefs({ inspectorCollapsed: false });
    store.setState({
      inspectorTab: "passages",
      compactPane: "inspector",
      highlightedEvidenceId: evidenceId,
    });
  },
  onSwitchToResearchWithQuery: (query) => {
    store.setState({ mode: "research", queryDraft: query });
    workspace.setInputValue(query);
    handleSubmitQuery(query);
  },
});

const inspector = createInspector({
  store,
  onSelectSource: (sourceId) => {
    loadSourceDetail(sourceId);
  },
  onJumpToSource: (sourceId) => {
    store.setState({ inspectorTab: "sources" });
    loadSourceDetail(sourceId);
  },
  onFilterPassages: (filter) => {
    store.setState({ passageFilter: filter });
  },
});

function renderAll(state) {
  toolbar.render(state);
  sidebar.render(state);
  workspace.render(state);
  inspector.render(state);
  if (settingsDialogEl.open) {
    settingsSheet.render(state);
  }
}

store.subscribe(renderAll);
initSplitResizers();
renderAll(store.getState());

async function bootstrap() {
  const [healthRes, configRes, runsRes] = await Promise.allSettled([
    api.getHealth(),
    api.getConfig(),
    api.listResearchRuns(),
  ]);

  const patch = {};
  const errors = [];

  if (healthRes.status === "fulfilled") {
    patch.health = healthRes.value;
  } else {
    errors.push(healthRes.reason?.message || "Health check failed");
  }

  if (configRes.status === "fulfilled") {
    patch.config = configRes.value;
  } else {
    errors.push(configRes.reason?.message || "Config load failed");
  }

  if (runsRes.status === "fulfilled") {
    patch.runs = Array.isArray(runsRes.value) ? runsRes.value : [];
  } else {
    errors.push(runsRes.reason?.message || "Run history unavailable");
  }

  patch.backendError = errors.length && !patch.health ? errors[0] : null;
  store.setState(patch);
}

bootstrap();
