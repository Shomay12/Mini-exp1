/**
 * Application Entrypoint — Connects the Workbench UI to FastAPI /api/v1 endpoints,
 * coordinates split-pane resizing, keyboard navigation, dialogs, and live announcements.
 */

import { ApiError, api, pollResearchUntilComplete } from "./api.js";
import { resolveSelectedModelEntry } from "./components/generation_workspace.js";
import { createInspector } from "./components/inspector.js";
import {
  createInitialLlmDraft,
  handleLlmActionClick,
  renderModelConnectionsPane,
  syncLlmDraftField,
} from "./components/llm_pane.js";
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
let activeGenerationAbortController = null;
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
    generationResult: null,
    generationRuns: [],
    generationRunning: false,
    generationActiveId: null,
    generationStreamContent: "",
    generationError: null,
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
      loadContextPreviewForRun(researchId);
      loadGenerationsForRun(researchId);
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
        use_local_llm: Boolean(prefs.useLocalLlm && store.getState().health?.active_llm),
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

    await Promise.allSettled([
      refreshSidebarRuns(),
      finalStatus === "completed" ? loadContextPreviewForRun(finalPkg.research_id) : Promise.resolve(),
      finalStatus === "completed" ? loadGenerationsForRun(finalPkg.research_id) : Promise.resolve(),
    ]);
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
  handleStopGeneration();
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
    contextPreview: null,
    contextCompressionPreview: null,
    contextError: null,
    generationResult: null,
    generationRuns: [],
    generationRunning: false,
    generationActiveId: null,
    generationStreamContent: "",
    generationStreamOutline: [],
    generationStreamSections: [],
    generationStreamCurrentSection: null,
    generationStreamWarnings: [],
    generationStreamCitations: [],
    generationError: null,
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
const llmDraft = createInitialLlmDraft();
let lastModelPaneSig = "";

const settingsRootEl = document.getElementById("settings-sheet-root");
if (settingsRootEl) {
  settingsRootEl.addEventListener("click", (event) => {
    const btn = event.target.closest("[data-llm-action]");
    if (btn && store.getState().settingsTab === "model") {
      Promise.resolve(handleLlmActionClick(btn, { store, draft: llmDraft, rootEl: settingsRootEl }))
        .finally(() => {
          refreshModelRegistry();
        });
    }
  });
  settingsRootEl.addEventListener("input", (event) => {
    if (store.getState().settingsTab === "model") {
      const needsRender = syncLlmDraftField(event.target, { store, draft: llmDraft });
      if (needsRender) {
        lastModelPaneSig = "";
        syncModelSettingsPane(store.getState());
      }
    }
  });
  settingsRootEl.addEventListener("change", (event) => {
    if (store.getState().settingsTab === "model") {
      const needsRender = syncLlmDraftField(event.target, { store, draft: llmDraft });
      if (needsRender) {
        lastModelPaneSig = "";
        syncModelSettingsPane(store.getState());
      }
    }
  });
}

function syncModelSettingsPane(state) {
  if (!settingsDialogEl?.open || state.settingsTab !== "model") {
    lastModelPaneSig = "";
    return;
  }
  const bodyEl = settingsRootEl?.querySelector(".sheet-body");
  if (!bodyEl) return;

  const sig = JSON.stringify({
    conns: state.llmConnections,
    discovered: state.llmDiscoveredModels,
    selectedId: state.llmSelectedConnectionId,
    busy: state.llmBusy,
    status: state.llmStatusMessage,
    err: state.llmError,
    promptRunning: state.llmPromptRunning,
    promptErr: state.llmPromptError,
    hasStreamText: Boolean(state.llmPromptStreamText),
    preset: llmDraft.preset,
  });

  if (sig !== lastModelPaneSig || !bodyEl.dataset.llmRendered) {
    lastModelPaneSig = sig;
    bodyEl.innerHTML = renderModelConnectionsPane(state, llmDraft);
    bodyEl.dataset.llmRendered = "1";
  } else {
    const outEl = bodyEl.querySelector("#llm-test-prompt-output");
    if (outEl && state.llmPromptStreamText) {
      outEl.textContent = state.llmPromptStreamText;
    }
  }
}

function syncToolbarLlmDisplay(state) {
  const engineEl = document.getElementById("toolbar-engine-status");
  const active = state.health?.active_llm;
  if (engineEl) {
    if (active) {
      const dotClass = active.status === "connected" ? "status-dot--ok" : "status-dot--pending";
      const label = active.model ? `${active.name}: ${active.model}` : active.name;
      engineEl.innerHTML = `<span class="status-dot ${dotClass}" aria-hidden="true"></span><span>LLM · ${escapeToolbarText(label)}</span>`;
      engineEl.title = `Active local model (${active.provider}, ${active.status}) — click to configure`;
      engineEl.style.cursor = "pointer";
      if (!engineEl.dataset.llmBound) {
        engineEl.dataset.llmBound = "1";
        engineEl.addEventListener("click", () => {
          store.setState({ settingsTab: "model" });
          settingsSheet.open();
          lastModelPaneSig = "";
          syncModelSettingsPane(store.getState());
        });
      }
    } else if (state.health) {
      engineEl.title = "Zero-LLM mode (Stage 4 active) — click to connect a local LLM";
      engineEl.style.cursor = "pointer";
      if (!engineEl.dataset.llmBound) {
        engineEl.dataset.llmBound = "1";
        engineEl.addEventListener("click", () => {
          store.setState({ settingsTab: "model" });
          settingsSheet.open();
          lastModelPaneSig = "";
          syncModelSettingsPane(store.getState());
        });
      }
    }
  }

  const budgetLabel = document.getElementById("toolbar-budget-label");
  const budgetFill = document.getElementById("toolbar-budget-fill");
  const alloc = state.researchPackage?.context_allocation;
  if (budgetLabel && budgetFill && (alloc || active)) {
    const evTokens = (state.researchPackage?.evidence || []).reduce(
      (sum, item) => sum + (Number(item.token_estimate) || 0),
      0
    );
    const used = alloc ? alloc.allocated_tokens : evTokens;
    const cap = alloc
      ? Math.max(1, alloc.available_evidence_tokens)
      : Math.max(256, (active.max_context_tokens || 8192) - (active.recommended_output_tokens || 1024));
    const pct = Math.min(100, Math.round((used / cap) * 100));
    budgetLabel.textContent = `${used.toLocaleString()} / ${cap.toLocaleString()} tok`;
    budgetFill.style.width = `${pct}%`;
  }
}

function escapeToolbarText(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

window.addEventListener("wi:open-settings", (event) => {
  const tab = event.detail?.tab;
  if (tab) store.setState({ settingsTab: tab });
  settingsSheet.open();
  lastModelPaneSig = "";
  syncModelSettingsPane(store.getState());
});

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
  onStartGeneration: () => handleStartGeneration(),
  onStopGeneration: () => handleStopGeneration(),
  onOpenGenInspector: () => {
    store.updatePrefs({ inspectorCollapsed: false });
    store.setState({ inspectorTab: "generation", compactPane: "inspector" });
    refreshModelRegistry();
  },
});

async function refreshContextMetadata() {
  try {
    const [capsRes, budgetRes, notesRes] = await Promise.all([
      api.getContextCapabilities(),
      api.getContextBudget(),
      api.listWorkspaceNotes(),
    ]);
    store.setState({
      contextCapabilities: capsRes?.capabilities || null,
      contextBudget: budgetRes || null,
      contextNotes: Array.isArray(notesRes) ? notesRes : [],
      contextError: null,
    });
  } catch (err) {
    store.setState({ contextError: err.message });
  }
}

async function loadContextPreviewForRun(researchId) {
  if (!researchId) return;
  store.setState({ contextLoading: true, contextError: null });
  try {
    const [previewRes, budgetRes] = await Promise.all([
      api.getResearchContext(researchId),
      api.getContextBudget(),
    ]);
    store.setState({
      contextPreview: previewRes,
      contextBudget: budgetRes || store.getState().contextBudget,
      contextCapabilities: previewRes?.context_package?.capabilities || store.getState().contextCapabilities,
      contextCompressionPreview: null,
      contextLoading: false,
      contextError: null,
    });
  } catch (err) {
    store.setState({ contextLoading: false, contextError: err.message });
  }
}

async function handleApplyContextBudget(payload) {
  store.setState({ contextLoading: true, contextError: null });
  try {
    const budgetRes = await api.updateContextBudget(payload);
    store.setState({
      contextBudget: budgetRes,
      contextCapabilities: budgetRes.capabilities,
      contextLoading: false,
    });
    const st = store.getState();
    if (st.activeResearchId) {
      await loadContextPreviewForRun(st.activeResearchId);
    } else if (st.queryDraft?.trim()) {
      await handlePreviewContext({});
    }
    announce("Updated context budget allocation.");
  } catch (err) {
    store.setState({ contextLoading: false, contextError: err.message });
  }
}

async function handlePreviewContext(overrides = {}) {
  const st = store.getState();
  const researchId = st.activeResearchId || st.researchPackage?.research_id || null;
  const question = st.researchPackage?.original_query || st.queryDraft?.trim() || "";
  if (!researchId && !question) return;

  store.setState({ contextLoading: true, contextError: null });
  try {
    const payload = {
      ...(researchId ? { research_id: researchId } : { question }),
      ...(overrides.profile ? { profile: overrides.profile } : {}),
      ...(overrides.compression_strategy ? { compression_strategy: overrides.compression_strategy } : {}),
      ...(overrides.user_context_limit ? { user_context_limit: overrides.user_context_limit } : {}),
      ...(overrides.output_reservation_tokens
        ? { output_reservation_tokens: overrides.output_reservation_tokens }
        : {}),
    };
    const previewRes = await api.previewContext(payload);
    store.setState({
      contextPreview: previewRes,
      contextCapabilities: previewRes?.context_package?.capabilities || st.contextCapabilities,
      contextCompressionPreview: null,
      contextLoading: false,
      contextError: null,
    });
  } catch (err) {
    store.setState({ contextLoading: false, contextError: err.message });
  }
}

async function handleCompressContext({ strategy } = {}) {
  const st = store.getState();
  const researchId = st.activeResearchId || st.researchPackage?.research_id;
  if (!researchId) return;
  store.setState({ contextLoading: true, contextError: null });
  try {
    const compRes = await api.compressContext({
      research_id: researchId,
      strategy: strategy || "extractive",
    });
    store.setState({
      contextCompressionPreview: compRes,
      contextLoading: false,
      contextError: null,
    });
  } catch (err) {
    store.setState({ contextLoading: false, contextError: err.message });
  }
}

async function handleAddWorkspaceNote({ title, content }) {
  try {
    await api.createWorkspaceNote({ title, content });
    const notes = await api.listWorkspaceNotes();
    store.setState({ contextNotes: Array.isArray(notes) ? notes : [], contextError: null });
    const st = store.getState();
    if (st.activeResearchId) await loadContextPreviewForRun(st.activeResearchId);
  } catch (err) {
    store.setState({ contextError: err.message });
  }
}

async function handleToggleWorkspaceNote(noteId, enabled) {
  try {
    await api.updateWorkspaceNote(noteId, { enabled });
    const notes = await api.listWorkspaceNotes();
    store.setState({ contextNotes: Array.isArray(notes) ? notes : [], contextError: null });
    const st = store.getState();
    if (st.activeResearchId) await loadContextPreviewForRun(st.activeResearchId);
  } catch (err) {
    store.setState({ contextError: err.message });
  }
}

async function handleDeleteWorkspaceNote(noteId) {
  try {
    await api.deleteWorkspaceNote(noteId);
    const notes = await api.listWorkspaceNotes();
    store.setState({ contextNotes: Array.isArray(notes) ? notes : [], contextError: null });
    const st = store.getState();
    if (st.activeResearchId) await loadContextPreviewForRun(st.activeResearchId);
  } catch (err) {
    store.setState({ contextError: err.message });
  }
}

async function refreshModelRegistry() {
  try {
    const entries = await api.listRegisteredModels();
    store.setState({ modelRegistry: Array.isArray(entries) ? entries : [] });
  } catch {
    // Ignore transient registry refresh errors
  }
}

async function loadGenerationsForRun(researchId) {
  if (!researchId) return;
  try {
    const runs = await api.listGenerations(researchId);
    const list = Array.isArray(runs) ? runs : [];
    const st = store.getState();
    if ((st.activeResearchId || st.researchPackage?.research_id) !== researchId) {
      return;
    }
    store.setState({
      generationRuns: list,
      generationResult: list.length > 0 ? list[0] : null,
    });
  } catch {
    // Preserve UI state cleanly on history fetch miss
  }
}

async function handleStartGeneration() {
  const st = store.getState();
  const researchId = st.activeResearchId || st.researchPackage?.research_id;
  if (!researchId || st.generationRunning) return;

  if (activeGenerationAbortController) {
    activeGenerationAbortController.abort();
  }
  activeGenerationAbortController = new AbortController();
  const { signal } = activeGenerationAbortController;

  const selectedEntry = resolveSelectedModelEntry(st);
  const prefs = st.prefs;
  const maxOut = Number.parseInt(String(prefs.genMaxOutputTokens || ""), 10);
  const payload = {
    research_id: researchId,
    ...(selectedEntry
      ? { connection_id: selectedEntry.connection_id, model_id: selectedEntry.model_id }
      : {}),
    template_type: prefs.genTemplate || "detailed_explanation",
    output_format: prefs.genFormat || "markdown",
    answer_length: prefs.genLength || "medium",
    generation_mode: prefs.genMode || "standard",
    temperature: Number.isFinite(Number(prefs.genTemperature)) ? Number(prefs.genTemperature) : 0.2,
    ...(Number.isFinite(maxOut) && maxOut >= 16 ? { max_output_tokens: maxOut } : {}),
    stream: Boolean(prefs.genStream),
  };

  store.setState({
    generationRunning: true,
    generationActiveId: null,
    generationStreamContent: "",
    generationStreamOutline: [],
    generationStreamSections: [],
    generationStreamCurrentSection: null,
    generationStreamWarnings: [],
    generationStreamCitations: [],
    generationStreamSourceMap: {},
    generationStreamContextStats: null,
    generationError: null,
  });
  announce("Starting local LLM answer generation.");

  try {
    if (!payload.stream) {
      const result = await api.generateAnswer(payload, signal);
      store.setState({
        generationRunning: false,
        generationActiveId: null,
        generationResult: result,
        generationError: null,
      });
      await loadGenerationsForRun(researchId);
      announce("Local answer generation completed.");
      return;
    }

    await api.streamGeneration(payload, {
      signal,
      onEvent: (ev) => {
        const cur = store.getState();
        if (ev.event === "start") {
          store.setState({
            generationActiveId: ev.generation_id || null,
            generationStreamSourceMap: ev.source_map || {},
            generationStreamContextStats: ev.context_stats || null,
          });
        } else if (ev.event === "outline") {
          store.setState({
            generationStreamOutline: Array.isArray(ev.outline) ? ev.outline : [],
          });
        } else if (ev.event === "section_start") {
          store.setState({
            generationStreamCurrentSection: {
              section_id: ev.section_id,
              order: ev.order,
              title: ev.title,
            },
          });
        } else if (ev.event === "section_complete" && ev.section) {
          store.setState({
            generationStreamSections: [...(cur.generationStreamSections || []), ev.section],
          });
        } else if (ev.event === "delta" && typeof ev.delta === "string") {
          store.setState({
            generationStreamContent: (cur.generationStreamContent || "") + ev.delta,
          });
        } else if (ev.event === "warning" && ev.warning) {
          store.setState({
            generationStreamWarnings: [...(cur.generationStreamWarnings || []), ev.warning],
          });
        } else if (ev.event === "citation") {
          store.setState({
            generationStreamCitations: Array.isArray(ev.citations) ? ev.citations : [],
          });
        } else if (ev.event === "complete" || ev.event === "cancelled") {
          store.setState({
            generationRunning: false,
            generationActiveId: null,
            generationStreamCurrentSection: null,
            generationResult: ev.generation || null,
            generationError: null,
          });
          announce(
            ev.event === "cancelled"
              ? "Generation stopped cleanly with partial output retained."
              : "Answer generation complete."
          );
        } else if (ev.event === "error") {
          store.setState({
            generationRunning: false,
            generationActiveId: null,
            generationStreamCurrentSection: null,
            generationResult: ev.generation || cur.generationResult,
            generationError: ev.detail || "Generation failed.",
          });
        }
      },
    });
    if (store.getState().generationRunning) {
      store.setState({ generationRunning: false, generationActiveId: null });
    }
    await loadGenerationsForRun(researchId);
  } catch (err) {
    if (err.name === "AbortError") {
      store.setState({
        generationRunning: false,
        generationActiveId: null,
        generationStreamCurrentSection: null,
      });
      await loadGenerationsForRun(researchId);
      return;
    }
    store.setState({
      generationRunning: false,
      generationActiveId: null,
      generationStreamCurrentSection: null,
      generationError: err instanceof ApiError ? err.message : err.message || "Generation failed.",
    });
    announce(`Generation error: ${err.message}`);
  } finally {
    if (activeGenerationAbortController?.signal === signal) {
      activeGenerationAbortController = null;
    }
  }
}

async function handleStopGeneration() {
  const st = store.getState();
  const activeId = st.generationActiveId;
  if (activeId) {
    try {
      const stopped = await api.cancelGeneration(activeId);
      if (stopped && stopped.content) {
        store.setState({ generationResult: stopped });
      }
      announce("Cancellation requested.");
    } catch {
      if (activeGenerationAbortController) {
        activeGenerationAbortController.abort();
      }
    }
    return;
  }
  if (activeGenerationAbortController) {
    activeGenerationAbortController.abort();
    activeGenerationAbortController = null;
  }
  store.setState({ generationRunning: false, generationActiveId: null });
}

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
  onOpenContextTab: () => {
    refreshContextMetadata();
    const st = store.getState();
    if (st.activeResearchId) {
      loadContextPreviewForRun(st.activeResearchId);
    }
  },
  onOpenGenerationTab: () => {
    refreshModelRegistry();
    const st = store.getState();
    if (st.activeResearchId) {
      loadGenerationsForRun(st.activeResearchId);
    }
  },
  onRefreshModelRegistry: refreshModelRegistry,
  onSelectGenerationRun: async (generationId) => {
    const existing = (store.getState().generationRuns || []).find(
      (r) => r.generation_id === generationId
    );
    if (existing) {
      store.setState({ generationResult: existing });
      return;
    }
    try {
      const loaded = await api.getGeneration(generationId);
      if (loaded) store.setState({ generationResult: loaded });
    } catch {
      // Ignore transient fetch failure
    }
  },
  onApplyContextBudget: handleApplyContextBudget,
  onPreviewContext: handlePreviewContext,
  onCompressContext: handleCompressContext,
  onAddWorkspaceNote: handleAddWorkspaceNote,
  onToggleWorkspaceNote: handleToggleWorkspaceNote,
  onDeleteWorkspaceNote: handleDeleteWorkspaceNote,
});

function renderAll(state) {
  toolbar.render(state);
  syncToolbarLlmDisplay(state);
  sidebar.render(state);
  workspace.render(state);
  inspector.render(state);
  if (settingsDialogEl.open) {
    settingsSheet.render(state);
    syncModelSettingsPane(state);
  }
}

store.subscribe(renderAll);
initSplitResizers();
renderAll(store.getState());

async function bootstrap() {
  const [healthRes, configRes, runsRes, presetsRes, llmConnsRes, ctxCapsRes, ctxBudgetRes, ctxNotesRes, regModelsRes] = await Promise.allSettled([
    api.getHealth(),
    api.getConfig(),
    api.listResearchRuns(),
    api.listLlmPresets(),
    api.listLlmConnections(),
    api.getContextCapabilities(),
    api.getContextBudget(),
    api.listWorkspaceNotes(),
    api.listRegisteredModels(),
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

  if (presetsRes.status === "fulfilled" && Array.isArray(presetsRes.value)) {
    patch.llmPresets = presetsRes.value;
  }

  if (llmConnsRes.status === "fulfilled" && Array.isArray(llmConnsRes.value)) {
    patch.llmConnections = llmConnsRes.value;
  }

  if (ctxCapsRes.status === "fulfilled" && ctxCapsRes.value?.capabilities) {
    patch.contextCapabilities = ctxCapsRes.value.capabilities;
  }

  if (ctxBudgetRes.status === "fulfilled" && ctxBudgetRes.value) {
    patch.contextBudget = ctxBudgetRes.value;
  }

  if (ctxNotesRes.status === "fulfilled" && Array.isArray(ctxNotesRes.value)) {
    patch.contextNotes = ctxNotesRes.value;
  }

  if (regModelsRes.status === "fulfilled" && Array.isArray(regModelsRes.value)) {
    patch.modelRegistry = regModelsRes.value;
  }

  patch.backendError = errors.length && !patch.health ? errors[0] : null;
  store.setState(patch);
}

bootstrap();
