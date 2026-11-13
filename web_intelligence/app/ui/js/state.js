/**
 * Centralized application state and local workspace preference persistence.
 */

const PREFS_KEY = "web_intelligence_workbench_prefs_v1";

const DEFAULT_PREFS = {
  sidebarCollapsed: false,
  inspectorCollapsed: false,
  sidebarWidth: 264,
  inspectorWidth: 380,
  contextBudgetTokens: 8192,
  reservedOutputTokens: 1024,
  maxResultsPerQuery: 6,
  maxFetchUrls: 8,
  chunkTargetTokens: 180,
  useLocalLlm: true,
  selectedProviders: [], // empty = backend default router
  genTemplate: "detailed_explanation",
  genFormat: "markdown",
  genLength: "medium",
  genMode: "standard",
  genTemperature: 0.2,
  genMaxOutputTokens: "",
  genStream: true,
  genSelectedRegistryEntryId: "",
};

function loadPrefs() {
  try {
    const raw = localStorage.getItem(PREFS_KEY);
    if (!raw) return { ...DEFAULT_PREFS };
    return { ...DEFAULT_PREFS, ...JSON.parse(raw) };
  } catch {
    return { ...DEFAULT_PREFS };
  }
}

function savePrefs(prefs) {
  try {
    localStorage.setItem(PREFS_KEY, JSON.stringify(prefs));
  } catch {
    // Ignore storage failures under private browsing quotas.
  }
}

export function createStore() {
  const prefs = loadPrefs();
  const state = {
    mode: "research", // "research" | "search"
    compactPane: "workspace", // "workspace" | "history" | "inspector"
    sidebarCollapsed: Boolean(prefs.sidebarCollapsed),
    inspectorCollapsed: Boolean(prefs.inspectorCollapsed),
    sidebarWidth: prefs.sidebarWidth,
    inspectorWidth: prefs.inspectorWidth,

    // Client-persisted workbench & budget defaults
    prefs,

    // Backend metadata
    health: null,
    config: null,
    backendError: null,

    // History & Filtering
    runs: [],
    historyFilter: "",
    activeResearchId: null,

    // Active Workspace Data
    queryDraft: "",
    isRunning: false,
    activeProgress: null,
    researchPackage: null,
    searchResponse: null,
    workspaceError: null,

    // Inspector Data
    inspectorTab: "sources", // "sources" | "passages" | "conflicts" | "context" | "audit"
    passageFilter: "all", // "all" | "corroborated" | "single_source" | "contested"
    selectedSourceId: null,
    selectedSourceDetail: null,
    sourceDetailLoading: false,
    sourceDetailError: null,

    // Stage 5 Adaptive Context Inspector Data
    contextCapabilities: null,
    contextBudget: null,
    contextPreview: null,
    contextCompressionPreview: null,
    contextNotes: [],
    contextLoading: false,
    contextError: null,
    contextShowFormattedPreview: false,

    // UI Modal State
    settingsTab: "providers", // "providers" | "privacy" | "budget" | "model"
    pendingDeleteRun: null,

    // Universal Local LLM Connection State
    llmPresets: [],
    llmConnections: [],
    llmDiscoveredModels: {},
    llmSelectedConnectionId: null,
    llmBusy: false,
    llmStatusMessage: null,
    llmError: null,
    llmPromptStreamText: "",
    llmPromptRunning: false,
    llmPromptError: null,
    llmPromptMeta: null,

    // Stage 6 Universal Local LLM Generation & ModelRegistry State
    modelRegistry: [],
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
    generationStreamSourceMap: {},
    generationStreamContextStats: null,
    generationError: null,
    generationExpandedSources: false,
    generationOptionsOpen: false,
  };

  const listeners = new Set();

  function notify() {
    for (const listener of listeners) {
      listener(state);
    }
  }

  return {
    getState() {
      return state;
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    setState(patch) {
      Object.assign(state, patch);
      notify();
    },
    updatePrefs(patch) {
      Object.assign(state.prefs, patch);
      if ("sidebarCollapsed" in patch) state.sidebarCollapsed = Boolean(patch.sidebarCollapsed);
      if ("inspectorCollapsed" in patch) state.inspectorCollapsed = Boolean(patch.inspectorCollapsed);
      if ("sidebarWidth" in patch) state.sidebarWidth = patch.sidebarWidth;
      if ("inspectorWidth" in patch) state.inspectorWidth = patch.inspectorWidth;
      savePrefs(state.prefs);
      notify();
    },
  };
}
