/**
 * FastAPI HTTP client for /api/v1 endpoints with clean error extraction and polling.
 */

const API_BASE = "/api/v1";

export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });

  const text = await response.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }

  if (!response.ok) {
    const message = extractErrorMessage(data, response.status);
    throw new ApiError(message, response.status, data);
  }

  return data;
}

function extractErrorMessage(data, status) {
  if (!data) return `Request failed with HTTP ${status}.`;
  if (typeof data.detail === "string") return data.detail;
  if (Array.isArray(data.detail)) {
    return data.detail
      .map((item) => {
        const loc = Array.isArray(item.loc) ? item.loc.join(".") : "";
        return loc ? `${loc}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }
  if (typeof data.message === "string") return data.message;
  if (typeof data.error === "string") return data.error;
  return `Request failed with HTTP ${status}.`;
}

export const api = {
  getHealth(signal) {
    return requestJson("/health", { method: "GET", signal });
  },

  getConfig(signal) {
    return requestJson("/config", { method: "GET", signal });
  },

  listResearchRuns(signal) {
    return requestJson("/research", { method: "GET", signal });
  },

  getResearchRun(researchId, signal) {
    return requestJson(`/research/${encodeURIComponent(researchId)}`, { method: "GET", signal });
  },

  deleteResearchRun(researchId) {
    return requestJson(`/research/${encodeURIComponent(researchId)}`, { method: "DELETE" });
  },

  getSourceDetail(sourceId, signal) {
    return requestJson(`/sources/${encodeURIComponent(sourceId)}`, { method: "GET", signal });
  },

  runSearch(payload, signal) {
    return requestJson("/search", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  runResearch(payload, signal) {
    return requestJson("/research", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  listLlmPresets(signal) {
    return requestJson("/llm/presets", { method: "GET", signal });
  },

  listLlmConnections(signal) {
    return requestJson("/llm/connections", { method: "GET", signal });
  },

  createLlmConnection(payload, signal) {
    return requestJson("/llm/connections", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  updateLlmConnection(connectionId, payload, signal) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
      signal,
    });
  },

  deleteLlmConnection(connectionId) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}`, {
      method: "DELETE",
    });
  },

  activateLlmConnection(connectionId) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}/activate`, {
      method: "POST",
    });
  },

  disconnectLlm() {
    return requestJson("/llm/disconnect", { method: "POST" });
  },

  testLlmConnection(connectionId, signal) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}/test`, {
      method: "POST",
      signal,
    });
  },

  discoverLlmModels(connectionId, signal) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}/models`, {
      method: "GET",
      signal,
    });
  },

  selectLlmModel(connectionId, model, signal) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}/model`, {
      method: "POST",
      body: JSON.stringify({ model }),
      signal,
    });
  },

  runLlmTestPrompt(connectionId, payload, signal) {
    return requestJson(`/llm/connections/${encodeURIComponent(connectionId)}/prompt`, {
      method: "POST",
      body: JSON.stringify({ ...payload, stream: false }),
      signal,
    });
  },

  getContextCapabilities(connectionId = null, refresh = false, signal) {
    const params = new URLSearchParams();
    if (connectionId) params.set("connection_id", connectionId);
    if (refresh) params.set("refresh", "true");
    const qs = params.toString();
    return requestJson(`/context/capabilities${qs ? `?${qs}` : ""}`, { method: "GET", signal });
  },

  getContextBudget(connectionId = null, signal) {
    const qs = connectionId ? `?connection_id=${encodeURIComponent(connectionId)}` : "";
    return requestJson(`/context/budget${qs}`, { method: "GET", signal });
  },

  updateContextBudget(payload, connectionId = null, signal) {
    const qs = connectionId ? `?connection_id=${encodeURIComponent(connectionId)}` : "";
    return requestJson(`/context/budget${qs}`, {
      method: "PUT",
      body: JSON.stringify(payload),
      signal,
    });
  },

  previewContext(payload, signal) {
    return requestJson("/context/preview", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  validateContext(payload, signal) {
    return requestJson("/context/validate", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  getResearchContext(researchId, connectionId = null, signal) {
    const qs = connectionId ? `?connection_id=${encodeURIComponent(connectionId)}` : "";
    return requestJson(`/context/${encodeURIComponent(researchId)}${qs}`, {
      method: "GET",
      signal,
    });
  },

  compressContext(payload, signal) {
    return requestJson("/context/compress", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  listWorkspaceNotes(signal) {
    return requestJson("/context/notes", { method: "GET", signal });
  },

  createWorkspaceNote(payload, signal) {
    return requestJson("/context/notes", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  updateWorkspaceNote(noteId, payload, signal) {
    return requestJson(`/context/notes/${encodeURIComponent(noteId)}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
      signal,
    });
  },

  deleteWorkspaceNote(noteId) {
    return requestJson(`/context/notes/${encodeURIComponent(noteId)}`, {
      method: "DELETE",
    });
  },

  listRegisteredModels(params = {}, signal) {
    const qs = new URLSearchParams();
    if (params.connection_id) qs.set("connection_id", params.connection_id);
    if (params.only_connected) qs.set("only_connected", "true");
    if (params.task_role) qs.set("task_role", params.task_role);
    const suffix = qs.toString() ? `?${qs.toString()}` : "";
    return requestJson(`/llm/registry${suffix}`, { method: "GET", signal });
  },

  registerManualModel(payload, signal) {
    return requestJson("/llm/registry", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  updateModelTaskRoles(entryId, taskRoles, signal) {
    const encodedPath = String(entryId)
      .split("/")
      .map((seg) => encodeURIComponent(seg))
      .join("/");
    return requestJson(`/llm/registry/${encodedPath}/roles`, {
      method: "PATCH",
      body: JSON.stringify({ task_roles: taskRoles }),
      signal,
    });
  },

  generateAnswer(payload, signal) {
    return requestJson("/generation", {
      method: "POST",
      body: JSON.stringify(payload),
      signal,
    });
  },

  listGenerations(researchId = null, signal = undefined) {
    const suffix = researchId ? `?research_id=${encodeURIComponent(researchId)}` : "";
    return requestJson(`/generation${suffix}`, { method: "GET", signal });
  },

  getGeneration(generationId, signal) {
    return requestJson(`/generation/${encodeURIComponent(generationId)}`, {
      method: "GET",
      signal,
    });
  },

  cancelGeneration(generationId) {
    return requestJson(`/generation/${encodeURIComponent(generationId)}/cancel`, {
      method: "POST",
    });
  },

  async streamGeneration(payload, { signal, onEvent } = {}) {
    const resp = await fetch(`${API_BASE}/generation/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });

    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`;
      try {
        const errBody = await resp.json();
        if (errBody?.detail) detail = String(errBody.detail);
      } catch {
        // ignore parse failure
      }
      throw new ApiError(detail, resp.status);
    }

    if (!resp.body) {
      throw new ApiError("Generation streaming response body unavailable.", 502);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "message";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const rawLine of lines) {
        const line = rawLine.replace(/\r$/, "");
        if (!line) {
          currentEvent = "message";
          continue;
        }
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
          continue;
        }
        if (!line.startsWith("data:")) continue;
        const dataStr = line.slice(5).trim();
        if (dataStr === "[DONE]") {
          return;
        }
        let parsed = null;
        try {
          parsed = JSON.parse(dataStr);
        } catch {
          continue;
        }
        if (parsed && !parsed.event) {
          parsed.event = currentEvent;
        }
        if (onEvent && parsed) {
          onEvent(parsed);
        }
      }
    }
  },

  async streamLlmTestPrompt(connectionId, payload, { signal, onChunk } = {}) {
    const resp = await fetch(`${API_BASE}/llm/connections/${encodeURIComponent(connectionId)}/prompt`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ ...payload, stream: true }),
      signal,
    });

    if (!resp.ok) {
      let detail = `${resp.status} ${resp.statusText}`;
      try {
        const errBody = await resp.json();
        if (errBody?.detail) detail = String(errBody.detail);
      } catch {
        // ignore parse failure
      }
      throw new ApiError(detail, resp.status);
    }

    if (!resp.body) {
      throw new ApiError("Streaming response body unavailable.", 502);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let currentEvent = "message";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
          currentEvent = "message";
          continue;
        }
        if (line.startsWith("event:")) {
          currentEvent = line.slice(6).trim();
          continue;
        }
        if (!line.startsWith("data:")) continue;
        const dataStr = line.slice(5).trim();
        if (dataStr === "[DONE]") {
          if (onChunk) onChunk({ delta: "", done: true });
          return;
        }
        let parsed = null;
        try {
          parsed = JSON.parse(dataStr);
        } catch {
          continue;
        }
        if (currentEvent === "error") {
          throw new ApiError(parsed?.detail || "LLM streaming failed.", 502);
        }
        if (onChunk && parsed) {
          onChunk(parsed);
        }
      }
    }
  },
};

const TERMINAL_STATUSES = new Set(["completed", "failed"]);

/**
 * Poll GET /api/v1/research/{id} until terminal state or AbortController abort.
 */
export async function pollResearchUntilComplete(researchId, { signal, onUpdate, intervalMs = 600 } = {}) {
  while (true) {
    if (signal?.aborted) {
      throw new DOMException("Research polling canceled", "AbortError");
    }
    const pkg = await api.getResearchRun(researchId, signal);
    if (onUpdate) onUpdate(pkg);
    const status = String(pkg?.research_status || "").toLowerCase();
    if (TERMINAL_STATUSES.has(status)) {
      return pkg;
    }
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, intervalMs);
      if (signal) {
        signal.addEventListener(
          "abort",
          () => {
            clearTimeout(timer);
            reject(new DOMException("Research polling canceled", "AbortError"));
          },
          { once: true }
        );
      }
    });
  }
}
