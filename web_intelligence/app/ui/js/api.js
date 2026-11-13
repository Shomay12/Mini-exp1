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
