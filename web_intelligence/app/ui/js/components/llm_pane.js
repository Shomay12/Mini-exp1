/**
 * Local & Self-Hosted LLM Connection Management Pane inside Settings Sheet.
 * Renders presets, connection lifecycle controls, model discovery/manual fallback,
 * verified vs configured capability inspection, and live streaming prompt tests.
 */

import { api } from "../api.js";
import { escapeHtml, formatNumber, renderStatusPill } from "../formatters.js";

export function createInitialLlmDraft() {
  return {
    preset: "ollama",
    name: "Local Ollama",
    provider: "ollama",
    baseUrl: "http://localhost:11434",
    apiEndpoint: "",
    apiKey: "",
    model: "",
    contextLength: 8192,
    maxOutputTokens: 1024,
    timeoutSeconds: 60,
    streaming: true,
    allowRemoteEndpoint: false,
    approvedByUser: true,
    activate: true,
    manualModel: "",
    testPrompt: "Reply in one concise sentence confirming model readiness.",
  };
}

function renderProvenanceBadge(source) {
  const normalized = String(source || "unknown").toLowerCase();
  if (normalized === "verified") {
    return `<span class="cap-badge cap-badge--verified" title="Observed directly from model server metadata">verified</span>`;
  }
  if (normalized === "configured") {
    return `<span class="cap-badge cap-badge--configured" title="User-configured assumption">configured</span>`;
  }
  return `<span class="cap-badge cap-badge--unknown" title="Not exposed by server or not yet inspected">unknown</span>`;
}

export function renderModelConnectionsPane(state, draft) {
  const presets = state.llmPresets || [];
  const connections = state.llmConnections || [];
  const activeConn = connections.find((c) => c.is_active) || null;
  const selectedConn =
    connections.find((c) => c.id === state.llmSelectedConnectionId) ||
    activeConn ||
    connections[0] ||
    null;

  const discoveredList = selectedConn
    ? state.llmDiscoveredModels?.[selectedConn.id] || []
    : [];

  const statusHtml = `
    ${
      state.llmError
        ? `<div class="banner--error" role="alert">${escapeHtml(state.llmError)}</div>`
        : ""
    }
    ${
      state.llmStatusMessage
        ? `<div class="llm-notice" role="status">${escapeHtml(state.llmStatusMessage)}</div>`
        : ""
    }
  `;

  const connectionsHtml = connections.length
    ? connections
        .map((conn) => {
          const isSelected = selectedConn && conn.id === selectedConn.id;
          return `
            <div class="llm-conn-card ${isSelected ? "is-selected" : ""}">
              <div class="llm-conn-card__top">
                <div class="llm-conn-card__identity">
                  <strong>${escapeHtml(conn.name)}</strong>
                  ${conn.is_active ? renderStatusPill("sufficient", "active") : ""}
                  ${renderStatusPill(
                    conn.status === "connected"
                      ? "completed"
                      : conn.status === "error"
                        ? "failed"
                        : "pending",
                    conn.status
                  )}
                </div>
                <span class="doc-section__sub mono">${escapeHtml(conn.provider)}</span>
              </div>

              <div class="llm-conn-card__meta">
                <span class="mono">${escapeHtml(conn.base_url)}</span>
                <span>· Model: <strong>${escapeHtml(conn.model || "unselected")}</strong></span>
                <span>· ${formatNumber(conn.context_length)} ctx</span>
                ${conn.has_api_key ? `<span>· Key saved</span>` : ""}
                ${conn.allow_remote_endpoint ? `<span>· Remote host allowed</span>` : `<span>· Loopback only</span>`}
              </div>
              ${
                conn.last_error
                  ? `<div class="llm-conn-card__err">${escapeHtml(conn.last_error)}</div>`
                  : ""
              }

              <div class="llm-conn-card__actions">
                <button class="btn btn--quiet" type="button" data-llm-action="inspect" data-conn-id="${escapeHtml(conn.id)}">
                  Inspect
                </button>
                ${
                  conn.is_active
                    ? `<button class="btn btn--quiet" type="button" data-llm-action="disconnect">Disconnect</button>`
                    : `<button class="btn" type="button" data-llm-action="activate" data-conn-id="${escapeHtml(conn.id)}">Use Connection</button>`
                }
                <button class="btn btn--quiet" type="button" data-llm-action="test" data-conn-id="${escapeHtml(conn.id)}" ${state.llmBusy ? "disabled" : ""}>
                  Test
                </button>
                <button class="btn btn--quiet" type="button" data-llm-action="discover" data-conn-id="${escapeHtml(conn.id)}" ${state.llmBusy ? "disabled" : ""}>
                  Discover Models
                </button>
                <button class="btn btn--danger" type="button" data-llm-action="delete" data-conn-id="${escapeHtml(conn.id)}" ${state.llmBusy ? "disabled" : ""}>
                  Remove
                </button>
              </div>
            </div>
          `;
        })
        .join("")
    : `
      <div class="settings-box">
        <div class="settings-row">
          <div>
            <div>No local model connections configured</div>
            <div class="settings-row__desc">Zero-LLM Stage 4 mode remains fully functional without any connected model.</div>
          </div>
          ${renderStatusPill("pending", "Zero-LLM")}
        </div>
      </div>
    `;

  let inspectorHtml = "";
  if (selectedConn) {
    const caps = selectedConn.capabilities || {
      model_name: selectedConn.model || "unselected",
      backend: selectedConn.provider,
      max_context_tokens: selectedConn.context_length,
      recommended_output_tokens: selectedConn.max_output_tokens,
      supports_streaming: selectedConn.streaming,
      supports_tool_calling: false,
      supports_json_mode: false,
      supports_vision: false,
      sources: {
        context_length: "configured",
        max_output_tokens: "configured",
        streaming: "configured",
        tool_calling: "unknown",
        json_mode: "unknown",
        vision: "unknown",
      },
      available_metadata: {},
    };
    const src = caps.sources || {};
    const metaEntries = Object.entries(caps.available_metadata || {});
    const manualModelVal = draft.manualModel !== undefined && draft.manualModel !== ""
      ? draft.manualModel
      : selectedConn.model || "";

    inspectorHtml = `
      <div class="settings-group">
        <h3 class="settings-group__title">Model Selection &amp; Capabilities — ${escapeHtml(selectedConn.name)}</h3>
        <div class="llm-panel-box">
          ${
            discoveredList.length
              ? `
              <div class="form-grid-row" style="margin-bottom:8px;">
                <label for="llm-discovered-select">Discovered models</label>
                <div class="llm-inline-input">
                  <select id="llm-discovered-select" class="select-control">
                    <option value="">Choose discovered model…</option>
                    ${discoveredList
                      .map(
                        (m) => `
                        <option value="${escapeHtml(m.id)}" ${m.id === selectedConn.model ? "selected" : ""}>
                          ${escapeHtml(m.name)}${m.parameter_size ? ` (${escapeHtml(m.parameter_size)})` : ""}${m.context_length ? ` · ${formatNumber(m.context_length)} ctx` : ""}
                        </option>`
                      )
                      .join("")}
                  </select>
                  <button class="btn" type="button" data-llm-action="apply-discovered" data-conn-id="${escapeHtml(selectedConn.id)}" ${state.llmBusy ? "disabled" : ""}>
                    Select
                  </button>
                </div>
              </div>
              `
              : ""
          }
          <div class="form-grid-row">
            <label for="llm-manual-model">Manual model ID</label>
            <div class="llm-inline-input">
              <input
                id="llm-manual-model"
                class="text-control mono"
                type="text"
                value="${escapeHtml(manualModelVal)}"
                placeholder="e.g. qwen3:4b or Qwen3-4B"
              />
              <button class="btn" type="button" data-llm-action="apply-manual-model" data-conn-id="${escapeHtml(selectedConn.id)}" ${state.llmBusy ? "disabled" : ""}>
                Apply Model
              </button>
            </div>
          </div>

          <div class="cap-grid" aria-label="Detected model capabilities">
            <div class="cap-item">
              <span class="cap-item__label">Context Window</span>
              <span class="cap-item__val mono">${formatNumber(caps.max_context_tokens)} tok</span>
              ${renderProvenanceBadge(src.context_length)}
            </div>
            <div class="cap-item">
              <span class="cap-item__label">Max Output</span>
              <span class="cap-item__val mono">${formatNumber(caps.recommended_output_tokens)} tok</span>
              ${renderProvenanceBadge(src.max_output_tokens)}
            </div>
            <div class="cap-item">
              <span class="cap-item__label">Streaming</span>
              <span class="cap-item__val">${caps.supports_streaming ? "Supported" : "Disabled"}</span>
              ${renderProvenanceBadge(src.streaming)}
            </div>
            <div class="cap-item">
              <span class="cap-item__label">Tool Calling</span>
              <span class="cap-item__val">${caps.supports_tool_calling ? "Supported" : "Unconfirmed"}</span>
              ${renderProvenanceBadge(src.tool_calling)}
            </div>
            <div class="cap-item">
              <span class="cap-item__label">JSON Mode</span>
              <span class="cap-item__val">${caps.supports_json_mode ? "Supported" : "Fallback JSON"}</span>
              ${renderProvenanceBadge(src.json_mode)}
            </div>
            <div class="cap-item">
              <span class="cap-item__label">Vision</span>
              <span class="cap-item__val">${caps.supports_vision ? "Supported" : "Text Only"}</span>
              ${renderProvenanceBadge(src.vision)}
            </div>
          </div>

          ${
            metaEntries.length
              ? `<div class="doc-section__sub mono" style="margin-top:8px;">
                  Metadata: ${metaEntries
                    .map(([k, v]) => `${escapeHtml(k)}=${escapeHtml(typeof v === "object" ? JSON.stringify(v) : String(v))}`)
                    .join(" · ")}
                </div>`
              : ""
          }

          <div class="llm-prompt-tester">
            <label for="llm-test-prompt-input" class="settings-row__desc" style="display:block; margin-bottom:4px;">
              Streaming Verification Prompt (${escapeHtml(selectedConn.model || "requires model")})
            </label>
            <div class="llm-inline-input">
              <input
                id="llm-test-prompt-input"
                class="text-control"
                type="text"
                value="${escapeHtml(draft.testPrompt)}"
                placeholder="Ask a quick sanity question…"
              />
              <button
                class="btn"
                type="button"
                data-llm-action="run-prompt"
                data-conn-id="${escapeHtml(selectedConn.id)}"
                ${state.llmPromptRunning || !selectedConn.model ? "disabled" : ""}
              >
                ${state.llmPromptRunning ? "Streaming…" : "Send Prompt"}
              </button>
            </div>
            ${
              state.llmPromptError
                ? `<div class="llm-conn-card__err" style="margin-top:6px;">${escapeHtml(state.llmPromptError)}</div>`
                : ""
            }
            ${
              state.llmPromptStreamText
                ? `<pre id="llm-test-prompt-output" class="llm-stream-output">${escapeHtml(state.llmPromptStreamText)}</pre>`
                : ""
            }
          </div>
        </div>
      </div>
    `;
  }

  return `
    ${statusHtml}

    <div class="settings-group">
      <h3 class="settings-group__title">Configured Model Connections</h3>
      <div class="llm-conn-list">
        ${connectionsHtml}
      </div>
    </div>

    ${inspectorHtml}

    <div class="settings-group">
      <h3 class="settings-group__title">Add Local or Self-Hosted Endpoint</h3>
      <div class="llm-panel-box llm-form">
        <div class="form-grid-row">
          <label for="llm-form-preset">Server Preset</label>
          <select id="llm-form-preset" class="select-control">
            ${(presets.length
              ? presets
              : [
                  { id: "ollama", label: "Ollama", provider: "ollama", default_base_url: "http://localhost:11434" },
                  { id: "lm-studio", label: "LM Studio", provider: "openai-compatible", default_base_url: "http://localhost:1234/v1" },
                  { id: "llamacpp", label: "llama.cpp", provider: "openai-compatible", default_base_url: "http://localhost:8080/v1" },
                  { id: "vllm", label: "vLLM", provider: "openai-compatible", default_base_url: "http://localhost:8000/v1" },
                  { id: "localai", label: "LocalAI", provider: "openai-compatible", default_base_url: "http://localhost:8080/v1" },
                  { id: "custom", label: "Custom OpenAI-Compatible", provider: "openai-compatible", default_base_url: "http://localhost:8000/v1" },
                ]
            )
              .map(
                (p) =>
                  `<option value="${escapeHtml(p.id)}" ${draft.preset === p.id ? "selected" : ""}>${escapeHtml(p.label)}</option>`
              )
              .join("")}
          </select>
        </div>

        <div class="form-grid-row">
          <label for="llm-form-name">Connection Name</label>
          <input id="llm-form-name" class="text-control" type="text" value="${escapeHtml(draft.name)}" placeholder="Local Ollama" />
        </div>

        <div class="form-grid-row">
          <label for="llm-form-provider">Protocol Adapter</label>
          <select id="llm-form-provider" class="select-control">
            <option value="ollama" ${draft.provider === "ollama" ? "selected" : ""}>Ollama (Native /api)</option>
            <option value="openai-compatible" ${draft.provider === "openai-compatible" ? "selected" : ""}>OpenAI-Compatible (/v1)</option>
          </select>
        </div>

        <div class="form-grid-row">
          <label for="llm-form-base-url">Base URL</label>
          <input id="llm-form-base-url" class="text-control mono" type="url" value="${escapeHtml(draft.baseUrl)}" placeholder="http://localhost:11434" />
        </div>

        <div class="form-grid-row">
          <label for="llm-form-api-endpoint">Chat Endpoint Override</label>
          <input id="llm-form-api-endpoint" class="text-control mono" type="text" value="${escapeHtml(draft.apiEndpoint)}" placeholder="Optional, e.g. /chat/completions" />
        </div>

        <div class="form-grid-row">
          <label for="llm-form-api-key">Optional API Key</label>
          <input id="llm-form-api-key" class="text-control" type="password" value="${escapeHtml(draft.apiKey)}" placeholder="Optional (stored locally, never returned)" autocomplete="off" />
        </div>

        <div class="form-grid-row">
          <label for="llm-form-model">Model Identifier</label>
          <input id="llm-form-model" class="text-control mono" type="text" value="${escapeHtml(draft.model)}" placeholder="Optional, e.g. Qwen3-4B" />
        </div>

        <div class="llm-two-col">
          <div class="form-grid-row">
            <label for="llm-form-context">Context Window</label>
            <input id="llm-form-context" class="number-control" type="number" min="512" max="2097152" step="512" value="${draft.contextLength}" />
          </div>
          <div class="form-grid-row">
            <label for="llm-form-max-output">Max Output</label>
            <input id="llm-form-max-output" class="number-control" type="number" min="64" max="131072" step="128" value="${draft.maxOutputTokens}" />
          </div>
        </div>

        <div class="form-grid-row">
          <label for="llm-form-timeout">Timeout (seconds)</label>
          <input id="llm-form-timeout" class="number-control" type="number" min="1" max="600" value="${draft.timeoutSeconds}" />
        </div>

        <div class="llm-check-row">
          <label><input id="llm-form-streaming" type="checkbox" ${draft.streaming ? "checked" : ""} /> Prefer streaming tokens</label>
          <label><input id="llm-form-activate" type="checkbox" ${draft.activate ? "checked" : ""} /> Set as active model after save</label>
        </div>

        <div class="llm-check-row">
          <label><input id="llm-form-approved" type="checkbox" ${draft.approvedByUser ? "checked" : ""} /> Explicitly approve endpoint for research inference</label>
        </div>

        <div class="llm-check-row">
          <label>
            <input id="llm-form-allow-remote" type="checkbox" ${draft.allowRemoteEndpoint ? "checked" : ""} />
            Allow non-loopback LAN or remote host (explicit security exception)
          </label>
          <div class="settings-row__desc">Loopback (localhost, 127.0.0.1, ::1) works by default. Metadata IPs remain always blocked.</div>
        </div>

        <div class="llm-form__submit">
          <button id="btn-llm-create-conn" class="btn btn--prominent" type="button" data-llm-action="create" ${state.llmBusy ? "disabled" : ""}>
            Save Connection
          </button>
        </div>
      </div>
    </div>
  `;
}

export async function refreshLlmConnections(store) {
  const [conns, health] = await Promise.all([
    api.listLlmConnections(),
    api.getHealth(),
  ]);
  store.setState({
    llmConnections: Array.isArray(conns) ? conns : [],
    health,
  });
}

export async function handleLlmActionClick(button, { store, draft, rootEl }) {
  const action = button.dataset.llmAction;
  const connId = button.dataset.connId || "";

  if (action === "inspect") {
    const conn = (store.getState().llmConnections || []).find((c) => c.id === connId);
    draft.manualModel = conn?.model || "";
    store.setState({
      llmSelectedConnectionId: connId,
      llmError: null,
      llmPromptError: null,
    });
    return;
  }

  if (action === "create") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const payload = {
        name: (draft.name || "").trim() || "Local Endpoint",
        provider: draft.provider || "openai-compatible",
        preset: draft.preset || null,
        base_url: (draft.baseUrl || "").trim(),
        api_endpoint: (draft.apiEndpoint || "").trim() || null,
        api_key: (draft.apiKey || "").trim() || null,
        model: (draft.model || "").trim(),
        context_length: Math.max(512, Number(draft.contextLength) || 8192),
        max_output_tokens: Math.max(64, Number(draft.maxOutputTokens) || 1024),
        request_timeout_seconds: Math.max(1, Number(draft.timeoutSeconds) || 60),
        streaming: Boolean(draft.streaming),
        allow_remote_endpoint: Boolean(draft.allowRemoteEndpoint),
        approved_by_user: Boolean(draft.approvedByUser),
        activate: Boolean(draft.activate),
      };
      const created = await api.createLlmConnection(payload);
      draft.apiKey = "";
      draft.manualModel = created.model || "";
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmSelectedConnectionId: created.id,
        llmStatusMessage: `Saved connection '${created.name}'. Click Test to verify reachability.`,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "activate") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const activated = await api.activateLlmConnection(connId);
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmSelectedConnectionId: activated.id,
        llmStatusMessage: `Activated '${activated.name}'.`,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "disconnect") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      await api.disconnectLlm();
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmStatusMessage: "Disconnected active local LLM (running in Zero-LLM mode).",
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "test") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const res = await api.testLlmConnection(connId);
      const nextDiscovered = {
        ...(store.getState().llmDiscoveredModels || {}),
        [connId]: res.discovered_models || [],
      };
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmSelectedConnectionId: connId,
        llmDiscoveredModels: nextDiscovered,
        llmStatusMessage: res.message,
        llmError: res.ok ? null : res.message,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "discover") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const res = await api.discoverLlmModels(connId);
      const nextDiscovered = {
        ...(store.getState().llmDiscoveredModels || {}),
        [connId]: res.models || [],
      };
      store.setState({
        llmBusy: false,
        llmSelectedConnectionId: connId,
        llmDiscoveredModels: nextDiscovered,
        llmStatusMessage: res.message,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "apply-discovered") {
    const selectEl = rootEl.querySelector("#llm-discovered-select");
    const chosen = selectEl?.value?.trim() || "";
    if (!chosen) return;
    draft.manualModel = chosen;
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const updated = await api.selectLlmModel(connId, chosen);
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmStatusMessage: `Selected model '${updated.model}' and refreshed capabilities.`,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "apply-manual-model") {
    const manualInput = rootEl.querySelector("#llm-manual-model");
    const chosen = (manualInput?.value || draft.manualModel || "").trim();
    if (!chosen) {
      store.setState({ llmError: "Enter a model identifier before applying." });
      return;
    }
    draft.manualModel = chosen;
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      const updated = await api.selectLlmModel(connId, chosen);
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmStatusMessage: `Configured model '${updated.model}'.`,
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "delete") {
    store.setState({ llmBusy: true, llmError: null, llmStatusMessage: null });
    try {
      await api.deleteLlmConnection(connId);
      await refreshLlmConnections(store);
      store.setState({
        llmBusy: false,
        llmSelectedConnectionId: null,
        llmStatusMessage: "Removed LLM connection.",
      });
    } catch (err) {
      store.setState({ llmBusy: false, llmError: err.message });
    }
    return;
  }

  if (action === "run-prompt") {
    const promptInput = rootEl.querySelector("#llm-test-prompt-input");
    const promptText = (promptInput?.value || draft.testPrompt || "").trim();
    if (!promptText) return;
    draft.testPrompt = promptText;
    store.setState({
      llmPromptRunning: true,
      llmPromptStreamText: "",
      llmPromptError: null,
    });
    let accumulated = "";
    try {
      await api.streamLlmTestPrompt(
        connId,
        { prompt: promptText, stream: true },
        {
          onChunk: (chunk) => {
            if (chunk?.delta) {
              accumulated += chunk.delta;
              store.setState({ llmPromptStreamText: accumulated });
            }
          },
        }
      );
      if (!accumulated) {
        accumulated = "(Model returned empty response)";
      }
      store.setState({
        llmPromptRunning: false,
        llmPromptStreamText: accumulated,
      });
    } catch (err) {
      store.setState({
        llmPromptRunning: false,
        llmPromptError: err.message,
      });
    }
  }
}

export function syncLlmDraftField(target, { store, draft }) {
  const id = target.id;
  if (!id) return false;

  if (id === "llm-form-preset") {
    const presetId = target.value;
    draft.preset = presetId;
    const presetObj = (store.getState().llmPresets || []).find((p) => p.id === presetId);
    if (presetObj) {
      draft.provider = presetObj.provider;
      draft.baseUrl = presetObj.default_base_url;
      draft.name = `${presetObj.label} Local`;
    }
    return true; // request re-render so Base URL / Provider inputs reflect preset
  }

  const fieldMap = {
    "llm-form-name": ["name", "value"],
    "llm-form-provider": ["provider", "value"],
    "llm-form-base-url": ["baseUrl", "value"],
    "llm-form-api-endpoint": ["apiEndpoint", "value"],
    "llm-form-api-key": ["apiKey", "value"],
    "llm-form-model": ["model", "value"],
    "llm-form-context": ["contextLength", "number"],
    "llm-form-max-output": ["maxOutputTokens", "number"],
    "llm-form-timeout": ["timeoutSeconds", "number"],
    "llm-form-streaming": ["streaming", "checked"],
    "llm-form-allow-remote": ["allowRemoteEndpoint", "checked"],
    "llm-form-approved": ["approvedByUser", "checked"],
    "llm-form-activate": ["activate", "checked"],
    "llm-manual-model": ["manualModel", "value"],
    "llm-test-prompt-input": ["testPrompt", "value"],
  };
  const mapping = fieldMap[id];
  if (!mapping) return false;
  const [key, kind] = mapping;
  if (kind === "checked") draft[key] = Boolean(target.checked);
  else if (kind === "number") draft[key] = Number(target.value);
  else draft[key] = target.value;
  return false;
}
