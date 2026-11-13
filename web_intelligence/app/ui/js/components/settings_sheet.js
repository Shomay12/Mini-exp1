/**
 * Research Configuration Sheet (⌘,) — Providers, Privacy, Retrieval & Budget, Phase 2 Local Model.
 * Follows macOS Settings & Sheets HIG: non-customizable pane switcher, real values only, explicit Done/Cancel.
 */

import { escapeHtml, formatBytes, formatNumber, ICONS } from "../formatters.js";

const SETTINGS_PANES = [
  { id: "providers", label: "Providers" },
  { id: "privacy", label: "Privacy & Network" },
  { id: "budget", label: "Retrieval & Budget" },
  { id: "model", label: "Local Model" },
];

export function createSettingsSheet({ store }) {
  const dialog = document.getElementById("settings-dialog");
  const root = document.getElementById("settings-sheet-root");

  function open() {
    render(store.getState());
    if (!dialog.open) {
      dialog.showModal();
    }
  }

  function close() {
    if (dialog.open) {
      dialog.close();
    }
  }

  root.addEventListener("click", (event) => {
    const tabBtn = event.target.closest("[data-settings-tab]");
    if (tabBtn) {
      store.setState({ settingsTab: tabBtn.dataset.settingsTab });
      return;
    }

    if (event.target.closest("#btn-settings-done")) {
      close();
      return;
    }

    if (event.target.closest("#btn-reset-budget-defaults")) {
      store.updatePrefs({
        contextBudgetTokens: 8192,
        reservedOutputTokens: 1024,
        maxResultsPerQuery: 6,
        maxFetchUrls: 8,
        chunkTargetTokens: 180,
        selectedProviders: [],
      });
      return;
    }

    const presetBtn = event.target.closest("[data-budget-preset]");
    if (presetBtn) {
      const target = Number(presetBtn.dataset.budgetPreset);
      if (Number.isFinite(target)) {
        store.updatePrefs({ contextBudgetTokens: target });
      }
    }
  });

  root.addEventListener("change", (event) => {
    const target = event.target;
    if (target.matches("[data-provider-checkbox]")) {
      const checked = Array.from(root.querySelectorAll("[data-provider-checkbox]:checked")).map(
        (el) => el.value
      );
      store.updatePrefs({ selectedProviders: checked });
      return;
    }

    if (target.id === "setting-budget-tokens") {
      const val = clampInt(target.value, 1024, 65536, 8192);
      store.updatePrefs({ contextBudgetTokens: val });
    } else if (target.id === "setting-reserved-tokens") {
      const val = clampInt(target.value, 128, 8192, 1024);
      store.updatePrefs({ reservedOutputTokens: val });
    } else if (target.id === "setting-max-results") {
      const val = clampInt(target.value, 1, 20, 6);
      store.updatePrefs({ maxResultsPerQuery: val });
    } else if (target.id === "setting-max-fetch") {
      const val = clampInt(target.value, 1, 25, 8);
      store.updatePrefs({ maxFetchUrls: val });
    } else if (target.id === "setting-chunk-tokens") {
      const val = clampInt(target.value, 40, 1000, 180);
      store.updatePrefs({ chunkTargetTokens: val });
    }
  });

  function clampInt(raw, min, max, fallback) {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed)) return fallback;
    return Math.min(max, Math.max(min, parsed));
  }

  function renderProvidersPane(state) {
    const providers = state.health?.providers || [];
    const creds = state.config?.config?.configured_credentials || {};
    const selected = new Set(state.prefs.selectedProviders || []);

    const providerRows = providers.length
      ? providers
          .map((p) => {
            const credConfigured =
              p.name in creds ? (creds[p.name] ? "API key configured" : "No API key") : p.is_mock ? "Offline fixture" : "";
            const statusBadge = p.enabled
              ? `<span class="status-pill status-pill--verified">${ICONS.check}<span>Enabled</span></span>`
              : `<span class="status-pill">${ICONS.missing}<span>Disabled</span></span>`;

            return `
              <div class="settings-row">
                <div>
                  <label style="display: flex; align-items: center; gap: 8px; font-weight: 600; cursor: pointer;">
                    <input
                      type="checkbox"
                      data-provider-checkbox
                      value="${escapeHtml(p.name)}"
                      ${selected.has(p.name) ? "checked" : ""}
                      ${!p.enabled ? "disabled" : ""}
                    />
                    <span>${escapeHtml(p.name)}</span>
                  </label>
                  <div class="settings-row__desc">
                    ${escapeHtml(p.notes || (p.is_mock ? "Deterministic offline mock provider" : "External search API adapter"))}
                    ${credConfigured ? ` · ${escapeHtml(credConfigured)}` : ""}
                  </div>
                </div>
                ${statusBadge}
              </div>
            `;
          })
          .join("")
      : `<div class="settings-row"><span class="settings-row__desc">No provider diagnostics available from /api/v1/health.</span></div>`;

    const priority = (state.config?.config?.search_providers || []).join(" → ") || "mock";

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Search Provider Override for Requests</h3>
        <div class="settings-box">
          ${providerRows}
        </div>
        <p class="settings-row__desc">
          Leave all unchecked to follow server priority (<span class="mono">${escapeHtml(priority)}</span>). Credential keys remain hidden on the server and are never sent to the UI.
        </p>
      </section>
    `;
  }

  function renderPrivacyPane(state) {
    const cfg = state.config?.config || {};
    const privacy = cfg.privacy || {};
    const limits = cfg.retrieval_limits || {};
    const boolBadge = (val, trueText = "Active", falseText = "Disabled") =>
      val
        ? `<span class="status-pill status-pill--verified">${ICONS.shield}<span>${trueText}</span></span>`
        : `<span class="status-pill">${ICONS.missing}<span>${falseText}</span></span>`;

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Enforced Server Privacy Policy (.env Runtime)</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <strong>Outbound Query Sanitization & Disclosure</strong>
              <div class="settings-row__desc">Strips tokens, emails, phone numbers, and local paths; logs disclosed outbound queries locally.</div>
            </div>
            ${boolBadge(privacy.disclose_outbound_queries ?? true, "Disclosed", "Hidden")}
          </div>
          <div class="settings-row">
            <div>
              <strong>Live Network Egress</strong>
              <div class="settings-row__desc">Controls whether external HTTP(S) search and page fetches are permitted.</div>
            </div>
            ${boolBadge(privacy.allow_live_network ?? true, "Allowed", "Offline Only")}
          </div>
          <div class="settings-row">
            <div>
              <strong>robots.txt Compliance & SSRF Defense</strong>
              <div class="settings-row__desc">Blocks loopback, private/link-local IPs, metadata hosts, and non-allowed ports (${escapeHtml((limits.allowed_ports || [80, 443]).join(", "))}).</div>
            </div>
            ${boolBadge(privacy.respect_robots_txt ?? true, "Enforced", "Ignored")}
          </div>
          <div class="settings-row">
            <div>
              <strong>Outbound Telemetry & Cloud LLM Dependency</strong>
              <div class="settings-row__desc">Phase 1 intelligence executes locally without external LLM APIs or telemetry.</div>
            </div>
            <span class="status-pill status-pill--verified">${ICONS.shield}<span>Disabled</span></span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Local SQLite Persistence & Retention</strong>
              <div class="settings-row__desc">Database path: <span class="mono">${escapeHtml(cfg.database_path || "./data/web_intelligence.db")}</span></div>
            </div>
            <span class="mono">${escapeHtml(String(cfg.retention_days ?? 30))} days</span>
          </div>
        </div>
      </section>
    `;
  }

  function renderBudgetPane(state) {
    const limits = state.config?.config?.retrieval_limits || {};
    const p = state.prefs;
    const effective = Math.max(256, p.contextBudgetTokens - p.reservedOutputTokens);

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Context Budget Allocation (Client & Export Window)</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <label for="setting-budget-tokens"><strong>Target Model Context Window</strong></label>
              <div class="settings-row__desc">Matches Phase 2 <span class="mono">ModelCapabilities.max_context_tokens</span>.</div>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
              <button class="btn" type="button" data-budget-preset="4096">4K</button>
              <button class="btn" type="button" data-budget-preset="8192">8K</button>
              <button class="btn" type="button" data-budget-preset="16384">16K</button>
              <input id="setting-budget-tokens" class="number-control" type="number" min="1024" max="65536" step="512" value="${p.contextBudgetTokens}" />
            </div>
          </div>
          <div class="settings-row">
            <div>
              <label for="setting-reserved-tokens"><strong>Reserved Output & Prompt Tokens</strong></label>
              <div class="settings-row__desc">Available evidence budget: <strong class="mono">${formatNumber(effective)} tokens</strong></div>
            </div>
            <input id="setting-reserved-tokens" class="number-control" type="number" min="128" max="8192" step="128" value="${p.reservedOutputTokens}" />
          </div>
        </div>
      </section>

      <section class="settings-group">
        <h3 class="settings-group__title">Per-Request Retrieval Defaults</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <label for="setting-max-results"><strong>Max Results per Subquery</strong></label>
              <div class="settings-row__desc">Sent in <span class="mono">ResearchRequest.max_results_per_query</span> (1–20).</div>
            </div>
            <input id="setting-max-results" class="number-control" type="number" min="1" max="20" value="${p.maxResultsPerQuery}" />
          </div>
          <div class="settings-row">
            <div>
              <label for="setting-max-fetch"><strong>Max Webpages Fetched</strong></label>
              <div class="settings-row__desc">Sent in <span class="mono">ResearchRequest.max_fetch_urls</span> (1–25).</div>
            </div>
            <input id="setting-max-fetch" class="number-control" type="number" min="1" max="25" value="${p.maxFetchUrls}" />
          </div>
          <div class="settings-row">
            <div>
              <label for="setting-chunk-tokens"><strong>Passage Chunk Target Tokens</strong></label>
              <div class="settings-row__desc">Sent in <span class="mono">ResearchRequest.chunk_target_tokens</span> (40–1000).</div>
            </div>
            <input id="setting-chunk-tokens" class="number-control" type="number" min="40" max="1000" step="10" value="${p.chunkTargetTokens}" />
          </div>
        </div>
      </section>

      <section class="settings-group">
        <h3 class="settings-group__title">Server Safety Limits (Read-Only)</h3>
        <div class="settings-box">
          <div class="settings-row">
            <span>Max Download Size / Redirects</span>
            <span class="mono">${formatBytes(limits.max_download_bytes ?? 2000000)} · ${limits.max_redirects ?? 5} redirects</span>
          </div>
          <div class="settings-row">
            <span>Request / Connect Timeouts</span>
            <span class="mono">${limits.request_timeout_seconds ?? 15}s / ${limits.connect_timeout_seconds ?? 5}s</span>
          </div>
          <div class="settings-row">
            <span>Concurrency & Per-Host Interval</span>
            <span class="mono">${limits.max_concurrent_fetches ?? 4} fetches · ${limits.per_host_min_interval_seconds ?? 0.5}s interval</span>
          </div>
        </div>
      </section>
    `;
  }

  function renderModelPane(state) {
    const effective = Math.max(256, state.prefs.contextBudgetTokens - state.prefs.reservedOutputTokens);
    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Local Small Language Model Integration</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <strong>Current Runtime Engine</strong>
              <div class="settings-row__desc">Phase 1 deterministic evidence retrieval, deduplication, BM25 ranking, claim extraction, and coverage analysis.</div>
            </div>
            <span class="status-pill status-pill--verified">${ICONS.check}<span>Active (Phase 1)</span></span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Local Inference Adapter (<span class="mono">LocalModelAdapter</span>)</strong>
              <div class="settings-row__desc">Reserved for Phase 2 local model execution (llama.cpp / Ollama / MLX). Zero cloud LLM calls required.</div>
            </div>
            <span class="status-pill">${ICONS.missing}<span>Not Attached</span></span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Context Budget Manager (<span class="mono">ContextBudgetManager</span>)</strong>
              <div class="settings-row__desc">Active UI budget tracker previews evidence chunk fit against ${formatNumber(effective)} effective evidence tokens.</div>
            </div>
            <span class="status-pill status-pill--privacy">${ICONS.shield}<span>Ready</span></span>
          </div>
        </div>
      </section>
    `;
  }

  function render(state) {
    if (!dialog.open && state.settingsTab === undefined) return;
    const activeTab = state.settingsTab || "providers";
    const paneTitle = SETTINGS_PANES.find((p) => p.id === activeTab)?.label || "Configuration";

    let bodyHtml = "";
    if (activeTab === "providers") bodyHtml = renderProvidersPane(state);
    else if (activeTab === "privacy") bodyHtml = renderPrivacyPane(state);
    else if (activeTab === "budget") bodyHtml = renderBudgetPane(state);
    else bodyHtml = renderModelPane(state);

    root.innerHTML = `
      <div class="sheet-header">
        <div class="sheet-header__row">
          <h2 id="settings-dialog-title" class="sheet-title">Configuration — ${escapeHtml(paneTitle)}</h2>
        </div>
        <div class="segmented segmented--full" role="tablist" aria-label="Configuration panes">
          ${SETTINGS_PANES.map(
            (pane) => `
              <button
                class="segmented__item"
                type="button"
                role="tab"
                aria-selected="${String(pane.id === activeTab)}"
                data-settings-tab="${pane.id}"
              >
                ${escapeHtml(pane.label)}
              </button>
            `
          ).join("")}
        </div>
      </div>
      <div class="sheet-body">
        ${bodyHtml}
      </div>
      <div class="sheet-footer">
        <button id="btn-reset-budget-defaults" class="btn btn--quiet" type="button">Reset Defaults</button>
        <button id="btn-settings-done" class="btn btn--prominent" type="button">Done</button>
      </div>
    `;
  }

  return { open, close, render };
}
