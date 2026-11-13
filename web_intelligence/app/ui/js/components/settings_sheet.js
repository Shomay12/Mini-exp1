/**
 * Research Configuration Sheet (⌘,) — Providers, Privacy, and Limits & Budget.
 */

import { escapeHtml, formatBytes, formatNumber } from "../formatters.js";

const SETTINGS_PANES = [
  { id: "providers", label: "Providers" },
  { id: "privacy", label: "Privacy" },
  { id: "budget", label: "Limits & Budget" },
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
      store.updatePrefs({ contextBudgetTokens: clampInt(target.value, 1024, 65536, 8192) });
    } else if (target.id === "setting-reserved-tokens") {
      store.updatePrefs({ reservedOutputTokens: clampInt(target.value, 128, 8192, 1024) });
    } else if (target.id === "setting-max-results") {
      store.updatePrefs({ maxResultsPerQuery: clampInt(target.value, 1, 20, 6) });
    } else if (target.id === "setting-max-fetch") {
      store.updatePrefs({ maxFetchUrls: clampInt(target.value, 1, 25, 8) });
    } else if (target.id === "setting-chunk-tokens") {
      store.updatePrefs({ chunkTargetTokens: clampInt(target.value, 40, 1000, 180) });
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
    const priority = (state.config?.config?.search_providers || []).join(", ") || "mock";

    const providerRows = providers.length
      ? providers
          .map((p) => {
            const credText =
              p.name in creds ? (creds[p.name] ? "Key configured" : "No key") : p.is_mock ? "Offline fixture" : "";
            return `
              <div class="settings-row">
                <div>
                  <label style="display: flex; align-items: center; gap: 8px; font-weight: 500; cursor: pointer;">
                    <input
                      type="checkbox"
                      data-provider-checkbox
                      value="${escapeHtml(p.name)}"
                      ${selected.has(p.name) ? "checked" : ""}
                      ${!p.enabled ? "disabled" : ""}
                    />
                    <span>${escapeHtml(p.name)}</span>
                  </label>
                  ${
                    p.notes || credText
                      ? `<div class="settings-row__desc">${escapeHtml(p.notes || credText)}${p.notes && credText ? ` · ${escapeHtml(credText)}` : ""}</div>`
                      : ""
                  }
                </div>
                <span class="doc-section__sub">${p.enabled ? "Enabled" : "Disabled"}</span>
              </div>
            `;
          })
          .join("")
      : `<div class="settings-row"><span class="settings-row__desc">No provider status returned.</span></div>`;

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Search Providers</h3>
        <div class="settings-box">${providerRows}</div>
        <div class="settings-row__desc">
          Unchecked uses server order (<span class="mono">${escapeHtml(priority)}</span>).
        </div>
      </section>
    `;
  }

  function renderPrivacyPane(state) {
    const cfg = state.config?.config || {};
    const privacy = cfg.privacy || {};
    const limits = cfg.retrieval_limits || {};

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Server Privacy Settings (Read-Only)</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <strong>Query Sanitization &amp; Disclosure</strong>
              <div class="settings-row__desc">Strips credentials, emails, phone numbers, and local paths before search.</div>
            </div>
            <span class="doc-section__sub">${privacy.disclose_outbound_queries ?? true ? "Disclosed" : "Hidden"}</span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Live Network</strong>
              <div class="settings-row__desc">External HTTP(S) search and page fetches.</div>
            </div>
            <span class="doc-section__sub">${privacy.allow_live_network ?? true ? "Allowed" : "Offline only"}</span>
          </div>
          <div class="settings-row">
            <div>
              <strong>robots.txt &amp; SSRF Protection</strong>
              <div class="settings-row__desc">Allowed ports: ${escapeHtml((limits.allowed_ports || [80, 443]).join(", "))}.</div>
            </div>
            <span class="doc-section__sub">${privacy.respect_robots_txt ?? true ? "Enforced" : "Off"}</span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Telemetry</strong>
            </div>
            <span class="doc-section__sub">${privacy.enable_telemetry ? "Enabled" : "Disabled"}</span>
          </div>
          <div class="settings-row">
            <div>
              <strong>Local Storage</strong>
              <div class="settings-row__desc"><span class="mono">${escapeHtml(cfg.database_path || "./data/web_intelligence.db")}</span></div>
            </div>
            <span class="doc-section__sub">${escapeHtml(String(cfg.retention_days ?? 30))}d retention</span>
          </div>
        </div>
      </section>
    `;
  }

  function renderBudgetPane(state) {
    const limits = state.config?.config?.retrieval_limits || {};
    const p = state.prefs;
    const totalBudget = Math.max(512, Number(p.contextBudgetTokens) || 8192);
    const reserved = Math.min(totalBudget - 256, Math.max(0, Number(p.reservedOutputTokens) || 1024));
    const effective = Math.max(256, totalBudget - reserved);

    return `
      <section class="settings-group">
        <h3 class="settings-group__title">Context Budget</h3>
        <div class="settings-box">
          <div class="settings-row">
            <div>
              <label for="setting-budget-tokens"><strong>Target Window (tokens)</strong></label>
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
              <label for="setting-reserved-tokens"><strong>Reserved Output Tokens</strong></label>
              <div class="settings-row__desc">Available evidence budget: ${formatNumber(effective)} tokens</div>
            </div>
            <input id="setting-reserved-tokens" class="number-control" type="number" min="128" max="8192" step="128" value="${p.reservedOutputTokens}" />
          </div>
        </div>
      </section>

      <section class="settings-group">
        <h3 class="settings-group__title">Request Retrieval Limits</h3>
        <div class="settings-box">
          <div class="settings-row">
            <label for="setting-max-results">Max Results / Subquery</label>
            <input id="setting-max-results" class="number-control" type="number" min="1" max="20" value="${p.maxResultsPerQuery}" />
          </div>
          <div class="settings-row">
            <label for="setting-max-fetch">Max Fetched URLs</label>
            <input id="setting-max-fetch" class="number-control" type="number" min="1" max="25" value="${p.maxFetchUrls}" />
          </div>
          <div class="settings-row">
            <label for="setting-chunk-tokens">Chunk Target Tokens</label>
            <input id="setting-chunk-tokens" class="number-control" type="number" min="40" max="1000" step="10" value="${p.chunkTargetTokens}" />
          </div>
        </div>
      </section>

      <section class="settings-group">
        <h3 class="settings-group__title">Server Bounds &amp; Runtime</h3>
        <div class="settings-box">
          <div class="settings-row">
            <span>Download Cap / Redirects</span>
            <span class="mono">${formatBytes(limits.max_download_bytes ?? 2000000)} · ${limits.max_redirects ?? 5} max</span>
          </div>
          <div class="settings-row">
            <span>Timeouts / Concurrency</span>
            <span class="mono">${limits.request_timeout_seconds ?? 15}s · ${limits.max_concurrent_fetches ?? 4} concurrent</span>
          </div>
          <div class="settings-row">
            <span>Local Model Status</span>
            <span class="doc-section__sub">${state.health?.cloud_llm_required ? "Cloud enabled" : "Local deterministic"}</span>
          </div>
        </div>
      </section>
    `;
  }

  function render(state) {
    const validIds = new Set(SETTINGS_PANES.map((p) => p.id));
    const activeTab = validIds.has(state.settingsTab) ? state.settingsTab : "providers";
    const paneTitle = SETTINGS_PANES.find((p) => p.id === activeTab)?.label || "Settings";

    if (activeTab === "model") {
      const activeBtn = root.querySelector('[data-settings-tab="model"]');
      const bodyEl = root.querySelector(".sheet-body");
      if (activeBtn?.getAttribute("aria-selected") === "true" && bodyEl?.dataset.llmRendered === "1") {
        return;
      }
    }

    let bodyHtml = "";
    if (activeTab === "providers") bodyHtml = renderProvidersPane(state);
    else if (activeTab === "privacy") bodyHtml = renderPrivacyPane(state);
    else if (activeTab === "model") bodyHtml = `<p class="coverage-summary-text">Loading local model connections…</p>`;
    else bodyHtml = renderBudgetPane(state);

    root.innerHTML = `
      <div class="sheet-header">
        <h2 id="settings-dialog-title" class="sheet-title">${escapeHtml(paneTitle)}</h2>
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
      <div class="sheet-body">${bodyHtml}</div>
      <div class="sheet-footer">
        <button id="btn-reset-budget-defaults" class="btn btn--quiet" type="button">Reset Defaults</button>
        <button id="btn-settings-done" class="btn btn--prominent" type="button">Done</button>
      </div>
    `;
  }

  return { open, close, render };
}
