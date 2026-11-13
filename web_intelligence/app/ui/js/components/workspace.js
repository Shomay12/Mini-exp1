/**
 * Main Research Workspace — Prominent Query Composer, Task Controls, Active Progress,
 * Privacy Disclosures, Long-Form Evidence Synthesis, Inline Citations, and Conflict Comparison.
 */

import {
  escapeHtml,
  formatDateTime,
  formatNumber,
  formatScore,
  ICONS,
  renderStatusPill,
  safeHttpUrl,
  truncateText,
} from "../formatters.js";

const PIPELINE_STAGES = [
  { id: "planning", label: "Plan" },
  { id: "searching", label: "Search" },
  { id: "retrieving", label: "Retrieve" },
  { id: "processing", label: "Extract & Verify" },
  { id: "completed", label: "Package" },
];

const STARTER_PROMPTS = [
  "Aurora-SLM-4B release date and small language models benchmarks and quantum entanglement routing",
  "How do 4-bit quantized small language models compare for local deployment and BM25 retrieval grounding?",
  "Evaluate hybrid BM25 + cross-encoder reranking trade-offs on small language model context budgets",
];

export function createWorkspace({
  store,
  onSubmitQuery,
  onCancelRun,
  onInspectSource,
  onInspectEvidence,
  onSwitchToResearchWithQuery,
}) {
  const root = document.getElementById("workspace-root");
  root.innerHTML = `
    <section id="ws-composer" aria-label="Research input"></section>
    <section id="ws-status" aria-label="Research progress and status"></section>
    <section id="ws-results" aria-label="Research results"></section>
  `;

  const composerEl = document.getElementById("ws-composer");
  const statusEl = document.getElementById("ws-status");
  const resultsEl = document.getElementById("ws-results");

  const expandedEvidenceIds = new Set();
  let outboundExpanded = false;
  let claimsTableExpanded = false;
  let limitsPopoverOpen = false;
  let lastComposerSig = "";
  let lastResultsSig = "";

  // Close inline limits popover when clicking outside
  document.addEventListener("click", (event) => {
    if (!limitsPopoverOpen) return;
    if (!event.target.closest("#limits-popover-anchor")) {
      limitsPopoverOpen = false;
      const pop = composerEl.querySelector("#limits-popover");
      const btn = composerEl.querySelector("#btn-limits-popover");
      if (pop) pop.hidden = true;
      if (btn) btn.setAttribute("aria-expanded", "false");
    }
  });

  composerEl.addEventListener("input", (event) => {
    if (event.target.id === "research-query-input") {
      store.getState().queryDraft = event.target.value;
      const runBtn = composerEl.querySelector("#btn-submit-query");
      if (runBtn && !store.getState().isRunning) {
        runBtn.disabled = event.target.value.trim().length < 2;
      }
      const counter = composerEl.querySelector("#composer-char-count");
      if (counter) counter.textContent = `${event.target.value.length} / 2,000`;
    }
  });

  composerEl.addEventListener("keydown", (event) => {
    if (event.target.id === "research-query-input" && event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      const q = event.target.value.trim();
      if (q.length >= 2 && !store.getState().isRunning) {
        onSubmitQuery(q);
      }
    }
  });

  composerEl.addEventListener("change", (event) => {
    const el = event.target;
    if (el.id === "inline-provider-select") {
      const val = el.value;
      store.updatePrefs({ selectedProviders: val === "auto" ? [] : [val] });
    } else if (el.id === "inline-max-results") {
      store.updatePrefs({ maxResultsPerQuery: clampInt(el.value, 1, 20, 6) });
    } else if (el.id === "inline-max-fetch") {
      store.updatePrefs({ maxFetchUrls: clampInt(el.value, 1, 25, 8) });
    } else if (el.id === "inline-chunk-tokens") {
      store.updatePrefs({ chunkTargetTokens: clampInt(el.value, 40, 1000, 180) });
    }
  });

  composerEl.addEventListener("click", (event) => {
    if (event.target.closest("#btn-limits-popover")) {
      limitsPopoverOpen = !limitsPopoverOpen;
      const pop = composerEl.querySelector("#limits-popover");
      const btn = composerEl.querySelector("#btn-limits-popover");
      if (pop) pop.hidden = !limitsPopoverOpen;
      if (btn) btn.setAttribute("aria-expanded", String(limitsPopoverOpen));
      return;
    }

    if (event.target.closest("#btn-submit-query")) {
      const ta = composerEl.querySelector("#research-query-input");
      const q = (ta ? ta.value : store.getState().queryDraft).trim();
      if (q.length >= 2) {
        limitsPopoverOpen = false;
        onSubmitQuery(q);
      }
      return;
    }

    if (event.target.closest("#btn-cancel-run")) {
      onCancelRun();
    }
  });

  statusEl.addEventListener("click", (event) => {
    if (event.target.closest("#btn-cancel-progress")) {
      onCancelRun();
      return;
    }
    const discBtn = event.target.closest("#btn-toggle-outbound");
    if (discBtn) {
      outboundExpanded = !outboundExpanded;
      discBtn.setAttribute("aria-expanded", String(outboundExpanded));
      const body = statusEl.querySelector("#outbound-disclosure-body");
      if (body) body.hidden = !outboundExpanded;
    }
  });

  resultsEl.addEventListener("click", (event) => {
    const starterBtn = event.target.closest("[data-starter-prompt]");
    if (starterBtn) {
      const prompt = starterBtn.dataset.starterPrompt;
      store.setState({ queryDraft: prompt });
      const ta = composerEl.querySelector("#research-query-input");
      if (ta) {
        ta.value = prompt;
        ta.focus();
        const runBtn = composerEl.querySelector("#btn-submit-query");
        if (runBtn) runBtn.disabled = false;
      }
      return;
    }

    const citeBtn = event.target.closest("[data-cite-source]");
    if (citeBtn) {
      onInspectSource(citeBtn.dataset.citeSource, citeBtn.dataset.citeEvidence || null);
      return;
    }

    const evDiscBtn = event.target.closest("[data-toggle-evidence]");
    if (evDiscBtn) {
      const evId = evDiscBtn.dataset.toggleEvidence;
      const expanded = expandedEvidenceIds.has(evId);
      if (expanded) expandedEvidenceIds.delete(evId);
      else expandedEvidenceIds.add(evId);
      evDiscBtn.setAttribute("aria-expanded", String(!expanded));
      const body = resultsEl.querySelector(`[data-evidence-body="${CSS.escape(evId)}"]`);
      if (body) body.hidden = expanded;
      return;
    }

    const claimsDiscBtn = event.target.closest("#btn-toggle-claims-table");
    if (claimsDiscBtn) {
      claimsTableExpanded = !claimsTableExpanded;
      claimsDiscBtn.setAttribute("aria-expanded", String(claimsTableExpanded));
      const body = resultsEl.querySelector("#claims-table-body");
      if (body) body.hidden = !claimsTableExpanded;
      return;
    }

    const promoteSearchBtn = event.target.closest("#btn-research-this-query");
    if (promoteSearchBtn) {
      const q = store.getState().searchResponse?.original_query || store.getState().queryDraft;
      if (q) onSwitchToResearchWithQuery(q);
    }
  });

  function clampInt(raw, min, max, fallback) {
    const n = Number.parseInt(raw, 10);
    if (!Number.isFinite(n)) return fallback;
    return Math.min(max, Math.max(min, n));
  }

  function focusInput() {
    const ta = composerEl.querySelector("#research-query-input");
    if (ta) ta.focus();
  }

  function setInputValue(text) {
    const ta = composerEl.querySelector("#research-query-input");
    if (ta) {
      ta.value = text;
      const runBtn = composerEl.querySelector("#btn-submit-query");
      if (runBtn) runBtn.disabled = text.trim().length < 2 || store.getState().isRunning;
      const counter = composerEl.querySelector("#composer-char-count");
      if (counter) counter.textContent = `${text.length} / 2,000`;
    }
  }

  function renderComposer(state) {
    const providers = (state.health?.providers || []).filter((p) => p.enabled);
    const sig = JSON.stringify({
      mode: state.mode,
      isRunning: state.isRunning,
      providers: providers.map((p) => p.name),
      prefs: state.prefs,
    });

    if (sig === lastComposerSig && composerEl.children.length > 0) {
      return;
    }
    lastComposerSig = sig;

    const currentDraft = state.queryDraft || "";
    const selectedProvider =
      state.prefs.selectedProviders && state.prefs.selectedProviders.length === 1
        ? state.prefs.selectedProviders[0]
        : "auto";
    const isSearch = state.mode === "search";
    const submitLabel = isSearch ? "Search Web" : "Run Research";
    const placeholder = isSearch
      ? "Enter a search query (PII, bearer tokens, and local paths are redacted before outbound search)…"
      : "Ask a research question to retrieve, deduplicate, rank, and cross-verify web evidence locally…";

    composerEl.innerHTML = `
      <div class="composer">
        <label class="sr-only" for="research-query-input">Research query</label>
        <textarea
          id="research-query-input"
          class="composer__textarea"
          maxlength="2000"
          placeholder="${escapeHtml(placeholder)}"
          ${state.isRunning ? "disabled" : ""}
        >${escapeHtml(currentDraft)}</textarea>

        <div class="composer__controls">
          <div class="composer__params">
            <label class="field-inline">
              <span>Provider</span>
              <select id="inline-provider-select" class="select-control" ${state.isRunning ? "disabled" : ""}>
                <option value="auto" ${selectedProvider === "auto" ? "selected" : ""}>Auto (Router)</option>
                ${providers
                  .map(
                    (p) =>
                      `<option value="${escapeHtml(p.name)}" ${selectedProvider === p.name ? "selected" : ""}>${escapeHtml(p.name)}${p.is_mock ? " (offline)" : ""}</option>`
                  )
                  .join("")}
              </select>
            </label>

            <div id="limits-popover-anchor" class="popover-anchor">
              <button
                id="btn-limits-popover"
                class="btn"
                type="button"
                aria-expanded="${String(limitsPopoverOpen)}"
                aria-controls="limits-popover"
              >
                <span>Limits: ${state.prefs.maxResultsPerQuery} res · ${state.prefs.maxFetchUrls} URLs…</span>
              </button>
              <div id="limits-popover" class="popover-panel" ${limitsPopoverOpen ? "" : "hidden"}>
                <div class="form-grid-row">
                  <label for="inline-max-results">Results / subquery</label>
                  <input id="inline-max-results" class="number-control" type="number" min="1" max="20" value="${state.prefs.maxResultsPerQuery}" />
                </div>
                <div class="form-grid-row">
                  <label for="inline-max-fetch">Max fetched URLs</label>
                  <input id="inline-max-fetch" class="number-control" type="number" min="1" max="25" value="${state.prefs.maxFetchUrls}" />
                </div>
                <div class="form-grid-row">
                  <label for="inline-chunk-tokens">Chunk target tokens</label>
                  <input id="inline-chunk-tokens" class="number-control" type="number" min="40" max="1000" step="10" value="${state.prefs.chunkTargetTokens}" />
                </div>
              </div>
            </div>

            <span id="composer-char-count" class="doc-section__sub mono">${currentDraft.length} / 2,000</span>
          </div>

          <div class="composer__actions">
            <span class="doc-section__sub toolbar-hide-compact"><span class="kbd">⌘Enter</span></span>
            ${
              state.isRunning
                ? `<button id="btn-cancel-run" class="btn" type="button">Cancel</button>`
                : ""
            }
            <button
              id="btn-submit-query"
              class="btn btn--prominent"
              type="button"
              ${state.isRunning || currentDraft.trim().length < 2 ? "disabled" : ""}
            >
              ${state.isRunning ? "Running…" : submitLabel}
            </button>
          </div>
        </div>
      </div>
    `;
  }

  function renderStatus(state) {
    const blocks = [];

    if (state.backendError) {
      blocks.push(`
        <div class="banner banner--error" role="alert">
          <div>
            <strong>Backend Connection Unavailable</strong>
            <div style="margin-top:4px;">${escapeHtml(state.backendError)}. Start FastAPI with <span class="mono">uvicorn web_intelligence.app.main:app --reload</span> and reload.</div>
          </div>
        </div>
      `);
    }

    if (state.workspaceError) {
      blocks.push(`
        <div class="banner banner--error" role="alert">
          <div>
            <strong>Execution Did Not Complete</strong>
            <div style="margin-top:4px;">${escapeHtml(state.workspaceError)}</div>
          </div>
        </div>
      `);
    }

    if (state.isRunning) {
      const p = state.activeProgress || { current_stage: "planning", completed_stages: [] };
      const stageIdx = Math.max(
        0,
        PIPELINE_STAGES.findIndex((s) => s.id === p.current_stage)
      );
      const pct = Math.min(95, Math.max(12, Math.round(((stageIdx + 1) / PIPELINE_STAGES.length) * 100)));
      const stageDescription = describeStage(p.current_stage, state.mode);

      blocks.push(`
        <div class="progress-panel" role="status" aria-live="polite">
          <div class="progress-panel__head">
            <div class="progress-panel__title">
              ${ICONS.running}
              <span>${escapeHtml(stageDescription)}</span>
            </div>
            <button id="btn-cancel-progress" class="btn" type="button">Stop Waiting</button>
          </div>
          <div class="progress-track" aria-hidden="true">
            <div class="progress-track__bar" style="width: ${pct}%;"></div>
          </div>
          <div class="progress-stages">
            ${PIPELINE_STAGES.map((st, idx) => {
              const isDone = (p.completed_stages || []).includes(st.id) || idx < stageIdx;
              const isCurrent = st.id === p.current_stage;
              const cls = isDone ? "is-done" : isCurrent ? "is-current" : "";
              return `<span class="progress-stage ${cls}">${isDone ? "✓" : isCurrent ? "●" : "○"} ${escapeHtml(st.label)}</span>`;
            }).join("")}
          </div>
          <div class="progress-telemetry">
            <span>Planned queries: ${p.queries_planned ?? 0}</span>
            <span>URLs found: ${p.urls_found ?? 0}</span>
            <span>Fetched: ${p.documents_fetched ?? 0}</span>
            <span>Deduplicated: ${p.duplicates_removed ?? 0}</span>
            <span>Evidence chunks: ${p.evidence_items_count ?? 0}</span>
          </div>
        </div>
      `);
    }

    // Outbound Query Disclosure & Privacy Notice when results are loaded
    const disclosed =
      state.mode === "research"
        ? state.researchPackage?.search_queries || []
        : state.searchResponse?.disclosed_queries || [];
    const privacyNotice = state.researchPackage?.metadata?.privacy_notice || "";

    if (disclosed.length > 0 || privacyNotice) {
      const totalRedactions = disclosed.reduce((sum, q) => sum + (q.redactions_applied?.length || 0), 0);
      blocks.push(`
        <div class="banner banner--privacy">
          <div style="width:100%;">
            <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
              <div style="display:flex; align-items:center; gap:8px;">
                ${ICONS.shield}
                <span>
                  <strong>Outbound Disclosure:</strong> ${disclosed.length} sanitized quer${disclosed.length === 1 ? "y" : "ies"}
                  ${totalRedactions > 0 ? `(${totalRedactions} redaction${totalRedactions === 1 ? "" : "s"} applied)` : "(no PII/secrets detected)"}
                </span>
              </div>
              <button
                id="btn-toggle-outbound"
                class="disclosure-btn"
                type="button"
                aria-expanded="${String(outboundExpanded)}"
                aria-controls="outbound-disclosure-body"
              >
                ${ICONS.triangle}
                <span>Inspect Outbound Queries</span>
              </button>
            </div>
            <div id="outbound-disclosure-body" class="disclosure-body" ${outboundExpanded ? "" : "hidden"}>
              ${privacyNotice ? `<p style="margin:0; color:var(--text-secondary);">${escapeHtml(privacyNotice)}</p>` : ""}
              ${disclosed
                .map(
                  (q) => `
                  <div style="display:flex; flex-direction:column; gap:2px; padding-top:4px;">
                    <div>
                      <span class="status-pill">${escapeHtml(q.provider)} · ${q.is_external ? "External" : "Local Mock"}</span>
                      <span class="mono">${escapeHtml(q.sanitized_query)}</span>
                    </div>
                    <div class="doc-section__sub">
                      Intent: ${escapeHtml(q.intent)} · Results: ${q.result_count ?? 0}
                      ${q.redactions_applied?.length ? ` · Redacted: ${escapeHtml(q.redactions_applied.join(", "))}` : ""}
                    </div>
                  </div>
                `
                )
                .join("")}
            </div>
          </div>
        </div>
      `);
    }

    statusEl.innerHTML = blocks.join("");
  }

  function describeStage(stage, mode) {
    if (mode === "search") return "Sanitizing query and querying configured search providers…";
    const copy = {
      pending: "Initializing local research pipeline…",
      planning: "Decomposing research question into subtopics and sanitized queries…",
      searching: "Querying search providers with outbound PII redaction…",
      retrieving: "Fetching candidate webpages through SSRF & robots.txt safeguards…",
      processing: "Deduplicating documents, ranking passages, extracting claims & checking coverage…",
      completed: "Finalizing local ResearchPackage…",
    };
    return copy[stage] || "Executing local research pipeline…";
  }

  function renderEmptyState(state) {
    return `
      <div class="empty-state">
        <h2 class="empty-state__title">Local-First Research Workbench</h2>
        <p class="empty-state__desc">
          Queries are sanitized locally, routed through SSRF-guarded webpage retrieval, deduplicated across four tiers,
          ranked with deterministic BM25 signals, and cross-checked for claim conflicts and coverage gaps — with zero cloud LLM dependency.
        </p>
        <div class="starter-list" aria-label="Sample research questions">
          ${STARTER_PROMPTS.map(
            (prompt) => `
              <button class="starter-prompt" type="button" data-starter-prompt="${escapeHtml(prompt)}">
                <span>${escapeHtml(prompt)}</span>
                <span class="doc-section__sub">Use Prompt</span>
              </button>
            `
          ).join("")}
        </div>
      </div>
    `;
  }

  function renderResearchPackage(pkg, state) {
    const meta = pkg.metadata || {};
    const cov = pkg.coverage || {};
    const plan = pkg.query_plan;
    const totalBudget = Math.max(512, Number(state.prefs.contextBudgetTokens) || 8192);
    const reserved = Math.min(totalBudget - 256, Math.max(0, Number(state.prefs.reservedOutputTokens) || 1024));
    const effectiveBudget = Math.max(256, totalBudget - reserved);

    // Map source_id -> 1-based citation index for readable inline citations
    const sourceIndexMap = new Map();
    (pkg.sources || []).forEach((s, idx) => sourceIndexMap.set(s.source_id, idx + 1));

    // Coverage section
    const subtopics = cov.subtopic_details || [];
    const coverageHtml = `
      <section class="doc-section" aria-labelledby="heading-coverage">
        <div class="doc-section__heading">
          <h2 id="heading-coverage" style="margin:0; font-size:inherit;">1. Query Decomposition & Research Coverage</h2>
          ${renderStatusPill(cov.coverage_status || "insufficient")}
        </div>
        <div class="coverage-box">
          ${
            plan
              ? `<div class="doc-section__sub">
                  Intent: <strong>${escapeHtml(plan.intent)}</strong> · Planner: <span class="mono">${escapeHtml(plan.planner_name)}</span>
                  ${plan.keywords?.length ? ` · Keywords: ${escapeHtml(plan.keywords.slice(0, 8).join(", "))}` : ""}
                </div>`
              : ""
          }
          ${cov.summary ? `<p class="coverage-summary-text">${escapeHtml(cov.summary)}</p>` : ""}
          ${
            subtopics.length
              ? `<div class="subtopic-grid">
                  ${subtopics
                    .map(
                      (st) => `
                      <div class="subtopic-item">
                        <div class="subtopic-item__top">
                          <span class="subtopic-item__name">${escapeHtml(st.subtopic)}</span>
                          ${renderStatusPill(st.status)}
                        </div>
                        <div class="subtopic-item__notes">
                          ${st.independent_domain_count} independent domain${st.independent_domain_count === 1 ? "" : "s"}
                          ${st.has_conflict ? " · Has conflict" : ""}
                          ${st.notes ? ` — ${escapeHtml(st.notes)}` : ""}
                        </div>
                      </div>
                    `
                    )
                    .join("")}
                </div>`
              : ""
          }
          ${
            cov.unanswered_questions?.length
              ? `<div style="font-size: var(--text-callout); color: var(--text-secondary);">
                  <strong>Unanswered Gaps:</strong> ${escapeHtml(cov.unanswered_questions.join(" · "))}
                </div>`
              : ""
          }
        </div>
      </section>
    `;

    // Ranked Evidence Synthesis section with running context-budget cutoff
    let runningTokens = 0;
    let cutoffInserted = false;
    const evidenceItems = pkg.evidence || [];
    const evidenceRowsHtml = evidenceItems.length
      ? evidenceItems
          .map((item) => {
            const tokens = Number(item.token_estimate) || 0;
            const prevTokens = runningTokens;
            runningTokens += tokens;
            const overBudget = runningTokens > effectiveBudget;
            let cutoffHtml = "";
            if (overBudget && !cutoffInserted && prevTokens > 0) {
              cutoffInserted = true;
              cutoffHtml = `
                <div class="budget-cutoff-divider" role="note">
                  <span>Context Budget Limit Reached (${formatNumber(effectiveBudget)} available tokens)</span>
                </div>
              `;
            }

            const citeNum = sourceIndexMap.get(item.source_id) || "?";
            const isExpanded = expandedEvidenceIds.has(item.evidence_id);
            const rel = item.relevance || {};
            const corr = item.corroboration || {};
            const heading = item.section_heading || item.source_title || item.source_domain;

            return `
              ${cutoffHtml}
              <article class="evidence-entry ${overBudget ? "is-over-budget" : ""}">
                <div class="evidence-entry__head">
                  <h3 class="evidence-entry__heading">${escapeHtml(heading)}</h3>
                  ${renderStatusPill(corr.corroboration_status || "unverified")}
                </div>
                <p class="evidence-entry__prose">${escapeHtml(item.passage_text)}</p>
                <div class="evidence-entry__footer">
                  <button
                    class="citation-pill"
                    type="button"
                    data-cite-source="${escapeHtml(item.source_id)}"
                    data-cite-evidence="${escapeHtml(item.evidence_id)}"
                    title="Inspect source ${escapeHtml(item.source_url)}"
                  >
                    <span>[${citeNum}] ${escapeHtml(item.source_domain)}</span>
                  </button>
                  <span class="doc-section__sub mono">${formatNumber(tokens)} tok · chars ${item.char_start}–${item.char_end}</span>
                  <span class="doc-section__sub">Relevance: <strong class="mono">${formatScore(rel.combined_score)}</strong></span>
                  <button
                    class="disclosure-btn"
                    type="button"
                    aria-expanded="${String(isExpanded)}"
                    data-toggle-evidence="${escapeHtml(item.evidence_id)}"
                  >
                    ${ICONS.triangle}
                    <span>Relevance &amp; Provenance</span>
                  </button>
                </div>
                <div class="disclosure-body" data-evidence-body="${escapeHtml(item.evidence_id)}" ${isExpanded ? "" : "hidden"}>
                  <div class="signal-metrics-grid">
                    <div class="metric-pair"><span class="metric-pair__label">Combined Relevance</span><span class="metric-pair__value">${formatScore(rel.combined_score)}</span></div>
                    <div class="metric-pair"><span class="metric-pair__label">BM25 Score</span><span class="metric-pair__value">${formatScore(rel.bm25_score)}</span></div>
                    <div class="metric-pair"><span class="metric-pair__label">Keyword Match</span><span class="metric-pair__value">${formatScore(rel.keyword_match_score)}</span></div>
                    <div class="metric-pair"><span class="metric-pair__label">Heading Match</span><span class="metric-pair__value">${formatScore(rel.heading_relevance_score)}</span></div>
                    <div class="metric-pair"><span class="metric-pair__label">Query Coverage</span><span class="metric-pair__value">${formatScore(rel.query_term_coverage)}</span></div>
                    <div class="metric-pair"><span class="metric-pair__label">Independent Sources</span><span class="metric-pair__value">${corr.independent_supporting_sources ?? 0}</span></div>
                  </div>
                  <div class="doc-section__sub">
                    <strong>Signal Separation:</strong> Relevance scores measure query-term alignment only and are not factual truth (${escapeHtml(item.provenance?.source_classification || "unknown")} source — ${escapeHtml(item.provenance?.classification_reason || "")}).
                  </div>
                </div>
              </article>
            `;
          })
          .join("")
      : `<p class="empty-state__desc">No evidence passages were extracted from the retrieved sources.</p>`;

    // Claims & Conflicts section
    const conflicts = pkg.conflicts || [];
    const claims = pkg.claims || [];
    const conflictsHtml = conflicts.length
      ? conflicts
          .map(
            (c) => `
            <div class="conflict-card">
              <div class="conflict-card__head">
                <div>
                  <strong>${escapeHtml(c.entity)} · ${escapeHtml(c.attribute)}</strong>
                  <div class="doc-section__sub">${escapeHtml(c.summary)}</div>
                </div>
                ${renderStatusPill("contested", c.conflict_type.replaceAll("_", " "))}
              </div>
              <div class="conflict-pair">
                <div class="conflict-claim">
                  <div class="conflict-claim__label">Claim A · ${escapeHtml(c.claim_a.source_domain)}</div>
                  <p class="conflict-claim__text">${escapeHtml(c.claim_a.statement_text)}</p>
                  <div class="doc-section__sub">
                    Normalized: <span class="mono">${escapeHtml(c.claim_a.normalized_value)}${c.claim_a.unit ? ` ${escapeHtml(c.claim_a.unit)}` : ""}</span>
                    · <button class="disclosure-btn" style="padding:0 4px; color:var(--accent);" type="button" data-cite-source="${escapeHtml(c.claim_a.source_id)}">Inspect Source</button>
                  </div>
                </div>
                <div class="conflict-claim">
                  <div class="conflict-claim__label">Claim B · ${escapeHtml(c.claim_b.source_domain)}</div>
                  <p class="conflict-claim__text">${escapeHtml(c.claim_b.statement_text)}</p>
                  <div class="doc-section__sub">
                    Normalized: <span class="mono">${escapeHtml(c.claim_b.normalized_value)}${c.claim_b.unit ? ` ${escapeHtml(c.claim_b.unit)}` : ""}</span>
                    · <button class="disclosure-btn" style="padding:0 4px; color:var(--accent);" type="button" data-cite-source="${escapeHtml(c.claim_b.source_id)}">Inspect Source</button>
                  </div>
                </div>
              </div>
              <p class="conflict-note">${escapeHtml(c.resolution_note)}</p>
            </div>
          `
          )
          .join("")
      : `<p class="empty-state__desc">No cross-source date, numeric, or polarity contradictions detected.</p>`;

    const claimsDisclosureHtml = claims.length
      ? `
        <div>
          <button
            id="btn-toggle-claims-table"
            class="disclosure-btn"
            type="button"
            aria-expanded="${String(claimsTableExpanded)}"
            aria-controls="claims-table-body"
          >
            ${ICONS.triangle}
            <span>Extracted Factual Claims (${claims.length})</span>
          </button>
          <div id="claims-table-body" class="disclosure-body" ${claimsTableExpanded ? "" : "hidden"}>
            ${claims
              .map(
                (cl) => `
                <div style="padding: 6px 0; border-bottom: 1px solid var(--border-subtle);">
                  <div><strong>${escapeHtml(cl.entity)}</strong> · ${escapeHtml(cl.attribute)}: <span class="mono">${escapeHtml(cl.normalized_value)}</span> (${escapeHtml(cl.value_type)})</div>
                  <div style="color:var(--text-secondary); margin-top:2px;">${escapeHtml(cl.statement_text)}</div>
                  <div class="doc-section__sub">
                    <button class="disclosure-btn" style="padding:0; color:var(--accent);" type="button" data-cite-source="${escapeHtml(cl.source_id)}">${escapeHtml(cl.source_domain)}</button>
                  </div>
                </div>
              `
              )
              .join("")}
          </div>
        </div>
      `
      : "";

    return `
      <article class="research-doc">
        <header class="doc-header">
          <h2 class="doc-header__title">${escapeHtml(pkg.original_query)}</h2>
          <div class="doc-header__meta">
            ${renderStatusPill(pkg.research_status)}
            <span>${meta.processed_documents ?? 0} processed (${meta.duplicate_documents ?? 0} deduplicated)</span>
            <span>· ${meta.evidence_items ?? 0} evidence passages</span>
            <span>· ${meta.conflicts_detected ?? 0} conflicts</span>
            <span>· <span class="mono">${formatNumber(meta.total_token_estimate ?? 0)}</span> tokens est.</span>
            <span>· ${escapeHtml(formatDateTime(meta.retrieved_at))}</span>
          </div>
        </header>

        ${coverageHtml}

        <section class="doc-section" aria-labelledby="heading-evidence">
          <div class="doc-section__heading">
            <h2 id="heading-evidence" style="margin:0; font-size:inherit;">2. Ranked Evidence Passages</h2>
            <span class="doc-section__sub">${evidenceItems.length} passages · sorted by relevance</span>
          </div>
          <div class="evidence-list">${evidenceRowsHtml}</div>
        </section>

        <section class="doc-section" aria-labelledby="heading-conflicts">
          <div class="doc-section__heading">
            <h2 id="heading-conflicts" style="margin:0; font-size:inherit;">3. Claim Verification &amp; Conflicting Claims</h2>
            <span class="doc-section__sub">${conflicts.length} conflict${conflicts.length === 1 ? "" : "s"} · ${claims.length} claims</span>
          </div>
          ${conflictsHtml}
          ${claimsDisclosureHtml}
        </section>
      </article>
    `;
  }

  function renderSearchResponse(resp) {
    const results = resp.results || [];
    const diagnostics = resp.provider_diagnostics || [];
    return `
      <article class="research-doc">
        <header class="doc-header">
          <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:12px; flex-wrap:wrap;">
            <h2 class="doc-header__title">${escapeHtml(resp.original_query)}</h2>
            <button id="btn-research-this-query" class="btn" type="button">Run Full Research on Query</button>
          </div>
          <div class="doc-header__meta">
            <span class="status-pill status-pill--privacy">${ICONS.shield}<span>Sanitized: <span class="mono">${escapeHtml(resp.sanitized_query)}</span></span></span>
            <span>${resp.total_results ?? results.length} results</span>
            ${diagnostics
              .map(
                (d) =>
                  `<span>· ${escapeHtml(d.provider)}: ${d.success ? `${d.result_count} hits (${Math.round(d.latency_ms)} ms)` : escapeHtml(d.error || "failed")}</span>`
              )
              .join("")}
          </div>
        </header>

        <section class="doc-section">
          ${
            results.length
              ? results
                  .map(
                    (r) => `
                  <div class="search-result-row">
                    <div class="doc-section__sub">#${r.rank} · ${escapeHtml(r.domain)} · via ${escapeHtml(r.provider)}${r.published_at ? ` · ${escapeHtml(formatDateTime(r.published_at))}` : ""}</div>
                    <a class="search-result-row__title" href="${escapeHtml(safeHttpUrl(r.url))}" target="_blank" rel="noopener noreferrer">${escapeHtml(r.title)}</a>
                    <p style="margin:0; color:var(--text-secondary); max-width:68ch; line-height:1.5;">${escapeHtml(r.snippet || "No snippet returned.")}</p>
                    <div class="doc-section__sub mono">${escapeHtml(truncateText(r.canonical_url, 96))}</div>
                  </div>
                `
                  )
                  .join("")
              : `<p class="empty-state__desc">No results matched this search query.</p>`
          }
        </section>
      </article>
    `;
  }

  function renderResults(state) {
    const sig = JSON.stringify({
      mode: state.mode,
      pkgId: state.researchPackage?.research_id || null,
      pkgStatus: state.researchPackage?.research_status || null,
      pkgEvLen: state.researchPackage?.evidence?.length || 0,
      searchQuery: state.searchResponse?.sanitized_query || null,
      searchLen: state.searchResponse?.results?.length || 0,
      budget: state.prefs.contextBudgetTokens,
      reserved: state.prefs.reservedOutputTokens,
      isRunning: state.isRunning,
    });
    if (sig === lastResultsSig && resultsEl.children.length > 0) {
      return;
    }
    lastResultsSig = sig;

    if (state.mode === "research" && state.researchPackage) {
      resultsEl.innerHTML = renderResearchPackage(state.researchPackage, state);
    } else if (state.mode === "search" && state.searchResponse) {
      resultsEl.innerHTML = renderSearchResponse(state.searchResponse);
    } else if (!state.isRunning) {
      resultsEl.innerHTML = renderEmptyState(state);
    } else {
      resultsEl.innerHTML = "";
    }
  }

  function render(state) {
    renderComposer(state);
    renderStatus(state);
    renderResults(state);
  }

  return { render, focusInput, setInputValue };
}
