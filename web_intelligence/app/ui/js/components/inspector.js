/**
 * Trailing Evidence & Provenance Inspector — Sources (with live GET /api/v1/sources/{id}),
 * Passages (corroboration filter), Conflicts (neutral comparison), and Research Audit.
 */

import {
  escapeHtml,
  formatBytes,
  formatDateTime,
  formatNumber,
  formatScore,
  ICONS,
  renderStatusPill,
  safeHttpUrl,
  truncateText,
} from "../formatters.js";

const PASSAGE_FILTERS = [
  { id: "all", label: "All" },
  { id: "corroborated", label: "Corroborated" },
  { id: "single_source", label: "Single" },
  { id: "contested", label: "Contested" },
];

export function createInspector({ store, onSelectSource, onFilterPassages, onJumpToSource }) {
  const paneEl = document.getElementById("pane-inspector");
  const countEl = document.getElementById("inspector-summary-count");
  const bodyEl = document.getElementById("inspector-body");
  const tabButtons = Array.from(paneEl.querySelectorAll("[data-inspector-tab]"));

  let redirectsExpanded = false;
  let headingsExpanded = false;
  let lastRenderSig = "";

  for (const btn of tabButtons) {
    btn.addEventListener("click", () => {
      store.setState({ inspectorTab: btn.dataset.inspectorTab });
    });
  }

  bodyEl.addEventListener("click", (event) => {
    const srcBtn = event.target.closest("[data-inspect-source]");
    if (srcBtn) {
      const sourceId = srcBtn.dataset.inspectSource;
      if (onJumpToSource && srcBtn.dataset.jumpSource === "true") {
        onJumpToSource(sourceId);
      } else {
        onSelectSource(sourceId);
      }
      return;
    }

    const closeDetailBtn = event.target.closest("#btn-close-source-detail");
    if (closeDetailBtn) {
      store.setState({
        selectedSourceId: null,
        selectedSourceDetail: null,
        sourceDetailLoading: false,
        sourceDetailError: null,
      });
      return;
    }

    const filterBtn = event.target.closest("[data-passage-filter]");
    if (filterBtn) {
      const nextFilter = filterBtn.dataset.passageFilter;
      if (onFilterPassages) onFilterPassages(nextFilter);
      else store.setState({ passageFilter: nextFilter });
      return;
    }

    const redirectBtn = event.target.closest("#btn-toggle-redirects");
    if (redirectBtn) {
      redirectsExpanded = !redirectsExpanded;
      redirectBtn.setAttribute("aria-expanded", String(redirectsExpanded));
      const target = bodyEl.querySelector("#inspector-redirects-body");
      if (target) target.hidden = !redirectsExpanded;
      return;
    }

    const headingsBtn = event.target.closest("#btn-toggle-headings");
    if (headingsBtn) {
      headingsExpanded = !headingsExpanded;
      headingsBtn.setAttribute("aria-expanded", String(headingsExpanded));
      const target = bodyEl.querySelector("#inspector-headings-body");
      if (target) target.hidden = !headingsExpanded;
    }
  });

  function buildSourceIndexMap(pkg) {
    const map = new Map();
    (pkg?.sources || []).forEach((s, idx) => map.set(s.source_id, idx + 1));
    return map;
  }

  function renderSourceDetail(state, sourceSummary, citeNum) {
    const detail =
      state.selectedSourceDetail && state.selectedSourceDetail.source_id === state.selectedSourceId
        ? state.selectedSourceDetail
        : sourceSummary;

    if (!detail) return "";

    const prov = detail.provenance || {};
    const meta = detail.source_metadata || {};
    const openUrl = safeHttpUrl(detail.final_url || detail.canonical_url || detail.original_url);
    const redirects = meta.redirect_chain || [];
    const warnings = meta.warnings || [];
    const headings = meta.headings || [];
    const publisher = prov.publisher || meta.publisher || "Not declared";
    const author = detail.author || prov.author || "Not declared";

    return `
      <section class="source-detail" aria-label="Selected source provenance and content">
        <div class="inspector-head__top">
          <div>
            <div class="doc-section__sub">
              ${citeNum ? `[${citeNum}] · ` : ""}Source Provenance
              ${state.sourceDetailLoading ? " · Refreshing…" : ""}
            </div>
            <h3 style="margin: 2px 0 0; font-size: var(--text-title-3); font-weight: 600; line-height: 1.3;">
              ${escapeHtml(detail.title || detail.domain || "Untitled Document")}
            </h3>
          </div>
          <button id="btn-close-source-detail" class="btn btn--quiet" type="button" title="Collapse source detail">Hide</button>
        </div>

        ${
          state.sourceDetailError
            ? `<div class="banner banner--error" role="alert" style="padding: 8px 10px; font-size: var(--text-caption);">
                <span>Could not refresh live source record: ${escapeHtml(state.sourceDetailError)}. Showing cached package record.</span>
              </div>`
            : ""
        }

        <div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px;">
          ${renderStatusPill(detail.extraction_status)}
          ${
            prov.source_classification
              ? `<span class="status-pill">${escapeHtml(prov.source_classification.replaceAll("_", " "))}</span>`
              : ""
          }
          ${
            openUrl !== "#"
              ? `<a class="inline-link" style="font-size: var(--text-caption);" href="${escapeHtml(openUrl)}" target="_blank" rel="noopener noreferrer">Open Original URL ↗</a>`
              : ""
          }
        </div>

        ${
          detail.extraction_error
            ? `<div class="banner banner--error" style="padding: 8px 10px; font-size: var(--text-caption);">
                Extraction note: ${escapeHtml(detail.extraction_error)}
              </div>`
            : ""
        }

        <dl class="kv-table" style="margin: 0;">
          <dt class="kv-table__key">Domain</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(detail.domain || "—")}</dd>

          <dt class="kv-table__key">Classification</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(prov.classification_reason || "—")}</dd>

          <dt class="kv-table__key">Publisher</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(publisher)}</dd>

          <dt class="kv-table__key">Author</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(author)}</dd>

          <dt class="kv-table__key">Published</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(formatDateTime(detail.published_at))}</dd>

          <dt class="kv-table__key">Retrieved</dt>
          <dd class="kv-table__val" style="margin:0;">${escapeHtml(formatDateTime(detail.retrieved_at))}</dd>

          <dt class="kv-table__key">HTTP / Type</dt>
          <dd class="kv-table__val mono" style="margin:0;">
            ${meta.http_status ?? "—"} · ${escapeHtml(meta.content_type || "unknown")} · ${formatBytes(meta.byte_length)}
          </dd>

          <dt class="kv-table__key">Extractor</dt>
          <dd class="kv-table__val mono" style="margin:0;">
            ${escapeHtml(meta.extractor_used || "none")} · relevance ${formatScore(detail.document_relevance_score)}
          </dd>

          <dt class="kv-table__key">Provider</dt>
          <dd class="kv-table__val mono" style="margin:0;">${escapeHtml(detail.search_provider || "—")}</dd>

          <dt class="kv-table__key">Canonical URL</dt>
          <dd class="kv-table__val mono" style="margin:0;">${escapeHtml(detail.canonical_url || detail.original_url || "—")}</dd>

          ${
            detail.content_hash
              ? `
            <dt class="kv-table__key">SHA-256</dt>
            <dd class="kv-table__val mono" style="margin:0;">${escapeHtml(truncateText(detail.content_hash, 28))}</dd>
          `
              : ""
          }
          ${
            detail.is_duplicate
              ? `
            <dt class="kv-table__key">Deduplicated</dt>
            <dd class="kv-table__val" style="margin:0;">
              ${escapeHtml(detail.duplicate_reason || "duplicate")}
              ${detail.duplicate_similarity != null ? ` (sim ${formatScore(detail.duplicate_similarity)})` : ""}
              ${detail.duplicate_of_source_id ? ` · of <span class="mono">${escapeHtml(truncateText(detail.duplicate_of_source_id, 16))}</span>` : ""}
            </dd>
          `
              : ""
          }
          ${
            detail.alias_urls?.length
              ? `
            <dt class="kv-table__key">Alias URLs</dt>
            <dd class="kv-table__val" style="margin:0;">${detail.alias_urls.length} merged alias URL${detail.alias_urls.length === 1 ? "" : "s"}</dd>
          `
              : ""
          }
        </dl>

        ${
          redirects.length || warnings.length
            ? `
          <div>
            <button
              id="btn-toggle-redirects"
              class="disclosure-btn"
              type="button"
              aria-expanded="${String(redirectsExpanded)}"
              aria-controls="inspector-redirects-body"
            >
              ${ICONS.triangle}
              <span>Redirects &amp; Diagnostics (${redirects.length} hops, ${warnings.length} notes)</span>
            </button>
            <div id="inspector-redirects-body" class="disclosure-body" ${redirectsExpanded ? "" : "hidden"}>
              ${
                redirects.length
                  ? `<div><strong>Redirect Chain:</strong><ol style="margin:4px 0 0; padding-left:18px;">${redirects
                      .map((u) => `<li class="mono" style="word-break:break-all;">${escapeHtml(u)}</li>`)
                      .join("")}</ol></div>`
                  : ""
              }
              ${
                warnings.length
                  ? `<div><strong>Warnings:</strong><ul style="margin:4px 0 0; padding-left:18px;">${warnings
                      .map((w) => `<li>${escapeHtml(w)}</li>`)
                      .join("")}</ul></div>`
                  : ""
              }
            </div>
          </div>
        `
            : ""
        }

        ${
          headings.length
            ? `
          <div>
            <button
              id="btn-toggle-headings"
              class="disclosure-btn"
              type="button"
              aria-expanded="${String(headingsExpanded)}"
              aria-controls="inspector-headings-body"
            >
              ${ICONS.triangle}
              <span>Document Outline (${headings.length} headings)</span>
            </button>
            <div id="inspector-headings-body" class="disclosure-body" ${headingsExpanded ? "" : "hidden"}>
              <ul style="margin:0; padding-left:16px; display:flex; flex-direction:column; gap:4px;">
                ${headings
                  .slice(0, 30)
                  .map(
                    (h) =>
                      `<li style="margin-left:${Math.max(0, ((h.level || 1) - 1) * 10)}px;">
                        <span class="mono" style="color:var(--text-tertiary);">H${h.level || 1}</span> ${escapeHtml(h.text)}
                      </li>`
                  )
                  .join("")}
              </ul>
            </div>
          </div>
        `
            : ""
        }

        <div>
          <div class="doc-section__sub" style="margin-bottom: 6px;">Extracted Clean Content</div>
          <div class="source-reader" aria-label="Extracted cleaned webpage content">${
            detail.main_content
              ? escapeHtml(detail.main_content)
              : escapeHtml(detail.search_snippet || "No cleaned body text available for this source.")
          }</div>
        </div>
      </section>
    `;
  }

  function renderSourcesTab(state, pkg, sourceIndexMap) {
    if (state.mode === "search" && state.searchResponse) {
      const results = state.searchResponse.results || [];
      if (!results.length) {
        return `<p class="empty-state__desc">No search candidates returned for inspection.</p>`;
      }
      return `
        <div class="doc-section__sub">Quick Search returns ranked snippets without fetching page bodies. Switch to Full Research for provenance and text extraction.</div>
        <div class="inspector-list">
          ${results
            .map(
              (r) => `
            <div class="inspector-row" style="cursor:default;">
              <div class="inspector-row__title">#${r.rank} · ${escapeHtml(r.title)}</div>
              <div class="inspector-row__meta">
                <span>${escapeHtml(r.domain)}</span>
                <span>· ${escapeHtml(r.provider)}</span>
                <a class="inline-link" href="${escapeHtml(safeHttpUrl(r.url))}" target="_blank" rel="noopener noreferrer">Open ↗</a>
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      `;
    }

    const sources = pkg?.sources || [];
    if (!sources.length) {
      return `<p class="empty-state__desc">Run or select a research task to inspect retrieved webpages, provenance metadata, and cleaned text.</p>`;
    }

    const selectedSummary = sources.find((s) => s.source_id === state.selectedSourceId) || null;
    const selectedCiteNum = selectedSummary ? sourceIndexMap.get(selectedSummary.source_id) : null;

    return `
      <div class="inspector-list" role="list" aria-label="Retrieved sources">
        ${sources
          .map((src, idx) => {
            const selected = src.source_id === state.selectedSourceId;
            const clsLabel = src.provenance?.source_classification
              ? src.provenance.source_classification.replaceAll("_", " ")
              : "";
            return `
              <button
                class="inspector-row ${selected ? "is-selected" : ""}"
                type="button"
                role="listitem"
                data-inspect-source="${escapeHtml(src.source_id)}"
                aria-pressed="${String(selected)}"
              >
                <div class="inspector-row__title">[${idx + 1}] ${escapeHtml(src.title || src.domain || src.original_url)}</div>
                <div class="inspector-row__meta">
                  <span>${escapeHtml(src.domain)}</span>
                  ${clsLabel ? `<span>· ${escapeHtml(clsLabel)}</span>` : ""}
                  <span>· rel ${formatScore(src.document_relevance_score)}</span>
                  ${src.is_duplicate ? `<span>· duplicate</span>` : ""}
                </div>
                <div style="margin-top:2px;">${renderStatusPill(src.extraction_status)}</div>
              </button>
            `;
          })
          .join("")}
      </div>
      ${selectedSummary ? renderSourceDetail(state, selectedSummary, selectedCiteNum) : ""}
    `;
  }

  function renderPassagesTab(state, pkg, sourceIndexMap) {
    const allPassages = pkg?.evidence || [];
    if (!allPassages.length) {
      return `<p class="empty-state__desc">No extracted evidence passages available for the current view.</p>`;
    }

    const filter = state.passageFilter || "all";
    const filtered =
      filter === "all"
        ? allPassages
        : allPassages.filter((item) => (item.corroboration?.corroboration_status || "unverified") === filter);

    return `
      <div class="segmented segmented--full" role="group" aria-label="Filter passages by corroboration status">
        ${PASSAGE_FILTERS.map(
          (f) => `
          <button
            class="segmented__item"
            type="button"
            aria-pressed="${String(f.id === filter)}"
            data-passage-filter="${f.id}"
          >
            ${escapeHtml(f.label)}
          </button>
        `
        ).join("")}
      </div>

      <div class="doc-section__sub">${filtered.length} of ${allPassages.length} extracted passages shown</div>

      <div class="inspector-list">
        ${
          filtered.length
            ? filtered
                .map((ev) => {
                  const corr = ev.corroboration || {};
                  const rel = ev.relevance || {};
                  const citeNum = sourceIndexMap.get(ev.source_id) || "?";
                  const highlighted = state.highlightedEvidenceId === ev.evidence_id;
                  return `
                    <div class="inspector-row ${highlighted ? "is-selected" : ""}" style="cursor:default;">
                      <div class="inspector-head__top">
                        <span class="doc-section__sub mono">[${citeNum}] ${escapeHtml(ev.source_domain)}</span>
                        ${renderStatusPill(corr.corroboration_status || "unverified")}
                      </div>
                      <div class="inspector-row__title">${escapeHtml(ev.section_heading || ev.source_title || ev.source_domain)}</div>
                      <p style="margin:0; font-size: var(--text-callout); color: var(--text-secondary); line-height:1.45;">
                        ${escapeHtml(truncateText(ev.passage_text, 220))}
                      </p>
                      <div class="inspector-row__meta" style="justify-content:space-between; width:100%;">
                        <span class="mono">rel ${formatScore(rel.combined_score)} · bm25 ${formatScore(rel.bm25_score)} · ${formatNumber(ev.token_estimate)} tok</span>
                        <button
                          class="citation-pill"
                          type="button"
                          data-inspect-source="${escapeHtml(ev.source_id)}"
                          data-jump-source="true"
                        >
                          Source
                        </button>
                      </div>
                    </div>
                  `;
                })
                .join("")
            : `<p class="empty-state__desc">No passages match the “${escapeHtml(filter.replaceAll("_", " "))}” filter.</p>`
        }
      </div>
    `;
  }

  function renderConflictsTab(pkg) {
    const conflicts = pkg?.conflicts || [];
    const claims = pkg?.claims || [];
    if (!conflicts.length && !claims.length) {
      return `<p class="empty-state__desc">No claim conflicts or structured assertions detected in the active run.</p>`;
    }

    return `
      ${
        conflicts.length
          ? `<div class="inspector-list">
              ${conflicts
                .map(
                  (c) => `
                <div class="conflict-card" style="padding: var(--space-3);">
                  <div class="conflict-card__head">
                    <strong>${escapeHtml(c.entity)} · ${escapeHtml(c.attribute)}</strong>
                    ${renderStatusPill("contested", c.conflict_type.replaceAll("_", " "))}
                  </div>
                  <div class="doc-section__sub">${escapeHtml(c.summary)}</div>
                  <div class="conflict-claim">
                    <div class="conflict-claim__label">A · ${escapeHtml(c.claim_a.source_domain)}</div>
                    <div class="conflict-claim__text" style="font-size: var(--text-callout);">${escapeHtml(c.claim_a.statement_text)}</div>
                    <div class="doc-section__sub">
                      <span class="mono">${escapeHtml(c.claim_a.normalized_value)}${c.claim_a.unit ? ` ${escapeHtml(c.claim_a.unit)}` : ""}</span>
                      · <button class="citation-pill" type="button" data-inspect-source="${escapeHtml(c.claim_a.source_id)}" data-jump-source="true">Inspect A</button>
                    </div>
                  </div>
                  <div class="conflict-claim">
                    <div class="conflict-claim__label">B · ${escapeHtml(c.claim_b.source_domain)}</div>
                    <div class="conflict-claim__text" style="font-size: var(--text-callout);">${escapeHtml(c.claim_b.statement_text)}</div>
                    <div class="doc-section__sub">
                      <span class="mono">${escapeHtml(c.claim_b.normalized_value)}${c.claim_b.unit ? ` ${escapeHtml(c.claim_b.unit)}` : ""}</span>
                      · <button class="citation-pill" type="button" data-inspect-source="${escapeHtml(c.claim_b.source_id)}" data-jump-source="true">Inspect B</button>
                    </div>
                  </div>
                  <p class="conflict-note">${escapeHtml(c.resolution_note)}</p>
                </div>
              `
                )
                .join("")}
            </div>`
          : `<p class="empty-state__desc">No cross-source contradictions detected across ${claims.length} extracted claims.</p>`
      }
    `;
  }

  function renderAuditTab(state, pkg) {
    if (state.mode === "search" && state.searchResponse) {
      const resp = state.searchResponse;
      return `
        <div class="settings-group">
          <h3 class="settings-group__title">Search Sanitization Audit</h3>
          <div class="inspector-row" style="cursor:default;">
            <div class="doc-section__sub">Sanitized Outbound Query</div>
            <div class="mono">${escapeHtml(resp.sanitized_query)}</div>
            <div class="doc-section__sub">
              Redactions: ${resp.redactions_applied?.length ? escapeHtml(resp.redactions_applied.join(", ")) : "None"}
            </div>
          </div>
        </div>
        <div class="settings-group">
          <h3 class="settings-group__title">Provider Diagnostics</h3>
          <div class="inspector-list">
            ${(resp.provider_diagnostics || [])
              .map(
                (d) => `
              <div class="inspector-row" style="cursor:default;">
                <div class="inspector-head__top">
                  <strong>${escapeHtml(d.provider)}</strong>
                  ${renderStatusPill(d.success ? "completed" : "failed", d.success ? "OK" : "Error")}
                </div>
                <div class="inspector-row__meta mono">
                  <span>${d.result_count} hits</span>
                  <span>· ${Math.round(d.latency_ms)} ms</span>
                </div>
                ${d.error ? `<div class="doc-section__sub" style="color:var(--signal-conflict);">${escapeHtml(d.error)}</div>` : ""}
              </div>
            `
              )
              .join("")}
          </div>
        </div>
      `;
    }

    if (!pkg) {
      return `<p class="empty-state__desc">Audit logs appear after running or selecting a local research task.</p>`;
    }

    const cov = pkg.coverage || {};
    const meta = pkg.metadata || {};
    const diagnostics = meta.provider_diagnostics || [];
    const queries = pkg.search_queries || [];

    return `
      <div class="settings-group">
        <h3 class="settings-group__title">Execution Summary</h3>
        <div class="inspector-row" style="cursor:default;">
          <dl class="kv-table" style="margin:0;">
            <dt class="kv-table__key">Research ID</dt>
            <dd class="kv-table__val mono" style="margin:0;">${escapeHtml(pkg.research_id)}</dd>
            <dt class="kv-table__key">Planner</dt>
            <dd class="kv-table__val mono" style="margin:0;">${escapeHtml(meta.planner_used || pkg.query_plan?.planner_name || "—")}</dd>
            <dt class="kv-table__key">Duration</dt>
            <dd class="kv-table__val mono" style="margin:0;">${formatNumber(Math.round(meta.duration_ms || 0))} ms</dd>
            <dt class="kv-table__key">Cloud LLM</dt>
            <dd class="kv-table__val" style="margin:0;">${meta.cloud_llm_used ? "Yes" : "Never used (Local deterministic)"}</dd>
            <dt class="kv-table__key">Docs</dt>
            <dd class="kv-table__val mono" style="margin:0;">
              ${meta.fetched_documents ?? 0} fetched · ${meta.duplicate_documents ?? 0} dedup · ${meta.failed_documents ?? 0} failed
            </dd>
            <dt class="kv-table__key">Coverage</dt>
            <dd class="kv-table__val" style="margin:0;">
              ${cov.covered_subtopics ?? 0} covered · ${cov.partial_subtopics ?? 0} partial · ${cov.missing_subtopics ?? 0} missing
            </dd>
          </dl>
        </div>
      </div>

      <div class="settings-group">
        <h3 class="settings-group__title">Outbound Query Disclosure (${queries.length})</h3>
        <div class="inspector-list">
          ${queries
            .map(
              (q) => `
            <div class="inspector-row" style="cursor:default;">
              <div class="inspector-head__top">
                <span class="status-pill">${escapeHtml(q.provider)} · ${q.is_external ? "External" : "Local Mock"}</span>
                <span class="doc-section__sub mono">${q.result_count ?? 0} hits</span>
              </div>
              <div class="mono" style="word-break:break-word;">${escapeHtml(q.sanitized_query)}</div>
              <div class="doc-section__sub">
                Intent: ${escapeHtml(q.intent)}
                ${q.redactions_applied?.length ? ` · Redacted: ${escapeHtml(q.redactions_applied.join(", "))}` : ""}
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      </div>

      <div class="settings-group">
        <h3 class="settings-group__title">Provider Diagnostics (${diagnostics.length})</h3>
        <div class="inspector-list">
          ${diagnostics
            .map(
              (d) => `
            <div class="inspector-row" style="cursor:default;">
              <div class="inspector-head__top">
                <strong>${escapeHtml(d.provider)}</strong>
                ${renderStatusPill(d.success ? "completed" : "failed", d.success ? "OK" : "Failed")}
              </div>
              <div class="inspector-row__meta mono">
                <span>${d.result_count} hits</span>
                <span>· ${Math.round(d.latency_ms)} ms</span>
              </div>
              ${d.error ? `<div class="doc-section__sub" style="color:var(--signal-conflict);">${escapeHtml(d.error)}</div>` : ""}
            </div>
          `
            )
            .join("")}
        </div>
      </div>
    `;
  }

  function render(state) {
    const activeTab = state.inspectorTab || "sources";
    for (const btn of tabButtons) {
      btn.setAttribute("aria-selected", String(btn.dataset.inspectorTab === activeTab));
    }

    const pkg = state.mode === "research" ? state.researchPackage : null;
    if (state.mode === "search" && state.searchResponse) {
      const n = state.searchResponse.results?.length ?? 0;
      countEl.textContent = `${n} result${n === 1 ? "" : "s"}`;
    } else if (pkg) {
      const sCount = pkg.sources?.length ?? 0;
      const eCount = pkg.evidence?.length ?? 0;
      countEl.textContent = `${sCount} src · ${eCount} ev`;
    } else {
      countEl.textContent = "No run loaded";
    }

    const sig = JSON.stringify({
      tab: activeTab,
      mode: state.mode,
      pkgId: pkg?.research_id || null,
      pkgStatus: pkg?.research_status || null,
      pkgSources: pkg?.sources?.length || 0,
      pkgEv: pkg?.evidence?.length || 0,
      searchQ: state.searchResponse?.sanitized_query || null,
      searchN: state.searchResponse?.results?.length || 0,
      passageFilter: state.passageFilter,
      selectedSourceId: state.selectedSourceId,
      selectedDetailId: state.selectedSourceDetail?.source_id || null,
      detailLoading: state.sourceDetailLoading,
      detailError: state.sourceDetailError,
      highlightedEvidenceId: state.highlightedEvidenceId || null,
    });

    if (sig === lastRenderSig && bodyEl.children.length > 0) {
      return;
    }
    lastRenderSig = sig;

    const sourceIndexMap = buildSourceIndexMap(pkg);
    if (activeTab === "sources") {
      bodyEl.innerHTML = renderSourcesTab(state, pkg, sourceIndexMap);
    } else if (activeTab === "passages") {
      bodyEl.innerHTML = renderPassagesTab(state, pkg, sourceIndexMap);
    } else if (activeTab === "conflicts") {
      bodyEl.innerHTML = renderConflictsTab(pkg);
    } else {
      bodyEl.innerHTML = renderAuditTab(state, pkg);
    }
  }

  return { render };
}
