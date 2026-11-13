/**
 * Stage 6 Research Answer Workspace — Local Model Selector, Generation Controls,
 * Live SSE Streaming Output, Interactive [S#] Citation Chips, and Source Drawer.
 */

import {
  escapeHtml,
  formatNumber,
  ICONS,
  renderStatusPill,
  safeHttpUrl,
  truncateText,
} from "../formatters.js";

const TEMPLATE_OPTIONS = [
  { id: "quick_answer", label: "Quick Answer" },
  { id: "detailed_explanation", label: "Detailed" },
  { id: "research_report", label: "Research Report" },
  { id: "comparative_analysis", label: "Comparative" },
  { id: "technical_report", label: "Technical" },
];

const FORMAT_OPTIONS = [
  { id: "markdown", label: "Markdown" },
  { id: "prose", label: "Prose" },
  { id: "bullets", label: "Bullets" },
  { id: "json", label: "JSON" },
];

const LENGTH_OPTIONS = [
  { id: "short", label: "Short" },
  { id: "medium", label: "Medium" },
  { id: "long", label: "Long" },
];

const MODE_OPTIONS = [
  { id: "standard", label: "Standard" },
  { id: "long_form_report", label: "Long-Form" },
];

/**
 * Resolve configured ModelRegistryEntry or active LLM fallback summary for display.
 */
export function resolveSelectedModelEntry(state) {
  const entries = state.modelRegistry || [];
  const chosenId = state.prefs.genSelectedRegistryEntryId;
  if (chosenId) {
    const exact = entries.find((e) => e.entry_id === chosenId);
    if (exact) return exact;
  }
  const activeEntry = entries.find((e) => e.is_active);
  if (activeEntry) return activeEntry;
  if (entries.length > 0) return entries[0];
  return null;
}

function linkifyInlineText(rawText, sourceMap, packageSources = []) {
  const srcLookup = new Map((packageSources || []).map((s) => [s.source_id, s]));

  // Escape HTML first so arbitrary webpage/model text can never inject markup
  let safe = escapeHtml(rawText || "");
  safe = safe.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/`([^`\n]+)`/g, '<code class="gen-inline-code">$1</code>');

  // Support both [S1] and comma-separated [S1, S2]
  safe = safe.replace(/\[((?:S\d+)(?:\s*,\s*S\d+)*)\]/g, (full, inner) => {
    const ids = inner.split(",").map((s) => s.trim());
    return ids
      .map((cid) => {
        const entry = sourceMap?.[cid];
        if (!entry) {
          return `<span class="citation-chip citation-chip--unknown" title="Unresolved citation ${escapeHtml(cid)}">[${escapeHtml(cid)}]</span>`;
        }
        const src = srcLookup.get(entry.source_id);
        const title = entry.title || src?.title || entry.domain;
        return `<button class="citation-link citation-chip" type="button" data-cite-source="${escapeHtml(entry.source_id)}" data-cite-id="${escapeHtml(cid)}" title="${escapeHtml(`${cid}: ${title} (${entry.domain})`)}">${escapeHtml(cid)}</button>`;
      })
      .join(" ");
  });

  return safe;
}

export function renderSafeStructuredBody(rawText, outputFormat, sourceMap, packageSources) {
  const text = String(rawText || "");
  if (!text.trim()) return "";

  if (outputFormat === "json") {
    return `<pre class="gen-json-pre mono">${linkifyInlineText(text, sourceMap, packageSources)}</pre>`;
  }

  const lines = text.split("\n");
  const blocks = [];
  let listItems = [];
  let paraLines = [];

  const flushList = () => {
    if (!listItems.length) return;
    blocks.push(
      `<ul class="gen-list">${listItems
        .map((item) => `<li>${linkifyInlineText(item, sourceMap, packageSources)}</li>`)
        .join("")}</ul>`
    );
    listItems = [];
  };

  const flushPara = () => {
    if (!paraLines.length) return;
    blocks.push(
      `<p class="gen-para">${linkifyInlineText(paraLines.join("\n"), sourceMap, packageSources).replaceAll("\n", "<br />")}</p>`
    );
    paraLines = [];
  };

  for (const rawLine of lines) {
    const trimmed = rawLine.trim();
    if (!trimmed) {
      flushList();
      flushPara();
      continue;
    }
    const headingMatch = trimmed.match(/^(#{1,3})\s+(.+)$/);
    if (headingMatch) {
      flushList();
      flushPara();
      const level = headingMatch[1].length;
      const cls = level === 1 ? "gen-h1" : level === 2 ? "gen-h2" : "gen-h3";
      blocks.push(
        `<div class="${cls}">${linkifyInlineText(headingMatch[2], sourceMap, packageSources)}</div>`
      );
      continue;
    }
    const bulletMatch = trimmed.match(/^[-*•]\s+(.+)$/);
    if (bulletMatch) {
      flushPara();
      listItems.push(bulletMatch[1]);
      continue;
    }
    flushList();
    paraLines.push(trimmed);
  }
  flushList();
  flushPara();

  return blocks.join("");
}

export function renderGenerationWorkspaceSection(state, pkg) {
  if (!pkg) return "";

  const prefs = state.prefs;
  const registryEntries = state.modelRegistry || [];
  const selectedEntry = resolveSelectedModelEntry(state);
  const activeLlmFallback = state.health?.active_llm || null;
  const hasAnyModel = Boolean(selectedEntry || activeLlmFallback);

  const isRunning = Boolean(state.generationRunning);
  const res = state.generationResult;
  const activeResultMatchesPkg = res && res.research_id === pkg.research_id ? res : null;

  const displayText = isRunning
    ? state.generationStreamContent || ""
    : activeResultMatchesPkg?.content || "";
  const activeSourceMap = isRunning
    ? state.generationStreamSourceMap || {}
    : activeResultMatchesPkg?.source_map || {};
  const activeFormat = isRunning
    ? prefs.genFormat
    : activeResultMatchesPkg?.output_format || prefs.genFormat;

  const warnings = isRunning
    ? state.generationStreamWarnings || []
    : activeResultMatchesPkg?.warnings || [];

  const citedList = isRunning
    ? state.generationStreamCitations || []
    : activeResultMatchesPkg?.citations || [];

  const currentModelLabel = selectedEntry
    ? `${selectedEntry.connection_name} · ${selectedEntry.model_id}`
    : activeLlmFallback
      ? `${activeLlmFallback.name} · ${activeLlmFallback.model || "default"}`
      : "No local model connected";

  let statusPillHtml = "";
  if (isRunning) {
    statusPillHtml = renderStatusPill("processing", "Streaming…");
  } else if (activeResultMatchesPkg) {
    const st = String(activeResultMatchesPkg.status || "").toLowerCase();
    if (st === "completed") {
      const valPassed = activeResultMatchesPkg.citation_validation?.passed;
      statusPillHtml = renderStatusPill(
        valPassed === false ? "partial" : "sufficient",
        valPassed === false ? "Citation Check" : "Completed"
      );
    } else if (st === "cancelled") {
      statusPillHtml = renderStatusPill("partial", "Stopped · Partial");
    } else if (st === "failed") {
      statusPillHtml = renderStatusPill("failed", "Failed");
    }
  }

  const selectorValue = selectedEntry ? selectedEntry.entry_id : "";

  const modelSelectHtml = registryEntries.length
    ? `
      <select
        id="gen-model-select"
        class="select-control gen-model-select"
        aria-label="Local model target"
        ${isRunning ? "disabled" : ""}
      >
        ${registryEntries
          .map((entry) => {
            const capNote = `${formatNumber(entry.context_limit)} ctx`;
            const statusMark = entry.connection_status === "connected" ? "" : " · unverified";
            const label = `${entry.connection_name} · ${entry.model_id} (${capNote}${statusMark})`;
            return `<option value="${escapeHtml(entry.entry_id)}" ${entry.entry_id === selectorValue ? "selected" : ""}>${escapeHtml(label)}</option>`;
          })
          .join("")}
      </select>
    `
    : activeLlmFallback
      ? `<span class="gen-model-chip mono">${escapeHtml(currentModelLabel)}</span>`
      : `<button id="btn-gen-connect-llm" class="btn btn--quiet" type="button">Connect Local LLM…</button>`;

  const optionsPopoverHtml = `
    <div id="gen-options-anchor" class="popover-anchor">
      <button
        id="btn-gen-options"
        class="btn btn--quiet"
        type="button"
        aria-expanded="${String(Boolean(state.generationOptionsOpen))}"
        aria-controls="gen-options-popover"
        ${isRunning ? "disabled" : ""}
      >
        <span>Tune</span>
      </button>
      <div id="gen-options-popover" class="popover-panel" ${state.generationOptionsOpen ? "" : "hidden"}>
        <div class="form-grid-row">
          <label for="gen-temperature-input">Temperature</label>
          <input
            id="gen-temperature-input"
            class="number-control"
            type="number"
            min="0"
            max="2"
            step="0.1"
            value="${escapeHtml(String(prefs.genTemperature ?? 0.2))}"
          />
        </div>
        <div class="form-grid-row">
          <label for="gen-max-tokens-input">Max output tok</label>
          <input
            id="gen-max-tokens-input"
            class="number-control"
            type="number"
            min="16"
            max="65536"
            step="64"
            placeholder="Auto"
            value="${escapeHtml(String(prefs.genMaxOutputTokens ?? ""))}"
          />
        </div>
        <div class="form-grid-row">
          <label for="gen-stream-toggle">Live streaming</label>
          <input
            id="gen-stream-toggle"
            type="checkbox"
            ${prefs.genStream ? "checked" : ""}
          />
        </div>
      </div>
    </div>
  `;

  const outline = isRunning
    ? state.generationStreamOutline || []
    : activeResultMatchesPkg?.long_form?.outline || [];
  const currentSec = state.generationStreamCurrentSection;

  const progressBannerHtml = isRunning
    ? `
      <div class="gen-progress" role="status" aria-live="polite">
        <div class="gen-progress__bar" aria-hidden="true"></div>
        <div class="gen-progress__text">
          ${ICONS.running}
          <span>
            ${
              currentSec
                ? `Drafting section ${currentSec.order}/${outline.length || "?"}: ${escapeHtml(currentSec.title)}`
                : `Generating grounded ${escapeHtml(prefs.genMode === "long_form_report" ? "long-form report" : "answer")} with ${escapeHtml(currentModelLabel)}…`
            }
          </span>
        </div>
      </div>
    `
    : "";

  const errorBannerHtml = state.generationError
    ? `<div class="banner--error" role="alert">${escapeHtml(state.generationError)}</div>`
    : "";

  const partialBannerHtml =
    !isRunning && activeResultMatchesPkg?.partial
      ? `<div class="ctx-issue ctx-issue--warning" role="status">
          <strong>Partial Response</strong>
          <span>Generation was stopped before completion; output above reflects tokens produced prior to cancellation.</span>
        </div>`
      : !isRunning && activeResultMatchesPkg?.truncated
        ? `<div class="ctx-issue ctx-issue--warning" role="status">
            <strong>Output Budget Reached</strong>
            <span>Model stopped at its maximum output token limit (<code>finish_reason=length</code>). Increase Max output tokens or pick a concise format.</span>
          </div>`
        : "";

  const answerHtml = displayText
    ? `<div class="generation-answer">${renderSafeStructuredBody(displayText, activeFormat, activeSourceMap, pkg.sources)}</div>`
    : !isRunning && pkg.research_status === "completed"
      ? `<p class="coverage-summary-text" style="margin:0;">
          ${
            hasAnyModel
              ? "Click <strong>Synthesize Answer</strong> to compile retrieved context and stream a source-cited answer using your local model."
              : "Connect Ollama, LM Studio, llama.cpp, vLLM, or any OpenAI-compatible local server to generate grounded answers."
          }
        </p>`
      : "";

  const warningsHtml =
    warnings.length > 0
      ? `<div class="synthesis-card__warnings">
          ${warnings.slice(0, 4).map((w) => `<div>• ${escapeHtml(w)}</div>`).join("")}
        </div>`
      : "";

  const usageMetaHtml =
    !isRunning && activeResultMatchesPkg
      ? (() => {
          const u = activeResultMatchesPkg.usage;
          const tokStr =
            u && (u.prompt_tokens != null || u.completion_tokens != null)
              ? `${u.prompt_tokens ?? "?"} in / ${u.completion_tokens ?? "?"} out tok`
              : "Token usage not reported";
          const ctxStats = activeResultMatchesPkg.context_stats;
          const ctxStr = ctxStats
            ? `${formatNumber(ctxStats.allocated_evidence_tokens)} ev tok · ${ctxStats.evidence_items_included} passages`
            : "";
          return `
            <div class="synthesis-card__meta">
              <span class="mono">${escapeHtml(activeResultMatchesPkg.model_id)}</span>
              <span>· ${formatNumber(activeResultMatchesPkg.duration_ms || 0)} ms</span>
              <span>· ${escapeHtml(tokStr)}</span>
              ${ctxStr ? `<span>· ${escapeHtml(ctxStr)}</span>` : ""}
              ${
                activeResultMatchesPkg.citation_validation
                  ? `<button id="btn-open-gen-inspector" class="inline-action-link" type="button">Inspect Telemetry</button>`
                  : ""
              }
            </div>
          `;
        })()
      : "";

  const sourcesDrawerHtml =
    citedList.length > 0
      ? `
        <div class="gen-sources-block">
          <button
            id="btn-toggle-gen-sources"
            class="disclosure-btn"
            type="button"
            aria-expanded="${String(Boolean(state.generationExpandedSources))}"
            aria-controls="gen-sources-drawer"
          >
            ${ICONS.triangle}
            <span>${citedList.length} cited source${citedList.length === 1 ? "" : "s"} in answer</span>
          </button>
          <div id="gen-sources-drawer" class="gen-sources-drawer" ${state.generationExpandedSources ? "" : "hidden"}>
            ${citedList
              .map((cit) => {
                const safeHref = safeHttpUrl(cit.url);
                return `
                  <div class="gen-source-item">
                    <div class="gen-source-item__head">
                      <button
                        class="citation-link citation-chip"
                        type="button"
                        data-cite-source="${escapeHtml(cit.source_id)}"
                        data-cite-id="${escapeHtml(cit.citation_id)}"
                      >${escapeHtml(cit.citation_id)}</button>
                      <button
                        class="inline-action-link"
                        type="button"
                        data-cite-source="${escapeHtml(cit.source_id)}"
                      >${escapeHtml(truncateText(cit.title || cit.domain, 76))}</button>
                      <span class="doc-section__sub">${escapeHtml(cit.domain)} · ${cit.occurrence_count}×</span>
                      ${safeHref !== "#" ? `<a class="doc-section__sub" href="${escapeHtml(safeHref)}" target="_blank" rel="noopener noreferrer">Open ↗</a>` : ""}
                    </div>
                    ${cit.passage_excerpt ? `<div class="doc-section__sub">${escapeHtml(truncateText(cit.passage_excerpt, 180))}</div>` : ""}
                  </div>
                `;
              })
              .join("")}
          </div>
        </div>
      `
      : "";

  const footnoteHtml =
    displayText && !isRunning
      ? `<p class="gen-footnote">
          Generated locally from retrieved web evidence. Citation links verify source provenance and do not independently prove factual truth.
        </p>`
      : "";

  return `
    <section class="doc-section generation-card" aria-labelledby="heading-gen-answer">
      <div class="doc-section__heading">
        <div class="gen-head-title">
          <h2 id="heading-gen-answer" style="margin:0; font-size:inherit;">Research Answer</h2>
          ${statusPillHtml}
        </div>

        <div class="gen-head-actions">
          ${modelSelectHtml}
          ${
            isRunning
              ? `<button id="btn-stop-generation" class="btn btn--destructive-outline" type="button">Stop Generating</button>`
              : `<button
                  id="btn-start-generation"
                  class="btn btn--prominent"
                  type="button"
                  ${!hasAnyModel || pkg.research_status !== "completed" ? "disabled" : ""}
                >Synthesize Answer</button>`
          }
        </div>
      </div>

      <div class="generation-controls" role="group" aria-label="Answer generation controls">
        <label class="field-inline">
          <span>Template</span>
          <select id="gen-template-select" class="select-control" ${isRunning ? "disabled" : ""}>
            ${TEMPLATE_OPTIONS.map(
              (o) => `<option value="${o.id}" ${prefs.genTemplate === o.id ? "selected" : ""}>${o.label}</option>`
            ).join("")}
          </select>
        </label>

        <label class="field-inline">
          <span>Format</span>
          <select id="gen-format-select" class="select-control" ${isRunning ? "disabled" : ""}>
            ${FORMAT_OPTIONS.map(
              (o) => `<option value="${o.id}" ${prefs.genFormat === o.id ? "selected" : ""}>${o.label}</option>`
            ).join("")}
          </select>
        </label>

        <label class="field-inline">
          <span>Length</span>
          <select id="gen-length-select" class="select-control" ${isRunning ? "disabled" : ""}>
            ${LENGTH_OPTIONS.map(
              (o) => `<option value="${o.id}" ${prefs.genLength === o.id ? "selected" : ""}>${o.label}</option>`
            ).join("")}
          </select>
        </label>

        <label class="field-inline">
          <span>Mode</span>
          <select id="gen-mode-select" class="select-control" ${isRunning ? "disabled" : ""}>
            ${MODE_OPTIONS.map(
              (o) => `<option value="${o.id}" ${prefs.genMode === o.id ? "selected" : ""}>${o.label}</option>`
            ).join("")}
          </select>
        </label>

        ${optionsPopoverHtml}
      </div>

      ${errorBannerHtml}
      ${progressBannerHtml}
      ${partialBannerHtml}
      ${answerHtml}
      ${warningsHtml}
      ${usageMetaHtml}
      ${sourcesDrawerHtml}
      ${footnoteHtml}
    </section>
  `;
}
