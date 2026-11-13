/**
 * Stage 5 Context Inspector — displays active model limits, adaptive allocation,
 * compression status, selected/excluded passages, workspace notes, validation,
 * and pre-submission formatted context preview.
 */

import { escapeHtml, formatNumber, renderStatusPill, truncateText } from "../formatters.js";

const PROFILE_OPTIONS = [
  { id: "quick_answer", label: "Quick Answer" },
  { id: "standard_research", label: "Standard Research" },
  { id: "deep_research", label: "Deep Research" },
  { id: "long_form", label: "Long-Form Generation" },
  { id: "custom", label: "Custom" },
];

const STRATEGY_OPTIONS = [
  { id: "extractive", label: "Extractive" },
  { id: "deduplication", label: "Deduplication" },
  { id: "structured", label: "Structured" },
  { id: "model_assisted", label: "Model-Assisted" },
  { id: "none", label: "None (Full)" },
];

export function renderContextInspectorTab(state, pkg) {
  const preview = state.contextPreview;
  const ctxPkg = preview?.context_package || null;
  const validation = preview?.validation || null;
  const adapterMessages = preview?.adapter_messages || [];
  const budgetResp = state.contextBudget;
  const caps = ctxPkg?.capabilities || budgetResp?.capabilities || state.contextCapabilities || null;
  const config = budgetResp?.config || null;
  const alloc = ctxPkg?.budget_allocation || budgetResp?.effective_allocation || null;

  const modelLabel = caps?.model_name
    ? `${caps.model_name} (${caps.provider})`
    : caps?.has_active_model
      ? caps.provider
      : "Unconfigured (fallback limits)";
  const limitSource = caps?.context_limit_source || "unknown";
  const maxCtx = caps?.max_context_tokens ?? 8192;
  const effCtx = alloc?.effective_context_limit ?? caps?.effective_context_limit ?? maxCtx;
  const approxPrefix = caps?.estimation_is_approximate !== false ? "~" : "";

  const activeProfile = alloc?.profile || config?.profile || "standard_research";
  const activeStrategy =
    ctxPkg?.compression?.strategy_requested ||
    config?.compression_strategy ||
    "extractive";
  const userCeilingVal = config?.user_context_limit ?? "";
  const outOverrideVal = config?.output_reservation_tokens ?? "";

  const currentInput = ctxPkg?.total_input_tokens ?? 0;
  const reservedOut =
    ctxPkg?.output_reservation_tokens ?? alloc?.output_reservation_tokens ?? 1024;
  const totalWithOut = ctxPkg ? ctxPkg.total_with_output_tokens : reservedOut;
  const usagePct = effCtx > 0 ? Math.min(100, Math.round((totalWithOut / effCtx) * 100)) : 0;

  const evAlloc = alloc?.research_evidence_tokens ?? 0;
  const evUsed = findSectionUsed(ctxPkg, "selected_evidence");
  const histAlloc = alloc?.conversation_history_tokens ?? 0;
  const histUsed = ctxPkg?.history?.total_tokens ?? findSectionUsed(ctxPkg, "conversation_history");

  const comp = state.contextCompressionPreview?.compression || ctxPkg?.compression || null;
  const selected = ctxPkg?.selected_evidence || [];
  const excluded = ctxPkg?.excluded_evidence || [];
  const notes = state.contextNotes || [];

  const formattedPreviewText = adapterMessages.length
    ? adapterMessages
        .map((m) => `[${String(m.role || "user").toUpperCase()}]\n${m.content || ""}`)
        .join("\n\n---\n\n")
    : ctxPkg?.rendered_user_prompt || "";

  return `
    ${state.contextError ? `<div class="ctx-issue ctx-issue--error">${escapeHtml(state.contextError)}</div>` : ""}

    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Model &amp; Context Window</h3>
        <span class="cap-badge cap-badge--${escapeHtml(limitSource)}">${escapeHtml(limitSource)}</span>
      </div>
      <dl class="kv-table">
        <dt class="kv-table__key">Active Model</dt>
        <dd class="kv-table__val" style="margin:0;">${escapeHtml(modelLabel)}</dd>
        <dt class="kv-table__key">Max Supported</dt>
        <dd class="kv-table__val mono" style="margin:0;">${formatNumber(maxCtx)} tok (eff. ${formatNumber(effCtx)})</dd>
        <dt class="kv-table__key">Token Counter</dt>
        <dd class="kv-table__val" style="margin:0;">${
          caps?.tokenizer_available
            ? `Exact (${escapeHtml(caps.tokenizer_id || "adapter")})`
            : "Approximate (conservative heuristic)"
        }</dd>
        <dt class="kv-table__key">Structured JSON</dt>
        <dd class="kv-table__val" style="margin:0;">${caps?.supports_structured_output ? "Supported" : "Standard text"}</dd>
      </dl>
    </div>

    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Token Budget &amp; Usage</h3>
        <span class="doc-section__sub mono">${approxPrefix}${formatNumber(totalWithOut)} / ${formatNumber(effCtx)} tok (${usagePct}%)</span>
      </div>
      <div class="ctx-meter" role="progressbar" aria-valuenow="${usagePct}" aria-valuemin="0" aria-valuemax="100" aria-label="Context budget usage">
        <div class="ctx-meter__fill ${usagePct > 92 ? "is-high" : ""}" style="width:${usagePct}%"></div>
      </div>
      <dl class="kv-table">
        <dt class="kv-table__key">Current Input</dt>
        <dd class="kv-table__val mono" style="margin:0;">${approxPrefix}${formatNumber(currentInput)} / ${formatNumber(alloc?.input_budget_tokens ?? Math.max(0, effCtx - reservedOut))} tok</dd>
        <dt class="kv-table__key">Output Reserved</dt>
        <dd class="kv-table__val mono" style="margin:0;">${formatNumber(reservedOut)} tok</dd>
        <dt class="kv-table__key">Evidence Alloc</dt>
        <dd class="kv-table__val mono" style="margin:0;">${approxPrefix}${formatNumber(evUsed)} / ${formatNumber(evAlloc)} tok</dd>
        <dt class="kv-table__key">History Alloc</dt>
        <dd class="kv-table__val mono" style="margin:0;">${approxPrefix}${formatNumber(histUsed)} / ${formatNumber(histAlloc)} tok${ctxPkg?.history?.compacted ? " · compacted" : ""}</dd>
        <dt class="kv-table__key">Meta / Safety</dt>
        <dd class="kv-table__val mono" style="margin:0;">${formatNumber(alloc?.structured_metadata_tokens ?? 0)} meta · ${formatNumber(alloc?.safety_margin_tokens ?? 0)} safety</dd>
      </dl>
    </div>

    <div class="settings-group">
      <h3 class="settings-group__title">Allocation Controls</h3>
      <form id="ctx-budget-form" class="ctx-controls-box">
        <div class="ctx-control-row">
          <label for="ctx-profile-select">Profile</label>
          <select id="ctx-profile-select" name="profile" class="select-control">
            ${PROFILE_OPTIONS.map(
              (opt) =>
                `<option value="${opt.id}" ${opt.id === activeProfile ? "selected" : ""}>${escapeHtml(opt.label)}</option>`
            ).join("")}
          </select>
        </div>
        <div class="ctx-control-row">
          <label for="ctx-strategy-select">Compression</label>
          <select id="ctx-strategy-select" name="compression_strategy" class="select-control">
            ${STRATEGY_OPTIONS.map(
              (opt) =>
                `<option value="${opt.id}" ${opt.id === activeStrategy ? "selected" : ""}>${escapeHtml(opt.label)}</option>`
            ).join("")}
          </select>
        </div>
        <div class="ctx-control-row">
          <label for="ctx-user-limit-input">Budget Ceiling</label>
          <input id="ctx-user-limit-input" name="user_context_limit" class="number-control" type="number" min="512" max="${maxCtx}" step="256" placeholder="${maxCtx}" value="${escapeHtml(String(userCeilingVal))}" />
        </div>
        <div class="ctx-control-row">
          <label for="ctx-output-res-input">Output Reserve</label>
          <input id="ctx-output-res-input" name="output_reservation_tokens" class="number-control" type="number" min="64" max="${maxCtx}" step="64" placeholder="${reservedOut}" value="${escapeHtml(String(outOverrideVal))}" />
        </div>
        <div class="ctx-actions-row">
          <button id="btn-ctx-apply-budget" class="btn btn--prominent" type="submit" ${state.contextLoading ? "disabled" : ""}>Apply Budget</button>
          <button id="btn-ctx-preview" class="btn" type="button" data-ctx-action="preview" ${state.contextLoading || (!pkg && !state.queryDraft) ? "disabled" : ""}>Preview</button>
          <button id="btn-ctx-compress" class="btn" type="button" data-ctx-action="compress" ${state.contextLoading || !pkg ? "disabled" : ""}>Compress</button>
        </div>
      </form>
    </div>

    ${renderValidationSection(validation)}
    ${renderCompressionSection(comp)}

    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Selected Evidence (${selected.length})</h3>
      </div>
      ${
        selected.length
          ? `<div class="inspector-list">
              ${selected
                .map(
                  (item) => `
                <div class="inspector-row" style="cursor:default;">
                  <div class="inspector-head__top">
                    <strong>${escapeHtml(item.citation_label)} · ${escapeHtml(item.source_domain)}</strong>
                    <span class="doc-section__sub mono">${item.allocated_token_count} tok</span>
                  </div>
                  <div class="inspector-row__meta">
                    ${item.is_conflict_evidence ? `<span class="pill pill--contested">conflict</span>` : ""}
                    <span>${escapeHtml(item.selection_reason)}</span>
                  </div>
                  <div class="doc-section__sub">${escapeHtml(truncateText(item.passage_text, 160))}</div>
                </div>`
                )
                .join("")}
            </div>`
          : `<div class="empty-state" style="padding:12px 0;">${
              pkg
                ? "Click Preview to evaluate passage selection for this run."
                : "Run or select research to inspect selected evidence passages."
            }</div>`
      }
    </div>

    ${
      excluded.length
        ? `<div class="settings-group">
            <h3 class="settings-group__title">Excluded Passages (${excluded.length})</h3>
            <div class="inspector-list">
              ${excluded
                .map(
                  (item) => `
                <div class="inspector-row" style="cursor:default;">
                  <div class="inspector-head__top">
                    <span>${escapeHtml(item.source_domain)}</span>
                    <span class="doc-section__sub mono">${item.original_token_count} tok</span>
                  </div>
                  <div class="inspector-row__meta mono">${escapeHtml(item.exclusion_reason)}</div>
                  <div class="doc-section__sub">${escapeHtml(truncateText(item.passage_preview, 110))}</div>
                </div>`
                )
                .join("")}
            </div>
          </div>`
        : ""
    }

    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Workspace Notes (${notes.length})</h3>
      </div>
      <div class="inspector-list">
        ${notes
          .map(
            (n) => `
          <div class="inspector-row" style="cursor:default;">
            <div class="inspector-head__top">
              <label style="display:flex;align-items:center;gap:6px;cursor:pointer;">
                <input type="checkbox" data-ctx-toggle-note="${escapeHtml(n.note_id)}" ${n.enabled ? "checked" : ""} />
                <strong>${escapeHtml(n.title)}</strong>
              </label>
              <button class="btn btn--quiet" type="button" data-ctx-delete-note="${escapeHtml(n.note_id)}" aria-label="Delete note">Remove</button>
            </div>
            <div class="doc-section__sub">${escapeHtml(truncateText(n.content, 120))}</div>
          </div>`
          )
          .join("")}
      </div>
      <form id="ctx-note-form" class="ctx-controls-box" style="margin-top:6px;">
        <input id="ctx-note-title" class="text-control" type="text" placeholder="Note title…" maxlength="200" />
        <input id="ctx-note-content" class="text-control" type="text" placeholder="Persistent workspace guidance…" maxlength="2000" />
        <button class="btn" type="submit">Add Note</button>
      </form>
    </div>

    ${
      formattedPreviewText
        ? `<div class="settings-group">
            <div class="inspector-head__top">
              <h3 class="settings-group__title">Model Input Preview</h3>
              <button id="btn-ctx-toggle-preview" class="btn btn--quiet" type="button" aria-expanded="${String(Boolean(state.contextShowFormattedPreview))}">
                ${state.contextShowFormattedPreview ? "Hide" : "Show"}
              </button>
            </div>
            ${
              state.contextShowFormattedPreview
                ? `<pre class="ctx-preview-pre mono">${escapeHtml(formattedPreviewText)}</pre>`
                : ""
            }
          </div>`
        : ""
    }
  `;
}

function findSectionUsed(ctxPkg, sectionName) {
  const row = (ctxPkg?.section_token_usage || []).find((s) => s.section === sectionName);
  return row ? row.estimated_tokens : 0;
}

function renderValidationSection(validation) {
  if (!validation) return "";
  const statusPill = renderStatusPill(
    validation.is_valid ? "completed" : "failed",
    validation.is_valid ? "Valid" : "Blocked"
  );
  const issuesHtml = (validation.issues || [])
    .map(
      (issue) => `
    <div class="ctx-issue ctx-issue--${issue.severity === "error" ? "error" : "warning"}">
      <div><strong>${escapeHtml(issue.code)}</strong>: ${escapeHtml(issue.message)}</div>
      <div class="doc-section__sub">${escapeHtml(issue.remediation)}</div>
    </div>`
    )
    .join("");

  return `
    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Pre-Inference Validation</h3>
        ${statusPill}
      </div>
      <div class="doc-section__sub mono">
        ${formatNumber(validation.total_required_tokens)} / ${formatNumber(validation.active_context_limit)} total · ${formatNumber(validation.remaining_tokens)} remaining
      </div>
      ${issuesHtml}
    </div>
  `;
}

function renderCompressionSection(comp) {
  if (!comp) return "";
  return `
    <div class="settings-group">
      <div class="inspector-head__top">
        <h3 class="settings-group__title">Compression</h3>
        <span class="doc-section__sub mono">${escapeHtml(comp.strategy_applied)} (${Math.round((comp.compression_ratio || 1) * 100)}%)</span>
      </div>
      <dl class="kv-table">
        <dt class="kv-table__key">Tokens</dt>
        <dd class="kv-table__val mono" style="margin:0;">${formatNumber(comp.original_total_tokens)} → ${formatNumber(comp.compressed_total_tokens)}</dd>
        <dt class="kv-table__key">Deduplicated</dt>
        <dd class="kv-table__val mono" style="margin:0;">${comp.deduplicated_count} merged · ${comp.independent_sources_preserved} domains</dd>
        <dt class="kv-table__key">Provenance</dt>
        <dd class="kv-table__val" style="margin:0;">${comp.provenance_preserved ? "All citations verified" : "Verification warning"}</dd>
      </dl>
      ${
        comp.model_assisted_fallback_reason
          ? `<div class="ctx-issue ctx-issue--warning">${escapeHtml(comp.model_assisted_fallback_reason)}</div>`
          : ""
      }
    </div>
  `;
}
