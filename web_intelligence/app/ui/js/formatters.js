/**
 * Safe DOM escaping, URL sanitization, formatting helpers, and accessible icon+label badges.
 */

const HTML_ESCAPE_MAP = {
  "&": "&amp;",
  "<": "&lt;",
  ">": "&gt;",
  '"': "&quot;",
  "'": "&#39;",
};

export function escapeHtml(value) {
  if (value === null || value === undefined) return "";
  return String(value).replace(/[&<>"']/g, (ch) => HTML_ESCAPE_MAP[ch]);
}

export function safeHttpUrl(url) {
  if (!url || typeof url !== "string") return "#";
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "http:" || parsed.protocol === "https:") {
      return parsed.href;
    }
  } catch {
    return "#";
  }
  return "#";
}

export function formatNumber(num) {
  const n = Number(num);
  if (!Number.isFinite(n)) return "0";
  return n.toLocaleString("en-US");
}

export function formatScore(score) {
  const n = Number(score);
  if (!Number.isFinite(n)) return "—";
  return n.toFixed(2);
}

export function formatBytes(bytes) {
  const n = Number(bytes);
  if (!Number.isFinite(n) || n <= 0) return "0 B";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(2)} MB`;
}

export function formatRelativeTime(isoString) {
  if (!isoString) return "";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return String(isoString);
  const diffSec = Math.round((Date.now() - date.getTime()) / 1000);
  if (diffSec < 45) return "Just now";
  const diffMin = Math.round(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.round(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export function formatDateTime(isoString) {
  if (!isoString) return "Not observed";
  const date = new Date(isoString);
  if (Number.isNaN(date.getTime())) return String(isoString);
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function truncateText(text, maxLen = 42) {
  if (!text) return "";
  if (text.length <= maxLen) return text;
  return `${text.slice(0, maxLen - 1).trimEnd()}…`;
}

/** Consistent 14px SVG icons paired with text labels (never color alone). */
export const ICONS = {
  check: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.8"></circle><path d="m5.5 8.1 1.7 1.7 3.5-3.6"></path></svg>`,
  partial: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.8"></circle><line x1="8" y1="5" x2="8" y2="8.5"></line><circle cx="8" cy="11" r="0.6" fill="currentColor"></circle></svg>`,
  conflict: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.3 14 13H2L8 2.3Z"></path><line x1="8" y1="6.3" x2="8" y2="9.5"></line><circle cx="8" cy="11.4" r="0.55" fill="currentColor"></circle></svg>`,
  missing: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.8"></circle><line x1="5.3" y1="8" x2="10.7" y2="8"></line></svg>`,
  shield: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.2 13 4.2v3.6c0 3-2.1 5.2-5 6-2.9-.8-5-3-5-6V4.2L8 2.2Z"></path></svg>`,
  running: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.8" stroke-dasharray="22 14"></circle></svg>`,
  triangle: `<svg class="disclosure-triangle" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 8.5 6 4 9.5Z" fill="currentColor" stroke="none"></path></svg>`,
};

export function renderStatusPill(status, labelOverride = null) {
  const normalized = String(status || "").toLowerCase();
  const map = {
    corroborated: { cls: "status-pill--verified", icon: ICONS.check, label: "Corroborated" },
    sufficient: { cls: "status-pill--verified", icon: ICONS.check, label: "Sufficient Coverage" },
    covered: { cls: "status-pill--verified", icon: ICONS.check, label: "Covered" },
    completed: { cls: "status-pill--verified", icon: ICONS.check, label: "Completed" },
    success: { cls: "status-pill--verified", icon: ICONS.check, label: "Extracted" },

    single_source: { cls: "status-pill--partial", icon: ICONS.partial, label: "Single Source" },
    partial: { cls: "status-pill--partial", icon: ICONS.partial, label: "Partial" },
    unverified: { cls: "", icon: ICONS.missing, label: "Unverified" },
    missing: { cls: "", icon: ICONS.missing, label: "Missing" },
    insufficient: { cls: "status-pill--partial", icon: ICONS.partial, label: "Insufficient" },
    skipped: { cls: "", icon: ICONS.missing, label: "Skipped" },

    contested: { cls: "status-pill--conflict", icon: ICONS.conflict, label: "Contested" },
    failed: { cls: "status-pill--conflict", icon: ICONS.conflict, label: "Failed" },
    blocked_ssrf: { cls: "status-pill--conflict", icon: ICONS.shield, label: "Blocked (SSRF)" },
    blocked_robots: { cls: "status-pill--partial", icon: ICONS.shield, label: "Blocked (robots.txt)" },
    timeout: { cls: "status-pill--conflict", icon: ICONS.conflict, label: "Timed Out" },
    too_large: { cls: "status-pill--partial", icon: ICONS.partial, label: "Size Exceeded" },
    http_error: { cls: "status-pill--conflict", icon: ICONS.conflict, label: "HTTP Error" },

    pending: { cls: "status-pill--active", icon: ICONS.running, label: "Pending" },
    planning: { cls: "status-pill--active", icon: ICONS.running, label: "Planning" },
    searching: { cls: "status-pill--active", icon: ICONS.running, label: "Searching" },
    retrieving: { cls: "status-pill--active", icon: ICONS.running, label: "Retrieving" },
    processing: { cls: "status-pill--active", icon: ICONS.running, label: "Processing" },
  };
  const cfg = map[normalized] || { cls: "", icon: ICONS.missing, label: status || "Unknown" };
  return `<span class="status-pill ${cfg.cls}">${cfg.icon}<span>${escapeHtml(labelOverride || cfg.label)}</span></span>`;
}
