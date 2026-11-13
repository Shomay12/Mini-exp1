/**
 * DOM escaping, URL safety, numeric/date formatters, and quiet status indicators.
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
  if (!isoString) return "Not reported";
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

export const ICONS = {
  check: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5"></circle><path d="m5.5 8.1 1.7 1.7 3.5-3.6"></path></svg>`,
  partial: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5"></circle><line x1="8" y1="5" x2="8" y2="8.5"></line><circle cx="8" cy="11" r="0.6" fill="currentColor"></circle></svg>`,
  conflict: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 2.5 13.8 13H2.2L8 2.5Z"></path><line x1="8" y1="6.3" x2="8" y2="9.4"></line><circle cx="8" cy="11.3" r="0.55" fill="currentColor"></circle></svg>`,
  missing: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5"></circle><line x1="5.4" y1="8" x2="10.6" y2="8"></line></svg>`,
  running: `<svg class="icon icon--sm" viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.5" stroke-dasharray="20 12"></circle></svg>`,
  triangle: `<svg class="disclosure-triangle" viewBox="0 0 12 12" aria-hidden="true"><path d="M4 2.5 8.5 6 4 9.5Z" fill="currentColor" stroke="none"></path></svg>`,
};

export function renderStatusPill(status, labelOverride = null) {
  const normalized = String(status || "").toLowerCase();
  const map = {
    corroborated: { cls: "status-label--positive", label: "Corroborated" },
    sufficient: { cls: "status-label--positive", label: "Covered" },
    covered: { cls: "status-label--positive", label: "Covered" },
    completed: { cls: "status-label--positive", label: "Completed" },
    success: { cls: "status-label--positive", label: "Extracted" },

    single_source: { cls: "", label: "Single source" },
    partial: { cls: "status-label--warning", label: "Partial" },
    unverified: { cls: "", label: "Unverified" },
    missing: { cls: "", label: "Missing" },
    insufficient: { cls: "status-label--warning", label: "Insufficient" },
    skipped: { cls: "", label: "Skipped" },

    contested: { cls: "status-label--negative", label: "Contested" },
    failed: { cls: "status-label--negative", label: "Failed" },
    blocked_ssrf: { cls: "status-label--negative", label: "Blocked (SSRF)" },
    blocked_robots: { cls: "status-label--warning", label: "Blocked (robots.txt)" },
    timeout: { cls: "status-label--negative", label: "Timed out" },
    too_large: { cls: "status-label--warning", label: "Too large" },
    http_error: { cls: "status-label--negative", label: "HTTP error" },

    pending: { cls: "status-label--active", label: "Pending" },
    planning: { cls: "status-label--active", label: "Planning" },
    searching: { cls: "status-label--active", label: "Searching" },
    retrieving: { cls: "status-label--active", label: "Retrieving" },
    processing: { cls: "status-label--active", label: "Processing" },
  };
  const cfg = map[normalized] || { cls: "", label: status || "Unknown" };
  return `<span class="status-label ${cfg.cls}"><span class="status-dot" aria-hidden="true"></span><span>${escapeHtml(labelOverride || cfg.label)}</span></span>`;
}
