"""Scoped research working memory and privacy guard for outbound search queries."""

from __future__ import annotations

import re

from web_intelligence.app.agent.models import ResearchSession
from web_intelligence.app.privacy.sanitizer import QuerySanitizer

_NOTE_MARKER_RE = re.compile(
    r"(?:\[workspace_note[^\]]*\]|private_note:|internal_context:|secret:|api_key=)",
    re.IGNORECASE,
)


class ResearchWorkingMemory:
    """Maintains cross-iteration session state while preventing private leaks into external search."""

    def __init__(self, sanitizer: QuerySanitizer | None = None) -> None:
        self._sanitizer = sanitizer or QuerySanitizer()

    @staticmethod
    def known_canonical_urls(session: ResearchSession) -> set[str]:
        urls: set[str] = set()
        for src in session.collected_sources:
            if src.canonical_url:
                urls.add(src.canonical_url)
            if src.original_url:
                urls.add(src.original_url)
            if src.final_url:
                urls.add(src.final_url)
        return urls

    @staticmethod
    def executed_search_queries(session: ResearchSession) -> set[str]:
        return {
            dq.sanitized_query.strip().lower()
            for dq in session.disclosed_queries
            if dq.sanitized_query and dq.sanitized_query.strip()
        }

    def sanitize_external_query(
        self,
        raw_query: str,
        *,
        blocked_private_terms: list[str] | None = None,
    ) -> tuple[str, list[str]]:
        """Sanitize an agent-proposed search query before any external provider sees it."""
        cleaned = _NOTE_MARKER_RE.sub("", raw_query).strip()
        redactions: list[str] = []

        for secret_phrase in blocked_private_terms or []:
            phrase = secret_phrase.strip()
            if len(phrase) >= 6 and phrase.lower() in cleaned.lower():
                cleaned = re.sub(re.escape(phrase), "", cleaned, flags=re.IGNORECASE).strip()
                redactions.append("private_workspace_context")

        san_result = self._sanitizer.sanitize(cleaned)
        sanitized = " ".join(san_result.sanitized_query.split())
        all_redactions = [*redactions, *list(san_result.redactions_applied)]
        return sanitized, list(dict.fromkeys(all_redactions))

    @staticmethod
    def iteration_brief(session: ResearchSession) -> str:
        """Build a concise factual snapshot of current session progress for planning follow-ups."""
        domains = sorted({s.domain for s in session.collected_sources if not s.is_duplicate})
        completed = [t.question for t in session.completed_tasks]
        unresolved = [g.description for g in session.identified_gaps if not g.resolved]
        parts = [
            f"Objective: {session.research_objective}",
            f"Iterations completed: {session.current_iteration}",
            f"Unique sources: {len(domains)} domains ({', '.join(domains[:6]) or 'none'})",
            f"Accepted evidence items: {len(session.evidence_records)}; conflicts: {len(session.conflicts)}",
            f"Completed tasks ({len(completed)}): {'; '.join(completed[:5]) or 'none'}",
            f"Open gaps ({len(unresolved)}): {'; '.join(unresolved[:5]) or 'none'}",
        ]
        return "\n".join(parts)
