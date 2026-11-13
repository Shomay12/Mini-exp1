"""Conversation history, session turn compaction, and user-controlled workspace notes."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from uuid import uuid4

from web_intelligence.app.context.models import (
    CompactedHistorySummary,
    ConversationTurn,
    HistoryPackage,
    WorkspaceNote,
)
from web_intelligence.app.context.token_estimator import (
    BaseContextTokenEstimator,
    ConservativeHeuristicEstimator,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_DECISION_RE = re.compile(
    r"\b(?:decided|decision|agreed|confirmed|concluded|selected|chose|chosen|resolved to|will use)\b",
    re.IGNORECASE,
)
_OPEN_Q_RE = re.compile(
    r"(?:\?\s*$|\b(?:unresolved|open question|unclear|todo|pending|need to verify|remains unknown)\b)",
    re.IGNORECASE,
)


class ConversationHistoryManager:
    """Manages current-turn, session-level, and workspace-note history within token limits."""

    def __init__(self, estimator: BaseContextTokenEstimator | None = None) -> None:
        self._estimator = estimator or ConservativeHeuristicEstimator()

    def build_history_package(
        self,
        *,
        turns: list[ConversationTurn] | None = None,
        workspace_notes: list[WorkspaceNote] | None = None,
        history_budget_tokens: int = 1024,
        max_history_turns: int = 8,
        estimator: BaseContextTokenEstimator | None = None,
    ) -> HistoryPackage:
        """Assemble current-turn, recent session turns, compacted older turns, and enabled notes."""
        counter = estimator or self._estimator
        all_turns = list(turns or [])
        active_notes = [note for note in (workspace_notes or []) if note.enabled]

        # Re-estimate token counts under the active model's estimator
        scored_notes: list[WorkspaceNote] = []
        notes_tokens = 0
        for note in active_notes:
            ntok = counter.estimate(f"{note.title}: {note.content}").tokens + 6
            scored_notes.append(note.model_copy(update={"token_count": ntok}))
            notes_tokens += ntok

        current_turn_items: list[ConversationTurn] = []
        session_turns: list[ConversationTurn] = []
        for turn in all_turns:
            ttok = counter.estimate(f"{turn.role}: {turn.content}").tokens + 6
            updated = turn.model_copy(update={"token_count": ttok})
            if turn.scope == "current_turn":
                current_turn_items.append(updated)
            else:
                session_turns.append(updated)

        warnings: list[str] = []
        current_tokens = sum(t.token_count for t in current_turn_items)
        fixed_tokens = notes_tokens + current_tokens
        remaining_for_session = max(0, history_budget_tokens - fixed_tokens)

        if fixed_tokens > history_budget_tokens:
            warnings.append(
                f"Enabled workspace notes + current turn ({fixed_tokens} tok) exceed "
                f"history budget ({history_budget_tokens} tok)."
            )

        # Keep most recent session turns within both max_history_turns and token budget.
        kept_recent: list[ConversationTurn] = []
        older_to_compact: list[ConversationTurn] = []
        session_used = 0

        # Reserve modest room for compacted summary if there are many older turns.
        total_session_tokens = sum(t.token_count for t in session_turns)
        needs_compaction = (
            len(session_turns) > max_history_turns
            or total_session_tokens > remaining_for_session
        )

        if not needs_compaction:
            kept_recent = session_turns
            session_used = total_session_tokens
            compacted_summary = None
            total_tokens = fixed_tokens + session_used
            return HistoryPackage(
                current_turn_items=current_turn_items,
                session_turns=kept_recent,
                compacted_summary=compacted_summary,
                workspace_notes=scored_notes,
                total_tokens=total_tokens,
                compacted=False,
                warnings=warnings,
            )

        # Reserve space for compaction summary when older turns exist
        summary_reserve = min(remaining_for_session // 3, 220) if remaining_for_session >= 96 else 0
        recent_cap = max(0, remaining_for_session - summary_reserve)

        rev_kept: list[ConversationTurn] = []
        for idx in range(len(session_turns) - 1, -1, -1):
            turn = session_turns[idx]
            if (
                len(rev_kept) < max_history_turns
                and session_used + turn.token_count <= recent_cap
            ):
                rev_kept.append(turn)
                session_used += turn.token_count
            else:
                older_to_compact = session_turns[: idx + 1]
                break

        kept_recent = list(reversed(rev_kept))
        compacted_summary: CompactedHistorySummary | None = None
        summary_tokens = 0
        if older_to_compact:
            available_for_summary = max(48, remaining_for_session - session_used)
            compacted_summary = self._compact_turns(
                older_to_compact,
                max_tokens=available_for_summary,
                counter=counter,
            )
            summary_tokens = compacted_summary.token_count
            if fixed_tokens + session_used + summary_tokens > history_budget_tokens:
                warnings.append(
                    "Compacted history summary exceeds history allocation; expand history budget or disable unused notes."
                )

        return HistoryPackage(
            current_turn_items=current_turn_items,
            session_turns=kept_recent,
            compacted_summary=compacted_summary,
            workspace_notes=scored_notes,
            total_tokens=fixed_tokens + session_used + summary_tokens,
            compacted=bool(older_to_compact),
            warnings=warnings,
        )

    def _compact_turns(
        self,
        turns: list[ConversationTurn],
        *,
        max_tokens: int,
        counter: BaseContextTokenEstimator,
    ) -> CompactedHistorySummary:
        """Deterministically summarize older turns while preserving decisions and open questions."""
        decisions: list[str] = []
        unresolved: list[str] = []
        seen_d: set[str] = set()
        seen_q: set[str] = set()

        for turn in turns:
            for d in turn.decisions:
                clean = d.strip()
                if clean and clean.lower() not in seen_d:
                    seen_d.add(clean.lower())
                    decisions.append(clean)
            for q in turn.unresolved_questions:
                clean = q.strip()
                if clean and clean.lower() not in seen_q:
                    seen_q.add(clean.lower())
                    unresolved.append(clean)

            for sentence in _SENTENCE_SPLIT_RE.split(turn.content):
                s = sentence.strip()
                if not s:
                    continue
                if _DECISION_RE.search(s) and s.lower() not in seen_d:
                    seen_d.add(s.lower())
                    decisions.append(s)
                if _OPEN_Q_RE.search(s) and s.lower() not in seen_q:
                    seen_q.add(s.lower())
                    unresolved.append(s)

        lines: list[str] = [f"Compacted {len(turns)} earlier session turn(s)."]
        if decisions:
            lines.append("Decisions:")
            for d in decisions:
                lines.append(f"- {d}")
        if unresolved:
            lines.append("Unresolved Questions:")
            for q in unresolved:
                lines.append(f"- {q}")
        if not decisions and not unresolved:
            topics = [
                f"{t.role}: {t.content[:90].strip()}..."
                if len(t.content) > 90
                else f"{t.role}: {t.content.strip()}"
                for t in turns[-2:]
            ]
            lines.append("Earlier turns: " + " | ".join(topics))

        summary_text = "\n".join(lines)
        est_tokens = counter.estimate(summary_text).tokens
        if est_tokens > max_tokens and decisions + unresolved:
            # Trim non-essential detail lines while still keeping all recorded decisions/open questions concise
            compact_lines = [f"Compacted {len(turns)} earlier turn(s)."]
            if decisions:
                compact_lines.append("Decisions: " + "; ".join(d[:100] for d in decisions))
            if unresolved:
                compact_lines.append("Open questions: " + "; ".join(q[:100] for q in unresolved))
            summary_text = "\n".join(compact_lines)
            est_tokens = counter.estimate(summary_text).tokens

        return CompactedHistorySummary(
            summary_text=summary_text,
            compacted_turn_ids=[t.turn_id for t in turns],
            preserved_decisions=decisions,
            preserved_unresolved_questions=unresolved,
            token_count=est_tokens,
        )

    @staticmethod
    def new_turn(
        *,
        role: str,
        content: str,
        research_id: str = "global",
        scope: str = "session",
        decisions: list[str] | None = None,
        unresolved_questions: list[str] | None = None,
    ) -> ConversationTurn:
        """Factory helper creating a timestamped ConversationTurn."""
        return ConversationTurn(
            turn_id=f"turn-{uuid4().hex[:12]}",
            research_id=research_id,
            scope=scope,
            role=role,
            content=content,
            created_at=datetime.now(timezone.utc),
            decisions=list(decisions or []),
            unresolved_questions=list(unresolved_questions or []),
        )

    @staticmethod
    def new_note(
        *,
        title: str,
        content: str,
        workspace_id: str = "default",
        enabled: bool = True,
    ) -> WorkspaceNote:
        """Factory helper creating a user-controlled WorkspaceNote."""
        now = datetime.now(timezone.utc)
        return WorkspaceNote(
            note_id=f"note-{uuid4().hex[:12]}",
            workspace_id=workspace_id,
            title=title.strip(),
            content=content.strip(),
            enabled=enabled,
            user_controlled=True,
            created_at=now,
            updated_at=now,
        )
