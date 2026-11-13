"""Chronological operational trace recorder for auditable Stage 7 decisions."""

from __future__ import annotations

import uuid
from typing import Any

from web_intelligence.app.agent.models import ResearchTraceEntry, TraceEventType
from web_intelligence.app.llm.security import redact_llm_error

_SENSITIVE_KEYS = frozenset({"api_key", "authorization", "token", "secret", "password"})


def _redact_mapping(payload: dict[str, Any] | None) -> dict[str, Any]:
    """Strip sensitive keys and redact secrets from trace metadata."""
    if not payload:
        return {}
    cleaned: dict[str, Any] = {}
    for key, val in payload.items():
        if str(key).lower() in _SENSITIVE_KEYS:
            cleaned[key] = "[REDACTED]"
        elif isinstance(val, dict):
            cleaned[key] = _redact_mapping(val)
        elif isinstance(val, str):
            cleaned[key] = redact_llm_error(val) if "bearer " in val.lower() else val[:600]
        else:
            cleaned[key] = val
    return cleaned


class ResearchTraceRecorder:
    """Builds typed operational trace entries explaining agent decisions without private CoT."""

    def __init__(self, research_id: str, *, initial_sequence: int = 0) -> None:
        self.research_id = research_id
        self._seq = initial_sequence
        self.entries: list[ResearchTraceEntry] = []

    @property
    def sequence(self) -> int:
        return self._seq

    def record(
        self,
        *,
        iteration: int,
        phase: str,
        event_type: TraceEventType,
        reason: str,
        task_id: str | None = None,
        tool_name: str | None = None,
        sanitized_inputs: dict[str, Any] | None = None,
        outcome: str = "",
        details: dict[str, Any] | None = None,
    ) -> ResearchTraceEntry:
        """Append and return one chronological operational trace entry."""
        self._seq += 1
        entry = ResearchTraceEntry(
            trace_id=f"tr-{uuid.uuid4().hex[:12]}",
            research_id=self.research_id,
            sequence=self._seq,
            iteration=max(0, iteration),
            phase=phase,
            event_type=event_type,
            task_id=task_id,
            tool_name=tool_name,
            sanitized_inputs=_redact_mapping(sanitized_inputs),
            outcome=outcome[:500],
            reason=reason.strip() or event_type.value,
            details=_redact_mapping(details),
        )
        self.entries.append(entry)
        return entry
