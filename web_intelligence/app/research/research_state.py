"""Mutable in-flight state tracker for research pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field

from web_intelligence.app.schemas.research import ResearchProgress, ResearchStatus


@dataclass
class ResearchState:
    """Tracks stage transitions and live execution metrics during a research run."""

    research_id: str
    original_query: str
    status: ResearchStatus = ResearchStatus.PENDING
    completed_stages: list[str] = field(default_factory=list)
    queries_planned: int = 0
    urls_found: int = 0
    documents_fetched: int = 0
    documents_processed: int = 0
    duplicates_removed: int = 0
    evidence_items_count: int = 0

    def advance(self, stage: str, status: ResearchStatus) -> None:
        """Record transition into a new pipeline stage."""
        if self.status != ResearchStatus.PENDING and stage not in self.completed_stages:
            prev = self.status.value
            if prev not in self.completed_stages and prev != stage:
                self.completed_stages.append(prev)
        self.status = status

    def mark_stage_done(self, stage_name: str) -> None:
        if stage_name not in self.completed_stages:
            self.completed_stages.append(stage_name)

    def to_progress(self) -> ResearchProgress:
        return ResearchProgress(
            current_stage=self.status.value,
            completed_stages=list(self.completed_stages),
            queries_planned=self.queries_planned,
            urls_found=self.urls_found,
            documents_fetched=self.documents_fetched,
            documents_processed=self.documents_processed,
            duplicates_removed=self.duplicates_removed,
            evidence_items_count=self.evidence_items_count,
        )
