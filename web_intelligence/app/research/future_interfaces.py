"""Future Phase 2 interfaces for local small language model integration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, Field

from web_intelligence.app.schemas.evidence import ClaimConflict, EvidenceItem, ExtractedClaim
from web_intelligence.app.schemas.research import QueryPlan, ResearchPackage


class ModelCapabilities(BaseModel):
    """Hardware and context characteristics of a local small language model."""

    model_name: str
    backend: str = "local"
    max_context_tokens: int = Field(default=8192, ge=512)
    recommended_output_tokens: int = Field(default=1024, ge=128)
    supports_json_mode: bool = False
    supports_tool_calling: bool = False


class TokenEstimator(Protocol):
    """Protocol for estimating token counts (swappable for model-specific tokenizers)."""

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for the given text."""
        ...


class HeuristicTokenEstimator:
    """Model-independent deterministic token estimator (~1.33 tokens/word + punct)."""

    def estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        words = text.split()
        char_estimate = max(1, len(text) // 4)
        word_estimate = max(1, round(len(words) * 1.33))
        return max(word_estimate, char_estimate)


class ContextBudgetManager(ABC):
    """Interface for dynamically allocating limited local-model context budgets."""

    @abstractmethod
    def allocate(
        self,
        package: ResearchPackage,
        capabilities: ModelCapabilities,
    ) -> list[EvidenceItem]:
        """Select and fit evidence items within the model's context budget."""


class LocalModelAdapter(ABC):
    """Backend-agnostic inference adapter for local small language models."""

    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        """Return capabilities and context window limits for the active local model."""

    @abstractmethod
    async def generate(self, prompt: str, *, max_tokens: int | None = None) -> str:
        """Run local inference and return generated text."""


class ResearchPlanner(ABC):
    """Interface for adaptive multi-turn research planning driven by a local model."""

    @abstractmethod
    async def plan_next_steps(
        self,
        question: str,
        current_package: ResearchPackage | None = None,
    ) -> QueryPlan:
        """Determine initial or follow-up search queries based on coverage gaps."""


class AnswerGenerator(ABC):
    """Interface for generating citation-grounded answers from a ResearchPackage."""

    @abstractmethod
    async def generate_answer(
        self,
        package: ResearchPackage,
        selected_evidence: list[EvidenceItem],
    ) -> str:
        """Produce a grounded synthesis referencing retrieved evidence IDs and URLs."""


class EvidenceVerifier(ABC):
    """Interface for verifying generated or extracted claims against source evidence."""

    @abstractmethod
    async def verify(
        self,
        claims: list[ExtractedClaim],
        evidence: list[EvidenceItem],
    ) -> list[ClaimConflict]:
        """Verify claims against retrieved evidence and report unsupported or conflicting items."""
