"""
AION Evaluation Interface
=========================
Defines the abstract contract for RAG evaluation providers.
Observes accepted generation candidates without deciding acceptance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class IEvaluationProvider(ABC):
    """
    Abstract contract for question evaluation providers.
    Evaluation providers observe final accepted candidates and compute
    standardized quality, grounding, and operational performance metrics.
    """

    @abstractmethod
    def evaluate_question(
        self,
        question_text: str,
        slot: Any,
        evidence: Any,
        latency_ms: float = 0.0,
        model_name: str = "",
        provider_name: str = "",
        attempt_index: int = 1,
        trace_id: str = "",
        generation_id: str = "",
    ) -> Any:
        """
        Evaluate a single accepted question against its slot specification
        and evidence context.
        """
        pass
