"""
AION Core Extraction — Consensus & Fusion Engine
=================================================
Merges extraction results from multiple adapters using fuzzy text matching,
confidence arbitration, and equation validation.
"""

from __future__ import annotations

from typing import List
from .contracts import ExtractionResult, TextBlock


class ExtractionConsensus:
    """Merges extraction outputs from multiple adapters."""

    @classmethod
    def merge(cls, result_a: ExtractionResult, result_b: ExtractionResult) -> ExtractionResult:
        if not result_a.success:
            return result_b
        if not result_b.success:
            return result_a

        # Base merge using adapter boundary method
        return result_a.merge_with(result_b)
