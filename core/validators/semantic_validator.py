"""
AION Core Validators — Semantic Validator & OR Distinctness Score
===================================================================
Enforces INV-3: Ensures OR-pair questions maintain semantic distinction (distinctness score).
Prevents near-identical questions from appearing as OR choices while accepting valid
alternative reasoning tasks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class DistinctnessReport:
    is_distinct       : bool
    similarity_score  : float
    distinctness_score: float
    issues            : List[str]


class SemanticValidator:
    """Evaluates semantic similarity and distinctness of OR-pair questions."""

    @classmethod
    def calculate_or_distinctness(cls, text_a: str, text_b: str) -> DistinctnessReport:
        if not text_a or not text_b:
            return DistinctnessReport(is_distinct=True, similarity_score=0.0, distinctness_score=1.0, issues=[])

        words_a = set(re.findall(r"\b\w{4,}\b", text_a.lower()))
        words_b = set(re.findall(r"\b\w{4,}\b", text_b.lower()))

        if not words_a or not words_b:
            return DistinctnessReport(is_distinct=True, similarity_score=0.0, distinctness_score=1.0, issues=[])

        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)

        # Jaccard similarity coefficient
        jaccard_sim = len(intersection) / max(len(union), 1)
        distinctness = 1.0 - jaccard_sim

        issues = []
        # Hard check for near-identical wording
        if jaccard_sim > 0.85:
            issues.append(f"OR-pair questions are nearly identical (jaccard_similarity={jaccard_sim:.2f})")
            is_distinct = False
        else:
            is_distinct = True

        return DistinctnessReport(
            is_distinct=is_distinct,
            similarity_score=jaccard_sim,
            distinctness_score=distinctness,
            issues=issues,
        )
