"""
AION Structural Architecture v2 — Joint Bloom × Mark Constraint Solver (JBMCS)
================================================================================
Jointly maps mark distributions to compatible Bloom cognitive levels under
position policies and emergency fallbacks.
"""

from __future__ import annotations

import random
from typing import List, Tuple
from .contracts import BloomLevel


MARK_BLOOM_TABLE = {
    1:  [BloomLevel.L1],
    2:  [BloomLevel.L1, BloomLevel.L2],
    3:  [BloomLevel.L1, BloomLevel.L2],
    4:  [BloomLevel.L2, BloomLevel.L3],
    5:  [BloomLevel.L2, BloomLevel.L3],
    6:  [BloomLevel.L2, BloomLevel.L3],   # 6-mark: L2/L3 not L1/L4
    7:  [BloomLevel.L3, BloomLevel.L4],
    8:  [BloomLevel.L3, BloomLevel.L4],
    9:  [BloomLevel.L4, BloomLevel.L5],
    10: [BloomLevel.L4, BloomLevel.L5],   # L6 never auto-assigned
}


class JointBloomMarkConstraintSolver:
    """Solves compatible Bloom levels per mark slot."""

    @classmethod
    def closest_bloom(cls, marks: int, bloom_levels: List[BloomLevel]) -> List[BloomLevel]:
        """Emergency fallback when mark-bloom intersection is empty."""
        ideal_val = max(1, min(6, marks // 2 if marks > 1 else 1))
        best = min(bloom_levels, key=lambda b: abs(b.value - ideal_val))
        return [best]

    @classmethod
    def solve(
        cls,
        D: Tuple[int, ...],
        bloom_levels: List[BloomLevel],
        sub_question_count: int,
        rng: random.Random,
        position_policy: str = "PRIMARY_DEMANDING",
    ) -> Tuple[BloomLevel, ...]:
        """
        Produce a Bloom profile P compatible with distribution D.
        Same profile is applied to both alternatives in an OR pair.
        """
        safe_bloom_levels = [b for b in bloom_levels if b != BloomLevel.L6] or bloom_levels
        P: List[BloomLevel] = []
        slots = D[:sub_question_count]

        for i, marks in enumerate(slots):
            allowed_by_marks = MARK_BLOOM_TABLE.get(
                marks, [BloomLevel.L2, BloomLevel.L3]
            )
            candidates = [b for b in allowed_by_marks if b in safe_bloom_levels]

            if not candidates:
                candidates = cls.closest_bloom(marks, safe_bloom_levels)

            sorted_c = sorted(candidates, key=lambda b: b.value)
            med_val = sorted_c[len(sorted_c) // 2].value

            if i == 0:
                preferred = [b for b in candidates if b.value <= med_val]
                if preferred:
                    candidates = preferred
            elif i == len(slots) - 1 and len(slots) > 1:
                preferred = [b for b in candidates if b.value >= med_val]
                if preferred:
                    candidates = preferred

            chosen = rng.choice(candidates)
            P.append(chosen)

        return tuple(P)
