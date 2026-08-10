"""
AION Tripartite Health & Quality Score System
==============================================
Splits single pipeline health score into 3 distinct evaluation scores:
  1. STRUCTURAL SCORE  (Marks, OR parity, CO/Bloom, format, slot completeness)
  2. GROUNDING SCORE   (Evidence coverage, module match, entity match, provenance)
  3. ACADEMIC SCORE    (Question quality, Bloom correctness, difficulty, math/visual validity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class TripartiteHealthScore:
    """Granular tripartite health breakdown."""
    structural_score: int = 100
    grounding_score: int = 100
    academic_score: int = 100
    violations: List[str] = field(default_factory=list)

    @property
    def overall_health(self) -> int:
        """Weighted combination of 3 health dimensions."""
        return int(
            0.35 * self.structural_score +
            0.40 * self.grounding_score +
            0.25 * self.academic_score
        )

    def print_summary(self):
        """Format and print tripartite health score box."""
        print(
            f"[HEALTH] Structural: {self.structural_score}/100 | "
            f"Grounding: {self.grounding_score}/100 | "
            f"Academic: {self.academic_score}/100 | "
            f"Overall: {self.overall_health}/100"
        )
