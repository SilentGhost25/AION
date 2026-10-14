"""
The QuestionPlan contract.

One plan per main-question slot. The Writing Agent consumes a plan
and produces a GeneratedQuestion. The plan carries everything the
Writing Agent needs and nothing it doesn't.
"""

# UNTESTED — run quickstart.py on server first
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class QuestionPlan:
    # Identity
    slot_id: str                    # "module_1_Q1"
    module_id: str                  # "module_1"
    module_idx: int                 # 1..5
    slot_idx: int                   # 0-indexed within module
    global_q_idx: int               # 1..N (1-10 for IAT1)

    # Marks contract
    partition: List[int] = field(default_factory=list)   # [6, 4] or [10]
    total_marks: int = 0

    # Pedagogical contract
    co: str = "CO1"
    bloom: str = "L2"

    # Topic grounding (from KG)
    topic: str = ""
    concept_neighborhood: List = field(default_factory=list)

    # Evidence
    evidence_blocks: List = field(default_factory=list)

    # Visual contract
    visual_required: bool = False
    figure: Optional[Any] = None            # FigureArtifact if visual_required

    # Additional constraints (free-form)
    constraints: dict = field(default_factory=dict)

    # Telemetry
    telemetry: dict = field(default_factory=dict)

    @property
    def is_multi_part(self) -> bool:
        return len(self.partition) > 1

    @property
    def difficulty_band(self) -> str:
        """Derive target difficulty band from marks and Bloom level."""
        if self.bloom in ("L1", "L2") or self.total_marks <= 4:
            return "easy"
        if self.bloom in ("L3",) or self.total_marks <= 6:
            return "medium"
        return "hard"  # L4, L5, L6 or marks >= 8

    def __repr__(self) -> str:
        vis = "V" if self.visual_required else "-"
        return (
            f"<Plan {self.slot_id} {self.bloom} {self.total_marks}M "
            f"{self.co} {vis} topic={self.topic[:30]!r}>"
        )
