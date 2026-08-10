"""
AION Master Production Specification — Paper Structure Plan Contract
=====================================================================
Frozen, immutable deterministic blueprint of the paper created ONCE
before any extraction, retrieval, or Qwen calls begin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


class ORParityViolationError(Exception):
    """Raised when an OR pair violates mark or sub-question structural parity."""
    pass


@dataclass(frozen=True)
class SlotDescriptor:
    """Describes one sub-question slot. Immutable once created."""
    slot_id       : str           # "Q1a", "Q1b", etc.
    question_no   : int           # 1-indexed question number
    sub_label     : str           # "a", "b", "c", "d"
    module_id     : int
    marks         : int           # LOCKED — never changes
    co            : str           # LOCKED
    bloom         : str           # LOCKED
    question_type : str           # LOCKED


@dataclass(frozen=True)
class ORPairDescriptor:
    """
    Describes one OR pair. Both alternatives share identical structure.
    This is the primary structural invariant of the paper.
    """
    module_id          : int
    alt_a_question_no  : int           # e.g. 1
    alt_b_question_no  : int           # e.g. 2
    total_marks        : int           # per alternative
    subquestion_count  : int
    mark_distribution  : Tuple[int, ...] # IDENTICAL for both alternatives

    # Both alternatives carry slots built from the same distribution
    slots_a            : Tuple[SlotDescriptor, ...]
    slots_b            : Tuple[SlotDescriptor, ...]

    def __post_init__(self):
        # Structural parity invariant
        assert len(self.slots_a) == self.subquestion_count, f"slots_a len {len(self.slots_a)} != {self.subquestion_count}"
        assert len(self.slots_b) == self.subquestion_count, f"slots_b len {len(self.slots_b)} != {self.subquestion_count}"
        assert sum(s.marks for s in self.slots_a) == self.total_marks, f"slots_a marks sum != {self.total_marks}"
        assert sum(s.marks for s in self.slots_b) == self.total_marks, f"slots_b marks sum != {self.total_marks}"

        marks_a = tuple(s.marks for s in self.slots_a)
        marks_b = tuple(s.marks for s in self.slots_b)
        if not (marks_a == marks_b == self.mark_distribution):
            raise ORParityViolationError(
                f"OR pair mark parity violated in Module {self.module_id}: "
                f"Alt A {marks_a} vs Alt B {marks_b} vs expected {self.mark_distribution}"
            )


@dataclass(frozen=True)
class PaperStructurePlan:
    """
    The complete deterministic paper blueprint.
    Created ONCE before any Qwen call.
    Immutable throughout the pipeline.
    """
    plan_id              : str
    request_id           : str
    created_at           : str

    # From GenerationRequest
    total_marks          : int
    module_count         : int
    marks_per_module     : int
    subquestion_count    : int
    distribution_policy  : str
    mark_distribution    : Tuple[int, ...]    # e.g. (6, 4)

    # The complete plan — one entry per module
    or_pairs             : Tuple[ORPairDescriptor, ...]

    # Derived totals for contract verification
    total_questions      : int    # module_count * 2
    total_attemptable    : int    # module_count * marks_per_module

    def __post_init__(self):
        assert len(self.or_pairs) == self.module_count, f"or_pairs len {len(self.or_pairs)} != module_count {self.module_count}"
        assert self.total_attemptable == self.total_marks, f"total_attemptable {self.total_attemptable} != total_marks {self.total_marks}"
        assert sum(self.mark_distribution) == self.marks_per_module, f"mark_distribution sum {sum(self.mark_distribution)} != marks_per_module {self.marks_per_module}"

    def get_all_slots(self) -> List[SlotDescriptor]:
        slots = []
        for pair in self.or_pairs:
            slots.extend(pair.slots_a)
            slots.extend(pair.slots_b)
        return slots

    def summary(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "total_marks": self.total_marks,
            "modules": self.module_count,
            "marks_per_module": self.marks_per_module,
            "subquestion_count": self.subquestion_count,
            "distribution": list(self.mark_distribution),
            "total_questions": self.total_questions,
            "or_pairs": len(self.or_pairs),
        }
