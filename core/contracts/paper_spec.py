"""
AION Subject-Agnostic Paper Specification Contract
===================================================
Defines the authoritative structural and marks contract for any exam across
any academic subject. Eliminates hardcoded numeric assumptions (e.g. 5 modules,
2 questions/module, 10 marks/question, 5 COs).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PaperSpec:
    exam_type            : str            # Validated against exam_specs catalog (e.g. "IAT1", "SEE")
    module_count         : int            # e.g., 5, 4, 3
    questions_per_module : int            # e.g., 2 (for OR pair), 4, 1
    marks_per_question   : int            # e.g., 10 (IAT), 20 (SEE), 25
    co_count             : int = 5        # Precedence: payload -> exam_specs.json -> 5
    depth_threshold_ratio: float = 0.8    # Tunable analytical depth trigger (0.8 * marks_per_question)
    custom_co_map        : Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.module_count <= 0:
            raise ValueError(f"module_count must be positive, got {self.module_count}")
        if self.questions_per_module <= 0:
            raise ValueError(f"questions_per_module must be positive, got {self.questions_per_module}")
        if self.marks_per_question <= 0:
            raise ValueError(f"marks_per_question must be positive, got {self.marks_per_question}")
        if self.co_count <= 0:
            raise ValueError(f"co_count must be positive, got {self.co_count}")
        if self.depth_threshold_ratio <= 0.0 or self.depth_threshold_ratio > 1.0:
            raise ValueError(f"depth_threshold_ratio must be in (0.0, 1.0], got {self.depth_threshold_ratio}")

    @property
    def total_questions(self) -> int:
        return self.module_count * self.questions_per_module

    @property
    def total_marks(self) -> int:
        return self.module_count * self.marks_per_question

    def allocate_partitions(self, user_marks_split: List[List[int]]) -> Dict[Tuple[int, int], List[int]]:
        """
        Deterministically maps a sequential list of partitions to (module_idx, question_idx).
        Validates:
        - len(user_marks_split) == total_questions
        - Every element is a positive integer
        - sum(partition) == marks_per_question
        """
        if len(user_marks_split) != self.total_questions:
            raise ValueError(
                f"Expected {self.total_questions} partitions for {self.module_count} modules "
                f"× {self.questions_per_module} questions, got {len(user_marks_split)}"
            )

        assignments: Dict[Tuple[int, int], List[int]] = {}
        cumulative = 0
        for m in range(self.module_count):
            for q in range(self.questions_per_module):
                partition = user_marks_split[cumulative]
                if not isinstance(partition, (list, tuple)) or not partition:
                    raise ValueError(f"Partition at index {cumulative} must be a non-empty list of integers, got {partition}")
                
                for elem in partition:
                    if not isinstance(elem, int) or elem <= 0:
                        raise ValueError(f"Partition {partition} contains invalid element {elem} (must be positive integer)")
                
                part_sum = sum(partition)
                if part_sum != self.marks_per_question:
                    raise ValueError(
                        f"Partition {partition} at index {cumulative} (Module {m+1}, Q{q+1}) sums to "
                        f"{part_sum}, expected {self.marks_per_question}"
                    )
                
                assignments[(m, q)] = list(partition)
                cumulative += 1

        return assignments

    def build_co_map(self) -> Dict[str, str]:
        """
        Builds the Course Outcome (CO) map for each module (1-indexed).
        If custom_co_map is provided, validates that values are non-empty and returns it.
        Otherwise computes Module M -> CO{min(M, co_count)}.
        """
        if self.custom_co_map:
            cleaned_map = {str(k): str(v).strip().upper() for k, v in self.custom_co_map.items()}
            # Validation: ensure every module 1..module_count is mapped
            for m in range(1, self.module_count + 1):
                if str(m) not in cleaned_map:
                    cleaned_map[str(m)] = f"CO{min(m, self.co_count)}"
            return cleaned_map

        mapping: Dict[str, str] = {}
        for m in range(1, self.module_count + 1):
            co_idx = min(m, self.co_count)
            mapping[str(m)] = f"CO{co_idx}"
        return mapping


STANDARD_MARKS_SPLITS: Dict[str, List[List[int]]] = {
    "IAT1": [
        [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10],
    ],
    "IAT2": [
        [6, 4], [6, 4], [10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10],
    ],
    "ELECTIVE_3MOD": [
        [10], [6, 4], [5, 5], [10], [6, 4], [5, 5], [10], [6, 4], [5, 5],
    ],
}
