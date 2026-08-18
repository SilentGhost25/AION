# AION-Trainer/ese/exam_blueprint.py
"""
Exam Blueprint — the paper-level coverage plan.

Before any question is generated, the Examiner Simulation Engine
plans the entire paper: how many questions, which concepts, which
bloom levels, what marks distribution, what question types.

This is the "Paper Blueprint" step — the paper is filled into this
plan, not generated freely.
"""

from core.contracts.question_slot import QuestionSlot
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any


class CoverageTarget:
    """Required coverage fractions by question type."""
    THEORY = "theory"
    ALGORITHM = "algorithm"
    COMPARISON = "comparison"
    APPLICATION = "application"
    CASE_STUDY = "case_study"
    NUMERICAL = "numerical"


VTU_STANDARD_DISTRIBUTION = {
    CoverageTarget.THEORY: 0.35,
    CoverageTarget.ALGORITHM: 0.25,
    CoverageTarget.COMPARISON: 0.15,
    CoverageTarget.APPLICATION: 0.15,
    CoverageTarget.CASE_STUDY: 0.10,
}


@dataclass
class QuestionSlot:
    """
    One slot in the exam blueprint.

    The slot is the PLAN. A GeneratedQuestion fills the slot.
    """
    slot_id: str
    module: int
    concept_id: str
    concept_name: str
    bloom_level: str
    marks: int
    difficulty: str
    question_type: str
    is_optional: bool = False           # VTU "OR" questions
    or_pair_id: Optional[str] = None    # partner slot for OR questions
    filled: bool = False
    question_text: str = ""
    question_metadata_id: str = ""

    def label(self) -> str:
        return f"{self.slot_id} [{self.marks}M / {self.bloom_level} / {self.difficulty}]"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionSlot":
        return cls(**data)


@dataclass
class ExamBlueprint:
    """
    Complete paper-level plan.

    The blueprint is produced before generation starts.
    The Chief Examiner verifies the blueprint for coverage,
    diversity, and balance before generation begins.
    """
    blueprint_id: str = ""
    subject_code: str = ""
    subject_name: str = ""
    university: str = "VTU"
    semester: int = 0
    total_marks: int = 100
    duration_hours: int = 3

    slots: List[QuestionSlot] = field(default_factory=list)

    # Coverage analysis
    coverage_by_type: Dict[str, float] = field(default_factory=dict)
    bloom_distribution: Dict[str, int] = field(default_factory=dict)
    module_distribution: Dict[int, int] = field(default_factory=dict)
    marks_distribution: Dict[int, int] = field(default_factory=dict)

    # Blueprint quality
    coverage_score: float = 0.0
    diversity_score: float = 0.0

    def required_slots(self) -> List[QuestionSlot]:
        return [s for s in self.slots if not s.is_optional]

    def optional_slots(self) -> List[QuestionSlot]:
        return [s for s in self.slots if s.is_optional]

    def unfilled_slots(self) -> List[QuestionSlot]:
        return [s for s in self.slots if not s.filled]

    def compute_distributions(self):
        """Recompute all distribution summaries from current slots."""
        self.bloom_distribution = {}
        self.module_distribution = {}
        self.marks_distribution = {}
        self.coverage_by_type = {}

        total_marks = sum(s.marks for s in self.required_slots()) or 1

        for slot in self.slots:
            self.bloom_distribution[slot.bloom_level] = (
                self.bloom_distribution.get(slot.bloom_level, 0) + 1
            )
            self.module_distribution[slot.module] = (
                self.module_distribution.get(slot.module, 0) + 1
            )
            self.marks_distribution[slot.marks] = (
                self.marks_distribution.get(slot.marks, 0) + 1
            )
            self.coverage_by_type[slot.question_type] = (
                self.coverage_by_type.get(slot.question_type, 0) + slot.marks
            )

        # Normalise coverage_by_type to fractions
        self.coverage_by_type = {
            qt: round(marks / total_marks, 3)
            for qt, marks in self.coverage_by_type.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExamBlueprint":
        if "slots" in data:
            data["slots"] = [
                QuestionSlot.from_dict(s) if isinstance(s, dict) else s
                for s in data["slots"]
            ]
        return cls(**data)

    def summary_text(self) -> str:
        lines = [
            f"Exam Blueprint: {self.subject_name} ({self.subject_code})",
            f"Total Slots: {len(self.slots)} | Marks: {self.total_marks}",
            "",
            "Coverage by Type:",
        ]
        for qt, frac in self.coverage_by_type.items():
            lines.append(f"  {qt:<20} {frac:.0%}")
        lines.append("")
        lines.append("Bloom Distribution:")
        for lvl, count in sorted(self.bloom_distribution.items()):
            lines.append(f"  {lvl}: {count} questions")
        return "\n".join(lines)


class ExamBlueprintBuilder:
    """
    Builds an ExamBlueprint from the concept store and configuration.

    Fully deterministic — selects concepts by importance and frequency,
    balances bloom levels and question types according to VTU norms.
    No language model involved.
    """

    VTU_MARKS_PER_MODULE = 20  # standard VTU: 5 modules × 20 marks

    def build(
        self,
        subject_code: str,
        subject_name: str,
        semester: int,
        concept_store,
        num_modules: int = 5,
        target_coverage: Dict[str, float] = None,
        include_optional: bool = True,
    ) -> ExamBlueprint:
        import uuid

        blueprint = ExamBlueprint(
            blueprint_id=str(uuid.uuid4())[:8],
            subject_code=subject_code,
            subject_name=subject_name,
            semester=semester,
            total_marks=100,
        )

        target_coverage = target_coverage or VTU_STANDARD_DISTRIBUTION
        slot_counter = 0

        for module_num in range(1, num_modules + 1):
            concepts = concept_store.concepts_for_module(subject_code, module_num)
            if not concepts:
                continue

            # Sort by importance × pyq_frequency combined signal
            concepts = sorted(
                concepts,
                key=lambda c: c.importance * 0.6 + min(c.previous_paper_frequency / 20, 1.0) * 0.4,
                reverse=True,
            )

            slots = self._build_module_slots(
                concepts, module_num, slot_counter, include_optional
            )
            blueprint.slots.extend(slots)
            slot_counter += len(slots)

        blueprint.compute_distributions()
        blueprint.coverage_score = self._compute_coverage_score(
            blueprint, target_coverage
        )
        blueprint.diversity_score = self._compute_diversity_score(blueprint)
        return blueprint

    def _build_module_slots(
        self,
        concepts: list,
        module_num: int,
        offset: int,
        include_optional: bool,
    ) -> List[QuestionSlot]:
        slots = []
        letter = "a"

        # Standard VTU module structure:
        # Q(2n-1)a: 10M required  Q(2n-1)b: 10M required
        # Q(2n)a:   10M optional  Q(2n)b:   10M optional
        question_num = (module_num * 2) - 1
        slot_configs = [
            (f"{question_num}{letter}", 10, False, self._bloom_for_importance(concepts, 0)),
            (f"{question_num}b", 10, False, self._bloom_for_importance(concepts, 1)),
        ]
        if include_optional:
            slot_configs += [
                (f"{question_num + 1}a", 10, True, self._bloom_for_importance(concepts, 2)),
                (f"{question_num + 1}b", 10, True, self._bloom_for_importance(concepts, 3)),
            ]

        for i, (slot_id, marks, is_optional, bloom) in enumerate(slot_configs):
            concept = concepts[min(i, len(concepts) - 1)]
            qtype = self._select_question_type(bloom, concept)
            difficulty = self._bloom_to_difficulty(bloom)

            slot = QuestionSlot(
                slot_id=slot_id,
                module=module_num,
                concept_id=concept.concept_id,
                concept_name=concept.name,
                bloom_level=bloom,
                marks=marks,
                difficulty=difficulty,
                question_type=qtype,
                is_optional=is_optional,
            )

            # Link OR pairs
            if is_optional and len(slots) >= 2:
                partner = slots[-1 if i % 2 == 1 else -2]
                if partner.is_optional:
                    slot.or_pair_id = partner.slot_id
                    partner.or_pair_id = slot.slot_id

            slots.append(slot)

        return slots

    def _bloom_for_importance(self, concepts: list, index: int) -> str:
        """Assign bloom levels: most important concept gets higher bloom."""
        mapping = {0: "L3", 1: "L2", 2: "L4", 3: "L2"}
        return mapping.get(index, "L2")

    def _select_question_type(self, bloom: str, concept) -> str:
        if bloom in ("L1",):
            return "definition"
        if bloom in ("L2",):
            return "explanation"
        if bloom in ("L3",) and concept.algorithms:
            return "algorithm"
        if bloom in ("L4",):
            return "comparison"
        if bloom in ("L5",):
            return "case_study"
        if bloom in ("L6",):
            return "case_study"
        return "explanation"

    def _bloom_to_difficulty(self, bloom: str) -> str:
        return {"L1": "easy", "L2": "easy", "L3": "medium",
                "L4": "medium", "L5": "hard", "L6": "hard"}.get(bloom, "medium")

    def _compute_coverage_score(
        self, blueprint: ExamBlueprint, target: Dict[str, float]
    ) -> float:
        if not blueprint.coverage_by_type:
            return 0.0
        diffs = []
        for qt, target_frac in target.items():
            actual = blueprint.coverage_by_type.get(qt, 0.0)
            diffs.append(abs(target_frac - actual))
        return max(0.0, 1.0 - sum(diffs) / len(diffs))

    def _compute_diversity_score(self, blueprint: ExamBlueprint) -> float:
        if not blueprint.slots:
            return 0.0
        bloom_levels = set(s.bloom_level for s in blueprint.slots)
        qtypes = set(s.question_type for s in blueprint.slots)
        modules = set(s.module for s in blueprint.slots)
        bloom_score = len(bloom_levels) / 6.0
        qtype_score = len(qtypes) / 5.0
        module_score = min(1.0, len(modules) / 5.0)
        return round((bloom_score + qtype_score + module_score) / 3.0, 4)
