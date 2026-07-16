# AION-Trainer/ese/chief_examiner.py
"""
Chief Examiner — the automatic quality gate that runs BEFORE
user review.

Catches what no individual slot validator can see:
    Repeated concepts across the paper
    Repeated action verbs across the paper
    Repeated question styles
    Repeated diagrams
    Repeated Bloom level sequences (e.g., all L2)
    Unbalanced module coverage

These are paper-level properties — they require seeing the entire
set of candidate questions together.

The Chief Examiner does NOT generate or modify questions.
It flags problems and can request regeneration for specific slots.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("aion.ese.chief_examiner")


@dataclass
class ChiefExaminerFlag:
    severity: str               # "error" | "warning" | "info"
    rule: str
    affected_slots: List[str]
    description: str
    recommendation: str


@dataclass
class ChiefExaminerReport:
    passed: bool = True
    flags: List[ChiefExaminerFlag] = field(default_factory=list)
    slots_to_regenerate: List[str] = field(default_factory=list)
    overall_quality: float = 1.0

    def add_flag(self, flag: ChiefExaminerFlag):
        self.flags.append(flag)
        if flag.severity == "error":
            self.passed = False

    def error_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "error")

    def warning_count(self) -> int:
        return sum(1 for f in self.flags if f.severity == "warning")


class ChiefExaminer:
    """
    Evaluates a fully populated ExamBlueprint for paper-level coherence,
    balance, and quality.
    """

    def __init__(self):
        pass

    def evaluate_paper(self, blueprint) -> ChiefExaminerReport:
        report = ChiefExaminerReport()
        slots = blueprint.slots

        if not slots:
            report.add_flag(ChiefExaminerFlag(
                severity="error",
                rule="Empty Paper",
                affected_slots=[],
                description="The exam blueprint has no question slots.",
                recommendation="Rebuild the exam blueprint with syllabus concepts."
            ))
            report.overall_quality = 0.0
            return report

        # 1. Concept Repetitions
        self._check_concept_repetitions(slots, report)

        # 2. Action Verb Variety
        self._check_verb_variety(slots, report)

        # 3. Bloom Distribution & Progression
        self._check_bloom_distribution(slots, report)

        # 4. Module & Marks Balance
        self._check_module_balance(blueprint, report)

        # 5. Diagram Over-utilization
        self._check_diagram_utilization(slots, report)

        # Compute overall quality score
        deductions = (report.error_count() * 0.2) + (report.warning_count() * 0.05)
        report.overall_quality = max(0.0, round(1.0 - deductions, 2))

        logger.info(
            f"[ChiefExaminer] Evaluated paper. Passed: {report.passed} | "
            f"Errors: {report.error_count()} | Warnings: {report.warning_count()} | "
            f"Quality: {report.overall_quality:.2f}"
        )
        return report

    def _check_concept_repetitions(self, slots, report: ChiefExaminerReport):
        concept_counts = Counter(s.concept_id for s in slots if s.concept_id)
        for concept_id, count in concept_counts.items():
            if count > 1:
                affected = [s.slot_id for s in slots if s.concept_id == concept_id]
                cname = next(s.concept_name for s in slots if s.concept_id == concept_id)
                
                # If they are in the same OR group, warning is sufficient, else error
                # (VTU allows choosing between two parts, but they shouldn't assess the same concept)
                is_same_or = False
                for s in slots:
                    if s.concept_id == concept_id and s.is_optional:
                        # Check if partner slot is in affected
                        if s.or_pair_id in affected:
                            is_same_or = True
                
                severity = "warning" if is_same_or else "error"
                desc = f"Concept '{cname}' is tested {count} times in slots {affected}."
                rec = "Replace duplicate concepts with other topics from the module."
                
                report.add_flag(ChiefExaminerFlag(
                    severity=severity,
                    rule="Concept Repetition",
                    affected_slots=affected,
                    description=desc,
                    recommendation=rec
                ))
                
                if severity == "error":
                    # Mark the duplicate slots for regeneration (excluding the first one)
                    report.slots_to_regenerate.extend(affected[1:])

    def _check_verb_variety(self, slots, report: ChiefExaminerReport):
        verbs = []
        for s in slots:
            if s.question_text:
                first_word = s.question_text.strip().split()[0].rstrip(",:").capitalize()
                verbs.append(first_word)
            
        if not verbs:
            return

        verb_counts = Counter(verbs)
        total = len(verbs)
        for verb, count in verb_counts.items():
            frac = count / total
            if frac > 0.5 and total >= 4:
                affected = [s.slot_id for s in slots if s.question_text and s.question_text.strip().capitalize().startswith(verb)]
                report.add_flag(ChiefExaminerFlag(
                    severity="warning",
                    rule="Monotonous Action Verbs",
                    affected_slots=affected,
                    description=f"Action verb '{verb}' dominates {frac:.0%} of the question paper.",
                    recommendation="Vary the phrasing. Use alternative cognitive verbs (e.g. Describe, Illustrate, Compare)."
                ))

    def _check_bloom_distribution(self, slots, report: ChiefExaminerReport):
        blooms = [s.bloom_level for s in slots if s.bloom_level]
        if not blooms:
            return
        
        bloom_counts = Counter(blooms)
        # If all questions are L2 (Explain)
        if len(bloom_counts) == 1 and "L2" in bloom_counts and len(slots) >= 4:
            report.add_flag(ChiefExaminerFlag(
                severity="warning",
                rule="Lacks Bloom Diversity",
                affected_slots=[s.slot_id for s in slots],
                description="Entire question paper is restricted to L2 (Understand/Explain) level.",
                recommendation="Introduce L3 (Apply/Illustrate) or L4 (Analyze/Compare) questions to test higher-order skills."
            ))

    def _check_module_balance(self, blueprint, report: ChiefExaminerReport):
        slots = blueprint.slots
        modules = set(s.module for s in slots if s.module)
        
        # Check module coverage
        if len(modules) < 5 and blueprint.university == "VTU":
            missing = list(set(range(1, 6)) - modules)
            report.add_flag(ChiefExaminerFlag(
                severity="error",
                rule="Incomplete Module Coverage",
                affected_slots=[],
                description=f"The paper has no questions covering Modules {missing}.",
                recommendation="Ensure the exam blueprint contains questions from all 5 syllabus modules."
            ))

        # Check marks distribution per module (standard VTU has 20 marks per module)
        module_marks = {}
        for s in slots:
            if not s.is_optional:  # only sum required slots
                module_marks[s.module] = module_marks.get(s.module, 0) + s.marks
                
        for mod, marks in module_marks.items():
            if marks != 20 and blueprint.university == "VTU":
                affected = [s.slot_id for s in slots if s.module == mod]
                report.add_flag(ChiefExaminerFlag(
                    severity="warning",
                    rule="Module Marks Unbalanced",
                    affected_slots=affected,
                    description=f"Module {mod} required slots total {marks} marks (expected: 20 marks).",
                    recommendation="Adjust question marks or slots to ensure exactly 20 marks are required per module."
                ))

    def _check_diagram_utilization(self, slots, report: ChiefExaminerReport):
        diagram_count = 0
        affected_slots = []
        for s in slots:
            if s.question_text and any(w in s.question_text.lower() for w in ["diagram", "sketch", "figure"]):
                diagram_count += 1
                affected_slots.append(s.slot_id)
                
        if len(slots) >= 4 and (diagram_count / len(slots)) > 0.6:
            report.add_flag(ChiefExaminerFlag(
                severity="warning",
                rule="Excessive Diagram Demands",
                affected_slots=affected_slots,
                description=f"Over 60% ({diagram_count} questions) demand diagrams, which may create a time constraint.",
                recommendation="Reduce diagram requirements by converting some slots to simple explanations or comparisons."
            ))
