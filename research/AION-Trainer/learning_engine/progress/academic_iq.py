# learning_engine/progress/academic_iq.py
"""
Academic IQ — calculates a single performance score out of 100 representing
AION's current academic mastery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from learning_engine.memory.concept_memory import ConceptMemory
from learning_engine.memory.relationship_memory import RelationshipMemory
from learning_engine.memory.question_memory import QuestionMemory
from learning_engine.memory.answer_memory import AnswerMemory
from learning_engine.memory.examiner_memory import ExaminerMemory

logger = logging.getLogger("aion.learning.academic_iq")


@dataclass
class AcademicIQDetails:
    iq_score: int
    concept_understanding: int
    relationships: int
    question_quality: int
    answer_quality: int
    examiner_style: int
    confidence: int


class AcademicIQCalculator:
    def __init__(
        self,
        concept_memory: ConceptMemory,
        relationship_memory: RelationshipMemory,
        question_memory: QuestionMemory,
        answer_memory: AnswerMemory,
        examiner_memory: ExaminerMemory,
    ):
        self.concept_memory = concept_memory
        self.relationship_memory = relationship_memory
        self.question_memory = question_memory
        self.answer_memory = answer_memory
        self.examiner_memory = examiner_memory

    def calculate(self, subject_code: str) -> AcademicIQDetails:
        # 1. Concept Understanding
        all_entries = []
        with self.concept_memory._lock:
            all_entries = [
                e for e in self.concept_memory._entries.values()
                if e.subject_code == subject_code
            ]

        if not all_entries:
            return AcademicIQDetails(0, 0, 0, 0, 0, 0, 0)

        sum_understand = sum(e.confidence.understand for e in all_entries)
        concept_understanding = int((sum_understand / len(all_entries)) * 100)

        # 2. Relationships
        total_rels = self.relationship_memory.total_count()
        learned_rels = self.relationship_memory.learned_count()
        relationships = int((learned_rels / total_rels) * 100) if total_rels > 0 else 0

        # 3. Question Quality
        question_quality = int(self.question_memory.acceptance_rate() * 100)

        # 4. Answer Quality
        answer_quality = int(self.answer_memory.average_quality() * 100)

        # 5. Examiner Style
        # Match rate or verb distribution variety
        pref_marks = self.examiner_memory.preferred_marks()
        examiner_style = int(min(1.0, len(pref_marks) * 0.2) * 100)

        # 6. Confidence
        sum_overall = sum(e.confidence.overall() for e in all_entries)
        confidence = int((sum_overall / len(all_entries)) * 100)

        # Calculate final composite score
        iq_score = int(
            (concept_understanding * 0.25) +
            (relationships * 0.15) +
            (question_quality * 0.20) +
            (answer_quality * 0.20) +
            (examiner_style * 0.10) +
            (confidence * 0.10)
        )

        # Bounds check
        iq_score = max(0, min(100, iq_score))

        details = AcademicIQDetails(
            iq_score=iq_score,
            concept_understanding=concept_understanding,
            relationships=relationships,
            question_quality=question_quality,
            answer_quality=answer_quality,
            examiner_style=examiner_style,
            confidence=confidence,
        )
        logger.info(f"[AcademicIQ] Calculated IQ details for {subject_code}: {details}")
        return details
