# learning_engine/stages/self_evaluator.py
"""
Self-Evaluation Engine — Stage 6 of the Academic Learning Loop.

After generating questions and answers, AION evaluates the quality
of its own output and identifies weak concepts that need revisiting.

This is the "Fail -> Re-read" step that makes the system self-improving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory, ConceptMemoryEntry
from learning_engine.memory.question_memory import QuestionMemory
from learning_engine.memory.answer_memory import AnswerMemory
from learning_engine.memory.mistake_memory import MistakeMemory

logger = logging.getLogger("aion.learning.self_evaluator")

WEAK_THRESHOLD = 0.70
STRONG_THRESHOLD = 0.88


@dataclass
class EvaluationResult:
    weak_concepts: List[str] = field(default_factory=list)    # concept_ids
    strong_concepts: List[str] = field(default_factory=list)
    question_acceptance_rate: float = 0.0
    answer_quality: float = 0.0
    avg_understanding: float = 0.0
    dimensions_to_improve: Dict[str, float] = field(default_factory=dict)


class SelfEvaluator:
    def __init__(
        self,
        concept_memory: ConceptMemory,
        question_memory: QuestionMemory,
        answer_memory: AnswerMemory,
        mistake_memory: MistakeMemory,
    ):
        self.concept_memory = concept_memory
        self.question_memory = question_memory
        self.answer_memory = answer_memory
        self.mistake_memory = mistake_memory

    def evaluate(self, subject_code: str, epoch: int) -> EvaluationResult:
        logger.info(f"[SelfEvaluator] Starting self-evaluation pass for {subject_code} in epoch {epoch}")
        
        # Get all concept memory entries for subject
        all_entries = []
        with self.concept_memory._lock:
            all_entries = [
                e for e in self.concept_memory._entries.values()
                if e.subject_code == subject_code
            ]

        if not all_entries:
            return EvaluationResult()

        weak_ids = []
        strong_ids = []
        sum_understand = 0.0

        dim_sums = {
            "understand": 0.0,
            "explain": 0.0,
            "compare": 0.0,
            "generate_question": 0.0,
            "generate_answer": 0.0,
            "predict_exam_use": 0.0,
        }

        for entry in all_entries:
            conf = entry.confidence
            overall = conf.overall()
            sum_understand += conf.understand

            # Accumulate dimensions
            for dim in dim_sums:
                dim_sums[dim] += getattr(conf, dim, 0.0)

            # Flag weak / strong
            if entry.needs_revisit(WEAK_THRESHOLD):
                weak_ids.append(entry.concept_id)
                self.concept_memory.mark_weak(entry.concept_id, f"Self-eval flagged weak in epoch {epoch}")
            else:
                self.concept_memory.clear_weak(entry.concept_id)
                if overall >= STRONG_THRESHOLD:
                    strong_ids.append(entry.concept_id)

        # Average dimensions
        num_concepts = len(all_entries)
        avg_dims = {dim: round(val / num_concepts, 4) for dim, val in dim_sums.items()}
        
        # Identify weakest dimension to improve
        weakest_dim = min(avg_dims, key=avg_dims.get)
        dimensions_to_improve = {weakest_dim: avg_dims[weakest_dim]}

        q_acc_rate = self.question_memory.acceptance_rate()
        ans_qual = self.answer_memory.average_quality()
        avg_und = round(sum_understand / num_concepts, 4)

        result = EvaluationResult(
            weak_concepts=weak_ids,
            strong_concepts=strong_ids,
            question_acceptance_rate=q_acc_rate,
            answer_quality=ans_qual,
            avg_understanding=avg_und,
            dimensions_to_improve=dimensions_to_improve,
        )

        logger.info(
            f"[SelfEvaluator] Evaluation done: "
            f"weak={len(weak_ids)} strong={len(strong_ids)} "
            f"q_acc={q_acc_rate:.2f} ans_qual={ans_qual:.2f} "
            f"weakest_dimension={weakest_dim} ({avg_dims[weakest_dim]:.2f})"
        )
        return result
