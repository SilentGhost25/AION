# learning_engine/stages/question_generator.py
"""
Question Generator — Stage 5 of the Academic Learning Loop.

Bridges to the existing ESE QuestionDiscoverer and QuestionRanker
without modifying them. Only invoked when concept is QUESTIONABLE.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory
from learning_engine.memory.question_memory import QuestionMemory, QuestionRecord
from learning_engine.memory.mistake_memory import MistakeMemory

logger = logging.getLogger("aion.learning.question_generator")


class LearningQuestionGenerator:
    def __init__(
        self,
        concept_memory: ConceptMemory,
        question_memory: QuestionMemory,
        mistake_memory: MistakeMemory,
    ):
        self.concept_memory = concept_memory
        self.question_memory = question_memory
        self.mistake_memory = mistake_memory

    def generate_for_concept(
        self,
        concept,
        answer_blueprint,
        intent,
        epoch: int,
    ) -> Optional[QuestionRecord]:
        entry = self.concept_memory.get(concept.concept_id)
        if entry is None or not entry.current_stage.can_generate_questions():
            return None

        # Use existing ESE discovery + ranking
        try:
            from ese.question_discoverer import QuestionDiscoverer
            from ese.question_ranker import QuestionRanker
            from ese.grammar_validator import GrammarValidator
            from ese.vtu_validator import VTUValidator

            discoverer = QuestionDiscoverer()
            ranker = QuestionRanker()
            grammar_validator = GrammarValidator()
            vtu_validator = VTUValidator()

            previously_asked = self.question_memory.accepted_texts(concept.concept_id)
            candidates = discoverer.discover(answer_blueprint, intent)
            if not candidates:
                return None

            best = ranker.best(candidates, answer_blueprint, intent, previously_asked)
            if best is None:
                return None

            candidate, ranking_score = best
            grammar_issues = grammar_validator.validate(candidate.text)
            vtu_issues = vtu_validator.validate(
                candidate.text, intent.bloom_level, intent.marks, intent.requires_diagram
            )

            accepted = not any(issue.severity == "error" for issue in grammar_issues)
            rejection_reason = ""
            if not accepted:
                rejection_reason = "; ".join(issue.message for issue in grammar_issues if issue.severity == "error")

            grammar_score = max(0.0, min(1.0, 1.0 - (len(grammar_issues) * 0.1)))
            vtu_score = max(0.0, min(1.0, 1.0 - (len(vtu_issues) * 0.1)))

            record = QuestionRecord(
                record_id=str(uuid.uuid4())[:8],
                concept_id=concept.concept_id,
                question_text=candidate.text,
                bloom_level=intent.bloom_level,
                marks=intent.marks,
                difficulty=answer_blueprint.difficulty,
                question_type=answer_blueprint.question_type,
                grammar_score=grammar_score,
                vtu_style_score=vtu_score,
                bloom_alignment=ranking_score.bloom_alignment,
                novelty_score=ranking_score.novelty,
                accepted=accepted,
                rejection_reason=rejection_reason,
                epoch=epoch,
            )
            self.question_memory.store(record)

            if accepted:
                self.concept_memory.update_confidence(
                    concept.concept_id,
                    "generate_question",
                    ranking_score.overall,
                )
                if entry.current_stage < ConceptStage.QUESTIONABLE:
                    self.concept_memory.advance_stage(concept.concept_id)
            else:
                # Record as a mistake for future learning
                self.mistake_memory.record(
                    mistake_id=record.record_id,
                    concept_id=concept.concept_id,
                    generated_text=candidate.text,
                    reason=rejection_reason,
                    categories=[issue.message for issue in grammar_issues],
                    epoch=epoch,
                )
                self.concept_memory.mark_weak(
                    concept.concept_id,
                    f"Question rejected in epoch {epoch}: {rejection_reason}",
                )

            return record

        except Exception as e:
            logger.error(f"[QuestionGenerator] Failed for {concept.name}: {e}")
            return None
