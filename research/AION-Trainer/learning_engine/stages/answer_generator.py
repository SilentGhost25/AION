# learning_engine/stages/answer_generator.py
"""
Answer Generator — Stage 4 of the Academic Learning Loop.

For every concept that has reached EXPLAINABLE stage, generate
an expected answer and store it in AnswerMemory.

Bridges to the existing AnswerBlueprintBuilder without modifying it.
"""

from __future__ import annotations

import logging
from typing import Optional

from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory
from learning_engine.memory.answer_memory import AnswerMemory, AnswerRecord

logger = logging.getLogger("aion.learning.answer_generator")


class LearningAnswerGenerator:
    def __init__(
        self,
        concept_memory: ConceptMemory,
        answer_memory: AnswerMemory,
        llm_client=None,
    ):
        self.concept_memory = concept_memory
        self.answer_memory = answer_memory
        self.llm = llm_client

    def generate_for_concept(self, concept, epoch: int) -> Optional[AnswerRecord]:
        entry = self.concept_memory.get(concept.concept_id)
        if entry is None or not entry.current_stage.can_generate_answers():
            return None

        answer_text = self._compose_answer(concept)
        if not answer_text:
            return None

        quality = self._score_answer_quality(answer_text, concept)

        record = AnswerRecord(
            concept_id=concept.concept_id,
            question_text=f"Explain {concept.name} with a suitable example.",
            expected_answer=answer_text,
            answer_components=self._detect_components(answer_text),
            quality_score=quality,
            marks=10,
            epoch_created=epoch,
        )
        self.answer_memory.store(record)

        self.concept_memory.update_confidence(
            concept.concept_id, "generate_answer", quality
        )

        # Advance to ANSWERABLE if quality is sufficient
        if quality >= 0.65 and entry.current_stage < ConceptStage.ANSWERABLE:
            self.concept_memory.advance_stage(concept.concept_id)
            logger.info(f"[AnswerGenerator] {concept.name} → ANSWERABLE")

        return record

    def _compose_answer(self, concept) -> str:
        parts = []
        if concept.definition:
            parts.append(concept.definition)
        if concept.explanation:
            parts.append(concept.explanation[:600])
        if concept.key_points:
            parts.append("Key properties: " + "; ".join(concept.key_points[:5]))
        if concept.algorithms:
            parts.append("Algorithm: " + "; ".join(concept.algorithms[:2]))
        if concept.applications:
            parts.append("Applications: " + ", ".join(concept.applications[:3]))
        if concept.requires_diagram:
            parts.append(f"[Diagram required: {concept.diagram_description}]")
        return "\n".join(parts)

    def _score_answer_quality(self, answer_text: str, concept) -> float:
        score = 0.0
        if concept.definition and concept.definition[:40] in answer_text:
            score += 0.25
        if len(answer_text.split()) >= 40:
            score += 0.20
        if concept.key_points and any(kp[:20] in answer_text for kp in concept.key_points):
            score += 0.20
        if concept.applications and any(app in answer_text for app in concept.applications):
            score += 0.15
        if concept.algorithms and any(alg in answer_text for alg in concept.algorithms):
            score += 0.10
        if concept.requires_diagram and "[Diagram" in answer_text:
            score += 0.10
        return round(min(1.0, score), 4)

    def _detect_components(self, text: str) -> list:
        components = []
        text_lower = text.lower()
        checks = {
            "definition": ["is defined", "is a", "refers to"],
            "explanation": ["explanation", "works by", "operates"],
            "algorithm": ["algorithm", "steps", "procedure"],
            "diagram": ["diagram", "figure", "draw"],
            "applications": ["application", "used in", "example"],
        }
        for comp, keywords in checks.items():
            if any(kw in text_lower for kw in keywords):
                components.append(comp)
        return components
