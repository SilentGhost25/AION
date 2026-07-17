# learning_engine/stages/concept_learner.py
"""
Concept Learner — Stage 1 and 2 of the Academic Learning Loop.

Read → Understand → Build Relationships

Integrates with the existing ACB ConceptStore and ESE AnswerBlueprint
without modifying either.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from learning_engine.stages import ConceptStage
from learning_engine.memory.concept_memory import ConceptMemory
from learning_engine.memory.relationship_memory import RelationshipMemory

logger = logging.getLogger("aion.learning.concept_learner")


class ConceptLearner:
    """
    Reads concepts from the ConceptStore and updates ConceptMemory
    with understanding scores derived from content completeness.
    """

    def __init__(
        self,
        concept_memory: ConceptMemory,
        relationship_memory: RelationshipMemory,
        llm_client=None,
    ):
        self.concept_memory = concept_memory
        self.relationship_memory = relationship_memory
        self.llm = llm_client

    def study_concept(self, concept, epoch: int) -> float:
        """
        Study one concept. Returns the understanding score achieved.
        Updates ConceptMemory in place.
        """
        entry = self.concept_memory.get(concept.concept_id)
        if entry is None:
            return 0.0

        # Compute understanding from content completeness (deterministic)
        understand_score = self._score_understanding(concept)

        # Compute explain score
        explain_score = self._score_explainability(concept)

        self.concept_memory.update_confidence(
            concept.concept_id, "understand", understand_score
        )
        self.concept_memory.update_confidence(
            concept.concept_id, "explain", explain_score
        )
        self.concept_memory.record_study(concept.concept_id)

        # Advance stage if scores justify it
        self._try_advance_stage(concept, entry)

        logger.debug(
            f"[ConceptLearner] {concept.name}: "
            f"understand={understand_score:.2f} explain={explain_score:.2f}"
        )
        return understand_score

    def build_relationships(self, concept, all_concepts: List) -> int:
        """
        Map relationships between this concept and others.
        Returns the number of new relationships recorded.
        """
        entry = self.concept_memory.get(concept.concept_id)
        if entry is None:
            return 0

        recorded = 0
        concept_names = {c.name.lower(): c.concept_id for c in all_concepts}

        # Prerequisites
        for prereq_name in concept.prerequisites:
            target_id = concept_names.get(prereq_name.lower())
            if target_id:
                self.relationship_memory.record(
                    concept.concept_id, target_id, "prerequisite", strength=0.8
                )
                recorded += 1

        # Related concepts
        for related in concept.related_concepts:
            target_id = concept_names.get(related.lower()) if isinstance(related, str) else related
            if target_id:
                self.relationship_memory.record(
                    concept.concept_id, target_id, "related", strength=0.6
                )
                recorded += 1

        if recorded > 0:
            self.concept_memory.update_confidence(
                concept.concept_id, "compare",
                min(1.0, 0.5 + recorded * 0.1),
            )
            self._try_advance_stage(concept, entry, to_stage=ConceptStage.CONNECTED)

        return recorded

    def _score_understanding(self, concept) -> float:
        score = 0.0
        if concept.definition:
            score += 0.30
        if concept.explanation:
            score += 0.25
        if concept.key_points:
            score += min(0.20, len(concept.key_points) * 0.04)
        if concept.examples:
            score += 0.10
        if concept.applications:
            score += 0.10
        if concept.algorithms:
            score += 0.05
        return round(min(1.0, score), 4)

    def _score_explainability(self, concept) -> float:
        score = 0.0
        if concept.definition and len(concept.definition) > 30:
            score += 0.35
        if concept.explanation and len(concept.explanation) > 80:
            score += 0.35
        if len(concept.key_points) >= 3:
            score += 0.20
        if concept.requires_diagram and concept.diagram_description:
            score += 0.10
        return round(min(1.0, score), 4)

    def _try_advance_stage(
        self,
        concept,
        entry,
        to_stage: Optional[ConceptStage] = None,
    ):
        conf = entry.confidence
        current = entry.current_stage

        target = to_stage or ConceptStage(current.value + 1)

        advance = False
        if target == ConceptStage.RECOGNISED and conf.understand >= 0.40:
            advance = True
        elif target == ConceptStage.UNDERSTOOD and conf.understand >= 0.65:
            advance = True
        elif target == ConceptStage.CONNECTED and conf.compare >= 0.50:
            advance = True
        elif target == ConceptStage.EXPLAINABLE and conf.explain >= 0.70:
            advance = True

        if advance and current < target:
            self.concept_memory.advance_stage(concept.concept_id)
            logger.info(
                f"[ConceptLearner] {concept.name} advanced to "
                f"{entry.current_stage.label()}"
            )
