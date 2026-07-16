# AION-Trainer/ese/question_planner.py
"""
Question Planner — maps ExamBlueprint slots to AssessmentIntents.

This is the "Step 2: Examiner Planning" stage.
It answers: "Given this slot, what EXACTLY should be assessed?"

The planner combines the slot specification (bloom, marks, type)
with the concept's academic profile (importance, relationships,
pyq history) to produce a precise AssessmentIntent — the complete
specification that drives both blueprint building and question discovery.

Fully deterministic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict

from ese.answer_blueprint import AnswerBlueprint, AnswerBlueprintBuilder
from ese.exam_blueprint import QuestionSlot
from server.prompt.assessment_intent import AssessmentIntent

logger = logging.getLogger("aion.ese.planner")

BLOOM_TO_ACTION_VERB: Dict[str, List[str]] = {
    "L1": ["Define", "List", "State", "Name"],
    "L2": ["Explain", "Describe", "Discuss", "Summarize"],
    "L3": ["Illustrate", "Apply", "Trace", "Demonstrate", "Solve"],
    "L4": ["Compare", "Analyze", "Differentiate", "Contrast"],
    "L5": ["Evaluate", "Justify", "Critique", "Assess"],
    "L6": ["Design", "Develop", "Propose", "Construct"],
}


@dataclass
class PlannerOutput:
    slot: QuestionSlot
    intent: AssessmentIntent
    blueprint: AnswerBlueprint


class QuestionPlanner:
    """
    Converts an ExamBlueprint slot into a concrete AssessmentIntent
    plus an AnswerBlueprint, ready for the Question Discovery Engine.
    """

    def __init__(self, concept_store, reference_library=None):
        self.concept_store = concept_store
        self.reference_library = reference_library
        self._blueprint_builder = AnswerBlueprintBuilder()

    def plan_slot(
        self,
        slot: QuestionSlot,
        previously_asked: List[str] = None,
        asked_concept_ids: List[str] = None,
    ) -> Optional[PlannerOutput]:
        concept = self.concept_store.get(slot.concept_id)
        if concept is None:
            logger.warning(f"[Planner] Concept not found: {slot.concept_id}")
            return None

        # Choose action verb: pick based on bloom + concept characteristics
        verb = self._choose_verb(slot.bloom_level, concept)

        # Determine if this should be a comparison question
        compare_with = None
        if slot.question_type == "comparison" and concept.related_concepts:
            compare_with = self._choose_comparison_target(
                concept, slot.bloom_level, asked_concept_ids or []
            )

        # Build AssessmentIntent
        intent = AssessmentIntent(
            topic=concept.name,
            subtopic=concept.key_points[0] if concept.key_points else "",
            definition=concept.definition,
            explanation=concept.explanation[:400] if concept.explanation else "",
            key_points=concept.key_points[:6],
            algorithms=concept.algorithms[:4],
            applications=concept.applications[:4],
            formulas=concept.formulas[:2],
            diagram_description=concept.diagram_description,
            bloom_level=slot.bloom_level,
            action_verb=verb,
            marks=slot.marks,
            question_type=slot.question_type,
            difficulty=slot.difficulty,
            requires_diagram=concept.requires_diagram,
            compare_with=compare_with,
            subject_code=slot.subject_code if hasattr(slot, "subject_code") else concept.primary_subject() or "",
            module=slot.module,
            previously_asked=previously_asked[-5:] if previously_asked else [],
        )

        # Attach reference questions from library
        if self.reference_library:
            intent.reference_questions = self.reference_library.get_references(
                topic=concept.name,
                bloom_level=slot.bloom_level,
                question_type=slot.question_type,
            )

        # Build answer blueprint
        blueprint = self._blueprint_builder.build(concept, intent)

        return PlannerOutput(slot=slot, intent=intent, blueprint=blueprint)

    def plan_all(
        self,
        exam_blueprint,
        previously_asked: List[str] = None,
    ) -> List[PlannerOutput]:
        outputs = []
        asked_concept_ids = []
        asked_texts = list(previously_asked or [])

        for slot in exam_blueprint.slots:
            output = self.plan_slot(slot, asked_texts, asked_concept_ids)
            if output:
                outputs.append(output)
                asked_concept_ids.append(slot.concept_id)

        logger.info(f"[Planner] Planned {len(outputs)} question slots")
        return outputs

    def _choose_verb(self, bloom_level: str, concept) -> str:
        candidates = BLOOM_TO_ACTION_VERB.get(bloom_level, ["Explain"])
        if bloom_level == "L3" and concept.algorithms:
            return "Illustrate"
        if bloom_level == "L3" and concept.requires_diagram:
            return "Trace"
        if bloom_level == "L4":
            return "Compare"
        return candidates[0]

    def _choose_comparison_target(
        self, concept, bloom_level: str, asked_ids: List[str]
    ) -> Optional[str]:
        for related_id in concept.related_concepts:
            if related_id not in asked_ids:
                related = self.concept_store.get(related_id)
                if related:
                    return related.name
        if concept.related_concepts:
            first = self.concept_store.get(concept.related_concepts[0])
            return first.name if first else None
        return None
