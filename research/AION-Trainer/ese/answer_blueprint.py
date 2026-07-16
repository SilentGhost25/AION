# AION-Trainer/ese/answer_blueprint.py
"""
Answer Blueprint — the internal academic representation that exists
BEFORE any English question is written.

This is the professor's handwritten notes, not the question.
It answers: "What must a student demonstrate to get full marks?"

The Question Discovery Engine's job is to find questions whose ideal
answer IS this blueprint. The Language Realizer's job is to convert
the selected question into grammatical English.

The blueprint is deterministic — built from the Concept Store and
Course Knowledge Graph, not from a language model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any


class AnswerComponent:
    """What a complete answer must contain."""
    DEFINITION = "definition"
    ALGORITHM = "algorithm"
    DIAGRAM = "diagram"
    EXAMPLE = "example"
    ADVANTAGES = "advantages"
    DISADVANTAGES = "disadvantages"
    COMPARISON = "comparison"
    COMPLEXITY = "complexity"
    PROOF = "proof"
    DERIVATION = "derivation"
    APPLICATIONS = "applications"
    PSEUDO_CODE = "pseudo_code"
    TRACE = "trace"
    FORMULA = "formula"
    CASE_STUDY = "case_study"


@dataclass
class ExpectedAnswerSection:
    """One section of the expected answer."""
    component: str                      # AnswerComponent constant
    required: bool = True               # must appear or bonus
    marks_weight: float = 1.0           # relative marks allocation
    content_hint: str = ""              # specific point(s) expected
    exemplar: str = ""                  # what a perfect answer looks like


@dataclass
class AnswerBlueprint:
    """
    The professor's internal answer model.

    Built entirely from the Concept Store — no language model
    involved. Contains everything needed to:
        1. Discover candidate questions (ESE Step 4)
        2. Rank those candidates (ESE Step 5)
        3. Evaluate a student's actual answer (future grading)
        4. Learn from professor corrections (what was missing)
    """

    # Core identity
    blueprint_id: str = ""
    concept_id: str = ""
    topic: str = ""
    subtopics: List[str] = field(default_factory=list)
    related_concepts: List[str] = field(default_factory=list)

    # Assessment targets
    bloom_level: str = "L2"
    marks: int = 10
    difficulty: str = "medium"
    question_type: str = "explanation"

    # What the answer must contain
    required_components: List[ExpectedAnswerSection] = field(default_factory=list)

    # Key content (populated from Concept Store)
    definition: str = ""
    algorithm_steps: List[str] = field(default_factory=list)
    key_properties: List[str] = field(default_factory=list)
    comparison_points: Dict[str, List[str]] = field(default_factory=dict)
    # e.g. {"A* vs UCS": ["A* uses heuristic", "UCS is optimal but slow"]}
    applications: List[str] = field(default_factory=list)
    formula: str = ""
    diagram_required: bool = False
    diagram_description: str = ""
    example_instance: str = ""          # concrete example to trace

    # Academic metadata
    subject_code: str = ""
    module: int = 0
    importance: float = 0.5
    pyq_frequency: int = 0
    syllabus_explicit: bool = False

    # Quality signal
    blueprint_confidence: float = 0.0   # how complete is this blueprint?

    def components_summary(self) -> List[str]:
        """Human-readable list of what a full answer needs."""
        return [
            f"{'[Required]' if s.required else '[Bonus]'} {s.component}: {s.content_hint}"
            for s in self.required_components
        ]

    def marks_per_component(self) -> Dict[str, float]:
        """Distribute marks across required components proportionally."""
        required = [s for s in self.required_components if s.required]
        if not required:
            return {}
        total_weight = sum(s.marks_weight for s in required)
        return {
            s.component: round((s.marks_weight / total_weight) * self.marks, 1)
            for s in required
        }

    def compute_confidence(self) -> float:
        """How complete is this blueprint? Used during ranking."""
        score = 0.0
        if self.definition:
            score += 0.25
        if self.required_components:
            score += 0.20
        if self.key_properties:
            score += 0.15
        if self.diagram_required and self.diagram_description:
            score += 0.10
        elif not self.diagram_required:
            score += 0.10
        if self.example_instance:
            score += 0.15
        if self.algorithm_steps:
            score += 0.15
        self.blueprint_confidence = min(1.0, score)
        return self.blueprint_confidence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerBlueprint":
        if "required_components" in data:
            data["required_components"] = [
                ExpectedAnswerSection(**s) if isinstance(s, dict) else s
                for s in data["required_components"]
            ]
        return cls(**data)


class AnswerBlueprintBuilder:
    """
    Constructs an AnswerBlueprint from a Concept (Concept Store entry)
    and an AssessmentIntent.

    Fully deterministic — no language model.
    """

    def build(self, concept, intent) -> AnswerBlueprint:
        blueprint = AnswerBlueprint(
            concept_id=concept.concept_id,
            topic=concept.name,
            subtopics=list(concept.key_points[:5]),
            related_concepts=list(concept.related_concepts[:4]),
            bloom_level=intent.bloom_level,
            marks=intent.marks,
            difficulty=intent.difficulty,
            question_type=intent.question_type,
            definition=concept.definition,
            algorithm_steps=list(concept.algorithms),
            key_properties=list(concept.key_points),
            applications=list(concept.applications),
            formula=concept.formulas[0] if concept.formulas else "",
            diagram_required=concept.requires_diagram,
            diagram_description=concept.diagram_description,
            subject_code=concept.primary_subject() or "",
            module=concept.primary_module() or 0,
            importance=concept.importance,
            pyq_frequency=concept.previous_paper_frequency,
            syllabus_explicit=concept.syllabus_mentions > 0,
        )

        # Build comparison points if a compare_with target is specified
        if intent.compare_with:
            blueprint.comparison_points[
                f"{concept.name} vs {intent.compare_with}"
            ] = self._build_comparison_points(concept, intent.compare_with)

        # Build required components based on bloom level + question type
        blueprint.required_components = self._build_components(blueprint, intent)
        blueprint.compute_confidence()
        return blueprint

    def _build_components(self, bp: AnswerBlueprint, intent) -> List[ExpectedAnswerSection]:
        components = []
        bloom = intent.bloom_level
        qtype = intent.question_type

        # Definition is almost always required
        if bloom in ("L1", "L2", "L3", "L4"):
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.DEFINITION,
                required=True, marks_weight=1.0,
                content_hint=bp.definition[:120] if bp.definition else f"Definition of {bp.topic}",
            ))

        # Algorithm if the concept has steps or type is algorithm
        if bp.algorithm_steps or qtype == "algorithm":
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.ALGORITHM,
                required=qtype == "algorithm",
                marks_weight=2.0,
                content_hint="; ".join(bp.algorithm_steps[:3]) or f"Steps of {bp.topic}",
            ))

        # Example for L3 and above
        if bloom in ("L3", "L4", "L5", "L6"):
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.EXAMPLE,
                required=True, marks_weight=1.5,
                content_hint=bp.example_instance or f"Worked example for {bp.topic}",
            ))

        # Algorithm trace for L3+
        if bloom in ("L3", "L4") and bp.algorithm_steps:
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.TRACE,
                required=bloom == "L3", marks_weight=1.5,
                content_hint=f"Step-by-step trace for {bp.topic}",
            ))

        # Comparison for L4
        if bloom in ("L4", "L5") or qtype == "comparison":
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.COMPARISON,
                required=qtype == "comparison", marks_weight=2.0,
                content_hint=str(list(bp.comparison_points.keys())[:1]),
            ))

        # Complexity for algorithms
        if qtype in ("algorithm", "explanation") and bp.algorithm_steps:
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.COMPLEXITY,
                required=bloom in ("L4", "L5"), marks_weight=1.0,
                content_hint="Time and space complexity",
            ))

        # Diagram
        if bp.diagram_required:
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.DIAGRAM,
                required=True, marks_weight=1.5,
                content_hint=bp.diagram_description or f"Diagram for {bp.topic}",
            ))

        # Applications for L2+
        if bloom in ("L2", "L3", "L4", "L5", "L6") and bp.applications:
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.APPLICATIONS,
                required=False, marks_weight=0.5,
                content_hint="; ".join(bp.applications[:3]),
            ))

        # Design/evaluation for L5/L6
        if bloom == "L5":
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.ADVANTAGES,
                required=True, marks_weight=1.0,
                content_hint=f"Advantages and limitations of {bp.topic}",
            ))
        if bloom == "L6":
            components.append(ExpectedAnswerSection(
                component=AnswerComponent.CASE_STUDY,
                required=True, marks_weight=2.0,
                content_hint=f"Design scenario using {bp.topic}",
            ))

        return components

    def _build_comparison_points(self, concept, compare_with: str) -> List[str]:
        return [
            f"{concept.name} uses {kp}" for kp in concept.key_points[:2]
        ] + [f"{compare_with} differs in approach"]
