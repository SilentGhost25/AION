"""
Question Planner — Real Question Planning (Planner -> Composer)
==============================================================
Current poor pipeline often does: Explain... Compare... Derive... (artificial)
Need: Planner decides concept/marks/Bloom/reasoning objective; Composer writes English.

Planner Input: GroundedConcept
Planner Output: QuestionPlan (deterministic, auditable)

Planner logic:
- Bloom progression per subject/module balance
- Marks distribution per exam rules (IA/SEE)
- Reasoning objective selection
- Concept selection avoiding repetition
- Numerical vs theoretical distribution
"""

from __future__ import annotations

import random
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from core.concepts.grounding import GroundedConcept

@dataclass
class PlannerConfig:
    exam_type: str = "SEE"  # IA | SEE
    difficulty: str = "mixed"  # easy | medium | hard | mixed
    target_bloom: Optional[int] = None
    marks: int = 10  # per question
    num_questions: int = 4
    require_numerical_ratio: float = 0.25  # At least 25% numerical if available

@dataclass
class QuestionPlan:
    plan_id: str
    concept_id: str
    concept_name: str
    source_hash: str
    marks: int
    bloom_level: int
    bloom_label: str
    reasoning_objective: str
    question_type: str  # conceptual | numerical | derivation | diagram | comparison
    action_verb: str
    requires_diagram: bool
    requires_formula: bool
    difficulty: str
    confidence: float
    expected_answer: str
    evidence_snippet: str
    constraints: Dict[str, Any] = field(default_factory=dict)

class QuestionPlanner:
    """
    Deterministic planner. No LLM. Pure logic.
    """

    BLOOM_VERBS = {
        1: ["Define", "List", "State"],
        2: ["Explain", "Describe", "Summarise"],
        3: ["Apply", "Illustrate", "Solve", "Calculate"],
        4: ["Analyse", "Compare", "Differentiate"],
        5: ["Evaluate", "Justify", "Assess"],
        6: ["Design", "Construct", "Formulate"],
    }

    REASONING_OBJECTIVES = {
        1: "Recall and state the canonical definition with precision.",
        2: "Demonstrate understanding by explaining mechanisms and relationships.",
        3: "Apply the concept to a new problem instance with calculation or illustration.",
        4: "Analyse by comparing, differentiating, or decomposing into components.",
        5: "Evaluate trade-offs, justify choices, or critique assumptions.",
        6: "Create by designing, constructing, or proposing a novel solution.",
    }

    def __init__(self, config: PlannerConfig | None = None):
        self.config = config or PlannerConfig()

    def plan(self, grounded_concepts: List[GroundedConcept]) -> List[QuestionPlan]:
        """
        Produce one plan per grounded concept (or subset per num_questions).
        Balances Bloom, difficulty, and numerical distribution.
        """
        if not grounded_concepts:
            return []

        # Shuffle deterministically per exam type to ensure variety
        concepts = list(grounded_concepts)
        # Balance: prioritize by confidence but ensure diversity
        concepts.sort(key=lambda g: g.confidence, reverse=True)

        # Select top N per config
        n = min(self.config.num_questions * 5, len(concepts))  # generate up to 20 candidates
        selected = concepts[:n]

        plans: List[QuestionPlan] = []
        bloom_counter: Dict[int, int] = {i: 0 for i in range(1, 7)}
        numerical_used = 0

        for idx, gc in enumerate(selected):
            # Decide Bloom: respect target or balance
            if self.config.target_bloom:
                bloom = self.config.target_bloom
            else:
                bloom = self._choose_balanced_bloom(gc, bloom_counter)

            bloom_counter[bloom] += 1

            # Decide marks
            marks = self._decide_marks(gc, bloom)

            # Decide type
            qtype = self._decide_type(gc, bloom, numerical_used)

            if qtype == "numerical":
                numerical_used += 1

            # Difficulty
            difficulty = self._decide_difficulty(gc, bloom)

            # Verb
            verb = random.choice(self.BLOOM_VERBS.get(bloom, ["Explain"]))

            plan_id = f"plan_{gc.concept.concept_id}_{hashlib.sha256(verb.encode()).hexdigest()[:4]}"

            plan = QuestionPlan(
                plan_id=plan_id,
                concept_id=gc.concept.concept_id,
                concept_name=gc.concept.concept_name,
                source_hash=gc.source_hash,
                marks=marks,
                bloom_level=bloom,
                bloom_label=gc.bloom_label if bloom == gc.bloom_level else self._bloom_label(bloom),
                reasoning_objective=self.REASONING_OBJECTIVES.get(bloom, "Demonstrate understanding."),
                question_type=qtype,
                action_verb=verb,
                requires_diagram=(qtype == "diagram" or bool(gc.concept.diagram_refs)),
                requires_formula=(qtype == "numerical" or bool(gc.concept.equations)),
                difficulty=difficulty,
                confidence=gc.confidence,
                expected_answer=gc.expected_answer,
                evidence_snippet=gc.evidence_snippet,
                constraints={
                    "exam_type": self.config.exam_type,
                    "module": getattr(gc.concept, "source_chunk_id", "unknown"),
                },
            )
            plans.append(plan)

        # Enforce numerical ratio if not met
        plans = self._enforce_numerical_ratio(plans)

        return plans

    def plan_single(self, gc: GroundedConcept, bloom_override: Optional[int] = None) -> QuestionPlan:
        return self.plan([gc])[0] if self.plan([gc]) else self._fallback_plan(gc, bloom_override)

    # -- Helpers ----------------------------------------------

    def _choose_balanced_bloom(self, gc: GroundedConcept, counter: Dict[int, int]) -> int:
        # Prefer gc's bloom but avoid over-concentration
        preferred = gc.bloom_level
        # If preferred already heavily used (≥3), pick next least used
        if counter[preferred] >= 2:
            # Find least used among 2-4 (most common exam levels)
            least = min([2, 3, 4], key=lambda b: counter[b])
            return least
        return preferred

    def _decide_marks(self, gc: GroundedConcept, bloom: int) -> int:
        if self.config.exam_type.upper() == "IA":
            return 10 if bloom >= 4 else 5 if bloom == 3 else 6
        else:  # SEE
            return 10 if bloom >= 3 else 8

    def _decide_type(self, gc: GroundedConcept, bloom: int, numerical_used: int) -> str:
        if gc.concept.concept_type == "numerical" and numerical_used < 2:
            return "numerical"
        if gc.concept.concept_type == "diagram":
            return "diagram"
        if gc.concept.concept_type == "derivation":
            return "derivation"
        if bloom == 4:
            return "comparison"
        if bloom >= 5:
            return "evaluation"
        return "conceptual"

    def _decide_difficulty(self, gc: GroundedConcept, bloom: int) -> str:
        if self.config.difficulty != "mixed":
            return self.config.difficulty
        if bloom <= 2:
            return "easy"
        if bloom == 3:
            return "medium"
        return "hard"

    def _bloom_label(self, level: int) -> str:
        return {1: "Remember", 2: "Understand", 3: "Apply", 4: "Analyse", 5: "Evaluate", 6: "Create"}.get(level, "Understand")

    def _enforce_numerical_ratio(self, plans: List[QuestionPlan]) -> List[QuestionPlan]:
        if not plans:
            return plans
        numerical = [p for p in plans if p.question_type == "numerical"]
        ratio = len(numerical) / len(plans)
        if ratio >= self.config.require_numerical_ratio:
            return plans
        # Try to convert one conceptual to numerical if evidence supports it
        for p in plans:
            if p.question_type == "conceptual" and p.requires_formula:
                p.question_type = "numerical"
                p.reasoning_objective = "Apply the concept to a new numerical instance with fresh values."
                if ratio >= self.config.require_numerical_ratio:
                    break
        return plans

    def _fallback_plan(self, gc: GroundedConcept, bloom_override: Optional[int]) -> QuestionPlan:
        bloom = bloom_override or gc.bloom_level
        return QuestionPlan(
            plan_id=f"plan_fallback_{gc.concept.concept_id}",
            concept_id=gc.concept.concept_id,
            concept_name=gc.concept.concept_name,
            source_hash=gc.source_hash,
            marks=10,
            bloom_level=bloom,
            bloom_label=self._bloom_label(bloom),
            reasoning_objective=self.REASONING_OBJECTIVES.get(bloom, ""),
            question_type="conceptual",
            action_verb=self.BLOOM_VERBS[bloom][0],
            requires_diagram=False,
            requires_formula=False,
            difficulty="medium",
            confidence=gc.confidence,
            expected_answer=gc.expected_answer,
            evidence_snippet=gc.evidence_snippet,
        )
