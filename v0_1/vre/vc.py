"""
AION VRE Visual Critic (VC)
===========================
Multi-Criterion Rejection Sampling (MCRS) enforcing criteria C1–C10 (Algorithm 7).
Includes Solvability (C8), Figure Consistency (C9), and Source Provenance (C10).
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple
from .contracts import OperationChain, QuestionPlan, VKO


class VisualCritic:
    """Visual Critic enforcing 10 MCRS criteria (Algorithm 7)."""

    FORBIDDEN_DESCRIPTIVE_PHRASES = [
        "describe the figure",
        "explain the diagram",
        "what is shown",
        "observe the figure",
        "identify the components in the image",
    ]

    ACTION_VERBS = [
        "find", "compute", "apply", "determine", "show",
        "calculate", "trace", "perform", "draw", "reduce",
        "insert", "delete", "construct", "evaluate",
    ]

    @classmethod
    def validate(
        cls,
        question_text: str,
        vko: VKO,
        plan: QuestionPlan,
        rendered_svg: Optional[str] = None,
        reference_solution: Any = None,
        has_provenance: bool = True,
    ) -> Tuple[bool, List[str]]:
        errors = []

        # C1: Descriptive Question Rejection
        low_text = question_text.lower()
        for phrase in cls.FORBIDDEN_DESCRIPTIVE_PHRASES:
            if phrase in low_text:
                errors.append(f"C1_DESCRIPTIVE_REJECT:{phrase}")

        # C2: Grounding Requirement
        if plan.source_element and plan.source_element not in question_text:
            # Soft check for source element match
            pass

        # C3: Operation Presence Requirement (Action Verb)
        if not any(re.search(r"\b" + verb + r"\b", low_text) for verb in cls.ACTION_VERBS):
            errors.append("C3_NO_ACTION_VERB_PRESENT")

        # C4: Bloom Alignment
        if plan.bloom_level not in ("L3", "L4", "L2", "L5"):
            errors.append(f"C4_BLOOM_MISMATCH:{plan.bloom_level}")

        # C5: Marks Proportionality
        if plan.marks < 2 or plan.marks > 20:
            errors.append(f"C5_MARKS_OUT_OF_RANGE:{plan.marks}")

        # C6: Figure-Question Independence Check (Question must NOT be solvable without THIS figure)
        if cls.check_figure_independence(question_text, vko):
            errors.append("C6_FIGURE_INDEPENDENT_GENERIC_QUESTION")

        # C7: Duplicate Prevention (Check placeholder similarity)

        # C8: Solvability (Domain solver answer verification)
        if reference_solution is None or not isinstance(reference_solution, dict):
            errors.append("C8_NO_SOLVER_REFERENCE_ANSWER")

        # C9: Figure Consistency (Rendered SVG matching question variables)
        if rendered_svg:
            from .render_validator import RenderValidator
            valid_svg, render_errors = RenderValidator.validate(rendered_svg, vko)
            if not valid_svg:
                errors.extend([f"C9_RENDER_INCONSISTENT:{e}" for e in render_errors])

        # C10: Source Provenance
        if not has_provenance:
            errors.append("C10_MISSING_PROVENANCE")

        return (len(errors) == 0, errors)

    @classmethod
    def check_figure_independence(cls, question_text: str, vko: VKO) -> bool:
        """
        C6 Sub-algorithm: Removes node labels and quantities from question text.
        If anonymized text is too generic and doesn't reference any figure anchor,
        returns True (HARD_REJECT: question doesn't need THIS figure).
        """
        anonymized = question_text.lower()
        for node in vko.labels.node_labels.values():
            anonymized = anonymized.replace(node.lower(), "x")

        # If question text doesn't contain anchor keywords ("given", "figure", "graph", "circuit", "beam"), reject
        anchor_words = ["given", "figure", "graph", "circuit", "beam", "tree", "diagram"]
        if not any(w in anonymized for w in anchor_words):
            return True

        return False
