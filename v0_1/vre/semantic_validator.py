"""
AION VRE Semantic Question Validator
====================================
Validates post-LLM generated question text to ensure it matches
the QuestionPlan's expected operation and contains zero descriptive-only phrasing.
"""

from __future__ import annotations

import re
from typing import List, Tuple
from .contracts import QuestionPlan


class SemanticQuestionValidator:
    """Post-LLM Semantic Question Validator."""

    @classmethod
    def validate(cls, question_text: str, plan: QuestionPlan) -> Tuple[bool, List[str]]:
        errors = []
        low_text = question_text.lower()

        # 1. Operation Verification
        expected_op = plan.operation.lower()
        if expected_op == "dijkstra" and "dijkstra" not in low_text:
            errors.append("SEMANTIC_VALIDATOR_MISSING_EXPECTED_OPERATION:dijkstra")
        elif expected_op in ("kvl", "equivalent_resistance") and not any(w in low_text for w in ["circuit", "resistance", "kvl"]):
            errors.append("SEMANTIC_VALIDATOR_MISSING_EXPECTED_OPERATION:circuit")

        # 2. Entity Verification (Source / Destination) for Graph Vertices
        if plan.operation == "DIJKSTRA":
            src = plan.anchors.get("source", plan.source_element)
            if src and len(src) == 1 and src.lower() not in low_text:
                errors.append(f"SEMANTIC_VALIDATOR_MISSING_SOURCE_ENTITY:{src}")

        # 3. Descriptive Rejection Check
        descriptive_patterns = ["describe the figure", "explain the diagram", "what is shown in the image"]
        for p in descriptive_patterns:
            if p in low_text:
                errors.append(f"SEMANTIC_VALIDATOR_DESCRIPTIVE_PHRASE:{p}")

        return (len(errors) == 0, errors)
