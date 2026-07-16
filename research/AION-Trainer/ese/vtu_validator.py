# AION-Trainer/ese/vtu_validator.py
"""
VTU Validator — enforces VTU-specific drafting rules and cognitive verb
alignments on the generated question texts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from ese.question_ranker import BLOOM_VERBS


@dataclass
class VTUValidationIssue:
    rule_name: str
    message: str
    severity: str  # "error" | "warning"


class VTUValidator:
    def __init__(self):
        pass

    def validate(
        self,
        text: str,
        bloom_level: str,
        marks: int,
        diagram_required: bool,
    ) -> List[VTUValidationIssue]:
        issues = []
        text_lower = text.lower()

        # 1. Action Verb Rule
        bloom_verbs = BLOOM_VERBS.get(bloom_level, [])
        first_word = text.strip().split()[0].lower().rstrip(",:.") if text.strip() else ""
        
        # Check if first word is a valid verb for this Bloom level
        if bloom_verbs and first_word not in bloom_verbs:
            issues.append(VTUValidationIssue(
                rule_name="bloom_verb_misalignment",
                message=(
                    f"Action verb '{first_word}' does not align with Bloom Level {bloom_level}. "
                    f"Expected one of: {bloom_verbs}"
                ),
                severity="warning"
            ))

        # 2. VTU Marks Values Rule
        valid_marks = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20]
        if marks not in valid_marks:
            issues.append(VTUValidationIssue(
                rule_name="invalid_vtu_marks",
                message=f"Marks value '{marks}' is atypical for VTU. Expected standard marks like: {valid_marks}",
                severity="warning"
            ))

        # 3. Diagram Requirement for High Marks
        if diagram_required and marks >= 8:
            if not any(w in text_lower for w in ["diagram", "sketch", "figure", "draw"]):
                issues.append(VTUValidationIssue(
                    rule_name="missing_diagram_instruction",
                    message="Concept requires diagram and marks >= 8, but no 'diagram' or 'sketch' instruction is in the question text.",
                    severity="error"
                ))

        # 4. Example Requirement for High Marks (if no diagram is requested)
        if marks >= 10 and not diagram_required:
            if not any(w in text_lower for w in ["example", "illustrate", "scenario", "numerical", "case study"]):
                issues.append(VTUValidationIssue(
                    rule_name="missing_example_instruction",
                    message="Question has high marks (>= 10) but does not ask for a 'suitable example' or 'worked illustration'.",
                    severity="warning"
                ))

        # 5. Tone and Formality Checks
        if "please" in text_lower:
            issues.append(VTUValidationIssue(
                rule_name="excessive_politeness",
                message="VTU question papers do not use polite phrasing like 'please'. Use direct command verbs.",
                severity="error"
            ))

        return issues
