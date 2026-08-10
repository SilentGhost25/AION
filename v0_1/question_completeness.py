"""
AION Question Completeness & Output Integrity Validator
========================================================
Detects and rejects truncated LLM outputs, dangling operators, incomplete equations,
or invalid sentence terminations before questions reach the paper assembler/renderer.
"""

from __future__ import annotations

import re
from typing import List, Tuple


class QuestionCompletenessValidator:
    """Validates structural completeness and sentence integrity of generated questions."""

    TRAILING_OPERATORS = (
        " and", " or", " with", " where", " equals", " is", "=", "+", "-", "*", "/", ",", ":"
    )

    @classmethod
    def validate(cls, text: str) -> Tuple[bool, List[str]]:
        errors = []
        clean = text.strip()

        if len(clean) < 15:
            errors.append("QUESTION_TRUNCATED_TOO_SHORT")
            return False, errors

        # 1. Trailing Operator / Cutoff Check
        low = clean.lower()
        for op in cls.TRAILING_OPERATORS:
            if low.endswith(op):
                errors.append(f"QUESTION_TRUNCATED_TRAILING_OPERATOR:{op}")
                break

        # 2. Sentence Termination Check
        last_char = clean[-1]
        if last_char not in (".", "?", "!", ":", ")", "}", "]"):
            errors.append(f"QUESTION_UNTERMINATED_SENTENCE:last_char='{last_char}'")

        # 3. Delimiter Balance Check
        if clean.count("(") != clean.count(")"):
            errors.append("QUESTION_UNBALANCED_PARENTHESES")
        if clean.count("[") != clean.count("]"):
            errors.append("QUESTION_UNBALANCED_BRACKETS")
        if clean.count("{") != clean.count("}"):
            errors.append("QUESTION_UNBALANCED_BRACES")
        if clean.count("$") % 2 != 0:
            errors.append("QUESTION_UNBALANCED_MATH_DELIMITER")

        # 4. Incomplete Equation Check (e.g., "R1 = 10Ω and R2 =")
        if re.search(r"=\s*$", clean):
            errors.append("QUESTION_INCOMPLETE_EQUATION")

        return (len(errors) == 0, errors)
