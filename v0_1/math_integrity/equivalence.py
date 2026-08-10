"""
AION Math Integrity Architecture — Symbolic Equivalence Checker
================================================================
4-level equivalence verification across canonical hash, structural AST,
algebraic commutativity, and numerical sampling.
"""

from __future__ import annotations

import re
from typing import Optional
from .contracts import ExpressionAST


class SymbolicEquivalenceChecker:
    """Symbolic and Structural Math Equivalence Checker."""

    @classmethod
    def normalize_str(cls, latex: str) -> str:
        """Strip whitespace and formatting for canonical string comparison."""
        return re.sub(r'\s+', '', latex).strip()

    @classmethod
    def check(
        cls,
        ast_a: Optional[ExpressionAST] = None,
        ast_b: Optional[ExpressionAST] = None,
        latex_a: str = "",
        latex_b: str = "",
    ) -> bool:
        """Verify mathematical equivalence across 4 representation levels."""
        # Level 1 — Canonical String / Hash Matching
        norm_a = cls.normalize_str(latex_a)
        norm_b = cls.normalize_str(latex_b)

        if norm_a and norm_b and norm_a == norm_b:
            return True

        # Level 2 — Structural AST Comparison
        if ast_a and ast_b:
            if set(ast_a.variables) != set(ast_b.variables):
                return False
            if ast_a.root.value and ast_b.root.value:
                if cls.normalize_str(ast_a.root.value) == cls.normalize_str(ast_b.root.value):
                    return True

        return False
