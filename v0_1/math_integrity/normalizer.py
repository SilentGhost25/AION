"""
AION Math Integrity Architecture — Math Normalizer
===================================================
Converts raw mathematical text into a canonical MathArtifact with LaTeX as M2 authority.
"""

from __future__ import annotations

import hashlib
import re
from typing import List, Optional
from .contracts import (
    EquationType,
    ExpressionAST,
    MathArtifact,
    MathNode,
    MathSourceType,
    MathValidationStatus,
)
from .registry import MATH_SYMBOL_REGISTRY


class MathNormalizer:
    """Canonical Math Normalizer establishing LaTeX as authoritative M2 representation."""

    @classmethod
    def convert_unicode_to_latex(cls, text: str) -> str:
        """Convert Unicode math symbols and sub/superscripts to canonical LaTeX."""
        result = text
        for symbol in sorted(MATH_SYMBOL_REGISTRY.values(), key=lambda s: len(s.unicode), reverse=True):
            if symbol.unicode in result and symbol.unicode != "":
                result = result.replace(symbol.unicode, symbol.latex)

        # Convert subscripts: ₀₁₂₃₄₅₆₇₈₉ -> _0_1...
        sub_map = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
        sup_map = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")

        for ch in "₀₁₂₃₄₅₆₇₈₉":
            if ch in result:
                result = result.replace(ch, f"_{ch.translate(sub_map)}")

        for ch in "⁰¹²³⁴⁵⁶⁷⁸⁹":
            if ch in result:
                result = result.replace(ch, f"^{ch.translate(sup_map)}")

        return result

    @classmethod
    def check_delimiter_balance(cls, latex: str) -> bool:
        """Verify structural balance of brackets and delimiters."""
        stack: List[str] = []
        pairs = {"{": "}", "(": ")", "[": "]"}
        i = 0
        while i < len(latex):
            char = latex[i]
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                opener = stack.pop()
                if pairs.get(opener) != char:
                    return False
            i += 1
        return len(stack) == 0

    @classmethod
    def build_ast(cls, normalized_latex: str) -> ExpressionAST:
        """Construct a lightweight ExpressionAST for symbolic equivalence."""
        tokens = re.findall(r'\\[a-zA-Z]+|[a-zA-Z0-9]+|[\+\-\*/=\(\)]', normalized_latex)
        variables = [t for t in tokens if t.isalpha() and len(t) == 1]
        operators = [t for t in tokens if t in ("+", "-", "*", "/", "=") or t.startswith("\\")]

        root = MathNode(
            node_type="expression",
            value=normalized_latex,
            children=[MathNode(node_type="token", value=t) for t in tokens],
        )

        return ExpressionAST(
            root=root,
            variables=list(set(variables)),
            constants=[],
            operators=list(set(operators)),
            complexity=len(tokens),
        )

    @classmethod
    def normalize(
        cls,
        raw_text: str,
        math_id: str,
        source_type: Optional[MathSourceType] = None,
        document_id: str = "doc_001",
        page: int = 1,
    ) -> MathArtifact:
        """Convert raw mathematical input into canonical MathArtifact."""
        clean_raw = raw_text.strip()

        # Step 1 — Detect Source Format
        if r"\frac" in clean_raw or r"\int" in clean_raw or r"\sum" in clean_raw or r"\begin" in clean_raw:
            detected_source = MathSourceType.NATIVE_LATEX
            latex = clean_raw
        elif any(s.unicode in clean_raw for s in MATH_SYMBOL_REGISTRY.values()):
            detected_source = MathSourceType.UNICODE_TEXT
            latex = cls.convert_unicode_to_latex(clean_raw)
        else:
            detected_source = source_type or MathSourceType.INLINE_TEXT
            latex = clean_raw

        # Step 2 — Equation Type Detection
        if clean_raw.startswith(r"\[") or clean_raw.startswith("$$") or clean_raw.startswith(r"\begin"):
            eq_type = EquationType.DISPLAY
        else:
            eq_type = EquationType.INLINE

        # Step 3 — Delimiter Balance Check & Normalization
        is_balanced = cls.check_delimiter_balance(latex)
        status = MathValidationStatus.VALID if is_balanced else MathValidationStatus.TRUNCATED

        normalized_latex = re.sub(r'\s+', ' ', latex).strip()

        # Step 4 — AST Construction
        ast = cls.build_ast(normalized_latex)

        # Step 5 — Compute SHA256 Hashes
        source_hash = hashlib.sha256(clean_raw.encode("utf-8")).hexdigest()
        canonical_hash = hashlib.sha256(normalized_latex.encode("utf-8")).hexdigest()

        return MathArtifact(
            math_id=math_id,
            latex=latex,
            normalized_latex=normalized_latex,
            ast=ast,
            unicode_text=clean_raw,
            source_type=detected_source,
            source_text=clean_raw,
            equation_type=eq_type,
            document_id=document_id,
            page=page,
            validation_status=status,
            source_hash=source_hash,
            canonical_hash=canonical_hash,
            placeholder=f"[MATH:{math_id}]",
        )
