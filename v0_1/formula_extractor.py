"""
AION Module: Formula & Expression Extractor
Finds mathematical formulas, equations, and expressions
from document text and includes them in questions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Formula:
    id:           str
    raw:          str        # Original text
    latex:        str        # LaTeX if detected
    context:      str        # Surrounding text
    page:         int
    formula_type: str        # equation | expression | definition | algorithm


# ── Detection patterns ────────────────────────────────────────

_LATEX_PATTERN = re.compile(
    r"\$\$?.+?\$\$?|"                  # $...$ or $$...$$
    r"\\begin\{.+?\}.*?\\end\{.+?\}|"  # \begin{equation}...\end
    r"\\[a-zA-Z]+\{[^}]+\}",           # \frac{}{}, \sum{}, etc.
    re.DOTALL
)

_INLINE_MATH = re.compile(
    r"[A-Za-z]\s*[=<>≤≥≠±]\s*[^,\.\n]{2,40}|"   # x = expr
    r"\b\d+\s*[+\-×÷*/]\s*\d+\s*[=<>]\s*\d+|"   # arithmetic
    r"[A-Za-z]+\([A-Za-z,\s]+\)\s*=|"            # f(x) =
    r"[∑∫∂∇√∞±×÷≤≥≠∈∀∃]",                        # unicode math
    re.UNICODE
)

_ALGO_PATTERN = re.compile(
    r"(Algorithm|Pseudocode|Procedure|Function)\s*\d*\s*[:\-]?\s*\w+",
    re.I
)

_DEF_PATTERN = re.compile(
    r"(Definition|Theorem|Lemma|Corollary|Proposition)\s*\d*\s*[:\-]",
    re.I
)


def extract_formulas(text: str, page: int = 0) -> list[Formula]:
    """
    Extract all formulas and mathematical expressions from text.
    Returns list of Formula objects.
    """
    formulas = []
    seen     = set()

    def _add(raw: str, ftype: str, context: str = ""):
        raw = raw.strip()
        if not raw or raw in seen or len(raw) < 3:
            return
        seen.add(raw)

        latex = raw if _LATEX_PATTERN.search(raw) else ""

        fid = f"formula_{len(formulas) + 1:03d}_p{page}"
        formulas.append(Formula(
            id           = fid,
            raw          = raw,
            latex        = latex,
            context      = context[:200],
            page         = page,
            formula_type = ftype,
        ))

    for m in _LATEX_PATTERN.finditer(text):
        start   = max(0, m.start() - 100)
        context = text[start: m.end() + 100]
        _add(m.group(), "equation", context)

    for m in _INLINE_MATH.finditer(text):
        sent_start = text.rfind(".", 0, m.start()) + 1
        sent_end   = text.find(".", m.end()) + 1
        sentence   = text[sent_start:sent_end].strip()
        _add(sentence or m.group(), "expression", sentence)

    for m in _ALGO_PATTERN.finditer(text):
        sent_end = text.find("\n\n", m.start())
        block    = text[m.start(): sent_end if sent_end > 0 else m.end() + 200]
        _add(block[:300], "algorithm", block[:200])

    for m in _DEF_PATTERN.finditer(text):
        sent_end = text.find("\n\n", m.start())
        block    = text[m.start(): sent_end if sent_end > 0 else m.end() + 200]
        _add(block[:300], "definition", block[:200])

    return formulas


def find_formulas_in_chunk(chunk: str) -> list[Formula]:
    """Find formulas in a text chunk for use in question generation."""
    return extract_formulas(chunk, page=0)


def format_formula_for_prompt(formula: Formula) -> str:
    """Format a formula for inclusion in generation prompt."""
    if formula.latex:
        return f"Formula: {formula.latex}"
    return f"Expression: {formula.raw}"
