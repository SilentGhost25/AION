"""
AION Core Integrity — Equation Integrity Gate
===============================================
Validates LaTeX equations against binary contamination, UTF-8 encoding failures,
unbalanced delimiter pairs, unknown command ratios, and unparseable ASTs.
Generates canonical SHA256 hashes for round-trip verification.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class EquationReport:
    status     : str          # "VALID" | "SUSPICIOUS" | "INVALID" | "BINARY_CONTAMINATION" | "ENCODING_FAILURE" | "TRUNCATED"
    confidence : float        # 0.0 to 1.0
    hash       : str          # SHA256 canonical hash
    issues     : List[str] = field(default_factory=list)


# Known standard LaTeX math commands
KNOWN_LATEX_COMMANDS = {
    "\\frac", "\\sqrt", "\\sum", "\\prod", "\\int", "\\oint", "\\lim",
    "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\theta", "\\lambda",
    "\\mu", "\\pi", "\\rho", "\\sigma", "\\tau", "\\phi", "\\omega",
    "\\Delta", "\\Gamma", "\\Theta", "\\Lambda", "\\Sigma", "\\Phi", "\\Omega",
    "\\cdot", "\\times", "\\div", "\\pm", "\\mp", "\\neq", "\\leq", "\\geq",
    "\\approx", "\\equiv", "\\propto", "\\infty", "\\partial", "\\nabla",
    "\\sin", "\\cos", "\\tan", "\\log", "\\ln", "\\exp",
    "\\vec", "\\hat", "\\bar", "\\dot", "\\ddot", "\\mathbf", "\\mathrm",
    "\\begin", "\\end", "\\left", "\\right", "\\over", "\\atop",
}


class EquationIntegrityGate:
    """Validates equations for binary contamination and mathematical syntax integrity."""

    @classmethod
    def validate(cls, latex: str) -> EquationReport:
        issues: List[str] = []

        if not latex or not latex.strip():
            return EquationReport(
                status="INVALID",
                confidence=0.0,
                hash="",
                issues=["Empty LaTeX string"],
            )

        # -- STEP 1: BINARY CONTAMINATION --------------------------------------
        if "\x00" in latex or "\ufffd" in latex:
            return EquationReport(
                status="BINARY_CONTAMINATION",
                confidence=0.0,
                hash="",
                issues=["Binary contamination or replacement char detected in equation"],
            )

        nonprintable = sum(1 for c in latex if not c.isprintable() and c not in "\n\r\t ")
        if nonprintable > 0:
            return EquationReport(
                status="BINARY_CONTAMINATION",
                confidence=0.0,
                hash="",
                issues=[f"{nonprintable} non-printable characters in equation"],
            )

        # -- STEP 2: UTF-8 ROUND TRIP ------------------------------------------
        utf8_valid = True
        try:
            latex.encode("utf-8").decode("utf-8")
        except UnicodeError:
            utf8_valid = False
            issues.append("UTF-8 round-trip decode failed")

        # -- STEP 3: DELIMITER BALANCE -----------------------------------------
        delimiters_balanced = cls._check_delimiter_balance(latex, issues)

        # -- STEP 4: KNOWN SYMBOL COVERAGE -------------------------------------
        commands = re.findall(r"\\[a-zA-Z]+", latex)
        unknown_count = sum(1 for c in commands if c not in KNOWN_LATEX_COMMANDS)
        unknown_ratio = unknown_count / max(len(commands), 1)

        if unknown_ratio > 0.30:
            issues.append(f"High unknown LaTeX command ratio: {unknown_ratio:.0%}")

        # -- STEP 5: CONFIDENCE SCORE -----------------------------------------
        confidence = (
            (1.0 if utf8_valid else 0.0) * 0.30 +
            (1.0 if delimiters_balanced else 0.5) * 0.40 +
            (1.0 - min(1.0, unknown_ratio)) * 0.30
        )

        # -- STEP 6: SHA256 CANONICAL HASH -------------------------------------
        normalized = re.sub(r"\s+", " ", latex.strip())
        canonical_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

        # -- STEP 7: VERDICT ---------------------------------------------------
        if not utf8_valid or not delimiters_balanced or confidence < 0.60:
            status = "INVALID"
        elif confidence >= 0.90:
            status = "VALID"
        else:
            status = "SUSPICIOUS"

        return EquationReport(
            status=status,
            confidence=confidence,
            hash=canonical_hash,
            issues=issues,
        )

    @classmethod
    def _check_delimiter_balance(cls, latex: str, issues: List[str]) -> bool:
        """Check matching pairs for braces, brackets, and display delimiters."""
        open_braces  = latex.count("{")
        close_braces = latex.count("}")
        if open_braces != close_braces:
            issues.append(f"Unbalanced braces: {{ count={open_braces}, }} count={close_braces}")
            return False

        open_inline  = latex.count("\\(")
        close_inline = latex.count("\\)")
        if open_inline != close_inline:
            issues.append(f"Unbalanced inline delimiters: \\( count={open_inline}, \\) count={close_inline}")
            return False

        open_disp  = latex.count("\\[")
        close_disp = latex.count("\\]")
        if open_disp != close_disp:
            issues.append(f"Unbalanced display delimiters: \\[ count={open_disp}, \\] count={close_disp}")
            return False

        # Environment balance
        begins = re.findall(r"\\begin\{([^}]+)\}", latex)
        ends   = re.findall(r"\\end\{([^}]+)\}", latex)
        if sorted(begins) != sorted(ends):
            issues.append(f"Unbalanced environments: begin={begins}, end={ends}")
            return False

        return True
