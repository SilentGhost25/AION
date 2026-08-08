"""
AION v2 Modular Academic Validator
==================================
Evaluates text chunks across 6 modular academic dimensions:
  1. Language  — English academic vocabulary & sentence structure
  2. Formula   — Mathematical equations, symbols, LaTeX, numerical expressions
  3. Table     — Tabular structured text & tabular alignment
  4. OCR       — Character clean ratio (rejects garbled OCR/binary noise)
  5. Diagram   — Diagram / figure references (Fig., Block Diagram, Schematic)
  6. Noise     — Rejects footers, watermarks, ISBNs, ads, index lists, boilerplate

Production-safe. Zero laptop-specific code.
"""

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Any, List


# Academic vocabulary triggers
ACADEMIC_TERMS = {
    "definition", "theorem", "equation", "formula", "proof", "derive", "calculate",
    "algorithm", "system", "architecture", "analysis", "module", "component",
    "principle", "property", "function", "variable", "parameter", "output", "input",
    "frequency", "signal", "voltage", "current", "matrix", "vector", "state", "diagram"
}

# Clutter / Non-Academic triggers
CLUTTER_PATTERNS = [
    re.compile(r"isbn[:\s]*[0-9\-]{10,17}", re.IGNORECASE),
    re.compile(r"copyright\s+©?\s*\d{4}", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"page\s+\d+\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"downloaded\s+from", re.IGNORECASE),
    re.compile(r"watermark", re.IGNORECASE),
    re.compile(r"www\.[a-z0-9\-]+\.[a-z]{2,}", re.IGNORECASE),
    re.compile(r"table\s+of\s+contents", re.IGNORECASE),
    re.compile(r"\[\s*ad\s*\]|advertisement", re.IGNORECASE),
]


@dataclass
class AcademicValidationResult:
    valid:           bool
    academic_score:  float                 # 0.0 to 1.0 overall score
    noise_score:     float                 # 0.0 = clean, 1.0 = total noise
    scores:          Dict[str, float]      # Modular breakdown
    rejection_reason: str = ""
    rejection_codes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid":           self.valid,
            "academic_score":  round(self.academic_score, 3),
            "noise_score":     round(self.noise_score, 3),
            "scores":          {k: round(v, 3) for k, v in self.scores.items()},
            "rejection_reason": self.rejection_reason,
            "rejection_codes": self.rejection_codes,
        }


def _check_language_score(text: str) -> float:
    """Score 1: Language quality & academic term density."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    if not words:
        return 0.0
    academic_hits = sum(1 for w in words if w in ACADEMIC_TERMS)
    ratio = academic_hits / len(words)
    return min(1.0, ratio * 15.0)  # Standard academic density scaling


def _check_formula_score(text: str) -> float:
    """Score 2: Formula & mathematical expression presence."""
    math_symbols = len(re.findall(r"[=+\-*/^√∫∑∏α-ωΑ-Ω∈∉⊂⊃⊆⊇]", text))
    latex_hits   = len(re.findall(r"\\(?:frac|sum|int|sqrt|alpha|beta|theta|partial)", text))
    numbers      = len(re.findall(r"\b\d+(?:\.\d+)?\b", text))
    if len(text) == 0:
        return 0.0
    score = (math_symbols * 2 + latex_hits * 4 + numbers * 0.5) / (len(text) / 100.0 + 1)
    return min(1.0, score)


def _check_table_score(text: str) -> float:
    """Score 3: Tabular alignment & table structure."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    table_lines = sum(1 for l in lines if "|" in l or "\t" in l or re.search(r"\s{3,}", l))
    return min(1.0, table_lines / len(lines))


def _check_ocr_score(text: str) -> float:
    """Score 4: OCR character printable clean ratio."""
    if not text:
        return 0.0
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    non_ascii = sum(1 for c in text if ord(c) > 127 and unicodedata.category(c).startswith("C"))
    clean_ratio = (printable - non_ascii) / len(text)
    return max(0.0, min(1.0, clean_ratio))


def _check_diagram_score(text: str) -> float:
    """Score 5: Diagram / Figure reference presence."""
    fig_hits = len(re.findall(r"\b(?:fig(?:ure)?|block diagram|circuit|schematic|architecture)\b", text, re.IGNORECASE))
    return min(1.0, fig_hits * 0.33)


def _check_noise_score(text: str) -> float:
    """Score 6: Noise & clutter detection (0.0 = clean, 1.0 = heavy clutter)."""
    if not text:
        return 1.0
    hits = sum(len(p.findall(text)) for p in CLUTTER_PATTERNS)
    # Check for excessive unprintable / control characters
    unprintable = sum(1 for c in text if not c.isprintable() and c not in "\n\r\t")
    noise = (hits * 0.2) + (unprintable / max(1, len(text)))
    return min(1.0, noise)


def validate_academic_quality(
    text: str,
    min_printable_ratio: float = 0.70,
    max_noise_score: float = 0.40,
) -> AcademicValidationResult:
    """
    Modular Academic Quality Validator.
    Evaluates text across 6 dimensions and returns structured result.
    """
    rejection_codes = []
    reason_parts    = []

    if not text or len(text.strip()) < 20:
        return AcademicValidationResult(
            valid=False,
            academic_score=0.0,
            noise_score=1.0,
            scores={"language": 0, "formula": 0, "table": 0, "ocr": 0, "diagram": 0, "noise": 1.0},
            rejection_reason="Text is empty or too short (< 20 chars)",
            rejection_codes=["ERR_EMPTY_TEXT"],
        )

    # 1. Modular Sub-Scores
    score_lang    = _check_language_score(text)
    score_formula = _check_formula_score(text)
    score_table   = _check_table_score(text)
    score_ocr     = _check_ocr_score(text)
    score_diagram = _check_diagram_score(text)
    score_noise   = _check_noise_score(text)

    scores = {
        "language": score_lang,
        "formula":  score_formula,
        "table":    score_table,
        "ocr":      score_ocr,
        "diagram":  score_diagram,
        "noise":    score_noise,
    }

    # 2. Rejection checks
    if score_ocr < min_printable_ratio:
        rejection_codes.append("ERR_OCR_GARBAGE")
        reason_parts.append(f"Printable OCR ratio ({score_ocr:.2f}) below minimum threshold ({min_printable_ratio:.2f})")

    if score_noise > max_noise_score:
        rejection_codes.append("ERR_CLUTTER_NOISE")
        reason_parts.append(f"Clutter noise score ({score_noise:.2f}) exceeds maximum threshold ({max_noise_score:.2f})")

    overall_academic = min(1.0, (score_lang * 0.4) + (score_ocr * 0.3) + (score_formula * 0.2) + (score_diagram * 0.1))
    valid = len(rejection_codes) == 0

    return AcademicValidationResult(
        valid            = valid,
        academic_score   = overall_academic,
        noise_score      = score_noise,
        scores           = scores,
        rejection_reason = " | ".join(reason_parts),
        rejection_codes  = rejection_codes,
    )
