"""
AION Core Evidence — Chunk-Level Unicode Integrity Gate
=========================================================
Enforces INV-2: Detects \ufffd replacement characters and null bytes at the chunk level.
Exempts valid mathematical and Greek Unicode characters (π, Ω, Σ, ∫, ∂, ∇, ≤, ≥, ∞).
Quarantines ONLY corrupted chunks without dropping good page content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List
from .taxonomy import EvidenceType

EXEMPT_UNICODE_RANGES = [
    (0x0370, 0x03FF),   # Greek and Coptic (α β γ Ω π etc.)
    (0x2200, 0x22FF),   # Mathematical Operators (∫ ∑ ∏ √ ∞ ∈ etc.)
    (0x2100, 0x214F),   # Letterlike Symbols (ℝ ℂ ℕ ℤ)
    (0x2190, 0x21FF),   # Arrows (-> ← ↔ ⇒)
    (0x00B0, 0x00B0),   # Degree sign °
    (0x00B1, 0x00B1),   # Plus-minus ±
    (0x00D7, 0x00D7),   # Multiplication ×
    (0x00F7, 0x00F7),   # Division ÷
    (0x2070, 0x209F),   # Superscripts and Subscripts
]


def is_math_unicode_exempt(code_point: int) -> bool:
    """Check if a Unicode code point belongs to an exempt math or Greek range."""
    for start, end in EXEMPT_UNICODE_RANGES:
        if start <= code_point <= end:
            return True
    return False


@dataclass
class UnicodeReport:
    clean             : bool
    replacement_chars : int = 0
    control_chars     : int = 0
    unicode_integrity : float = 1.0
    violations        : List[Dict[str, Any]] = field(default_factory=list)
    evidence_type     : EvidenceType = EvidenceType.TEXT_PROSE


class UnicodeIntegrityGate:
    """Chunk-level Unicode integrity scanner."""

    @classmethod
    def check(cls, text: str) -> UnicodeReport:
        if not text:
            return UnicodeReport(clean=True, replacement_chars=0, control_chars=0, unicode_integrity=1.0)

        violations: List[Dict[str, Any]] = []
        replacement_count = 0
        control_count = 0

        for i, char in enumerate(text):
            cp = ord(char)

            # Hard violation: Replacement character
            if char in ("\ufffd", "\x00", "\ufffe", "\uffff"):
                replacement_count += 1
                violations.append({
                    "position": i,
                    "char": repr(char),
                    "code_point": hex(cp),
                    "type": "REPLACEMENT_CHAR" if char == "\ufffd" else "NULL_BYTE",
                })

            # Control characters (excluding math exempt and whitespace)
            elif cp < 32 and char not in ("\n", "\r", "\t", "\x0b", "\x0c"):
                if not is_math_unicode_exempt(cp):
                    control_count += 1

        is_clean = replacement_count == 0 and control_count == 0
        tot = max(len(text), 1)
        integrity = 1.0 if replacement_count == 0 else max(0.0, 1.0 - (replacement_count / tot) * 10.0)

        evidence_type = EvidenceType.UNICODE_CORRUPT if not is_clean else EvidenceType.TEXT_PROSE

        return UnicodeReport(
            clean=is_clean,
            replacement_chars=replacement_count,
            control_chars=control_count,
            unicode_integrity=integrity,
            violations=violations,
            evidence_type=evidence_type,
        )
