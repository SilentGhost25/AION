"""
AION VRE Quantity Parser & Symbol Normalizer
============================================
OCR Raw Text -> Symbol Normalizer -> Quantity & Unit Parser -> Validation.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from .contracts import QuantityType


class QuantityParser:
    """Parses numeric values, normalizes OCR symbols (e.g. 10O -> 10Ω), and extracts units."""

    # OCR symbol replacements
    SYMBOL_MAP = {
        r"(\d+)\s*O\b": r"\1 Ω",
        r"(\d+)\s*ohm(s)?\b": r"\1 Ω",
        r"(\d+)\s*kO\b": r"\1 kΩ",
        r"(\d+)\s*v\b": r"\1 V",
        r"(\d+)\s*a\b": r"\1 A",
        r"(\d+)\s*kn\b": r"\1 kN",
        r"(\d+)\s*m\b": r"\1 m",
    }

    @classmethod
    def normalize_symbol(cls, raw_text: str) -> str:
        text = raw_text.strip()
        for pattern, replacement in cls.SYMBOL_MAP.items():
            text = re.sub(pattern, replacement, text, flags=re.I)
        return text

    @classmethod
    def parse_quantity(
        cls,
        raw_text: str,
        expected_type: QuantityType = QuantityType.GENERIC,
    ) -> Tuple[Optional[float], str, float]:
        """
        Parses value, unit, and confidence from text.
        Returns: (numeric_value, unit_string, confidence_score)
        """
        normalized = cls.normalize_symbol(raw_text)

        # Regex for number + optional unit
        match = re.search(r"([-+]?\d*\.?\d+)\s*([a-zA-ZΩμkM]*)" , normalized)
        if not match:
            return (None, "", 0.0)

        val_str, unit_str = match.group(1), match.group(2).strip()

        try:
            val = float(val_str)
        except ValueError:
            return (None, "", 0.0)

        # Confidence calculation
        confidence = 0.90
        if raw_text != normalized:
            confidence = 0.82  # Symbol was corrected

        if expected_type == QuantityType.RESISTANCE and ("Ω" in unit_str or "kΩ" in unit_str):
            confidence += 0.08

        return (val, unit_str, round(min(1.0, confidence), 2))
