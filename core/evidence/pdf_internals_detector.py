"""
AION Core Evidence — Weighted PDF Internals Detector
=====================================================
Enforces INV-1: Scans extracted text for PDF internal object markers
using weighted pattern scoring to prevent PDF metadata leakage without
false positives on legitimate academic text (e.g. the word 'object').
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from .taxonomy import EvidenceType


@dataclass
class PDFInternalReport:
    has_internals       : bool
    matched_patterns    : List[Dict[str, Any]] = field(default_factory=list)
    total_score         : int = 0
    contamination_ratio : float = 0.0
    evidence_type       : Optional[EvidenceType] = None


# Weighted pattern definitions: (pattern_regex, weight)
WEIGHTED_PDF_PATTERNS = [
    # High-confidence PDF structure markers (Weight = 5)
    (r"/FontFile\d*", 5),
    (r"/ToUnicode", 5),
    (r"/FlateDecode", 5),
    (r"/CIDSystemInfo", 5),
    (r"/Subtype\s*/CIDFont", 5),
    (r"/Subtype\s*/Type[0-9]", 5),

    # Medium-high structural operators (Weight = 4)
    (r"endobj", 4),
    (r"endstream", 4),
    (r"\bstream\b", 4),
    (r"\bxref\b", 4),
    (r"startxref", 4),
    (r"/Type\s*/Font", 4),
    (r"/Type\s*/Page", 4),
    (r"%%EOF", 4),

    # Document container dictionaries (Weight = 3)
    (r"/Contents\s+\d+", 3),
    (r"/XObject", 3),
    (r"/Resources", 3),
    (r"/MediaBox", 3),
    (r"/CropBox", 3),
    (r"/Widths\s*\[", 3),

    # Low-weight contextual words (Weight = 1, insufficient alone)
    (r"\b\d+\s+\d+\s+obj\b", 2),
]


def detect_pdf_internals(text: str) -> PDFInternalReport:
    """
    Scan text for PDF internal structure markers using weighted scoring.
    Score >= 5 triggers PDF_METADATA classification and quarantine.
    """
    if not text or not text.strip():
        return PDFInternalReport(has_internals=False, total_score=0)

    matched: List[Dict[str, Any]] = []
    total_score = 0

    for pattern_str, weight in WEIGHTED_PDF_PATTERNS:
        match = re.search(pattern_str, text, re.IGNORECASE)
        if match:
            total_score += weight
            matched.append({
                "pattern": pattern_str,
                "weight": weight,
                "snippet": match.group()[:60],
            })

    # Threshold for PDF internal classification
    has_internals = total_score >= 5

    contamination_ratio = min(1.0, (total_score * 20.0) / max(len(text), 1)) if has_internals else 0.0
    evidence_type = EvidenceType.PDF_METADATA if has_internals else None

    return PDFInternalReport(
        has_internals=has_internals,
        matched_patterns=matched,
        total_score=total_score,
        contamination_ratio=contamination_ratio,
        evidence_type=evidence_type,
    )
