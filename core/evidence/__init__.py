"""
AION Core Evidence Subsystem
=============================
Provides evidence classification, weighted PDF internals detection,
chunk-level Unicode integrity analysis, and provenance tracking.
"""

from .taxonomy import EvidenceType, RETRIEVAL_ELIGIBLE_TYPES, QUARANTINE_TYPES, EXCLUDED_TYPES
from .pdf_internals_detector import detect_pdf_internals, PDFInternalReport
from .unicode_gate import UnicodeIntegrityGate, UnicodeReport
from .evidence_classifier import EvidenceClassifier

__all__ = [
    "EvidenceType",
    "RETRIEVAL_ELIGIBLE_TYPES",
    "QUARANTINE_TYPES",
    "EXCLUDED_TYPES",
    "detect_pdf_internals",
    "PDFInternalReport",
    "UnicodeIntegrityGate",
    "UnicodeReport",
    "EvidenceClassifier",
]
