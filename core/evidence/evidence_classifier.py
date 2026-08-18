"""
AION Core Evidence — Evidence Classifier
==========================================
Classifies extracted text blocks and evidence chunks according to the EvidenceType taxonomy.
Enforces hard disqualifiers (PDF_METADATA, UNICODE_CORRUPT, BINARY_STREAM) before retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict
from .pdf_internals_detector import detect_pdf_internals
from .taxonomy import EvidenceType
from .unicode_gate import UnicodeIntegrityGate


class EvidenceClassifier:
    """Classifies extracted evidence chunks."""

    @classmethod
    def classify(cls, text: str, meta: Dict[str, Any] = None) -> EvidenceType:
        if not text or len(text.strip()) < 10:
            return EvidenceType.BLANK

        meta = meta or {}

        # -- PRIORITY 1: HARD DISQUALIFIERS ------------------------------------

        # PDF Internals check (INV-1)
        pdf_report = detect_pdf_internals(text)
        if pdf_report.has_internals:
            return EvidenceType.PDF_METADATA

        # Unicode Integrity check (INV-2)
        unicode_report = UnicodeIntegrityGate.check(text)
        if not unicode_report.clean:
            return EvidenceType.UNICODE_CORRUPT

        # Binary check
        if "\x00" in text:
            return EvidenceType.BINARY_STREAM

        # -- PRIORITY 2: STRUCTURAL EXCLUSIONS ----------------------------------

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        first_line = lines[0].lower() if lines else ""

        if re.search(r"\b(references|bibliography|works cited)\b", first_line):
            return EvidenceType.REFERENCE_LIST

        if re.search(r"\b(table of contents|contents)\b", first_line):
            return EvidenceType.TOC

        if len(text.strip()) < 80 and re.search(r"^\d+\s*\|\s*page", first_line):
            return EvidenceType.HEADER_FOOTER

        # -- PRIORITY 3: ACADEMIC CONTENT CLASSIFICATION ----------------------

        # Equation block
        if meta.get("equation_ids") or "LaTeX" in meta or re.search(r"\\(begin|frac|sum|int|sqrt)", text):
            return EvidenceType.EQUATION

        # Table data
        if meta.get("table_ids") or (text.count("|") > 4 and text.count("\n") > 2):
            return EvidenceType.TABLE_DATA

        # Figure diagram
        if meta.get("figure_ids") or meta.get("content_type") == "FIGURE":
            return EvidenceType.FIGURE_DIAGRAM

        # Code block
        if "```" in text or re.search(r"\b(def |class |function |return |void |int main)\b", text):
            return EvidenceType.TEXT_CODE

        # Theorem / Lemma / Proof
        if re.search(r"^(theorem|lemma|corollary|proof|proposition)\b", text, re.IGNORECASE):
            return EvidenceType.TEXT_THEOREM

        # Definition
        if re.search(r"^(definition|defined as|refers to)\b", text, re.IGNORECASE):
            return EvidenceType.TEXT_DEFINITION

        # Worked Example
        if re.search(r"^(example|problem|solution)\b", text, re.IGNORECASE):
            return EvidenceType.TEXT_EXAMPLE

        # List / Enumeration
        if re.search(r"^\s*([1-9]\.|\*|-|\([a-z]\))\s+", text, re.MULTILINE):
            return EvidenceType.LIST_ENUMERATION

        return EvidenceType.TEXT_PROSE
