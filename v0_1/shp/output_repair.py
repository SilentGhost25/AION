"""
AION SHP Stage 5 — Output Repair
==================================
Repairs generated questions before they reach the renderer.
One repair pass maximum. Never silently accepts bad output.
"""

from __future__ import annotations

import re
from typing import Optional

from .error_knowledge import ErrorKnowledgeBase, Severity

PDF_QUESTION_KEYWORDS = frozenset({
    "xref", "stream", "endstream", "endobj", "obj",
    "/kids", "/parent", "/contents", "/mediabox", "/font",
    "pdf structure", "pdf object", "pdf document",
    "binary", "metadata", "parser", "extractor",
})

FALLBACK_TEMPLATES = {
    1: "Define {topic} and state its importance in {subject}.",
    2: "Explain {topic} with a suitable example from {subject}.",
    3: "Illustrate {topic} with a neat diagram and describe its working principle.",
    4: "Compare {topic} with a related concept and analyze the key differences.",
    5: "Evaluate the significance of {topic} in the context of {subject}.",
    6: "Design a system that applies {topic} to solve a problem in {subject}.",
}


class OutputRepair:
    """
    Stage 5: Validate and repair generated question text.
    One pass. Either repaired or explicitly failed.
    """

    def __init__(self, kb: ErrorKnowledgeBase):
        self.kb = kb

    def repair(
        self,
        question:    str,
        evidence:    list[str],
        bloom_level: int  = 2,
        verb:        str  = "Explain",
        topic:       str  = "the concept",
        subject:     str  = "engineering",
        slot_id:     str  = "",
    ) -> tuple[str, bool]:
        """
        Validate and repair a single question.
        Returns (repaired_text, was_repaired).
        Returns ("", False) if unrecoverable.
        """
        q_lower = question.lower()
        found_kw = [kw for kw in PDF_QUESTION_KEYWORDS if kw in q_lower]
        if found_kw:
            rec = self.kb.record(
                "SH-032", "S5_OUTPUT",
                f"{slot_id}: Question references parser artifacts: {found_kw[:2]}",
                Severity.ERROR,
                {"keywords": found_kw, "question": question[:100]},
            )
            repaired = self._apply_fallback(bloom_level, verb, topic, subject)
            self.kb.resolve(rec, "Replaced with fallback template")
            return repaired, True

        if re.search(r'[\x00-\x1f\x7f-\x9f]', question):
            rec = self.kb.record(
                "SH-032", "S5_OUTPUT",
                f"{slot_id}: Question contains binary characters",
                Severity.ERROR,
                {"question": question[:100]},
            )
            repaired = self._apply_fallback(bloom_level, verb, topic, subject)
            self.kb.resolve(rec, "Replaced binary text with fallback template")
            return repaired, True

        if len(question.split()) < 6:
            rec = self.kb.record(
                "SH-032", "S5_OUTPUT",
                f"{slot_id}: Question too short ({len(question.split())}w)",
                Severity.WARNING,
            )
            repaired = self._apply_fallback(bloom_level, verb, topic, subject)
            self.kb.resolve(rec, "Replaced short question with fallback template")
            return repaired, True

        cleaned = re.sub(r'\s+', ' ', question).strip()
        return cleaned, False

    def _apply_fallback(
        self, bloom_level: int, verb: str, topic: str, subject: str
    ) -> str:
        tpl = FALLBACK_TEMPLATES.get(
            bloom_level,
            "Explain {topic} and its importance in {subject}."
        )
        return tpl.format(topic=topic, subject=subject)
