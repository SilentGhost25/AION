"""
AION Visual: 4-Gate Visual Verifier
Gated validation pipeline before attaching any image to a question.
Fail-closed: if verification fails, fallback to text-only question.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple
from .figure_card import FigureCard


class VisualVerifier:
    """
    4-Gate Verifier for Visual Questions:
    Gate 1: Provenance Gate (Document / Module match)
    Gate 2: Visual Necessity Gate (Question mentions/requires figure)
    Gate 3: Grounding Gate (No hallucinated entities)
    Gate 4: Ambiguity Gate (Sufficient confidence separation)
    """

    def __init__(self, min_provenance_score: float = 0.20):
        self.min_provenance_score = min_provenance_score

    def verify(
        self,
        question_dict: dict,
        card:          FigureCard,
        target_module: str,
    ) -> tuple[bool, str]:
        """
        Runs all 4 verification gates.
        Returns (passed: bool, reason: str).
        """
        # ── Gate 1: Provenance Gate ───────────────────────────
        if card.provenance_score < self.min_provenance_score:
            return False, f"Gate 1 Fail: low provenance ({card.provenance_score:.2f})"

        if card.module_id != target_module and card.module_id != "module_1":
            return False, f"Gate 1 Fail: module mismatch ({card.module_id} != {target_module})"

        # ── Gate 2: Visual Necessity Gate ────────────────────
        q_text = question_dict.get("text", "").lower()
        has_visual_marker = any(
            marker in q_text for marker in [
                "figure", "diagram", "chart", "graph", "circuit",
                "flowchart", "architecture", "given", "shown",
                "attached", "image", "illustration"
            ]
        )
        if not has_visual_marker:
            return False, "Gate 2 Fail: question does not reference a visual figure"

        # ── Gate 3: Grounding Gate ────────────────────────────
        # Ensure question mentions at least one entity or keyword from context/facts
        card_words = set(
            re.findall(r"\b[a-z]{4,}\b", card.full_context().lower())
        )
        q_words = set(
            re.findall(r"\b[a-z]{4,}\b", q_text)
        )

        overlap = card_words.intersection(q_words)
        # Exclude generic question words
        generic = {
            "question", "figure", "diagram", "explain", "analyze",
            "describe", "given", "shown", "module", "system", "vtu"
        }
        domain_overlap = overlap - generic

        if not domain_overlap and len(card_words) > 5:
            return False, "Gate 3 Fail: question concept not grounded in figure evidence"

        # ── Gate 4: Ambiguity Gate ────────────────────────────
        if not card.eligible:
            return False, f"Gate 4 Fail: card ineligible ({card.skip_reason})"

        return True, "Passed all 4 verification gates"
