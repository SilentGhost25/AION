"""
AION Structural Architecture v2 — Visual Decision Pipeline
===========================================================
Multi-factor decision engine mapping QuestionSlot visual priors and VRE dependencies
to definitive VisualDecision outcomes (IMAGE_REQUIRED, IMAGE_OPTIONAL, IMAGE_NOT_NEEDED).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from .contracts import BloomLevel, QuestionSlot, VisualDecision, VisualPrior


class VisualUnavailableError(Exception):
    """Raised when a hard visual dependency cannot be satisfied by VRE."""
    pass


class VisualDecisionPipeline:
    """Multi-factor Visual Decision Pipeline."""

    @classmethod
    def decide(
        cls,
        slot: QuestionSlot,
        concept: Any,
        evidence: Optional[List[Dict[str, Any]]] = None,
        vre: Optional[Any] = None,
    ) -> VisualDecision:
        evidence = evidence or []

        # STEP 1 — HARD GATES
        if slot.visual_prior == VisualPrior.FORBIDDEN:
            return VisualDecision.IMAGE_NOT_NEEDED

        if slot.marks <= 2 and slot.bloom <= BloomLevel.L2:
            return VisualDecision.IMAGE_NOT_NEEDED

        # STEP 2 — VRE DEPENDENCY ANALYSIS
        dependency = "NO_DEPENDENCY"
        if vre and hasattr(vre, "analyze_dependency"):
            dependency = vre.analyze_dependency(
                concept=concept,
                question_type=slot.question_type,
                operation=slot.bloom,
            )
        else:
            # Fallback heuristic based on question type
            hard_types = {"CIRCUIT_NUMERICAL", "KVL_ANALYSIS", "GRAPH_SHORTEST_PATH", "SFD_BMD"}
            if slot.question_type in hard_types and slot.visual_prior == VisualPrior.PREFERRED:
                dependency = "HARD_DEPENDENCY"
            elif slot.visual_prior in (VisualPrior.PREFERRED, VisualPrior.OPTIONAL):
                dependency = "SOFT_DEPENDENCY"

        if dependency == "HARD_DEPENDENCY":
            return VisualDecision.IMAGE_REQUIRED

        # STEP 3 — SOLVER AVAILABILITY CHECK
        if dependency in ("HARD_DEPENDENCY", "SOFT_DEPENDENCY"):
            solver_available = True
            figure_available = True
            if vre:
                if hasattr(vre, "solver_available"):
                    solver_available = vre.solver_available(slot.question_type)
                if hasattr(vre, "figure_available"):
                    figure_available = vre.figure_available(concept)

            if not solver_available or not figure_available:
                if dependency == "HARD_DEPENDENCY":
                    raise VisualUnavailableError(f"Hard visual dependency unmet for slot {slot.slot_id}")
                else:
                    return VisualDecision.IMAGE_NOT_NEEDED

        # STEP 4 — EVIDENCE SCAN
        visual_evidence = [e for e in evidence if e.get("has_figure", False)]
        if slot.visual_prior == VisualPrior.PREFERRED and visual_evidence:
            return VisualDecision.IMAGE_OPTIONAL

        # STEP 5 — MARKS-BASED HEURISTIC
        if slot.marks >= 6 and slot.bloom >= BloomLevel.L3:
            if slot.visual_prior in (VisualPrior.OPTIONAL, VisualPrior.PREFERRED):
                return VisualDecision.IMAGE_OPTIONAL

        # STEP 6 — DEFAULT
        return VisualDecision.IMAGE_NOT_NEEDED
