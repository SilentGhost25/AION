"""
AION Module: Difficulty Policy
==============================
Modular difficulty easing patch with config toggle.
Toggle EASE_PAPER_DIFFICULTY = False to fully revert to
original resolve_co_bl_from_marks behavior with zero side effects.
"""

from __future__ import annotations

# ============================================================
# MODULAR DIFFICULTY EASING PATCH
# Toggle EASE_PAPER_DIFFICULTY = False to fully revert to
# original resolve_co_bl_from_marks behavior with zero side effects.
# ============================================================
EASE_PAPER_DIFFICULTY: bool = True   # <-- single on/off switch


def _resolve_co_bl_from_marks_original(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
) -> tuple[str, int]:
    """ORIGINAL, UNMODIFIED policy — kept as permanent backup/reference."""
    ptype = (planned_type or "CONCEPTUAL").upper()
    if marks <= 4:
        return "CO1", (1 if ptype == "CONCEPTUAL" else 2)
    if marks <= 6:
        if module_idx <= 4:
            return "CO2", 3
        return "CO3", 4
    return "CO3", 4


def _resolve_co_bl_from_marks_eased(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
) -> tuple[str, int]:
    """
    EASED policy: shifts the marks threshold so 6M sub-questions also
    fall into the easy CO1/CO2 range, reserving CO3/L4 (hard tier)
    only for 8M/10M full-weight questions. Targets ~60% of paper marks
    at L1-L2, ~40% at L4, matching user-requested 60/40 split.
    Applies uniformly across ALL mark splits (4M, 6M, 8M, 10M).
    """
    ptype = (planned_type or "CONCEPTUAL").upper()

    # 4M -> foundational (CO1 / L1-L2)
    if marks <= 4:
        return "CO1", (1 if ptype == "CONCEPTUAL" else 2)

    # 6M -> moderate / easy-tier application (CO2 / L2-L3) across ALL modules
    if marks <= 6:
        return "CO2", (2 if ptype == "CONCEPTUAL" else 3)

    # 8M/10M unchanged -> analytical hard tier (CO3 / L4)
    return "CO3", 4


def resolve_co_bl_from_marks(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
) -> tuple[str, int]:
    """
    Public entry point — routes to eased or original policy based on
    the EASE_PAPER_DIFFICULTY toggle. All existing call sites continue
    to call this exact function name/signature unchanged.
    """
    if EASE_PAPER_DIFFICULTY:
        return _resolve_co_bl_from_marks_eased(module_idx, marks, total_parts, planned_type)
    return _resolve_co_bl_from_marks_original(module_idx, marks, total_parts, planned_type)
