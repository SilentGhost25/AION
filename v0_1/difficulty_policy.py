"""
AION Module: Difficulty Policy
==============================
Strict Course Outcome (CO) <-> Bloom Level (BL) Mapping & Printing Policy.
Enforces institutional OBE rules:
- CO1 <-> L1/L2 (Printed RBT: "L1/L2")
- CO2 <-> L3    (Printed RBT: "L3")
- CO3 <-> L4    (Printed RBT: "L4")
- CO4 <-> L5    (Printed RBT: "L5")
- CO5 <-> L6    (Printed RBT: "L6")
"""

from __future__ import annotations

import os
import re
from typing import Any
from core.contracts.module_identity import make_co

# ============================================================
# MODULAR DIFFICULTY POLICY CONFIG
# ============================================================
EASE_PAPER_DIFFICULTY: bool = True   # <-- single on/off switch

# Suggestive Bloom range by marks
MARKS_TO_BLOOM_RANGE = {
    (1, 4): [1, 2, 3],      # Suggest L1-L2, allow L3 for numerical/applied
    (5, 7): [2, 3, 4],      # Suggest L2-L3, allow L4 for analytical
    (8, 20): [3, 4, 5],     # Suggest L3-L5 (analytical, design, evaluation)
}

# Compatibility mapping for callers expecting strict constants
STRICT_CO_TO_BLOOM = {
    "CO1": (1, 2),
    "CO2": (2, 3),
    "CO3": (3, 4),
    "CO4": (4, 5),
    "CO5": (5, 6),
}

STRICT_BLOOM_TO_CO = {
    1: "CO1",
    2: "CO1",
    3: "CO2",
    4: "CO3",
    5: "CO4",
    6: "CO5",
}

STRICT_MODULE_TO_CO = {
    1: "CO1",
    2: "CO2",
    3: "CO3",
    4: "CO4",
    5: "CO5",
}

STRICT_CO_TO_PRINTED_RBT = {
    "CO1": "L1/L2",
    "CO2": "L3",
    "CO3": "L4",
    "CO4": "L5",
    "CO5": "L6",
}


def format_co_and_rbt(co: Any = None, bloom: Any = None, module_idx: int = 1) -> tuple[str, str]:
    """
    Format CO and RBT strings safely without destructive overrides.
    Preserves actual pipeline CO and Bloom levels.
    """
    co_str = str(co or "").strip().upper()
    if not co_str or not co_str.startswith("CO"):
        co_str = make_co(min(max(1, int(module_idx or 1)), 5))

    b_str = str(bloom or "").strip().upper()
    if b_str.isdigit():
        b_str = f"L{b_str}"
    elif not b_str.startswith("L"):
        m = re.search(r"L?([1-6])", b_str)
        b_str = f"L{m.group(1)}" if m else "L2"

    return co_str, b_str


def _resolve_co_by_mode(
    module_idx: int,
    marks: int,
    mode: str = "marks-based",
) -> str:
    """
    Resolves Course Outcome (CO) based on the specified assignment mode:
    - 'marks-based' (default): legacy backward-compatible OBE mapping (<=4M -> CO1, 6M -> CO2, 8M+ -> CO3)
    - 'module-based': strict syllabus module outcome (Module M -> COM)
    - 'hybrid': module-based for modules 1-3, marks-based capped at CO3 for modules 4-5
    """
    m_str = (mode or "marks-based").lower().strip()
    if m_str == "module-based":
        return make_co(module_idx) if module_idx else "CO1"
    elif m_str == "hybrid":
        if module_idx <= 3:
            return make_co(module_idx)
        return make_co(min(max(1, marks // 2), 3))
    else:
        # Default: marks-based
        if marks <= 4:
            return "CO1"
        elif marks <= 6:
            return "CO2"
        return "CO3"


def _resolve_co_bl_from_marks_original(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
    co_mode: str = None,
) -> tuple[str, int]:
    """ORIGINAL, UNMODIFIED policy — kept as permanent backup/reference."""
    ptype = (planned_type or "CONCEPTUAL").upper()
    mode = co_mode or os.getenv("AION_CO_MODE", "marks-based")
    co = _resolve_co_by_mode(module_idx, marks, mode)
    if marks <= 4:
        return co, (1 if ptype == "CONCEPTUAL" else 2)
    if marks <= 6:
        bloom = 3 if module_idx <= 4 else 4
        return co, bloom
    return co, 4


def _resolve_co_bl_from_marks_eased(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
    co_mode: str = None,
) -> tuple[str, int]:
    """
    EASED policy: shifts the marks threshold so 6M sub-questions also
    fall into the easy CO1/CO2 range, reserving CO3/L4 (hard tier)
    only for 8M/10M full-weight questions. Targets ~60% of paper marks
    at L1-L2, ~40% at L4, matching user-requested 60/40 split.
    Applies uniformly across ALL mark splits (4M, 6M, 8M, 10M).
    """
    ptype = (planned_type or "CONCEPTUAL").upper()
    mode = co_mode or os.getenv("AION_CO_MODE", "marks-based")
    co = _resolve_co_by_mode(module_idx, marks, mode)

    # 4M -> foundational (L1-L2)
    if marks <= 4:
        return co, (1 if ptype == "CONCEPTUAL" else 2)

    # 6M -> moderate / easy-tier application (L2-L3) across ALL modules
    if marks <= 6:
        return co, (2 if ptype == "CONCEPTUAL" else 3)

    # 8M/10M -> analytical hard tier (L4-L5)
    bloom = 5 if ptype in ("EVALUATE", "EVALUATION") else 4
    return co, bloom


def resolve_co_bl_from_marks(
    module_idx: int,
    marks: int,
    total_parts: int,
    planned_type: str = "CONCEPTUAL",
    co_mode: str = None,
) -> tuple[str, int]:
    """
    Public entry point — routes to eased or original policy based on
    the EASE_PAPER_DIFFICULTY toggle. All existing call sites continue
    to call this exact function name/signature unchanged.
    """
    if EASE_PAPER_DIFFICULTY:
        return _resolve_co_bl_from_marks_eased(module_idx, marks, total_parts, planned_type, co_mode)
    return _resolve_co_bl_from_marks_original(module_idx, marks, total_parts, planned_type, co_mode)
