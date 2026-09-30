"""
AION Paper Specification Resolver
=================================
Resolves authoritative PaperSpec contracts from the central catalog (exam_specs.json)
or request-level overrides. Enforces precedence rules and validates exam structures.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.contracts.paper_spec import PaperSpec

LOG = logging.getLogger("aion.paper_spec_resolver")

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "exam_specs.json"


def load_exam_specs_catalog() -> Dict[str, Any]:
    """Loads the exam specification catalog from disk."""
    if not CONFIG_PATH.is_file():
        LOG.warning(f"[PAPER_SPEC] Catalog missing at {CONFIG_PATH}. Using fallback presets.")
        return {}
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("exam_types", {})
    except Exception as e:
        LOG.error(f"[PAPER_SPEC] Failed loading catalog from {CONFIG_PATH}: {e}")
        return {}


def resolve_paper_spec(
    exam_type: str = "IAT1",
    override: Optional[Dict[str, Any]] = None,
) -> PaperSpec:
    """
    Resolves an immutable PaperSpec.
    
    Precedence:
    1. Direct override fields from request payload.
    2. Catalog definitions from exam_specs.json.
    3. Safe defaults (5 modules, 2 questions/mod, 10 marks, 5 COs).
    
    If exam_type is not recognized in catalog and override does not define essential
    dimensions, raises ValueError.
    """
    catalog = load_exam_specs_catalog()
    norm_type = (exam_type or "IAT1").strip().upper()
    override = override or {}

    entry = catalog.get(norm_type)

    if entry is None and norm_type in ("VTU", "MID", "IA"):
        entry = catalog.get("IAT1")
    elif entry is None and norm_type in ("FINAL", "ANNUAL", "EXTERNAL"):
        entry = catalog.get("SEE")

    if entry is None and not (
        override.get("module_count")
        and override.get("questions_per_module")
        and override.get("marks_per_question")
    ):
        available = list(catalog.keys())
        raise ValueError(
            f"Unknown exam_type '{exam_type}'. Available catalog types: {available}, "
            f"or supply full override (module_count, questions_per_module, marks_per_question)."
        )

    base = entry or {
        "module_count": 5,
        "questions_per_module": 2,
        "marks_per_question": 10,
        "co_count": 5,
        "depth_threshold_ratio": 0.8,
    }

    # Precedence: override -> base catalog
    module_count = int(override.get("module_count") or base.get("module_count", 5))
    questions_per_module = int(override.get("questions_per_module") or base.get("questions_per_module", 2))
    marks_per_question = int(override.get("marks_per_question") or base.get("marks_per_question", 10))

    # CO Count Precedence:
    # 1. override["co_count"]
    # 2. override["co_mapping"] length (if provided)
    # 3. base["co_count"]
    # 4. default 5
    co_count_raw = override.get("co_count")
    if co_count_raw is not None:
        co_count = int(co_count_raw)
    elif override.get("custom_co_map"):
        co_count = len(override["custom_co_map"])
    else:
        co_count = int(base.get("co_count", 5))

    depth_ratio = float(override.get("depth_threshold_ratio") or base.get("depth_threshold_ratio", 0.8))
    custom_co_map = override.get("custom_co_map")

    spec = PaperSpec(
        exam_type             = norm_type,
        module_count          = module_count,
        questions_per_module  = questions_per_module,
        marks_per_question    = marks_per_question,
        co_count              = co_count,
        depth_threshold_ratio = depth_ratio,
        custom_co_map         = custom_co_map,
    )

    LOG.info(
        f"[PAPER_SPEC] Resolved spec for {norm_type}: {spec.module_count} modules × "
        f"{spec.questions_per_module} questions = {spec.total_questions} total questions, "
        f"{spec.marks_per_question} marks/question ({spec.total_marks} marks total), "
        f"{spec.co_count} COs."
    )
    return spec


class PaperSpecResolver:
    """Class-style resolver interface for compatibility."""
    @staticmethod
    def resolve(exam_type: str = "IAT1", override: Optional[Dict[str, Any]] = None) -> PaperSpec:
        return resolve_paper_spec(exam_type=exam_type, override=override)
