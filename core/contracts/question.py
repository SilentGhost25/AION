"""
AION Master Production Specification — Question Intent & Output Contracts
==========================================================================
Defines the QuestionIntent passed to Qwen and the GeneratedQuestion produced by Qwen.
Qwen performs linguistic realization ONLY; structural parameters come from the plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class QuestionIntent:
    """
    The complete specification for one sub-question slot.
    Qwen receives this. Qwen does not make decisions about anything in this object.
    """
    slot_id          : str
    question_no      : int
    sub_label        : str
    module_id        : int
    marks            : int          # LOCKED
    bloom            : str          # LOCKED e.g. "L3"
    bloom_verb       : str          # LOCKED e.g. "Calculate"
    bloom_operation  : str          # LOCKED e.g. "APPLY"
    co               : str          # LOCKED
    question_type    : str          # LOCKED e.g. "NUMERICAL"
    difficulty_band  : str          # LOCKED e.g. "MEDIUM"
    visual_decision  : str          # LOCKED "IMAGE_REQUIRED"|"IMAGE_NOT_NEEDED"

    # Evidence
    concept          : str
    evidence_text    : List[str]
    evidence_pages   : List[int]
    grounding_score  : float

    # Math artifacts (references only — M4 invariant)
    math_artifacts   : Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Solver (for numerical/graph questions)
    solver_context   : Optional[Dict[str, Any]] = None

    # Constraints for Qwen
    expected_answer_type : str = "descriptive"
    forbidden_phrases    : List[str] = field(default_factory=list)
    required_entities    : List[str] = field(default_factory=list)

    # Generation seed for reproducibility
    generation_seed  : int = 42


@dataclass
class GeneratedQuestion:
    """Output from Qwen for one slot. Structural fields come from the plan."""
    slot_id          : str
    question_text    : str          # natural language from Qwen
    marks            : int          # FROM PLAN — not from Qwen
    bloom            : str          # FROM PLAN — not from Qwen
    co               : str          # FROM PLAN — not from Qwen
    math_placeholders: List[str] = field(default_factory=list)
    visual_asset     : Optional[Dict[str, Any]] = None

    # Validation
    integrity_score  : float = 100.0
    warnings         : List[str] = field(default_factory=list)
    status           : str = "PASS" # "PASS" | "WARN" | "FAIL"
