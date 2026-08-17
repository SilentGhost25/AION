# core/contracts/demand_profile.py

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.contracts.question_slot import QuestionContract


def _compute_min_dimensions(bloom: str, marks: int) -> int:
    """Bloom-and-marks-aware minimum answer dimensions required."""
    if bloom == "L1":
        return 1
    elif bloom == "L2":
        return 1 if marks <= 5 else 2
    elif bloom in ("L3", "L4"):
        if marks <= 3:  return 1
        if marks <= 5:  return 2
        if marks <= 8:  return 3
        return 4
    else:  # L5, L6
        if marks <= 5:  return 2
        if marks <= 8:  return 3
        return 4


@dataclass(frozen=True)
class DemandProfile:
    """
    H2 — Structural demand specification derived from contract.
    Replaces keyword-scanning AnswerDemandValidator.
    Validator checks question text against this profile.
    """
    bloom_level         : str
    marks               : int
    min_dimensions      : int       # minimum distinct answer dimensions required
    requires_comparison : bool
    requires_calculation: bool
    requires_justification: bool

    @property
    def min_outputs(self) -> int:
        return self.min_dimensions

    @classmethod
    def from_contract(cls, contract: "QuestionContract") -> "DemandProfile":
        sig = contract.task_signature
        return cls(
            bloom_level          = contract.bloom_level,
            marks                = contract.marks,
            min_dimensions       = _compute_min_dimensions(contract.bloom_level, contract.marks),
            requires_comparison  = sig.requires_comparison,
            requires_calculation = sig.requires_calculation,
            requires_justification = sig.requires_justification,
        )
