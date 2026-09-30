"""
Contracts for the Refinement Agent.

RefinementOutcome — one repaired (or unresolvable) slot
RefinementReport  — the batch of outcomes with tier telemetry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# -----------------------------------------------------------------------------
# Tier identifiers
# -----------------------------------------------------------------------------


TIER_DETERMINISTIC = "deterministic"
TIER_LLM = "llm"
TIER_SYMPY = "sympy"          # verification tier, not a repair tier
TIER_NONE = "none"            # no repair succeeded


# -----------------------------------------------------------------------------
# Outcome types
# -----------------------------------------------------------------------------


@dataclass
class RefinementOutcome:
    """
    Result of attempting to repair one slot.

    resolved   : True if the slot now passes local + API checks
    tier_used  : "deterministic" | "llm" | "none"
    reason_in  : the failure code we were trying to fix
    reason_out : the reason it's still failing (only if not resolved)
    detail     : free-form diagnostic
    """
    slot_id: str
    resolved: bool
    tier_used: str
    reason_in: str
    reason_out: Optional[str] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if not self.resolved and not self.reason_out:
            raise ValueError("unresolved outcome must specify reason_out")
        if self.tier_used not in (
            TIER_DETERMINISTIC, TIER_LLM, TIER_NONE
        ):
            raise ValueError(f"invalid tier_used: {self.tier_used!r}")


@dataclass
class RefinementReport:
    """Aggregate result of the Refinement Agent's run."""
    outcomes: List[RefinementOutcome] = field(default_factory=list)
    repaired_questions: list = field(default_factory=list)     # GeneratedQuestion objects
    unresolved_slots: List[str] = field(default_factory=list)
    deterministic_repairs: int = 0
    llm_repairs: int = 0
    llm_budget_exhausted: int = 0
    sympy_rejections: int = 0
    elapsed_seconds: float = 0.0
    telemetry: dict = field(default_factory=dict)

    def outcome_for(self, slot_id: str) -> Optional[RefinementOutcome]:
        for o in self.outcomes:
            if o.slot_id == slot_id:
                return o
        return None

    @property
    def all_resolved(self) -> bool:
        return len(self.unresolved_slots) == 0
