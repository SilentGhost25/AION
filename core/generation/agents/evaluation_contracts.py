"""
Contracts for the Evaluation Agent.

SlotVerdict   — the outcome of evaluating one question
AuditReport   — the batch of verdicts + API metadata

Reason codes are string constants (not an Enum) so they can be
serialized directly to JSON and matched by the Refinement Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# -----------------------------------------------------------------------------
# Reason codes
# -----------------------------------------------------------------------------


# Local-check codes
BLOOM_MISMATCH = "BLOOM_MISMATCH"
MARKING_SCHEME_MISMATCH = "MARKING_SCHEME_MISMATCH"
MISSING_FIGURE_REFERENCE = "MISSING_FIGURE_REFERENCE"
QUESTION_TOO_SHORT = "QUESTION_TOO_SHORT"
QUESTION_TOO_LONG = "QUESTION_TOO_LONG"
SOLUTION_MISSING = "SOLUTION_MISSING"
SOLUTION_TOO_SHORT = "SOLUTION_TOO_SHORT"
MULTI_PART_STRUCTURE_MISMATCH = "MULTI_PART_STRUCTURE_MISMATCH"

# API-check codes
FACTUAL_ERROR = "FACTUAL_ERROR"
UNGROUNDED = "UNGROUNDED"
CIRCULAR = "CIRCULAR"
OFF_TOPIC = "OFF_TOPIC"
VERB_TASK_MISMATCH = "VERB_TASK_MISMATCH"

# Degraded-mode code
UNVERIFIED = "UNVERIFIED"

ALL_REASON_CODES = frozenset({
    BLOOM_MISMATCH, MARKING_SCHEME_MISMATCH, MISSING_FIGURE_REFERENCE,
    QUESTION_TOO_SHORT, QUESTION_TOO_LONG, SOLUTION_MISSING,
    SOLUTION_TOO_SHORT, MULTI_PART_STRUCTURE_MISMATCH,
    FACTUAL_ERROR, UNGROUNDED, CIRCULAR, OFF_TOPIC, VERB_TASK_MISMATCH,
    UNVERIFIED,
})


# -----------------------------------------------------------------------------
# Verdicts
# -----------------------------------------------------------------------------


@dataclass
class SlotVerdict:
    """Evaluation outcome for one slot."""
    slot_id: str
    verdict: str                                    # "pass" | "fail" | "unverified"
    source: str                                     # "local" | "api" | "degraded"
    reason_code: Optional[str] = None
    detail: str = ""
    suggested_fix: Optional[str] = None

    def __post_init__(self) -> None:
        if self.verdict not in ("pass", "fail", "unverified"):
            raise ValueError(f"invalid verdict: {self.verdict!r}")
        if self.verdict == "fail" and not self.reason_code:
            raise ValueError("failed verdict requires a reason_code")

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @property
    def failed(self) -> bool:
        return self.verdict == "fail"


@dataclass
class AuditReport:
    """Aggregate result of the Evaluation Agent's run."""
    verdicts: List[SlotVerdict] = field(default_factory=list)
    degraded_mode: bool = False
    api_used: bool = False
    api_provider: Optional[str] = None
    api_calls: int = 0
    elapsed_seconds: float = 0.0
    telemetry: dict = field(default_factory=dict)

    # -------------------------------------------------------------- Convenience

    def failed_slots(self) -> List[SlotVerdict]:
        return [v for v in self.verdicts if v.failed]

    def passed_slots(self) -> List[SlotVerdict]:
        return [v for v in self.verdicts if v.passed]

    def unverified_slots(self) -> List[SlotVerdict]:
        return [v for v in self.verdicts if v.verdict == "unverified"]

    def verdict_for(self, slot_id: str) -> Optional[SlotVerdict]:
        for v in self.verdicts:
            if v.slot_id == slot_id:
                return v
        return None

    @property
    def all_passed(self) -> bool:
        return len(self.verdicts) > 0 and all(v.passed for v in self.verdicts)
