"""
Contracts for the Checking Agent — the final gate before assembly.

CheckingOutcome — result of the check
QAReport        — aggregated telemetry and verdict for the whole paper

The QAReport is the single source of truth for downstream consumers
(DOCX export, API response, telemetry dashboards).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# -----------------------------------------------------------------------------
# Status constants
# -----------------------------------------------------------------------------


STATUS_PASS = "PASS"
STATUS_DEGRADED = "DEGRADED"
STATUS_BLOCKED = "BLOCKED"

ALL_STATUSES = frozenset({STATUS_PASS, STATUS_DEGRADED, STATUS_BLOCKED})


# -----------------------------------------------------------------------------
# Block reasons
# -----------------------------------------------------------------------------


BLOCK_REASON_UNRESOLVED_SLOTS = "UNRESOLVED_SLOTS"
BLOCK_REASON_WRITING_FAILURES = "WRITING_FAILURES"
BLOCK_REASON_INSUFFICIENT_QUESTIONS = "INSUFFICIENT_QUESTIONS"
BLOCK_REASON_INVALID_STATE = "INVALID_STATE"

DEGRADED_REASON_UNVERIFIED_EVALUATION = "UNVERIFIED_EVALUATION"


# -----------------------------------------------------------------------------
# Contracts
# -----------------------------------------------------------------------------


@dataclass
class CheckingOutcome:
    """Result of one paper-level check."""
    status: str                             # PASS | DEGRADED | BLOCKED
    blocked_reasons: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALL_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.status == STATUS_BLOCKED and not self.blocked_reasons:
            raise ValueError("BLOCKED status requires at least one reason")


@dataclass
class QAReport:
    """
    Aggregated quality report for a generated paper.

    Fields:
        status              — final verdict
        slot_count          — total slots attempted
        resolved_slots      — slots that passed all gates
        unresolved_slots    — slots that failed and could not be repaired
        degraded_reasons    — non-fatal issues (only present if status=DEGRADED)
        blocked_reasons     — fatal issues (only present if status=BLOCKED)
        telemetry           — aggregated per-agent telemetry
        cost_telemetry      — API cost tracking (Phase 4 wires this)
        cache_telemetry     — semantic cache hit/miss data
    """
    status: str
    slot_count: int = 0
    resolved_slots: int = 0
    unresolved_slots: List[str] = field(default_factory=list)
    degraded_reasons: List[str] = field(default_factory=list)
    blocked_reasons: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)
    cost_telemetry: Dict[str, Any] = field(default_factory=dict)
    cache_telemetry: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALL_STATUSES:
            raise ValueError(f"invalid status: {self.status!r}")
        if self.status == STATUS_BLOCKED and not self.blocked_reasons:
            raise ValueError("BLOCKED QAReport requires blocked_reasons")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response and JSON output."""
        return {
            "status": self.status,
            "slot_count": self.slot_count,
            "resolved_slots": self.resolved_slots,
            "unresolved_slots": list(self.unresolved_slots),
            "degraded_reasons": list(self.degraded_reasons),
            "blocked_reasons": list(self.blocked_reasons),
            "telemetry": dict(self.telemetry),
            "cost_telemetry": dict(self.cost_telemetry),
            "cache_telemetry": dict(self.cache_telemetry),
        }

    @property
    def passes(self) -> bool:
        return self.status == STATUS_PASS

    @property
    def degraded(self) -> bool:
        return self.status == STATUS_DEGRADED

    @property
    def blocked(self) -> bool:
        return self.status == STATUS_BLOCKED


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class PaperBlockedError(RuntimeError):
    """
    Raised by the orchestrator when the Checking Agent returns BLOCKED
    and the hard-block flag is enabled.

    Carries the QAReport so callers can log/inspect the failure.
    """
    def __init__(self, report: QAReport) -> None:
        self.report = report
        super().__init__(
            f"Paper blocked: {', '.join(report.blocked_reasons)} "
            f"(unresolved slots: {len(report.unresolved_slots)})"
        )
