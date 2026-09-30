"""
Contracts for the Agent Orchestrator.

OrchestratorResult — the aggregate output of the full five-agent pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OrchestratorResult:
    """
    Complete output of the five-agent pipeline.

    success     : True if the orchestrator completed all stages without
                  an unexpected agent crash. Note: a successful run can
                  still produce a BLOCKED paper (see `qa_report.status`).

    qa_report   : The final QAReport. Always present when success=True.

    plans, questions, audit_report, refinement_report:
                  Intermediate artifacts for downstream consumers
                  (DOCX export, telemetry dashboards).

    failure_code, failure_detail:
                  Populated when success=False (agent crash).

    telemetry   : Aggregated telemetry from all agents plus orchestrator
                  timing.
    """
    success: bool
    qa_report: Optional[object] = None
    plans: List = field(default_factory=list)
    questions: List = field(default_factory=list)
    audit_report: Optional[object] = None
    refinement_report: Optional[object] = None
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None
    telemetry: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.success and not self.failure_code:
            raise ValueError("failed OrchestratorResult requires failure_code")
        if self.success and self.qa_report is None:
            raise ValueError("successful OrchestratorResult requires qa_report")

    @property
    def paper_blocked(self) -> bool:
        """True if the paper was blocked by the Checking Agent."""
        if self.qa_report is None:
            return False
        return getattr(self.qa_report, "status", None) == "BLOCKED"

    @property
    def paper_degraded(self) -> bool:
        if self.qa_report is None:
            return False
        return getattr(self.qa_report, "status", None) == "DEGRADED"

    @property
    def paper_passed(self) -> bool:
        if self.qa_report is None:
            return False
        return getattr(self.qa_report, "status", None) == "PASS"
