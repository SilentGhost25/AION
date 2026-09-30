"""
Agent Orchestrator — sequences the five agents.

Sequence:
    1. Planning Agent    → context.plans
    2. Writing Agent     → context.questions (+ records failures in unresolved_slots)
    3. Evaluation Agent  → context.audit_report          (optional)
    4. Refinement Agent  → updates context.questions     (optional)
    5. Checking Agent    → context.qa_report

The orchestrator does not raise on expected outcomes. It returns a
fully-populated OrchestratorResult. For pipeline integration, use
`run_or_raise()` which raises PaperBlockedError when the paper is
BLOCKED and the hard-block flag is enabled.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

from .base import AgentContext
from .checking_agent import CheckingAgent
from .checking_contracts import (
    PaperBlockedError,
    QAReport,
    STATUS_BLOCKED,
)
from .evaluation_agent import EvaluationAgent
from .orchestrator_contracts import OrchestratorResult
from .planning_agent import PlanningAgent
from .refinement_agent import RefinementAgent
from .writing_agent import WritingAgent


HARD_BLOCK_ENV_VAR = "AION_ENABLE_UNRESOLVED_HARD_BLOCK"
DEFAULT_HARD_BLOCK = True


def _hard_block_enabled() -> bool:
    val = os.getenv(HARD_BLOCK_ENV_VAR, str(DEFAULT_HARD_BLOCK)).strip().lower()
    return val in ("true", "1", "yes", "on")


class AgentOrchestrator:
    """
    Sequences the five agents over a single AgentContext.

    Parameters
    ----------
    planning_agent   : required
    writing_agent    : required
    checking_agent   : required (fail-closed gate)
    evaluation_agent : optional (skip if no API caller configured)
    refinement_agent : optional (skip if no LLM caller or no API audit)
    clock            : optional monotonic clock for timing
    """

    def __init__(
        self,
        planning_agent: PlanningAgent,
        writing_agent: WritingAgent,
        checking_agent: CheckingAgent,
        evaluation_agent: Optional[EvaluationAgent] = None,
        refinement_agent: Optional[RefinementAgent] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if planning_agent is None:
            raise ValueError("planning_agent is required")
        if writing_agent is None:
            raise ValueError("writing_agent is required")
        if checking_agent is None:
            raise ValueError("checking_agent is required")

        self._planning = planning_agent
        self._writing = writing_agent
        self._evaluation = evaluation_agent
        self._refinement = refinement_agent
        self._checking = checking_agent
        self._clock = clock or time.monotonic

        # Pairing warning: evaluation without refinement means unverified
        # verdicts never get a chance to be repaired. Log but do not raise.
        if (evaluation_agent is None) ^ (refinement_agent is None):
            print(
                "[ORCHESTRATOR] warning: evaluation_agent and "
                "refinement_agent should typically be provided together",
                flush=True,
            )

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> OrchestratorResult:
        start = self._clock()

        # ---- Stage 1: Planning ----
        plan_result = self._planning.run(context)
        if not plan_result.success:
            return self._failure(
                "PLANNING_FAILED",
                plan_result.failure_detail or "planning agent failed",
            )
        context.plans = plan_result.payload
        context.telemetry["planning"] = plan_result.telemetry

        # ---- Stage 2: Writing ----
        write_result = self._writing.run(context)
        if not write_result.success:
            return self._failure(
                "WRITING_FAILED",
                write_result.failure_detail or "writing agent failed",
            )
        writing_output = write_result.payload
        context.questions = list(writing_output.questions)
        context.telemetry["writing"] = write_result.telemetry

        # Record writing failures as unresolved slots immediately
        if writing_output.failures:
            for f in writing_output.failures:
                if f.slot_id not in context.unresolved_slots:
                    context.unresolved_slots.append(f.slot_id)
            context.telemetry["writing_failures"] = [
                {"slot_id": f.slot_id, "failure_code": f.failure_code}
                for f in writing_output.failures
            ]

        # ---- Stage 3: Evaluation (optional) ----
        if self._evaluation is not None and context.questions:
            eval_result = self._evaluation.run(context)
            if not eval_result.success:
                return self._failure(
                    "EVALUATION_FAILED",
                    eval_result.failure_detail or "evaluation agent failed",
                )
            context.audit_report = eval_result.payload
            context.telemetry["evaluation"] = eval_result.telemetry

        # ---- Stage 4: Refinement (optional) ----
        if self._refinement is not None and context.audit_report is not None:
            refine_result = self._refinement.run(context)
            if not refine_result.success:
                return self._failure(
                    "REFINEMENT_FAILED",
                    refine_result.failure_detail or "refinement agent failed",
                )
            context.telemetry["refinement"] = refine_result.telemetry
            # The Refinement Agent updates context.questions and
            # context.unresolved_slots in place.

        # ---- Stage 5: Checking (required) ----
        check_result = self._checking.run(context)
        if not check_result.success:
            return self._failure(
                "CHECKING_FAILED",
                check_result.failure_detail or "checking agent failed",
            )
        qa_report: QAReport = check_result.payload
        context.qa_report = qa_report
        context.telemetry["checking"] = check_result.telemetry

        elapsed = self._clock() - start
        context.telemetry["orchestrator"] = {
            "elapsed_seconds": round(elapsed, 3),
            "stages_completed": 5,
        }

        self._log_done(qa_report, elapsed)

        return OrchestratorResult(
            success=True,
            qa_report=qa_report,
            plans=context.plans,
            questions=context.questions,
            audit_report=context.audit_report,
            refinement_report=context.telemetry.get("refinement"),
            telemetry=dict(context.telemetry),
        )

    def run_or_raise(self, context: AgentContext) -> OrchestratorResult:
        """
        Same as `run`, but raises PaperBlockedError when the paper is
        BLOCKED and AION_ENABLE_UNRESOLVED_HARD_BLOCK is enabled.
        """
        result = self.run(context)
        if (
            result.success
            and result.paper_blocked
            and _hard_block_enabled()
        ):
            raise PaperBlockedError(result.qa_report)
        return result

    # -------------------------------------------------------------- Internals

    def _failure(self, code: str, detail: str) -> OrchestratorResult:
        print(f"[ORCHESTRATOR] failure: {code} — {detail}", flush=True)
        return OrchestratorResult(
            success=False,
            failure_code=code,
            failure_detail=detail,
        )

    def _log_done(self, qa_report: QAReport, elapsed: float) -> None:
        status = getattr(qa_report, "status", "unknown")
        resolved = getattr(qa_report, "resolved_slots", 0)
        unresolved = len(getattr(qa_report, "unresolved_slots", []) or [])
        print(
            f"[ORCHESTRATOR] done status={status} "
            f"resolved={resolved} unresolved={unresolved} "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
