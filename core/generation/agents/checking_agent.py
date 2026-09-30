"""
Checking Agent — the fail-closed gate.

Reads the accumulated state from the context:
    - context.questions           (post-Refinement)
    - context.unresolved_slots
    - context.telemetry           (aggregated)
    - context.audit_report        (from Evaluation Agent)
    - context.request             (for config flags)

Produces a QAReport and returns it in the AgentResult.

The agent never raises. Failures are reported as BLOCKED status.
The orchestrator decides whether to raise PaperBlockedError.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, List, Optional

from .base import Agent, AgentContext, AgentResult
from .checking_contracts import (
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    BLOCK_REASON_WRITING_FAILURES,
    DEGRADED_REASON_UNVERIFIED_EVALUATION,
    QAReport,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_PASS,
)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


HARD_BLOCK_ENV_VAR = "AION_ENABLE_UNRESOLVED_HARD_BLOCK"
DEFAULT_HARD_BLOCK = True


def _hard_block_enabled() -> bool:
    val = os.getenv(HARD_BLOCK_ENV_VAR, str(DEFAULT_HARD_BLOCK)).strip().lower()
    return val in ("true", "1", "yes", "on")


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class CheckingAgent(Agent):
    name = "checking"

    def __init__(
        self,
        clock: Optional[Callable[[], float]] = None,
        min_questions_ratio: float = 0.9,
    ) -> None:
        """
        min_questions_ratio:
            Fraction of expected slots that must be resolved for the paper
            to be PASS. E.g., 0.9 means at least 90% of slots must resolve.
            Below that, the paper is BLOCKED (with insufficient_questions).
            Default 0.9.
        """
        if not (0.0 <= min_questions_ratio <= 1.0):
            raise ValueError("min_questions_ratio must be in [0, 1]")
        self._clock = clock or time.monotonic
        self._min_ratio = min_questions_ratio

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> AgentResult:
        start = self._clock()

        questions = list(context.questions or [])
        unresolved = list(context.unresolved_slots or [])
        expected = self._expected_slot_count(context)

        slot_count = len(questions) + len(unresolved)
        resolved = len([q for q in questions if self._is_resolvable(q)])

        blocked_reasons: List[str] = []
        degraded_reasons: List[str] = []

        # Check 1: unresolved slots
        if unresolved:
            # If hard block is enabled, this is fatal.
            if _hard_block_enabled():
                blocked_reasons.append(BLOCK_REASON_UNRESOLVED_SLOTS)
            else:
                degraded_reasons.append(BLOCK_REASON_UNRESOLVED_SLOTS)

        # Check 2: insufficient questions
        if expected > 0:
            ratio = resolved / expected
            if ratio < self._min_ratio:
                blocked_reasons.append(BLOCK_REASON_INSUFFICIENT_QUESTIONS)

        # Check 3: writing failures recorded in context (belt-and-suspenders)
        writing_failures = self._extract_writing_failures(context)
        if writing_failures and _hard_block_enabled():
            # Writing failures that survived to here (no repair possible)
            # are fatal.
            if BLOCK_REASON_WRITING_FAILURES not in blocked_reasons:
                blocked_reasons.append(BLOCK_REASON_WRITING_FAILURES)

        # Check 4: degraded evaluation (API unavailable or unverified slots)
        if context.audit_report is not None and (
            getattr(context.audit_report, "degraded_mode", False)
            or any(getattr(v, "verdict", None) == "unverified" for v in getattr(context.audit_report, "verdicts", []))
        ):
            if DEGRADED_REASON_UNVERIFIED_EVALUATION not in degraded_reasons:
                degraded_reasons.append(DEGRADED_REASON_UNVERIFIED_EVALUATION)

        # Compute final status
        if blocked_reasons:
            status = STATUS_BLOCKED
        elif degraded_reasons:
            status = STATUS_DEGRADED
        else:
            status = STATUS_PASS

        # Aggregate telemetry
        telemetry = self._aggregate_telemetry(context)
        telemetry["checking"] = {
            "expected_slots": expected,
            "resolved_slots": resolved,
            "unresolved_slots": len(unresolved),
            "hard_block_enabled": _hard_block_enabled(),
        }

        report = QAReport(
            status=status,
            slot_count=slot_count,
            resolved_slots=resolved,
            unresolved_slots=unresolved,
            degraded_reasons=degraded_reasons,
            blocked_reasons=blocked_reasons,
            telemetry=telemetry,
            cost_telemetry=self._extract_cost_telemetry(context),
            cache_telemetry=self._extract_cache_telemetry(context),
        )

        elapsed = self._clock() - start
        self.log(
            "done",
            status=status,
            resolved=resolved,
            unresolved=len(unresolved),
            blocked_reasons=",".join(blocked_reasons) if blocked_reasons else "none",
            elapsed=f"{elapsed:.3f}s",
        )

        return AgentResult(
            success=True,
            payload=report,
            telemetry={
                "status": status,
                "resolved": resolved,
                "unresolved": len(unresolved),
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    # -------------------------------------------------------------- Internals

    def _expected_slot_count(self, context: AgentContext) -> int:
        """Expected number of slots from PaperSpec, if available."""
        spec = getattr(context, "paper_spec", None)
        if spec is not None and hasattr(spec, "total_questions"):
            try:
                return int(spec.total_questions)
            except Exception:
                return 0
        return 0

    def _is_resolvable(self, question) -> bool:
        """A question is resolvable if its text and solution are non-empty."""
        text = (getattr(question, "question_text", "") or "").strip()
        solution = (getattr(question, "solution", "") or "").strip()
        return bool(text) and bool(solution)

    def _extract_writing_failures(self, context: AgentContext) -> List[Any]:
        """
        Writing Agent records its failures in the WritingAgentOutput payload.
        The orchestrator forwards them into context.telemetry["writing_failures"].
        """
        telemetry = context.telemetry or {}
        return list(telemetry.get("writing_failures", []) or [])

    def _aggregate_telemetry(self, context: AgentContext) -> dict:
        """
        Merge all per-agent telemetry into a single dict.
        Context.telemetry is expected to contain sub-dicts by agent name.
        """
        telemetry = dict(context.telemetry or {})
        # Keep only JSON-serializable values
        cleaned = {}
        for k, v in telemetry.items():
            try:
                import json
                json.dumps(v)
                cleaned[k] = v
            except (TypeError, ValueError):
                cleaned[k] = str(v)
        return cleaned

    def _extract_cost_telemetry(self, context: AgentContext) -> dict:
        telemetry = context.telemetry or {}
        return dict(telemetry.get("cost", {}) or {})

    def _extract_cache_telemetry(self, context: AgentContext) -> dict:
        telemetry = context.telemetry or {}
        return dict(telemetry.get("cache", {}) or {})
