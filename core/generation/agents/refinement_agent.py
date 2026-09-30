"""
Refinement Agent — repairs flagged questions.

Pipeline per run:
    1. For each failed verdict in the AuditReport:
       a. Try deterministic repair.
       b. If that fails, try LLM repair (if budget remains).
       c. If numerical, verify with SymPy.
       d. On success, replace the question in the output.
       e. On failure, mark the slot UNRESOLVED.
    2. Return a RefinementReport and update context.questions.

Determinism:
    - In the absence of LLM repair, the output is deterministic.
    - When LLM repair fires, the paper is marked DEGRADED_DETERMINISM
      (the Checking Agent or telemetry layer will surface this).
"""

from __future__ import annotations

import copy
import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Protocol

from .base import Agent, AgentContext, AgentResult
from .deterministic_repairs import try_deterministic_repair
from .evaluation_contracts import AuditReport, SlotVerdict
from .generated_question import GeneratedQuestion
from .refinement_contracts import (
    TIER_DETERMINISTIC,
    TIER_LLM,
    TIER_NONE,
    RefinementOutcome,
    RefinementReport,
)
from .refinement_prompts import build_repair_prompt


# -----------------------------------------------------------------------------
# Protocols
# -----------------------------------------------------------------------------


class LLMCaller(Protocol):
    def call(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Any:
        ...


class SympyVerifier(Protocol):
    """Verifies that a numerical question's solution is internally consistent."""
    def verify(self, question_text: str, solution_text: str) -> bool:
        ...


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


REQUIRED_REPAIR_KEYS = ("question_text", "solution", "marking_scheme")

# Heuristic: a question is "numerical" if it contains equations or
# explicit numerical givens with math operators.
_NUMERICAL_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?\s*(?:%|km|m/s|Hz|GHz|MHz|dB|W|V|A|kg|s)\b)"
    r"|(=\s*[A-Za-z0-9\(\)\+\-\*/]+\b)"
    r"|(\$[^$]+\$)"
    r"|(\\frac|\\sqrt|\\int|\\sum)",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class RefinementAgent(Agent):
    name = "refinement"

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        sympy_verifier: Optional[SympyVerifier] = None,
        max_llm_repairs_per_paper: int = 5,
        max_attempts_per_slot: int = 2,
        seed: int = 42,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if max_llm_repairs_per_paper < 0:
            raise ValueError("max_llm_repairs_per_paper must be >= 0")
        if max_attempts_per_slot < 1:
            raise ValueError("max_attempts_per_slot must be >= 1")
        self._llm = llm_caller
        self._sympy = sympy_verifier
        self._llm_budget = max_llm_repairs_per_paper
        self._max_attempts = max_attempts_per_slot
        self._seed = seed
        self._clock = clock or time.monotonic

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> AgentResult:
        questions: List[GeneratedQuestion] = list(context.questions or [])
        report: AuditReport = context.audit_report

        if report is None or not report.verdicts:
            return AgentResult(
                success=True,
                payload=RefinementReport(),
            )

        # Build a quick lookup: slot_id -> question
        q_by_slot = {q.slot_id: q for q in questions}

        start = self._clock()
        outcomes: List[RefinementOutcome] = []
        repaired_questions: List[GeneratedQuestion] = []
        unresolved: List[str] = []
        det_count = 0
        llm_count = 0
        budget_hits = 0
        sympy_rejections = 0

        for verdict in report.verdicts:
            original = q_by_slot.get(verdict.slot_id)
            if original is None:
                # Verdict references a slot we don't have — skip.
                continue

            if not verdict.failed:
                repaired_questions.append(original)
                continue

            outcome, repaired = self._refine_one(
                original=original,
                verdict=verdict,
                llm_budget_used=llm_count,
            )

            if outcome.resolved and repaired is not None:
                # SymPy verification for numerical repairs
                if (
                    outcome.tier_used == TIER_LLM
                    and self._sympy is not None
                    and self._is_numerical(repaired)
                ):
                    if not self._verify_with_sympy(repaired):
                        outcome = RefinementOutcome(
                            slot_id=outcome.slot_id,
                            resolved=False,
                            tier_used=outcome.tier_used,
                            reason_in=outcome.reason_in,
                            reason_out="SYMPY_VERIFICATION_FAILED",
                            detail="repaired numerical question failed SymPy check",
                        )
                        sympy_rejections += 1

            outcomes.append(outcome)

            if outcome.resolved and repaired is not None:
                repaired_questions.append(repaired)
                if outcome.tier_used == TIER_DETERMINISTIC:
                    det_count += 1
                elif outcome.tier_used == TIER_LLM:
                    llm_count += 1
            else:
                unresolved.append(outcome.slot_id)
                repaired_questions.append(original)

            if outcome.reason_out == "LLM_BUDGET_EXHAUSTED":
                budget_hits += 1

        elapsed = self._clock() - start

        refinement_report = RefinementReport(
            outcomes=outcomes,
            repaired_questions=repaired_questions,
            unresolved_slots=unresolved,
            deterministic_repairs=det_count,
            llm_repairs=llm_count,
            llm_budget_exhausted=budget_hits,
            sympy_rejections=sympy_rejections,
            elapsed_seconds=round(elapsed, 3),
            telemetry={
                "total_slots": len(questions),
                "flagged_slots": len([v for v in report.verdicts if v.failed]),
                "resolved": len(repaired_questions) - len(unresolved),
                "unresolved": len(unresolved),
            },
        )

        # Update the context in place
        context.questions = repaired_questions
        context.unresolved_slots = unresolved

        self.log(
            "done",
            flagged=refinement_report.telemetry["flagged_slots"],
            resolved=refinement_report.telemetry["resolved"],
            unresolved=len(unresolved),
            deterministic=det_count,
            llm=llm_count,
        )

        return AgentResult(
            success=True,
            payload=refinement_report,
            telemetry=refinement_report.telemetry,
        )

    # -------------------------------------------------------------- Internals

    def _refine_one(
        self,
        original: GeneratedQuestion,
        verdict: SlotVerdict,
        llm_budget_used: int,
    ) -> tuple:
        """
        Try to repair one question.
        Returns (RefinementOutcome, repaired_question_or_None).
        """
        reason_code = verdict.reason_code or "UNKNOWN"
        detail = verdict.detail or ""
        suggested_fix = verdict.suggested_fix

        # Tier 1: deterministic
        deterministic_repaired = try_deterministic_repair(
            question=original,
            reason_code=reason_code,
            detail=detail,
        )
        if deterministic_repaired is not None:
            return (
                RefinementOutcome(
                    slot_id=original.slot_id,
                    resolved=True,
                    tier_used=TIER_DETERMINISTIC,
                    reason_in=reason_code,
                    detail="deterministic repair applied",
                ),
                deterministic_repaired,
            )

        # Tier 2: LLM
        if self._llm is None:
            return (
                RefinementOutcome(
                    slot_id=original.slot_id,
                    resolved=False,
                    tier_used=TIER_NONE,
                    reason_in=reason_code,
                    reason_out="NO_LLM_CALLER",
                    detail="no LLM caller configured",
                ),
                None,
            )

        if llm_budget_used >= self._llm_budget:
            return (
                RefinementOutcome(
                    slot_id=original.slot_id,
                    resolved=False,
                    tier_used=TIER_NONE,
                    reason_in=reason_code,
                    reason_out="LLM_BUDGET_EXHAUSTED",
                    detail=f"budget={self._llm_budget} exhausted",
                ),
                None,
            )

        llm_repaired = self._try_llm_repair(
            question=original,
            reason_code=reason_code,
            detail=detail,
            suggested_fix=suggested_fix,
        )
        if llm_repaired is not None:
            return (
                RefinementOutcome(
                    slot_id=original.slot_id,
                    resolved=True,
                    tier_used=TIER_LLM,
                    reason_in=reason_code,
                    detail="llm repair applied",
                ),
                llm_repaired,
            )

        return (
            RefinementOutcome(
                slot_id=original.slot_id,
                resolved=False,
                tier_used=TIER_NONE,
                reason_in=reason_code,
                reason_out="LLM_REPAIR_FAILED",
                detail=f"llm returned invalid output after {self._max_attempts} attempts",
            ),
            None,
        )

    def _try_llm_repair(
        self,
        question: GeneratedQuestion,
        reason_code: str,
        detail: str,
        suggested_fix: Optional[str],
    ) -> Optional[GeneratedQuestion]:
        for attempt in range(1, self._max_attempts + 1):
            prompt = build_repair_prompt(
                question=question,
                reason_code=reason_code,
                detail=detail,
                suggested_fix=suggested_fix,
                strict_json=(attempt > 1),
            )
            try:
                raw = self._llm.call(prompt, schema=None, seed=self._seed)
            except Exception as e:
                self.log("llm_error", slot=question.slot_id, error=str(e))
                continue

            parsed = self._parse_json(raw)
            if parsed is None:
                continue
            missing = [k for k in REQUIRED_REPAIR_KEYS if k not in parsed]
            if missing:
                continue

            return self._build_repaired(question, parsed)
        return None

    def _parse_json(self, raw: Any) -> Optional[dict]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None
        text = raw.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _build_repaired(
        self, original: GeneratedQuestion, parsed: dict
    ) -> GeneratedQuestion:
        q = copy.deepcopy(original)
        q.question_text = str(parsed.get("question_text", original.question_text)).strip()
        q.solution = str(parsed.get("solution", original.solution)).strip()
        q.marking_scheme = list(parsed.get("marking_scheme", original.marking_scheme))
        q.diagram_request = parsed.get("diagram_request", original.diagram_request)
        q.references_image = bool(
            parsed.get("references_image", original.references_image)
        )
        # Update telemetry
        q.telemetry = dict(q.telemetry or {})
        q.telemetry["repair_change_summary"] = str(
            parsed.get("change_summary", "")
        )[:300]
        return q

    def _is_numerical(self, question: GeneratedQuestion) -> bool:
        text = (question.question_text or "") + " " + (question.solution or "")
        return bool(_NUMERICAL_PATTERN.search(text))

    def _verify_with_sympy(self, question: GeneratedQuestion) -> bool:
        try:
            return bool(
                self._sympy.verify(question.question_text, question.solution)
            )
        except Exception as e:
            self.log("sympy_error", slot=question.slot_id, error=str(e))
            return False
