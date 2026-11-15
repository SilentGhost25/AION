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
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

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


def _aggregate_unresolved_metadata(
    unresolved_verdicts: List[SlotVerdict],
) -> Tuple[str, Tuple[str, ...], str, str]:
    """
    Apply conservative aggregation over unresolved findings for a single slot.
    Returns (unresolved_scope, unresolved_subparts, failure_code, failure_detail).

    Rules:
    - If any finding has empty affected_subparts (question-level or ambiguous),
      final scope is "question" and unresolved_subparts is ().
    - Otherwise, scope is "subquestion" and unresolved_subparts is the sorted union
      of all affected subparts.
    - failure_code is a sorted semicolon-joined list of unique reason_codes.
    - failure_detail is a semicolon-joined list of finding details.
    """
    if not unresolved_verdicts:
        return ("question", (), "UNKNOWN", "Unresolved failure")

    is_question_level = any(not getattr(v, "affected_subparts", ()) for v in unresolved_verdicts)
    if is_question_level:
        scope = "question"
        subparts: Tuple[str, ...] = ()
    else:
        scope = "subquestion"
        all_parts = set()
        for v in unresolved_verdicts:
            all_parts.update(getattr(v, "affected_subparts", ()))
        subparts = tuple(sorted(all_parts))

    unique_codes = sorted(set(v.reason_code for v in unresolved_verdicts if v.reason_code))
    code = ";".join(unique_codes) if unique_codes else "UNKNOWN"
    details = [v.detail for v in unresolved_verdicts if v.detail]
    detail = "; ".join(details) if details else "Repair unsuccessful"

    return scope, subparts, code, detail


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class RefinementAgent(Agent):
    name = "refinement"

    def __init__(
        self,
        llm_caller: Optional[LLMCaller] = None,
        sympy_verifier: Optional[SympyVerifier] = None,
        max_llm_repairs_per_paper: int = 8,
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

        # Group verdicts by slot_id while preserving report order
        verdicts_by_slot: Dict[str, List[SlotVerdict]] = {}
        for v in report.verdicts:
            verdicts_by_slot.setdefault(v.slot_id, []).append(v)

        for q in questions:
            slot_id = q.slot_id
            slot_verdicts = verdicts_by_slot.get(slot_id, [])
            failed_verdicts = [v for v in slot_verdicts if v.failed]

            if not failed_verdicts:
                repaired_questions.append(q)
                continue

            original = q

            # Diagnostic telemetry registry: capture unaliased deep snapshot ONCE pre-repair
            diag_registry = context.telemetry.setdefault("diagnostic_unresolved", {})
            if slot_id not in diag_registry:
                diag_registry[slot_id] = {
                    "slot_id": original.slot_id,
                    "module_id": original.module_id,
                    "global_q_idx": original.global_q_idx,
                    "original_question": {
                        "question_text": original.question_text,
                        "solution": original.solution,
                        "marking_scheme": copy.deepcopy(original.marking_scheme),
                        "partition": list(original.partition),
                        "bloom": original.bloom,
                        "co": original.co,
                    },
                    "findings": [],
                    "repair_attempts": [],
                }

            # Record all failed findings in append-only telemetry
            for fv in failed_verdicts:
                diag_registry[slot_id]["findings"].append({
                    "reason_code": fv.reason_code,
                    "detail": fv.detail,
                    "affected_subparts": list(getattr(fv, "affected_subparts", ())),
                    "suggested_fix": fv.suggested_fix,
                    "timestamp": self._clock(),
                })

            current_q = original
            slot_resolved = True
            unresolved_verdicts_for_slot: List[SlotVerdict] = []

            for fv in failed_verdicts:
                outcome, repaired = self._refine_one(
                    original=current_q,
                    verdict=fv,
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

                # Record repair attempt in append-only telemetry
                diag_registry[slot_id]["repair_attempts"].append({
                    "attempt_index": len(diag_registry[slot_id]["repair_attempts"]) + 1,
                    "tier_attempted": outcome.tier_used,
                    "resolved": outcome.resolved,
                    "reason_in": outcome.reason_in,
                    "reason_out": outcome.reason_out,
                    "detail": outcome.detail,
                    "timestamp": self._clock(),
                })

                outcomes.append(outcome)

                if outcome.resolved and repaired is not None:
                    current_q = repaired
                    if outcome.tier_used == TIER_DETERMINISTIC:
                        det_count += 1
                    elif outcome.tier_used == TIER_LLM:
                        llm_count += 1
                else:
                    slot_resolved = False
                    unresolved_verdicts_for_slot.append(fv)

                if outcome.reason_out == "LLM_BUDGET_EXHAUSTED":
                    budget_hits += 1

            if slot_resolved and current_q is not original:
                # Successfully repaired: clear all unresolved fields
                current_q.is_unresolved = False
                current_q.unresolved_scope = None
                current_q.unresolved_subparts = ()
                current_q.failure_code = None
                current_q.failure_detail = None
                repaired_questions.append(current_q)
            else:
                # Unresolved failure: populate deterministic metadata on original
                unresolved.append(slot_id)
                scope, subparts, code, detail = _aggregate_unresolved_metadata(unresolved_verdicts_for_slot)
                original.is_unresolved = True
                original.unresolved_scope = scope
                original.unresolved_subparts = subparts
                original.failure_code = code
                original.failure_detail = detail
                repaired_questions.append(original)

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
