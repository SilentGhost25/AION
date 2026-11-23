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
from .generated_question import GeneratedQuestion, SubPart
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
    - failure_code is a single deterministic primary canonical code:
      * Question-level finding takes precedence if present.
      * Otherwise, the first unresolved finding in report order defines the primary code.
    - failure_detail is a semicolon-joined summary of all active finding details.
    - Full details of all findings remain intact in diagnostic telemetry["diagnostic_unresolved"].
    """
    if not unresolved_verdicts:
        return ("question", (), "UNKNOWN", "Unresolved failure")

    is_question_level = any(not getattr(v, "affected_subparts", ()) for v in unresolved_verdicts)
    if is_question_level:
        scope = "question"
        subparts: Tuple[str, ...] = ()
        question_level_verdicts = [v for v in unresolved_verdicts if not getattr(v, "affected_subparts", ())]
        code = question_level_verdicts[0].reason_code or "UNKNOWN"
    else:
        scope = "subquestion"
        all_parts = set()
        for v in unresolved_verdicts:
            all_parts.update(getattr(v, "affected_subparts", ()))
        subparts = tuple(sorted(all_parts))
        code = unresolved_verdicts[0].reason_code or "UNKNOWN"

    details = [v.detail for v in unresolved_verdicts if v.detail]
    detail = "; ".join(details) if details else "Repair unsuccessful"

    return scope, subparts, code, detail


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


def _validate_and_normalize_marking_scheme(
    raw_scheme: Any,
    expected_marks: int,
) -> Optional[List[Dict[str, Any]]]:
    """
    Strictly validate and normalize marking scheme criteria:
    - Must be a non-empty list of dicts.
    - Each dict must have a non-empty 'criterion' or 'description' string.
    - Each dict must have a 'marks' field that is a positive integer,
      or a float/string representing an exact positive integer (e.g., 2.0 or "2").
    - Booleans and non-integral numbers are rejected.
    - Sum of marks across all criteria must strictly equal expected_marks.
    - Returns normalized list of criteria with integer marks, or None if invalid.
    - Strictly rejects substantive mark changes; NEVER rescales drifted marks.
    """
    if not isinstance(raw_scheme, list) or not raw_scheme or expected_marks <= 0:
        return None

    normalized: List[Dict[str, Any]] = []
    total = 0

    for item in raw_scheme:
        if not isinstance(item, dict):
            return None
        desc = item.get("criterion") or item.get("description")
        if not desc or not str(desc).strip():
            return None

        raw_marks = item.get("marks")
        if isinstance(raw_marks, bool) or raw_marks is None:
            return None

        # Accept positive integer or exact integer float/str representation
        if isinstance(raw_marks, int):
            m_val = raw_marks
        elif isinstance(raw_marks, float):
            if not raw_marks.is_integer():
                return None
            m_val = int(raw_marks)
        elif isinstance(raw_marks, str):
            s = raw_marks.strip()
            try:
                f_val = float(s)
                if not f_val.is_integer():
                    return None
                m_val = int(f_val)
            except ValueError:
                return None
        else:
            return None

        if m_val <= 0:
            return None

        crit_dict = dict(item)
        crit_dict["criterion"] = str(desc).strip()
        crit_dict["marks"] = m_val
        normalized.append(crit_dict)
        total += m_val

    if total != expected_marks:
        # Strictly reject substantive mark changes; no rescaling
        return None

    return normalized


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

        affected_subparts = getattr(verdict, "affected_subparts", ())
        llm_repaired = self._try_llm_repair(
            question=original,
            reason_code=reason_code,
            detail=detail,
            suggested_fix=suggested_fix,
            affected_subparts=affected_subparts,
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
        affected_subparts: Tuple[str, ...] = (),
    ) -> Optional[GeneratedQuestion]:
        is_subpart_repair = bool(
            getattr(question, "sub_parts", None)
            and (
                len(affected_subparts) == 1
                or len(re.findall(r"SubPart\s*\(([a-z])\)", detail, re.IGNORECASE)) == 1
            )
        )
        for attempt in range(1, self._max_attempts + 1):
            prompt = build_repair_prompt(
                question=question,
                reason_code=reason_code,
                detail=detail,
                suggested_fix=suggested_fix,
                affected_subparts=affected_subparts,
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

            if is_subpart_repair:
                has_text = "text" in parsed or "question_text" in parsed
                has_sol = "solution" in parsed
                has_scheme = "marking_scheme" in parsed
                if not (has_text and has_sol and has_scheme):
                    continue
            else:
                missing = [k for k in REQUIRED_REPAIR_KEYS if k not in parsed]
                if missing and "sub_parts" not in parsed:
                    continue

            repaired = self._build_repaired(question, parsed, affected_subparts=affected_subparts)
            if repaired is not None:
                return repaired
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
        self,
        original: GeneratedQuestion,
        parsed: dict,
        affected_subparts: Tuple[str, ...] = (),
    ) -> Optional[GeneratedQuestion]:
        q = copy.deepcopy(original)

        # 1. Target SubPart isolation
        if q.sub_parts:
            target_label = None
            if affected_subparts and len(affected_subparts) == 1:
                target_label = affected_subparts[0].lower()
            elif "label" in parsed:
                target_label = str(parsed["label"]).strip().lower()
            else:
                m = re.findall(r"SubPart\s*\(([a-z])\)", str(parsed.get("detail", "")), re.IGNORECASE)
                if len(m) == 1:
                    target_label = m[0].lower()

            if target_label:
                target_sp = next((sp for sp in q.sub_parts if sp.label.lower() == target_label), None)
                if target_sp is not None:
                    # Reject substantive marks change if attempted in response
                    if "marks" in parsed:
                        try:
                            if int(parsed["marks"]) != target_sp.marks:
                                return None
                        except (ValueError, TypeError):
                            return None

                    new_text = str(parsed.get("text") or parsed.get("question_text", target_sp.text)).strip()
                    # Strip leading redundant label like "(b) " if model prepended it
                    redundant_pattern = rf"^\s*\(?\s*{re.escape(target_label)}\s*[\)\.\:\-]\s*|^\s*\(\s*{re.escape(target_label)}\s*\)\s*"
                    new_text = re.sub(redundant_pattern, "", new_text, count=1, flags=re.IGNORECASE).strip()
                    if not new_text:
                        return None

                    new_solution = str(parsed.get("solution", target_sp.solution)).strip()
                    raw_scheme = list(parsed.get("marking_scheme", target_sp.marking_scheme))

                    # Strict criteria validation and marks conservation:
                    # Accept permitted numerical normalization, reject substantive discrepancies
                    new_scheme = _validate_and_normalize_marking_scheme(raw_scheme, target_sp.marks)
                    if new_scheme is None:
                        return None

                    # Sibling subparts remain bit-for-bit unchanged; update target only
                    target_sp.text = new_text
                    target_sp.solution = new_solution
                    target_sp.marking_scheme = new_scheme

                    valid, err = target_sp.validate()
                    if not valid:
                        return None

                    q.sync_materialized_projections()
                    consistent, err = q.verify_projection_consistency()
                    if not consistent:
                        return None

                    q.telemetry = dict(q.telemetry or {})
                    q.telemetry["repair_change_summary"] = str(
                        parsed.get("change_summary", "")
                    )[:300]
                    return q

            # Full multi-part replacement if sub_parts provided in response
            if "sub_parts" in parsed and isinstance(parsed["sub_parts"], list):
                raw_sps = parsed["sub_parts"]
                if len(raw_sps) != len(q.sub_parts):
                    return None
                new_subparts = []
                for i, raw_sp in enumerate(raw_sps):
                    orig_sp = q.sub_parts[i]
                    sp_label = str(raw_sp.get("label", orig_sp.label)).strip().lower()
                    if "marks" in raw_sp:
                        try:
                            sp_marks = int(raw_sp["marks"])
                            if sp_marks != orig_sp.marks:
                                return None
                        except (ValueError, TypeError):
                            return None
                    else:
                        sp_marks = orig_sp.marks

                    sp_text = str(raw_sp.get("text", orig_sp.text)).strip()
                    sp_sol = str(raw_sp.get("solution", orig_sp.solution)).strip()
                    raw_scheme = list(raw_sp.get("marking_scheme", orig_sp.marking_scheme))

                    sp_scheme = _validate_and_normalize_marking_scheme(raw_scheme, orig_sp.marks)
                    if sp_scheme is None:
                        return None

                    sp_obj = SubPart(
                        label=sp_label,
                        marks=sp_marks,
                        text=sp_text,
                        topic=orig_sp.topic,
                        solution=sp_sol,
                        marking_scheme=sp_scheme,
                        figure_id=orig_sp.figure_id,
                        image_path=orig_sp.image_path,
                        figure_caption=orig_sp.figure_caption,
                        image_sha256=orig_sp.image_sha256,
                        bloom=getattr(orig_sp, "bloom", "") or "",
                        co=getattr(orig_sp, "co", "") or "",
                    )
                    valid, err = sp_obj.validate()
                    if not valid:
                        return None
                    new_subparts.append(sp_obj)

                q.sub_parts = new_subparts
                q.sync_materialized_projections()
                consistent, err = q.verify_projection_consistency()
                if not consistent:
                    return None
                q.telemetry = dict(q.telemetry or {})
                q.telemetry["repair_change_summary"] = str(
                    parsed.get("change_summary", "")
                )[:300]
                return q

        # 2. Legacy single-part replacement
        if "marks" in parsed:
            try:
                if int(parsed["marks"]) != original.marks:
                    return None
            except (ValueError, TypeError):
                return None

        q.question_text = str(parsed.get("question_text", original.question_text)).strip()
        q.solution = str(parsed.get("solution", original.solution)).strip()
        raw_scheme = list(parsed.get("marking_scheme", original.marking_scheme))
        if raw_scheme or original.marking_scheme or "marking_scheme" in parsed:
            norm_scheme = _validate_and_normalize_marking_scheme(raw_scheme, original.marks)
            if norm_scheme is None:
                return None
            q.marking_scheme = norm_scheme
        else:
            q.marking_scheme = []

        q.diagram_request = parsed.get("diagram_request", original.diagram_request)
        q.references_image = bool(
            parsed.get("references_image", original.references_image)
        )
        if q.sub_parts and len(q.sub_parts) == 1 and not q.is_multi_part:
            q.sub_parts[0].text = q.question_text
            q.sub_parts[0].solution = q.solution
            q.sub_parts[0].marking_scheme = q.marking_scheme
            valid, err = q.sub_parts[0].validate()
            if not valid:
                return None
            q.sync_materialized_projections()
            consistent, _ = q.verify_projection_consistency()
            if not consistent:
                return None

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
