"""
Checking Agent — the fail-closed gate.

Reads the accumulated state from the context:
    - context.questions           (post-Refinement)
    - context.unresolved_slots
    - context.telemetry           (aggregated)
    - context.audit_report        (from Evaluation Agent)
    - context.request             (for config flags)
    - context.plans               (authoritative expected slots from PlanningAgent)
    - context.paper_spec          (fallback / dimension verification)

Produces a QAReport and returns it in the AgentResult.

The agent never raises during regular processing. Specification corruptions
or fatal defects are reported as BLOCKED status.
The orchestrator decides whether to raise PaperBlockedError.
"""

from __future__ import annotations

import os
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

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
    STATUS_OK,
    STATUS_PARTIAL,
)


# -----------------------------------------------------------------------------
# Config & Helpers
# -----------------------------------------------------------------------------


HARD_BLOCK_ENV_VAR = "AION_ENABLE_UNRESOLVED_HARD_BLOCK"
DEFAULT_HARD_BLOCK = True
BLOCK_REASON_MISSING_VISUAL_ASSET = "MISSING_VISUAL_ASSET"


def _hard_block_enabled() -> bool:
    val = os.getenv(HARD_BLOCK_ENV_VAR, str(DEFAULT_HARD_BLOCK)).strip().lower()
    return val in ("true", "1", "yes", "on")


def _is_positive_int(val: Any) -> bool:
    """Strictly validate positive integer, rejecting booleans, floats, strings, and None."""
    return isinstance(val, int) and not isinstance(val, bool) and val > 0


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

        (
            sorted_f,
            sorted_p,
            sorted_b,
            sorted_unexp_obs,
            sorted_unexp_reg,
            fatal_integrity_defects,
            all_accounting_defects,
            accounting_telemetry,
        ) = self._account_slots(context)

        expected = len(sorted_f) + len(sorted_p) + len(sorted_b)
        resolved = len(sorted_f)
        unresolved_expected = sorted_p + sorted_b
        total_unresolved = len(unresolved_expected)
        slot_count = len(sorted_f) + total_unresolved

        blocked_reasons: List[str] = []
        degraded_reasons: List[str] = []

        # Check 1: Fatal integrity defects fail closed immediately
        if fatal_integrity_defects:
            for defect in fatal_integrity_defects:
                blocked_reasons.append(f"INTEGRITY_DEFECT:{defect}")

        # Check 2: Unresolved slots
        if total_unresolved > 0:
            # Hard block only when nothing usable survived.
            # Partial resolution -> PARTIAL, not BLOCKED.
            if _hard_block_enabled() and resolved == 0:
                if BLOCK_REASON_UNRESOLVED_SLOTS not in blocked_reasons:
                    blocked_reasons.append(BLOCK_REASON_UNRESOLVED_SLOTS)
            else:
                if BLOCK_REASON_UNRESOLVED_SLOTS not in degraded_reasons:
                    degraded_reasons.append(BLOCK_REASON_UNRESOLVED_SLOTS)

        # Check 3: Insufficient questions ratio
        if expected > 0:
            ratio = resolved / expected
            if ratio < self._min_ratio:
                is_degraded_audit = (
                    context.audit_report is not None
                    and getattr(context.audit_report, "degraded_mode", False)
                )
                if is_degraded_audit and resolved > 0:
                    if BLOCK_REASON_INSUFFICIENT_QUESTIONS not in degraded_reasons:
                        degraded_reasons.append(BLOCK_REASON_INSUFFICIENT_QUESTIONS)
                else:
                    if BLOCK_REASON_INSUFFICIENT_QUESTIONS not in blocked_reasons:
                        blocked_reasons.append(BLOCK_REASON_INSUFFICIENT_QUESTIONS)

        # Check 4: Writing failures in context
        writing_failures = self._extract_writing_failures(context)
        if writing_failures and _hard_block_enabled():
            if BLOCK_REASON_WRITING_FAILURES not in blocked_reasons:
                blocked_reasons.append(BLOCK_REASON_WRITING_FAILURES)

        # Check 5: Degraded evaluation (API unavailable or unverified slots)
        if context.audit_report is not None and (
            getattr(context.audit_report, "degraded_mode", False)
            or any(getattr(v, "verdict", None) == "unverified" for v in getattr(context.audit_report, "verdicts", []))
        ):
            if DEGRADED_REASON_UNVERIFIED_EVALUATION not in degraded_reasons:
                degraded_reasons.append(DEGRADED_REASON_UNVERIFIED_EVALUATION)

        # Check 5b: Physical validation of referenced visual assets (Stage G2-D, hardened in Stage G4-C)
        from core.contracts.document_artifact import validate_physical_image
        from core.generation.agents.prompts import IMAGE_REFERENCE_PHRASES
        ws_root = getattr(context, "workspace_dir", None)
        if ws_root is None and hasattr(context, "request") and isinstance(context.request, dict):
            ws_root = context.request.get("workspace_dir")
        for q in getattr(context, "questions", []):
            if not getattr(q, "is_unresolved", False):
                sub_parts = getattr(q, "sub_parts", []) or []

                # 1. Validate parent-level referenced image if parent indicates visual reference
                parent_refs_image = getattr(q, "references_image", False)
                p_path = getattr(q, "image_path", None)
                p_hash = getattr(q, "image_sha256", None)
                parent_text = getattr(q, "question_text", "") or ""
                parent_text_ref = any(phrase in parent_text.lower() for phrase in IMAGE_REFERENCE_PHRASES) if not sub_parts else False

                if parent_refs_image or p_path or parent_text_ref:
                    if not p_path:
                        if BLOCK_REASON_MISSING_VISUAL_ASSET not in blocked_reasons:
                            blocked_reasons.append(BLOCK_REASON_MISSING_VISUAL_ASSET)
                        blocked_reasons.append(f"{BLOCK_REASON_MISSING_VISUAL_ASSET}:parent:missing_path")
                    else:
                        valid_img, img_err = validate_physical_image(
                            p_path,
                            expected_sha256=p_hash,
                            workspace_root=ws_root,
                        )
                        if not valid_img:
                            if BLOCK_REASON_MISSING_VISUAL_ASSET not in blocked_reasons:
                                blocked_reasons.append(BLOCK_REASON_MISSING_VISUAL_ASSET)
                            blocked_reasons.append(f"{BLOCK_REASON_MISSING_VISUAL_ASSET}:parent:{img_err}")

                # 2. Validate every visual-bearing subpart independently (NON-SUPPRESSION)
                for sp in sub_parts:
                    sp_path = getattr(sp, "image_path", None)
                    sp_hash = getattr(sp, "image_sha256", None)
                    sp_fid = getattr(sp, "figure_id", None)
                    sp_text = getattr(sp, "text", "") or ""
                    sp_text_ref = any(phrase in sp_text.lower() for phrase in IMAGE_REFERENCE_PHRASES)
                    sp_label = getattr(sp, "label", "?")

                    if sp_path or sp_fid or sp_text_ref:
                        if not sp_path and sp_fid:
                            if BLOCK_REASON_MISSING_VISUAL_ASSET not in blocked_reasons:
                                blocked_reasons.append(BLOCK_REASON_MISSING_VISUAL_ASSET)
                            blocked_reasons.append(f"{BLOCK_REASON_MISSING_VISUAL_ASSET}:subpart_{sp_label}:missing_path_for_figure_{sp_fid}")
                        elif not sp_path and sp_text_ref:
                            if BLOCK_REASON_MISSING_VISUAL_ASSET not in blocked_reasons:
                                blocked_reasons.append(BLOCK_REASON_MISSING_VISUAL_ASSET)
                            blocked_reasons.append(f"{BLOCK_REASON_MISSING_VISUAL_ASSET}:subpart_{sp_label}:text_references_figure_without_asset")
                        elif sp_path:
                            valid_img, img_err = validate_physical_image(
                                sp_path,
                                expected_sha256=sp_hash,
                                workspace_root=ws_root,
                            )
                            if not valid_img:
                                if BLOCK_REASON_MISSING_VISUAL_ASSET not in blocked_reasons:
                                    blocked_reasons.append(BLOCK_REASON_MISSING_VISUAL_ASSET)
                                blocked_reasons.append(f"{BLOCK_REASON_MISSING_VISUAL_ASSET}:subpart_{sp_label}:{img_err}")

        # Check 6: Record subpart defects
        for sid, subparts in accounting_telemetry.get("subquestion_defects", {}).items():
            degraded_reasons.append(f"SUBPART_UNRESOLVED:{sid}:{','.join(subparts)}")

        # Compute final status
        if blocked_reasons:
            status = STATUS_BLOCKED
        elif total_unresolved > 0 and resolved > 0:
            status = STATUS_PARTIAL
        elif degraded_reasons:
            status = STATUS_DEGRADED
        elif resolved == expected and not fatal_integrity_defects:
            status = STATUS_OK
        else:
            status = STATUS_BLOCKED

        # Aggregate telemetry
        telemetry = self._aggregate_telemetry(context)
        telemetry["checking"] = {
            "expected_slots": expected,
            "resolved_slots": resolved,
            "unresolved_slots": total_unresolved,
            "hard_block_enabled": _hard_block_enabled(),
            "accounting": accounting_telemetry,
            "all_accounting_defects": all_accounting_defects,
        }

        report = QAReport(
            status=status,
            slot_count=slot_count,
            resolved_slots=resolved,
            unresolved_slots=unresolved_expected,
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
            unresolved=total_unresolved,
            blocked_reasons=",".join(blocked_reasons) if blocked_reasons else "none",
            elapsed=f"{elapsed:.3f}s",
        )

        return AgentResult(
            success=True,
            payload=report,
            telemetry={
                "status": status,
                "resolved": resolved,
                "unresolved": total_unresolved,
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    # -------------------------------------------------------------- Internals

    def _get_validated_expected_slots(self, context: AgentContext) -> List[str]:
        spec = getattr(context, "paper_spec", None)
        plans = getattr(context, "plans", None)

        # 1. Authoritative plans branch
        if plans is not None:
            if not isinstance(plans, list):
                raise ValueError(f"Invalid specification: context.plans must be a list, got {type(plans).__name__}")
            if len(plans) == 0:
                raise ValueError("Invalid specification: context.plans exists but is empty (EMPTY_AUTHORITATIVE_PLAN)")

            validated_ids = []
            seen = set()
            for idx, p in enumerate(plans):
                sid = getattr(p, "slot_id", None)
                if sid is None:
                    raise ValueError(f"Invalid specification: plan[{idx}] has slot_id=None")
                if not isinstance(sid, str):
                    raise ValueError(f"Invalid specification: plan[{idx}] slot_id is not a string (got {type(sid).__name__})")
                clean_sid = sid.strip()
                if not clean_sid:
                    raise ValueError(f"Invalid specification: plan[{idx}] has empty slot_id")
                if clean_sid in seen:
                    raise ValueError(f"Invalid specification: duplicate expected slot_id '{clean_sid}' at plan[{idx}]")
                seen.add(clean_sid)
                validated_ids.append(clean_sid)

            # Cross-validate against PaperSpec if both are present
            if spec is not None:
                mod_count = getattr(spec, "module_count", None)
                q_per_mod = getattr(spec, "questions_per_module", None)
                if not _is_positive_int(mod_count) or not _is_positive_int(q_per_mod):
                    raise ValueError(
                        f"Invalid PaperSpec dimensions: module_count={mod_count} ({type(mod_count).__name__}), "
                        f"questions_per_module={q_per_mod} ({type(q_per_mod).__name__})"
                    )

                expected_total = mod_count * q_per_mod
                if len(validated_ids) != expected_total:
                    raise ValueError(
                        f"Plan count mismatch: context.plans has {len(validated_ids)} slots, "
                        f"expected {expected_total} from PaperSpec (module_count={mod_count}, questions_per_module={q_per_mod})"
                    )

                for idx, actual_sid in enumerate(validated_ids):
                    m = (idx // q_per_mod) + 1
                    expected_sid = f"module_{m}_Q{idx + 1}"
                    if actual_sid != expected_sid:
                        raise ValueError(
                            f"Plan naming/ordering contract violation at index {idx}: "
                            f"got '{actual_sid}', expected '{expected_sid}'"
                        )

            return validated_ids

        # 2. Isolated test fallback from PaperSpec
        elif spec is not None:
            mod_count = getattr(spec, "module_count", None)
            q_per_mod = getattr(spec, "questions_per_module", None)
            if not _is_positive_int(mod_count) or not _is_positive_int(q_per_mod):
                raise ValueError(
                    f"Invalid PaperSpec dimensions: module_count={mod_count} ({type(mod_count).__name__}), "
                    f"questions_per_module={q_per_mod} ({type(q_per_mod).__name__})"
                )

            expected_ids = []
            for m in range(1, mod_count + 1):
                for q in range(1, q_per_mod + 1):
                    global_idx = (m - 1) * q_per_mod + q
                    expected_ids.append(f"module_{m}_Q{global_idx}")
            return expected_ids

        # 3. Neither plans nor spec present
        else:
            raise ValueError("Cannot establish expected slots: context has neither plans nor paper_spec")

    def _expected_slot_count(self, context: AgentContext) -> int:
        """Expected number of slots from PaperSpec or plans, if available."""
        try:
            return len(self._get_validated_expected_slots(context))
        except Exception:
            return 0

    def _is_resolvable(self, question: Any) -> bool:
        """
        A question is resolvable if and only if it is in the clean, fully-resolved state:
        1. question is not None.
        2. question_text and solution are valid non-empty strings.
        3. is_unresolved is strictly boolean False.
        4. failure_code is None.
        5. failure_detail is None.
        6. unresolved_scope is None.
        7. unresolved_subparts is a valid empty collection (list or tuple). None is NOT supported.
        Any active marker, non-None failure detail, malformed metadata, or cross-field contradiction returns False.
        """
        if question is None:
            return False

        raw_text = getattr(question, "question_text", None)
        raw_sol = getattr(question, "solution", None)
        if not isinstance(raw_text, str) or not isinstance(raw_sol, str):
            return False
        if not raw_text.strip() or not raw_sol.strip():
            return False

        raw_is_unres = getattr(question, "is_unresolved", False)
        if not isinstance(raw_is_unres, bool) or raw_is_unres:
            return False

        if getattr(question, "failure_code", None) is not None:
            return False

        if getattr(question, "failure_detail", None) is not None:
            return False

        if getattr(question, "unresolved_scope", None) is not None:
            return False

        raw_subparts = getattr(question, "unresolved_subparts", ())
        if not isinstance(raw_subparts, (list, tuple)):
            return False
        for item in raw_subparts:
            if not isinstance(item, str) or not item.strip():
                return False
        if len(raw_subparts) > 0:
            return False

        return True

    def _account_slots(
        self,
        context: AgentContext,
    ) -> Tuple[
        List[str],          # sorted_f: fully resolved slots
        List[str],          # sorted_p: partially resolved slots
        List[str],          # sorted_b: blocked slots
        List[str],          # sorted_unexp_obs: unexpected observed slots
        List[str],          # sorted_unexp_reg: unexpected unresolved registry slots
        List[str],          # fatal_integrity_defects: fatal integrity defects (trigger blocked_reasons)
        List[str],          # all_accounting_defects: complete audit log of all defects
        Dict[str, Any],     # telemetry: detailed accounting metrics
    ]:
        """
        Partition authoritative expected set E into three mutually exclusive sets:
          F (fully resolved), P (partially resolved), B (blocked).
        Separately identifies fatal_integrity_defects vs ordinary slot failures.
        Validates metadata independently and ahead of content checks.
        Treats any missing question object as a fatal defect regardless of registry membership.
        """
        expected_ids = self._get_validated_expected_slots(context)
        expected_set = set(expected_ids)

        questions = list(context.questions or [])
        raw_unresolved_registry = list(context.unresolved_slots or [])

        fatal_integrity_defects: List[str] = []
        all_accounting_defects: List[str] = []

        # 1. Type-safe inspection of observed question slot IDs
        seen_observed_counts: Dict[str, int] = {}
        q_by_slot: Dict[str, Any] = {}

        for idx, q in enumerate(questions):
            sid = getattr(q, "slot_id", None)
            if sid is None or not isinstance(sid, str) or not sid.strip():
                defect = f"MALFORMED_OBSERVED_SLOT_ID:question[{idx}]:type_{type(sid).__name__}"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                continue
            clean_sid = sid.strip()
            seen_observed_counts[clean_sid] = seen_observed_counts.get(clean_sid, 0) + 1
            if clean_sid not in q_by_slot:
                q_by_slot[clean_sid] = q

        duplicate_observed = {sid for sid, c in seen_observed_counts.items() if c > 1}
        for sid in sorted(duplicate_observed):
            defect = f"DUPLICATE_SLOT_ID:{sid}"
            fatal_integrity_defects.append(defect)
            all_accounting_defects.append(defect)

        unexpected_observed = {sid for sid in seen_observed_counts.keys() if sid not in expected_set}
        for sid in sorted(unexpected_observed):
            defect = f"UNEXPECTED_SLOT_ID:{sid}"
            fatal_integrity_defects.append(defect)
            all_accounting_defects.append(defect)

        # 2. Type-safe inspection of unresolved_slots registry
        clean_unresolved_registry: Set[str] = set()
        for idx, r in enumerate(raw_unresolved_registry):
            r_sid = getattr(r, "slot_id", r)
            if isinstance(r_sid, str) and r_sid.strip():
                clean_unresolved_registry.add(r_sid.strip())
            else:
                defect = f"MALFORMED_UNRESOLVED_REGISTRY_ENTRY:index_{idx}"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)

        unexpected_unresolved = clean_unresolved_registry - expected_set
        for sid in sorted(unexpected_unresolved):
            defect = f"UNEXPECTED_UNRESOLVED_REGISTRY_ID:{sid}"
            fatal_integrity_defects.append(defect)
            all_accounting_defects.append(defect)

        # 3. Partition expected set E
        fully_resolved: Set[str] = set()
        partially_resolved: Set[str] = set()
        blocked: Set[str] = set()
        subquestion_defects: Dict[str, Tuple[str, ...]] = {}

        for sid in expected_ids:
            # Check A: Duplicated slot
            if sid in duplicate_observed:
                blocked.add(sid)
                continue

            # Check B: Missing question object (fatal defect regardless of registry membership)
            if sid not in q_by_slot:
                blocked.add(sid)
                if sid in clean_unresolved_registry:
                    defect = f"UNRESOLVED_SLOT_WITHOUT_QUESTION:{sid}"
                else:
                    defect = f"MISSING_EXPECTED_SLOT:{sid}"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                continue

            q = q_by_slot[sid]
            in_unresolved_registry = (sid in clean_unresolved_registry)

            slot_has_fatal_metadata = False

            # Check C: Prioritized Metadata Validation (executed ahead of content checks)
            raw_is_unres = getattr(q, "is_unresolved", False)
            if not isinstance(raw_is_unres, bool):
                defect = f"MALFORMED_QUESTION_METADATA:{sid}:is_unresolved_type_{type(raw_is_unres).__name__}"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                slot_has_fatal_metadata = True

            failure_code = getattr(q, "failure_code", None)
            if failure_code is not None and (not isinstance(failure_code, str) or not failure_code.strip()):
                defect = f"MALFORMED_QUESTION_METADATA:{sid}:invalid_failure_code"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                slot_has_fatal_metadata = True

            failure_detail = getattr(q, "failure_detail", None)
            if failure_detail is not None and (not isinstance(failure_detail, str) or not failure_detail.strip()):
                defect = f"MALFORMED_QUESTION_METADATA:{sid}:invalid_failure_detail"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                slot_has_fatal_metadata = True

            scope = getattr(q, "unresolved_scope", None)
            if scope is not None and (not isinstance(scope, str) or not scope.strip()):
                defect = f"MALFORMED_QUESTION_METADATA:{sid}:invalid_unresolved_scope"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                slot_has_fatal_metadata = True

            raw_subparts = getattr(q, "unresolved_subparts", ())
            subparts: Tuple[str, ...] = ()
            if not isinstance(raw_subparts, (list, tuple)):
                defect = f"MALFORMED_UNRESOLVED_SUBPARTS:{sid}:invalid_type_{type(raw_subparts).__name__}"
                fatal_integrity_defects.append(defect)
                all_accounting_defects.append(defect)
                slot_has_fatal_metadata = True
            else:
                parsed_subparts: List[str] = []
                subparts_valid = True
                for elem in raw_subparts:
                    if not isinstance(elem, str) or not elem.strip():
                        subparts_valid = False
                        break
                    parsed_subparts.append(elem.strip())
                if not subparts_valid:
                    defect = f"MALFORMED_UNRESOLVED_SUBPARTS:{sid}:invalid_element"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True
                else:
                    subparts = tuple(parsed_subparts)

            # Check D: Cross-Field Stage 2A Consistency Validation
            if not raw_is_unres:
                # Case 1: Question claims clean status (is_unresolved=False)
                has_active_failure_markers = (
                    (scope is not None)
                    or (len(subparts) > 0)
                    or (failure_code is not None)
                    or (failure_detail is not None)
                )
                if has_active_failure_markers:
                    defect = (
                        f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:"
                        f"is_unresolved_false_with_active_markers(scope={scope},subparts={subparts},code={failure_code},detail={failure_detail})"
                    )
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True

                if in_unresolved_registry:
                    defect = f"UNRESOLVED_REGISTRY_CONTRADICTION:{sid}:registry_unresolved_but_question_clean"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True
            else:
                # Case 2: Question claims unresolved status (is_unresolved=True)
                if not in_unresolved_registry:
                    defect = f"UNRESOLVED_REGISTRY_CONTRADICTION:{sid}:question_unresolved_but_absent_from_registry"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True

                if failure_code is None:
                    defect = f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:unresolved_true_with_missing_failure_code"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True

                if failure_detail is None:
                    defect = f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:unresolved_true_with_missing_failure_detail"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True

                if scope is None:
                    defect = f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:unresolved_true_with_missing_scope"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True
                elif scope == "question":
                    if len(subparts) > 0:
                        defect = (
                            f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:"
                            f"question_scope_with_nonempty_subparts_{','.join(subparts)}"
                        )
                        fatal_integrity_defects.append(defect)
                        all_accounting_defects.append(defect)
                        slot_has_fatal_metadata = True
                elif scope == "subquestion":
                    if len(subparts) == 0:
                        defect = f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:subquestion_scope_without_subparts"
                        fatal_integrity_defects.append(defect)
                        all_accounting_defects.append(defect)
                        slot_has_fatal_metadata = True
                else:
                    defect = f"INCONSISTENT_UNRESOLVED_METADATA:{sid}:unknown_scope_{scope}"
                    fatal_integrity_defects.append(defect)
                    all_accounting_defects.append(defect)
                    slot_has_fatal_metadata = True

            # Check E: Content Usability Validation (Evaluated independently, retaining all defects)
            raw_text = getattr(q, "question_text", None)
            raw_sol = getattr(q, "solution", None)
            text_valid = isinstance(raw_text, str) and bool(raw_text.strip())
            sol_valid = isinstance(raw_sol, str) and bool(raw_sol.strip())

            if not text_valid or not sol_valid:
                all_accounting_defects.append(f"EMPTY_CONTENT:{sid}")

            # Check F: Slot Classification
            if slot_has_fatal_metadata:
                # Fatal metadata or contradictory lifecycle always fails closed into B
                blocked.add(sid)
                continue

            if not text_valid or not sol_valid:
                # Clean metadata but empty content is an ordinary blocked slot
                blocked.add(sid)
                continue

            if raw_is_unres:
                if scope == "subquestion":
                    partially_resolved.add(sid)
                    subquestion_defects[sid] = subparts
                    all_accounting_defects.append(f"SUBPART_UNRESOLVED:{sid}:{','.join(subparts)}")
                elif scope == "question":
                    blocked.add(sid)
                    all_accounting_defects.append(f"QUESTION_UNRESOLVED:{sid}:{failure_code}")
                continue

            # Clean, resolvable question
            fully_resolved.add(sid)

        # 4. Strict Partition Verification
        assert (fully_resolved | partially_resolved | blocked) == expected_set, "F U P U B must equal E"
        assert len(fully_resolved & partially_resolved) == 0, "F and P must be disjoint"
        assert len(fully_resolved & blocked) == 0, "F and B must be disjoint"
        assert len(partially_resolved & blocked) == 0, "P and B must be disjoint"

        sorted_f = [s for s in expected_ids if s in fully_resolved]
        sorted_p = [s for s in expected_ids if s in partially_resolved]
        sorted_b = [s for s in expected_ids if s in blocked]
        sorted_unexp_obs = sorted(unexpected_observed)
        sorted_unexp_reg = sorted(unexpected_unresolved)

        telemetry = {
            "expected_slots": len(expected_ids),
            "fully_resolved_count": len(sorted_f),
            "partially_resolved_count": len(sorted_p),
            "blocked_count": len(sorted_b),
            "unexpected_observed_count": len(sorted_unexp_obs),
            "unexpected_unresolved_count": len(sorted_unexp_reg),
            "fatal_integrity_defects_count": len(fatal_integrity_defects),
            "fully_resolved_slots": sorted_f,
            "partially_resolved_slots": sorted_p,
            "blocked_slots": sorted_b,
            "unexpected_observed_slots": sorted_unexp_obs,
            "unexpected_unresolved_slots": sorted_unexp_reg,
            "subquestion_defects": subquestion_defects,
        }

        return (
            sorted_f,
            sorted_p,
            sorted_b,
            sorted_unexp_obs,
            sorted_unexp_reg,
            fatal_integrity_defects,
            all_accounting_defects,
            telemetry,
        )

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
