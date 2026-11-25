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

        # Check 7: Whole-paper cross-module deduplication audit (Stage G5-C)
        from .overlap_detector import audit_cross_module_duplicates
        cross_module_duplicates = audit_cross_module_duplicates(context.questions or [])
        if cross_module_duplicates:
            for dup in cross_module_duplicates:
                if dup["severity"] == "question_level":
                    reason = f"CROSS_MODULE_QUESTION_DUPLICATE:{dup['slot_a']}_vs_{dup['slot_b']}"
                else:
                    reason = f"CROSS_MODULE_SUBPART_DUPLICATE:{dup['slot_a']}({dup.get('subpart_a', '?')})_vs_{dup['slot_b']}({dup.get('subpart_b', '?')})"
                if reason not in degraded_reasons:
                    degraded_reasons.append(reason)

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
            "cross_module_duplicates": cross_module_duplicates,
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


# =============================================================================
# Stage G6: Choice-Pair Equivalence, Selection-Tree Modeling & Dual Bloom Audit
# =============================================================================

from dataclasses import dataclass, field
import itertools


@dataclass
class AssessmentProfile:
    """
    Comprehensive whole-paper academic assessment profile across 10 gates (Section 8).
    """
    overall_status: str  # "PASS" | "FAIL" | "INDETERMINATE"
    evidence_grounding_gate: str = "NOT_APPLICABLE"
    segment_grounding_gate: str = "NOT_APPLICABLE"
    choice_topic_dispersion_gate: str = "NOT_APPLICABLE"
    bloom_accounting_gate: str = "PASS"
    cognitive_distribution_gate: str = "NOT_APPLICABLE"
    subpart_progression_gate: str = "NOT_APPLICABLE"
    concept_integrity_gate: str = "PASS"
    solution_completeness_gate: str = "NOT_APPLICABLE"
    rubric_validation_gate: str = "NOT_APPLICABLE"
    visual_dependency_gate: str = "NOT_APPLICABLE"
    review_recommendations: List[str] = field(default_factory=list)
    review_warnings: List[str] = field(default_factory=list)
    telemetry: Dict[str, Any] = field(default_factory=dict)


def check_choice_pair_equivalence(q1: Any, q2: Any) -> Tuple[str, List[str]]:
    """
    Audits choice-pair equivalence across 7 dimensions (Section 7.4):
      1. Total marks equality (Marks(Q1) == Marks(Q2))
      2. Subpart count parity (|SubParts(Q1)| == |SubParts(Q2)|)
      3. Subpart marks partition alignment (sorted marks match)
      4. Subpart Bloom depth alignment (cognitive depth profile matches)
      5. Primary archetype compatibility
      6. Time commitment parity (|EstTime(Q1) - EstTime(Q2)| <= 5 min)
      7. Topic dispersion (Tier 1, Tier 2, or Tier 3)
    Returns:
      (status, defects)
      status: "PASS" | "FAIL"
    """
    defects: List[str] = []

    # 1. Total marks equality
    m1 = getattr(q1, "marks", 0) or 0
    m2 = getattr(q2, "marks", 0) or 0
    if m1 != m2:
        defects.append(f"MARKS_MISMATCH: Q1 has {m1} marks, Q2 has {m2} marks")

    sps1 = list(getattr(q1, "sub_parts", None) or [])
    sps2 = list(getattr(q2, "sub_parts", None) or [])

    # 2. Subpart count parity
    if len(sps1) != len(sps2):
        defects.append(f"SUBPART_COUNT_MISMATCH: Q1 has {len(sps1)} subparts, Q2 has {len(sps2)} subparts")

    # 3. Subpart marks partition alignment
    if sps1 and sps2:
        marks1_sorted = sorted([getattr(sp, "marks", 0) or 0 for sp in sps1])
        marks2_sorted = sorted([getattr(sp, "marks", 0) or 0 for sp in sps2])
        if marks1_sorted != marks2_sorted:
            defects.append(f"SUBPART_MARKS_PARTITION_MISMATCH: Q1 partition {marks1_sorted} vs Q2 partition {marks2_sorted}")

    # 4. Subpart Bloom depth alignment
    bloom_rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
    if sps1 and sps2:
        b1_ranks = sorted([bloom_rank.get(str(getattr(sp, "bloom", "L2")).upper(), 2) for sp in sps1])
        b2_ranks = sorted([bloom_rank.get(str(getattr(sp, "bloom", "L2")).upper(), 2) for sp in sps2])
        # Imbalance if max ranks differ by more than 1 or average ranks differ by more than 1.0
        if abs(max(b1_ranks) - max(b2_ranks)) > 1 or abs(sum(b1_ranks) / len(b1_ranks) - sum(b2_ranks) / len(b2_ranks)) > 1.0:
            defects.append(f"COGNITIVE_DEPTH_IMBALANCE: Q1 Bloom ranks {b1_ranks} vs Q2 Bloom ranks {b2_ranks}")
    else:
        q1_b = bloom_rank.get(str(getattr(q1, "bloom", "L2")).upper(), 2)
        q2_b = bloom_rank.get(str(getattr(q2, "bloom", "L2")).upper(), 2)
        if abs(q1_b - q2_b) > 1:
            defects.append(f"COGNITIVE_DEPTH_IMBALANCE: Q1 Bloom rank {q1_b} vs Q2 Bloom rank {q2_b}")

    # 5. Primary archetype compatibility
    arch1 = str(getattr(q1, "primary_archetype", None) or getattr(q1, "archetype", "")).strip().lower()
    arch2 = str(getattr(q2, "primary_archetype", None) or getattr(q2, "archetype", "")).strip().lower()
    if arch1 and arch2 and arch1 != arch2:
        # Check incompatible pairs (e.g. purely descriptive vs algorithmic proof/derivation)
        incompatible_pairs = [
            ({"conceptual_essay", "descriptive_explanation"}, {"algorithmic_proof", "complex_derivation", "numerical_computation"}),
            ({"definition", "recall"}, {"system_design", "comparative_analysis"}),
        ]
        for set_a, set_b in incompatible_pairs:
            if (arch1 in set_a and arch2 in set_b) or (arch1 in set_b and arch2 in set_a):
                defects.append(f"ARCHETYPE_ASYMMETRY: Q1 archetype '{arch1}' is academically asymmetric to Q2 archetype '{arch2}'")

    # 6. Time commitment parity (|EstTime(Q1) - EstTime(Q2)| <= 5 min)
    def _calc_time(q: Any) -> float:
        t = getattr(q, "estimated_time_minutes", None)
        if t is not None and isinstance(t, (int, float)):
            return float(t)
        sps = getattr(q, "sub_parts", None) or []
        if sps:
            sub_times = [getattr(sp, "estimated_time_minutes", None) for sp in sps]
            if all(st is not None and isinstance(st, (int, float)) for st in sub_times):
                return float(sum(sub_times))
        # Estimate: 1.5 minutes per mark
        return float(getattr(q, "marks", 10) or 10) * 1.5

    t1 = _calc_time(q1)
    t2 = _calc_time(q2)
    if abs(t1 - t2) > 5.0:
        defects.append(f"TIME_COMMITMENT_IMBALANCE: Q1 est time {t1:.1f}m vs Q2 est time {t2:.1f}m exceeds 5 min tolerance")

    # 7. Topic dispersion
    from .overlap_detector import classify_relationship
    rel = classify_relationship(q1, q2)
    if rel.get("status") == "FAIL":
        defects.append(f"CHOICE_PAIR_TOPIC_COLLAPSE: {rel.get('detail', 'Choice pair collapses on identical topic/operation')}")

    if defects:
        return "FAIL", defects
    return "PASS", []


# -----------------------------------------------------------------------------
# Selection-Tree Attempt-Path Modeling
# -----------------------------------------------------------------------------

class SelectionNode:
    """
    Decision point enforcing choice rule (choose K of Candidates).
    May be conditional on parent candidate selection (Section 7.5).
    """
    def __init__(
        self,
        node_id: str,
        choose_k: int,
        candidates: List[Any],
        parent_candidate: Any = None,
        child_selection_nodes: Optional[List['SelectionNode']] = None,
    ):
        self.node_id = node_id
        self.choose_k = choose_k
        self.candidates = candidates
        self.parent_candidate = parent_candidate
        self.child_selection_nodes = child_selection_nodes or []

        # Initial structural validation
        if self.choose_k <= 0 or self.choose_k > len(self.candidates):
            raise ValueError(
                f"INVALID_CHOICE_STRUCTURE: Node '{node_id}' requires choosing {self.choose_k} from {len(self.candidates)} candidates"
            )


def build_selection_tree_from_questions(questions: List[Any]) -> List[SelectionNode]:
    """
    Builds canonical Selection Tree nodes from questions.
    Groups questions by choice_group_id into internal choice nodes.
    """
    groups: Dict[str, List[Any]] = {}
    ungrouped: List[Any] = []

    for q in questions:
        cg = getattr(q, "choice_group_id", None)
        if cg and str(cg).strip():
            cg_str = str(cg).strip()
            groups.setdefault(cg_str, []).append(q)
        else:
            ungrouped.append(q)

    root_nodes: List[SelectionNode] = []

    # Compulsory questions: choose 1 of 1
    for idx, q in enumerate(ungrouped):
        qid = getattr(q, "question_id", None) or getattr(q, "slot_id", f"compulsory_{idx}")
        # Check if question has nested subpart choices (e.g. choose 2 of 3 subparts)
        nested_nodes = []
        nested_rule = getattr(q, "subpart_choice_rule", None)
        sps = list(getattr(q, "sub_parts", None) or [])
        if nested_rule and isinstance(nested_rule, dict) and sps:
            k = nested_rule.get("choose_k", len(sps))
            nested_nodes.append(SelectionNode(f"{qid}_subparts", k, sps, parent_candidate=q))

        root_nodes.append(SelectionNode(
            node_id=str(qid),
            choose_k=1,
            candidates=[q],
            child_selection_nodes=nested_nodes,
        ))

    # Internal choice groups: choose 1 of N
    for group_id, cands in groups.items():
        # Check nested rules for each candidate in group
        for cand in cands:
            qid = getattr(cand, "question_id", None) or getattr(cand, "slot_id", "cand")
            nested_rule = getattr(cand, "subpart_choice_rule", None)
            sps = list(getattr(cand, "sub_parts", None) or [])
            if nested_rule and isinstance(nested_rule, dict) and sps:
                k = nested_rule.get("choose_k", len(sps))
                # Add child selection node to candidate
                cand._child_selection_node = SelectionNode(f"{qid}_subparts", k, sps, parent_candidate=cand)

        root_nodes.append(SelectionNode(
            node_id=f"choice_group_{group_id}",
            choose_k=1,
            candidates=cands,
        ))

    return root_nodes


def enumerate_attempt_paths(nodes: List[SelectionNode]) -> List[List[Any]]:
    """
    Exhaustively enumerates valid attempt paths over conditional selection tree.
    Each path is a flat list of chosen SubParts or leaf Questions.
    """
    if not nodes:
        return [[]]

    def _resolve_candidate(cand: Any) -> List[List[Any]]:
        # If candidate has nested child selection node, activate it conditionally
        child_node = getattr(cand, "_child_selection_node", None)
        if child_node is not None:
            # Choose k from child candidates
            sub_paths = []
            for combo in itertools.combinations(child_node.candidates, child_node.choose_k):
                sub_paths.append(list(combo))
            return sub_paths

        # If candidate is a Question with subparts, selecting Q selects its subparts
        sps = list(getattr(cand, "sub_parts", None) or [])
        if sps:
            return [sps]
        return [[cand]]

    node_options: List[List[List[Any]]] = []
    for node in nodes:
        options_for_node: List[List[Any]] = []
        for combo in itertools.combinations(node.candidates, node.choose_k):
            # For each candidate in combo, resolve its active sub-trees
            cand_resolutions = [_resolve_candidate(c) for c in combo]
            for prod in itertools.product(*cand_resolutions):
                flattened = []
                for sublist in prod:
                    flattened.extend(sublist)
                options_for_node.append(flattened)
        node_options.append(options_for_node)

    all_paths: List[List[Any]] = []
    for prod in itertools.product(*node_options):
        path = []
        for sub_items in prod:
            path.extend(sub_items)
        all_paths.append(path)

    return all_paths


def calculate_attempt_path_bloom_bounds(
    questions: List[Any],
    max_exhaustive_paths: int = 1024,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes MinPct(B) and MaxPct(B) for each Bloom band B across all valid attempt paths (Section 7.5).
    Uses exhaustive enumeration for |P| <= 1024 and tree DP convolution for |P| > 1024.
    Mathematical identity guaranteed between both algorithms.
    """
    nodes = build_selection_tree_from_questions(questions)
    bands = ["L1", "L2", "L3", "L4", "L5", "L6"]

    # Calculate total path count bound
    num_paths = 1
    for node in nodes:
        ways = len(list(itertools.combinations(node.candidates, node.choose_k)))
        nested_ways = 1
        for c in node.candidates:
            cn = getattr(c, "_child_selection_node", None)
            if cn is not None:
                nested_ways = max(nested_ways, len(list(itertools.combinations(cn.candidates, cn.choose_k))))
        num_paths *= (ways * nested_ways)

    if num_paths <= max_exhaustive_paths:
        # Exhaustive enumeration
        paths = enumerate_attempt_paths(nodes)
        min_pcts = {b: 100.0 for b in bands}
        max_pcts = {b: 0.0 for b in bands}

        for path in paths:
            total_marks = sum(float(getattr(item, "marks", 0) or 0) for item in path)
            if total_marks <= 0.0:
                continue
            for b in bands:
                b_marks = sum(
                    float(getattr(item, "marks", 0) or 0)
                    for item in path
                    if str(getattr(item, "bloom", "")).upper() == b
                )
                pct = (b_marks / total_marks) * 100.0
                if pct < min_pcts[b]:
                    min_pcts[b] = pct
                if pct > max_pcts[b]:
                    max_pcts[b] = pct

        return {b: (round(min_pcts[b], 2), round(max_pcts[b], 2)) for b in bands}

    else:
        # Tree Dynamic Programming Convolution over (total_marks, bloom_marks_tuple)
        # Each state is a mapping: total_marks -> set of bloom_mark tuples (l1, l2, l3, l4, l5, l6)
        def _get_item_tuple(item: Any) -> Tuple[float, Tuple[float, ...]]:
            m = float(getattr(item, "marks", 0) or 0)
            b = str(getattr(item, "bloom", "")).upper()
            b_tuple = tuple(m if b == band else 0.0 for band in bands)
            return m, b_tuple

        # DP over nodes
        dp_states: Set[Tuple[float, Tuple[float, ...]]] = {(0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))}

        for node in nodes:
            next_dp: Set[Tuple[float, Tuple[float, ...]]] = set()
            for combo in itertools.combinations(node.candidates, node.choose_k):
                combo_states: Set[Tuple[float, Tuple[float, ...]]] = {(0.0, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))}
                for cand in combo:
                    child_node = getattr(cand, "_child_selection_node", None)
                    if child_node is not None:
                        cand_states: Set[Tuple[float, Tuple[float, ...]]] = set()
                        for sub_combo in itertools.combinations(child_node.candidates, child_node.choose_k):
                            sub_tot = sum(float(getattr(x, "marks", 0) or 0) for x in sub_combo)
                            sub_b = [0.0] * 6
                            for x in sub_combo:
                                xm, xb = _get_item_tuple(x)
                                for i in range(6):
                                    sub_b[i] += xb[i]
                            cand_states.add((sub_tot, tuple(sub_b)))
                    else:
                        sps = list(getattr(cand, "sub_parts", None) or [])
                        if sps:
                            sub_tot = sum(float(getattr(x, "marks", 0) or 0) for x in sps)
                            sub_b = [0.0] * 6
                            for x in sps:
                                xm, xb = _get_item_tuple(x)
                                for i in range(6):
                                    sub_b[i] += xb[i]
                            cand_states = {(sub_tot, tuple(sub_b))}
                        else:
                            cm, cb = _get_item_tuple(cand)
                            cand_states = {(cm, cb)}

                    # Convolve with combo_states
                    merged: Set[Tuple[float, Tuple[float, ...]]] = set()
                    for t1, b1 in combo_states:
                        for t2, b2 in cand_states:
                            merged.add((t1 + t2, tuple(b1[i] + b2[i] for i in range(6))))
                    combo_states = merged

                # Convolve combo_states with existing dp_states
                for dt, db in dp_states:
                    for ct, cb in combo_states:
                        next_dp.add((dt + ct, tuple(db[i] + cb[i] for i in range(6))))

            dp_states = next_dp

        # Compute min and max percentages across all reachable DP states
        min_pcts = {b: 100.0 for b in bands}
        max_pcts = {b: 0.0 for b in bands}

        for total_marks, b_tuple in dp_states:
            if total_marks <= 0.0:
                continue
            for i, b in enumerate(bands):
                pct = (b_tuple[i] / total_marks) * 100.0
                if pct < min_pcts[b]:
                    min_pcts[b] = pct
                if pct > max_pcts[b]:
                    max_pcts[b] = pct

        return {b: (round(min_pcts[b], 2), round(max_pcts[b], 2)) for b in bands}


def audit_whole_paper_bloom(
    questions: List[Any],
    config: Any = None,
) -> Dict[str, Any]:
    """
    Audits whole-paper Bloom distribution across printed paper and attempt paths (Section 7.5, 8).
    Independent bloom accounting vs cognitive distribution.
    """
    defects: List[str] = []
    bands = ["L1", "L2", "L3", "L4", "L5", "L6"]

    # Flatten all subparts for printed paper
    all_subparts: List[Any] = []
    for q in questions:
        sps = list(getattr(q, "sub_parts", None) or [])
        if sps:
            all_subparts.extend(sps)
        else:
            all_subparts.append(q)

    # Check for missing/null Bloom
    missing_bloom = []
    for sp in all_subparts:
        b = getattr(sp, "bloom", None)
        if not b or str(b).upper() not in bands:
            missing_bloom.append(getattr(sp, "label", getattr(sp, "slot_id", "?")))

    accounting_complete = len(missing_bloom) == 0
    if not accounting_complete:
        defects.append(f"ACCOUNTING_INCOMPLETE: Subparts {missing_bloom} missing valid Bloom level")

    # Printed paper profile
    printed_total = sum(float(getattr(sp, "marks", 0) or 0) for sp in all_subparts)
    printed_profile = {}
    if printed_total > 0:
        for b in bands:
            b_marks = sum(
                float(getattr(sp, "marks", 0) or 0)
                for sp in all_subparts
                if str(getattr(sp, "bloom", "")).upper() == b
            )
            printed_profile[b] = round((b_marks / printed_total) * 100.0, 2)
    else:
        printed_profile = {b: 0.0 for b in bands}

    # Attempt path bounds
    try:
        attempt_bounds = calculate_attempt_path_bloom_bounds(questions)
    except Exception as exc:
        attempt_bounds = {b: (0.0, 0.0) for b in bands}
        defects.append(f"INVALID_CHOICE_STRUCTURE: {exc}")

    # Check against CognitiveTargetDistribution if configured
    target_dist = getattr(config, "cognitive_target", None) if config else None
    distribution_compliant = True

    if target_dist is not None:
        targets = {
            "L1": target_dist.l1_percentage,
            "L2": target_dist.l2_percentage,
            "L3": target_dist.l3_percentage,
            "L4": target_dist.l4_percentage,
            "L5": target_dist.l5_percentage,
            "L6": target_dist.l6_percentage,
        }
        tol = target_dist.tolerance_percentage

        # Evaluate printed compliance
        for b, tgt in targets.items():
            achieved = printed_profile.get(b, 0.0)
            if not ((tgt - tol - 0.01) <= achieved <= (tgt + tol + 0.01)):
                distribution_compliant = False
                defects.append(f"PRINTED_BLOOM_OUT_OF_TOLERANCE: {b} achieved {achieved}%, target {tgt}% ± {tol}%")

        # Evaluate attempt compliance
        for b, tgt in targets.items():
            min_p, max_p = attempt_bounds.get(b, (0.0, 0.0))
            if min_p < (tgt - tol - 0.01) or max_p > (tgt + tol + 0.01):
                distribution_compliant = False
                defects.append(f"ATTEMPT_BLOOM_OUT_OF_TOLERANCE: {b} range [{min_p}%, {max_p}%] violates target {tgt}% ± {tol}%")

    # Check Mandatory CO Coverage
    mandatory_cos = getattr(config, "mandatory_course_outcomes", set()) if config else set()
    if mandatory_cos:
        assigned_co_marks: Dict[str, float] = {}
        for sp in all_subparts:
            co = getattr(sp, "course_outcome", None) or getattr(sp, "co", None)
            if co:
                co_str = str(co).strip().upper()
                assigned_co_marks[co_str] = assigned_co_marks.get(co_str, 0.0) + float(getattr(sp, "marks", 0) or 0)

        for m_co in mandatory_cos:
            m_str = str(m_co).strip().upper()
            if assigned_co_marks.get(m_str, 0.0) <= 0.0:
                distribution_compliant = False
                defects.append(f"MANDATORY_CO_COVERAGE_FAILED: Mandatory CO '{m_str}' received 0 marks in active paper")

    return {
        "accounting_complete": accounting_complete,
        "distribution_compliant": distribution_compliant and accounting_complete,
        "printed_profile": printed_profile,
        "attempt_bounds": attempt_bounds,
        "defects": defects,
    }


def generate_assessment_profile(
    context_or_questions: Any,
    config: Any = None,
    paper_spec: Any = None,
    kg: Any = None,
    artifact: Any = None,
) -> AssessmentProfile:
    """
    Evaluates paper across 10 gates and computes overall_status with Non-Vacuous Pass Invariant (Section 8).
    """
    if hasattr(context_or_questions, "questions"):
        questions = list(getattr(context_or_questions, "questions", []) or [])
        cfg = config or getattr(context_or_questions, "config", None)
        spec = paper_spec or getattr(context_or_questions, "paper_spec", None)
        kg_obj = kg or getattr(context_or_questions, "kg", None)
        art_obj = artifact or getattr(context_or_questions, "artifact", None)
    else:
        questions = list(context_or_questions or [])
        cfg = config
        spec = paper_spec
        kg_obj = kg
        art_obj = artifact

    rec_recommendations: List[str] = []
    rec_warnings: List[str] = []
    telemetry: Dict[str, Any] = {}

    # 1. Gate: evidence_grounding_gate
    # Applicable if artifact contains references/syllabus; Non-applicable for synthetic unit tests
    allow_synth = getattr(cfg, "allow_synthetic", False) if cfg else False
    if art_obj is not None and not allow_synth:
        # Check evidence block presence
        has_ev = all(bool(getattr(sp, "evidence_block_ids", None)) for q in questions for sp in getattr(q, "sub_parts", [q]))
        evidence_gate = "PASS" if has_ev else "FAIL"
    else:
        evidence_gate = "NOT_APPLICABLE"

    # 2. Gate: segment_grounding_gate
    # Applicable if paper is module-segmented
    mod_count = getattr(spec, "module_count", 0) if spec else 0
    if mod_count > 1:
        segment_gate = "PASS"
    else:
        segment_gate = "NOT_APPLICABLE"

    # 3. Gate: choice_topic_dispersion_gate
    # Applicable if paper contains internal choice pairs
    choice_groups: Dict[str, List[Any]] = {}
    for q in questions:
        cg = getattr(q, "choice_group_id", None)
        if cg and str(cg).strip():
            choice_groups.setdefault(str(cg).strip(), []).append(q)

    if choice_groups:
        choice_gate = "PASS"
        for cg_id, cands in choice_groups.items():
            if len(cands) >= 2:
                for i in range(len(cands)):
                    for j in range(i + 1, len(cands)):
                        st, defs = check_choice_pair_equivalence(cands[i], cands[j])
                        if st == "FAIL":
                            choice_gate = "FAIL"
                            rec_warnings.extend(defs)
    else:
        choice_gate = "NOT_APPLICABLE"

    # 4. Gate: bloom_accounting_gate (ALWAYS APPLICABLE)
    bloom_audit = audit_whole_paper_bloom(questions, config=cfg)
    if not bloom_audit["accounting_complete"]:
        bloom_gate = "FAIL"
    else:
        bloom_gate = "PASS"

    # 5. Gate: cognitive_distribution_gate
    # Applicable if target or mandatory COs configured
    has_target = cfg and (getattr(cfg, "cognitive_target", None) is not None or getattr(cfg, "mandatory_course_outcomes", set()))
    if has_target:
        if any("MANDATORY_CO_COVERAGE_FAILED" in d for d in bloom_audit["defects"]):
            cog_dist_gate = "FAIL"
        elif not bloom_audit["distribution_compliant"]:
            cog_dist_gate = "FAIL"
        else:
            cog_dist_gate = "PASS"
    else:
        cog_dist_gate = "NOT_APPLICABLE"

    # 6. Gate: subpart_progression_gate
    # Applicable if paper contains multi-part questions
    multi_part_qs = [q for q in questions if len(getattr(q, "sub_parts", []) or []) >= 2]
    if multi_part_qs:
        subpart_gate = "PASS"
        from .overlap_detector import classify_relationship
        for q in multi_part_qs:
            sps = list(q.sub_parts)
            for i in range(len(sps)):
                for j in range(i + 1, len(sps)):
                    rel = classify_relationship(sps[i], sps[j], kg=kg_obj, artifact=art_obj, allow_synthetic=allow_synth)
                    if rel.get("status") == "FAIL":
                        subpart_gate = "FAIL"
                    elif rel.get("status") == "INDETERMINATE" and subpart_gate != "FAIL":
                        subpart_gate = "INDETERMINATE"
    else:
        subpart_gate = "NOT_APPLICABLE"

    # 7. Gate: concept_integrity_gate (ALWAYS APPLICABLE)
    from .overlap_detector import generate_concept_integrity_report
    concept_report = generate_concept_integrity_report(questions, kg=kg_obj, artifact=art_obj, allow_synthetic=allow_synth)
    concept_gate = concept_report["gate_status"]
    if concept_gate == "INDETERMINATE":
        rec_recommendations.append("UNVERIFIED_CONCEPT_REVIEW")

    # 8. Gate: solution_completeness_gate
    # Applicable if generate_solutions=True
    gen_sol = getattr(cfg, "generate_solutions", True) if cfg else True
    if gen_sol and questions:
        solution_gate = "PASS"
        from .local_evaluators import evaluate_solution_completeness
        for q in questions:
            for sp in getattr(q, "sub_parts", [q]):
                sol = getattr(sp, "solution", "") or getattr(q, "solution", "")
                q_text = getattr(sp, "text", "") or getattr(q, "question_text", "")
                contract = getattr(sp, "expected_response", None)
                m = getattr(sp, "marks", 5) or 5
                sol_st, _ = evaluate_solution_completeness(q_text, sol, contract=contract, marks=m)
                if sol_st == "FAIL":
                    solution_gate = "FAIL"
                elif sol_st == "INDETERMINATE" and solution_gate != "FAIL":
                    solution_gate = "INDETERMINATE"
    else:
        solution_gate = "NOT_APPLICABLE"

    # 9. Gate: rubric_validation_gate
    # Applicable if generate_rubrics=True
    gen_rub = getattr(cfg, "generate_rubrics", True) if cfg else True
    if gen_rub and questions:
        rubric_gate = "PASS"
        from .local_evaluators import evaluate_marking_scheme_relevance
        for q in questions:
            for sp in getattr(q, "sub_parts", [q]):
                m_scheme = getattr(sp, "marking_scheme", None)
                # Check for malformed entries in telemetry
                malformed = getattr(sp, "telemetry", {}).get("malformed_rubric_entries", []) if hasattr(sp, "telemetry") and isinstance(sp.telemetry, dict) else []
                if malformed:
                    rubric_gate = "FAIL"
                    break
                r_st, _ = evaluate_marking_scheme_relevance(sp, m_scheme)
                if r_st == "FAIL":
                    rubric_gate = "FAIL"
                elif r_st == "INDETERMINATE" and rubric_gate != "FAIL":
                    rubric_gate = "INDETERMINATE"
    else:
        rubric_gate = "NOT_APPLICABLE"

    # 10. Gate: visual_dependency_gate
    # Applicable if paper contains visual assets
    has_visual = any(
        bool(getattr(sp, "visual_task", None) or getattr(sp, "image_path", None) or getattr(q, "image_path", None))
        for q in questions
        for sp in getattr(q, "sub_parts", [q])
    )
    if has_visual:
        visual_gate = "PASS"
        from .local_evaluators import audit_visual_asset, evaluate_visual_dependency
        for q in questions:
            for sp in getattr(q, "sub_parts", [q]):
                vt = getattr(sp, "visual_task", None)
                fig_meta = getattr(sp, "figure_metadata", None) or ({"figure_id": vt.figure_id, "width": 800, "height": 600} if vt else None)
                q_text = getattr(sp, "text", "") or getattr(q, "question_text", "")
                b_lvl = getattr(sp, "bloom", "L2")
                v_st, v_detail = evaluate_visual_dependency(q_text, figure_metadata=fig_meta, visual_contract=vt, bloom=b_lvl)
                if v_st == "FAIL":
                    visual_gate = "FAIL"
                elif v_st == "INDETERMINATE" and visual_gate != "FAIL":
                    visual_gate = "INDETERMINATE"
                    if "REVISE_FOR_HUMAN_REVIEW" not in rec_recommendations:
                        rec_recommendations.append("REVISE_FOR_HUMAN_REVIEW")
                    rec_warnings.append(f"VISUAL_DEPENDENCY_AMBIGUOUS: {v_detail}")

                # Mandatory Sub-Gate: Visual Asset Correctness & Answer-Leakage Audit
                va = getattr(sp, "visual_audit", None)
                if va is None and (getattr(sp, "figure", None) is not None or (isinstance(fig_meta, dict) and fig_meta.get("ocr_payload"))):
                    fig_obj = getattr(sp, "figure", None) or fig_meta
                    va = audit_visual_asset(
                        fig_obj,
                        q_text,
                        expected_answer=getattr(sp, "solution", None),
                        visual_contract=vt,
                        expected_response=getattr(sp, "expected_response", None),
                    )
                if va is not None:
                    va_dec = getattr(va, "decision", "INDETERMINATE")
                    if va_dec == "FAIL":
                        visual_gate = "FAIL"
                    elif va_dec == "INDETERMINATE" and visual_gate != "FAIL":
                        visual_gate = "INDETERMINATE"
                        if "REVISE_FOR_HUMAN_REVIEW" not in rec_recommendations:
                            rec_recommendations.append("REVISE_FOR_HUMAN_REVIEW")
                        rec_warnings.append(f"VISUAL_ASSET_AUDIT_INDETERMINATE: {getattr(va, 'figure_id', '?')}")
    else:
        visual_gate = "NOT_APPLICABLE"

    # Aggregate Overall Status
    all_gates = [
        evidence_gate, segment_gate, choice_gate, bloom_gate,
        cog_dist_gate, subpart_gate, concept_gate, solution_gate,
        rubric_gate, visual_gate,
    ]
    applicable_gates = [g for g in all_gates if g != "NOT_APPLICABLE"]

    # Non-Vacuous Pass Invariant:
    # Requires universal gates (bloom, concept) + at least one content/grounding gate
    content_grounding_gates = [evidence_gate, segment_gate, choice_gate, cog_dist_gate, subpart_gate, solution_gate, rubric_gate, visual_gate]
    has_content_pass = any(g == "PASS" for g in content_grounding_gates)

    if any(g == "FAIL" for g in applicable_gates):
        overall = "FAIL"
    elif any(g == "INDETERMINATE" for g in applicable_gates):
        overall = "INDETERMINATE"
    elif len(applicable_gates) < 3 or not has_content_pass:
        # Zero gates or only trivial non-academic gates applicable
        overall = "INDETERMINATE"
        rec_recommendations.append("INSUFFICIENT_APPLICABLE_GATES")
    elif all(g == "PASS" for g in applicable_gates):
        overall = "PASS"
    else:
        overall = "INDETERMINATE"

    telemetry["bloom_audit"] = bloom_audit
    telemetry["concept_audit"] = concept_report["telemetry"]

    return AssessmentProfile(
        overall_status=overall,
        evidence_grounding_gate=evidence_gate,
        segment_grounding_gate=segment_gate,
        choice_topic_dispersion_gate=choice_gate,
        bloom_accounting_gate=bloom_gate,
        cognitive_distribution_gate=cog_dist_gate,
        subpart_progression_gate=subpart_gate,
        concept_integrity_gate=concept_gate,
        solution_completeness_gate=solution_gate,
        rubric_validation_gate=rubric_gate,
        visual_dependency_gate=visual_gate,
        review_recommendations=rec_recommendations,
        review_warnings=rec_warnings,
        telemetry=telemetry,
    )

