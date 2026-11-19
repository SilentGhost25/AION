"""
Pipeline bridge — the single entry point for v3.

Callers use `run_v3_pipeline()` if they want to opt into v3 unconditionally.
Callers use `run_pipeline_v3_or_legacy()` if they want flag-driven dispatch.

The v3 pipeline:
    1. Builds the KG from the artifact
    2. Constructs the agent graph via DependencyFactory
    3. Runs AgentOrchestrator
    4. Converts the OrchestratorResult to legacy shape
    5. Reconciles post-adaptation population with pre-adaptation QA invariants
    6. Returns `(paper_parts, full_paper, meta)`
"""

from __future__ import annotations

import collections.abc
import copy
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from .agent_orchestrator import AgentOrchestrator
from .base import AgentContext
from .checking_contracts import (
    BLOCK_REASON_INSUFFICIENT_QUESTIONS,
    BLOCK_REASON_UNRESOLVED_SLOTS,
    QAReport,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_PARTIAL,
    STATUS_PASS,
)
from .dependency_factory import (
    DependencyFactory,
    FactoryResult,
    default_api_caller_factory,
    default_llm_caller_factory,
    default_sympy_verifier_factory,
    default_vlm_caller_factory,
)
from .legacy_adapter import to_legacy_shape
from .orchestrator_contracts import OrchestratorResult


# -----------------------------------------------------------------------------
# Feature flag
# -----------------------------------------------------------------------------

V3_FLAG_ENV = "AION_ENABLE_V3_AGENTS"


def v3_enabled() -> bool:
    val = os.getenv(V3_FLAG_ENV, "false").strip().lower()
    return val in ("true", "1", "yes", "on")


# -----------------------------------------------------------------------------
# 3.1 Bounded Safety & Serialization Primitives
# -----------------------------------------------------------------------------


def _safe_str(val: Any, max_len: int = 200) -> str:
    """Safely convert any value to string with bounded length, handling failing __str__."""
    try:
        s = str(val)
        return s[:max_len]
    except Exception:
        return "<unstringable_exception>"


def _safe_repr(val: Any, max_len: int = 500) -> str:
    """Safely convert any value to repr with bounded length, handling failing __repr__."""
    try:
        r = repr(val)
        return r[:max_len]
    except Exception:
        return "<unrepresentable_object>"


def _safe_type_name(val: Any) -> str:
    """Safely extract the class name of an object with bounded length."""
    try:
        t = type(val)
        name = getattr(t, "__name__", None)
        if isinstance(name, str) and name:
            return name[:50]
        return _safe_str(t, 50)
    except Exception:
        return "UnknownType"


def _safe_float_ratio(val: Any) -> Optional[float]:
    """Strictly convert and validate a finite float in [0.0, 1.0], rejecting booleans."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return None
    try:
        f = float(val)
        if math.isfinite(f) and 0.0 <= f <= 1.0:
            return f
    except (OverflowError, ValueError, TypeError):
        pass
    return None


def _safe_telemetry_copy(
    telemetry_obj: Any,
    fatal_defects: Optional[List[str]] = None,
    label: str = "telemetry",
) -> dict:
    """Safely deep-copy any mapping structure without silent data loss."""
    if telemetry_obj is None:
        return {}

    if not isinstance(telemetry_obj, collections.abc.Mapping):
        if fatal_defects is not None:
            fatal_defects.append(
                f"MALFORMED_TELEMETRY_MAPPING:{label}:type_{_safe_type_name(telemetry_obj)}"
            )
        return {
            "_is_corrupted": True,
            "_raw_repr": _safe_repr(telemetry_obj, 500),
            "_conversion_error": "not_a_mapping",
        }

    try:
        dict_copy = dict(telemetry_obj)
        return copy.deepcopy(dict_copy)
    except Exception as exc:
        if fatal_defects is not None:
            fatal_defects.append(
                f"TELEMETRY_COPY_ERROR:{label}:{_safe_type_name(exc)}"
            )
        return {
            "_is_corrupted": True,
            "_raw_repr": _safe_repr(telemetry_obj, 500),
            "_copy_error": _safe_str(exc, 200),
        }


def _safe_serialize_qa_report(
    qa: Any,
    label: str,
    fatal_defects: Optional[List[str]] = None,
) -> Tuple[bool, Optional[dict], Optional[str]]:
    """
    Safely serialize a QAReport object into an independent, JSON-serializable dict snapshot
    using strict allow_nan=False. Returns (is_success, snapshot_or_fallback, defect_code).
    """
    if qa is None:
        return True, None, None

    # Step 1: Protected attribute/property access
    try:
        to_dict_fn = getattr(qa, "to_dict", None)
    except Exception as exc:
        defect = f"QA_REPORT_ACCESSOR_ERROR:{label}:{_safe_type_name(exc)}"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        return (
            False,
            {
                "_is_corrupted": True,
                "_raw_repr": _safe_repr(qa, 500),
                "_serialization_error": f"accessor_error:{_safe_str(exc, 200)}",
            },
            defect,
        )

    if not callable(to_dict_fn):
        defect = f"QA_REPORT_NOT_SERIALIZABLE:{label}:no_callable_to_dict"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        return (
            False,
            {
                "_is_corrupted": True,
                "_raw_repr": _safe_repr(qa, 500),
                "_serialization_error": "no_callable_to_dict",
            },
            defect,
        )

    # Step 2: Protected invocation
    try:
        raw_dict = to_dict_fn()
    except Exception as exc:
        defect = f"QA_REPORT_SERIALIZATION_ERROR:{label}:{_safe_type_name(exc)}"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        return (
            False,
            {
                "_is_corrupted": True,
                "_raw_repr": _safe_repr(qa, 500),
                "_serialization_error": f"call_error:{_safe_str(exc, 200)}",
            },
            defect,
        )

    if not isinstance(raw_dict, dict):
        defect = (
            f"QA_REPORT_SERIALIZATION_TYPE_ERROR:{label}:type_{_safe_type_name(raw_dict)}"
        )
        if fatal_defects is not None:
            fatal_defects.append(defect)
        return (
            False,
            {
                "_is_corrupted": True,
                "_raw_repr": _safe_repr(raw_dict, 500),
                "_serialization_error": "not_a_dict",
            },
            defect,
        )

    # Step 3: Strict JSON roundtrip validation (rejects NaN and Infinity)
    try:
        json_str = json.dumps(raw_dict, allow_nan=False)
        snapshot = json.loads(json_str)
        return True, snapshot, None
    except Exception as exc:
        defect = f"QA_REPORT_JSON_SERIALIZATION_ERROR:{label}:{_safe_type_name(exc)}"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        return (
            False,
            {
                "_is_corrupted": True,
                "_raw_repr": _safe_repr(raw_dict, 500),
                "_serialization_error": f"json_error:{_safe_str(exc, 200)}",
            },
            defect,
        )


def _validate_and_extract_reasons(
    raw: Any,
    field_name: str,
    fatal_defects: List[str],
) -> List[str]:
    """Strictly validate container and element types of reason fields. Accepts ONLY list and tuple."""
    if not isinstance(raw, (list, tuple)) or isinstance(raw, (str, bytes)):
        fatal_defects.append(
            f"INVALID_PRE_QA_{field_name.upper()}_TYPE:type_{_safe_type_name(raw)}"
        )
        return []

    validated_reasons: List[str] = []
    seen: Set[str] = set()

    for idx, item in enumerate(raw):
        if not isinstance(item, str):
            fatal_defects.append(
                f"INVALID_PRE_QA_{field_name.upper()}_ELEMENT:index_{idx}:type_{_safe_type_name(item)}"
            )
            continue
        if not item.strip():
            fatal_defects.append(
                f"MALFORMED_PRE_QA_{field_name.upper()}_ELEMENT:index_{idx}:empty"
            )
            continue
        if item != item.strip():
            fatal_defects.append(
                f"MALFORMED_PRE_QA_{field_name.upper()}_ELEMENT_WHITESPACE:{_safe_str(item, 100)}"
            )
            continue
        if item not in seen:
            seen.add(item)
            validated_reasons.append(item)

    return validated_reasons


def _resolve_min_ratio(
    context: Optional[AgentContext],
    pre_qa: Optional[QAReport],
    fatal_defects: List[str],
) -> float:
    """Resolve min_questions_ratio strictly without silent fallback."""
    tel = getattr(pre_qa, "telemetry", None) if pre_qa is not None else None
    if tel is not None:
        if not isinstance(tel, collections.abc.Mapping):
            fatal_defects.append(
                f"MALFORMED_TELEMETRY_STRUCTURE:type_{_safe_type_name(tel)}"
            )
        elif "checking" in tel:
            chk = tel["checking"]
            if not isinstance(chk, collections.abc.Mapping):
                fatal_defects.append(
                    f"MALFORMED_TELEMETRY_CHECKING_STRUCTURE:type_{_safe_type_name(chk)}"
                )
            elif "min_questions_ratio" in chk:
                val = chk["min_questions_ratio"]
                ratio = _safe_float_ratio(val)
                if ratio is not None:
                    return ratio
                fatal_defects.append(
                    f"INVALID_MIN_QUESTIONS_RATIO_CONFIG:telemetry:{_safe_str(val, 100)}"
                )
                return 0.9

    if context is not None:
        req = getattr(context, "request", None)
        if req is not None:
            if not isinstance(req, collections.abc.Mapping):
                fatal_defects.append(
                    f"MALFORMED_REQUEST_CONTEXT:type_{_safe_type_name(req)}"
                )
            elif "min_questions_ratio" in req:
                val = req["min_questions_ratio"]
                ratio = _safe_float_ratio(val)
                if ratio is not None:
                    return ratio
                fatal_defects.append(
                    f"INVALID_MIN_QUESTIONS_RATIO_CONFIG:request:{_safe_str(val, 100)}"
                )
                return 0.9

    return 0.9


def _validate_stage2a_lifecycle_contract(
    sid: str,
    q: Any,
    slots_with_fatal_defects: Set[str],
    fatal_defects: List[str],
) -> None:
    """Exhaustively validate Stage 2A lifecycle metadata across all states."""
    raw_is_unres = getattr(q, "is_unresolved", None)
    if type(raw_is_unres) is not bool:
        slots_with_fatal_defects.add(sid)
        fatal_defects.append(
            f"MALFORMED_QUESTION_LIFECYCLE_FLAG:{sid}:type_{_safe_type_name(raw_is_unres)}"
        )
        return

    scope = getattr(q, "unresolved_scope", None)
    subparts = getattr(q, "unresolved_subparts", None)
    f_code = getattr(q, "failure_code", None)
    f_detail = getattr(q, "failure_detail", None)

    if not raw_is_unres:
        if scope is not None:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_UNRESOLVED_SCOPE:{sid}:{_safe_str(scope, 100)}"
            )
        if subparts is not None and (
            not isinstance(subparts, (list, tuple))
            or isinstance(subparts, (str, bytes))
            or len(subparts) > 0
        ):
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_UNRESOLVED_SUBPARTS:{sid}:{_safe_type_name(subparts)}"
            )
        if f_code is not None:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_FAILURE_CODE:{sid}:{_safe_str(f_code, 100)}"
            )
        if f_detail is not None:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_FAILURE_DETAIL:{sid}:{_safe_str(f_detail, 100)}"
            )
        return

    if scope not in ("question", "subquestion"):
        slots_with_fatal_defects.add(sid)
        fatal_defects.append(f"INVALID_UNRESOLVED_SCOPE:{sid}:{_safe_str(scope, 100)}")

    if not isinstance(f_code, str) or not f_code.strip():
        slots_with_fatal_defects.add(sid)
        fatal_defects.append(
            f"MISSING_OR_MALFORMED_FAILURE_CODE:{sid}:{_safe_type_name(f_code)}"
        )

    if not isinstance(f_detail, str) or not f_detail.strip():
        slots_with_fatal_defects.add(sid)
        fatal_defects.append(
            f"MISSING_OR_MALFORMED_FAILURE_DETAIL:{sid}:{_safe_type_name(f_detail)}"
        )

    if scope == "question":
        if subparts is not None and (
            not isinstance(subparts, (list, tuple))
            or isinstance(subparts, (str, bytes))
            or len(subparts) > 0
        ):
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_UNRESOLVED_SUBPARTS:{sid}:{_safe_type_name(subparts)}"
            )
    elif scope == "subquestion":
        if not isinstance(subparts, (list, tuple)) or isinstance(
            subparts, (str, bytes)
        ):
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"INVALID_UNRESOLVED_SUBPARTS_TYPE:{sid}:{_safe_type_name(subparts)}"
            )
        elif len(subparts) == 0:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(f"EMPTY_UNRESOLVED_SUBPARTS:{sid}")
        else:
            seen_labels = set()
            for idx, item in enumerate(subparts):
                if not isinstance(item, str) or not item.strip():
                    slots_with_fatal_defects.add(sid)
                    fatal_defects.append(
                        f"MALFORMED_SUBPART_IDENTIFIER:{sid}:index_{idx}:{_safe_type_name(item)}"
                    )
                    break
                if item != item.strip():
                    slots_with_fatal_defects.add(sid)
                    fatal_defects.append(
                        f"MALFORMED_SUBPART_IDENTIFIER_WHITESPACE:{sid}:{_safe_str(item, 100)}"
                    )
                    break
                if item in seen_labels:
                    slots_with_fatal_defects.add(sid)
                    fatal_defects.append(
                        f"DUPLICATE_SUBPART_IDENTIFIER:{sid}:{_safe_str(item, 100)}"
                    )
                    break
                seen_labels.add(item)


def _safe_reset_evaluation(
    factory: Any,
    errors_list: Optional[List[str]] = None,
    stage_label: str = "CLEANUP",
) -> None:
    """Safely access factory.evaluation and call reset() without throwing."""
    try:
        eval_agent = getattr(factory, "evaluation", None)
        if eval_agent is not None:
            reset_fn = getattr(eval_agent, "reset", None)
            if callable(reset_fn):
                reset_fn()
    except Exception as exc:
        if errors_list is not None:
            errors_list.append(
                f"{stage_label}_RESET_ERROR:{_safe_type_name(exc)}:{_safe_str(exc, 200)}"
            )


# -----------------------------------------------------------------------------
# 3.2 Canonical & Emergency Metadata Builders
# -----------------------------------------------------------------------------


def _build_pipeline_meta(
    *,
    validated_exec_success: bool,
    raw_success: Any,
    paper_status: str,
    export_eligible: bool,
    failure_code: Optional[str],
    failure_detail: Optional[str],
    telemetry: dict,
    factory_notes: List[Any],
    qa_report: Optional[dict],
    pre_adaptation_qa_report: Optional[dict],
    adaptation_defects: List[str],
    reconciliation_meta: dict,
    blocked_reasons: List[str],
    error: Optional[str] = None,
    cleanup_errors: Optional[List[str]] = None,
) -> dict:
    """Canonical metadata constructor ensuring identical schema across normal and handled error paths."""
    meta = {
        "success": bool(validated_exec_success),
        "raw_execution_success": raw_success,
        "paper_status": paper_status,
        "export_eligible": bool(export_eligible),
        "failure_code": failure_code,
        "failure_detail": failure_detail,
        "telemetry": telemetry,
        "factory_notes": list(factory_notes or []),
        "qa_report": qa_report,
        "pre_adaptation_qa_report": pre_adaptation_qa_report,
        "adaptation_defects": list(adaptation_defects or []),
        "reconciliation_meta": reconciliation_meta,
        "blocked_reasons": list(blocked_reasons or []),
    }
    if error is not None:
        meta["error"] = error
    if cleanup_errors:
        meta["cleanup_errors"] = list(cleanup_errors)
    return meta


def _safe_json_snapshot(
    data: Any,
    label: str,
    fatal_defects: Optional[List[str]] = None,
) -> Tuple[bool, Any, Optional[str]]:
    """
    Strict, recursively safe serialization boundary enforcing allow_nan=False.
    Detects circular references, nested non-serializable objects, and dictionary subclasses.
    Produces an independent deep copy via JSON round-trip, preventing mutable reference aliasing.
    Diagnostics use primitive-only strings and NEVER embed the original non-serializable object.
    Returns (is_success, snapshot_or_fallback, defect_code).
    """
    if data is None:
        return True, None, None

    try:
        json_str = json.dumps(data, allow_nan=False)
        snapshot = json.loads(json_str)
        return True, snapshot, None
    except (ValueError, TypeError, OverflowError) as exc:
        defect = f"SERIALIZATION_ERROR:{label}:{_safe_type_name(exc)}:{_safe_str(exc, 100)}"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        fallback = {
            "_is_corrupted": True,
            "_raw_repr": _safe_repr(data, 200),
            "_serialization_error": f"{_safe_type_name(exc)}:{_safe_str(exc, 100)}",
        }
        return False, fallback, defect
    except Exception as exc:
        defect = f"SERIALIZATION_EXCEPTION:{label}:{_safe_type_name(exc)}:{_safe_str(exc, 100)}"
        if fatal_defects is not None:
            fatal_defects.append(defect)
        fallback = {
            "_is_corrupted": True,
            "_raw_repr": _safe_repr(data, 200),
            "_serialization_error": f"unhandled:{_safe_type_name(exc)}:{_safe_str(exc, 100)}",
        }
        return False, fallback, defect


def _safe_dict_convert(val: Any) -> dict:
    """Safely convert val to a dictionary, returning empty dict on failure."""
    if isinstance(val, dict):
        try:
            return dict(val)
        except Exception:
            pass
    return {}


def _safe_list_convert(val: Any) -> list:
    """Safely convert val to a list, returning empty list on failure or non-sequence."""
    if isinstance(val, (list, tuple)) and not isinstance(val, (str, bytes)):
        try:
            return list(val)
        except Exception:
            pass
    return []


def _build_emergency_meta_dict(
    *,
    validated_exec_success: bool,
    raw_success: Any,
    paper_status: str,
    export_eligible: bool,
    failure_code: Optional[str],
    failure_detail: Optional[str],
    telemetry: Any,
    factory_notes: Any,
    qa_report: Any,
    pre_adaptation_qa_report: Any,
    adaptation_defects: Any,
    reconciliation_meta: Any,
    blocked_reasons: Any,
    error: Optional[str] = None,
    cleanup_errors: Optional[List[str]] = None,
) -> dict:
    """
    Direct literal dictionary construction for emergency fallback.
    Completely independent of _build_pipeline_meta to guarantee unhandled recovery.
    Guarded against non-finite floats, circular references, non-serializable nested objects,
    and mutable reference aliasing via _safe_json_snapshot across all dict and list structures.
    Diagnostics are maintained in a separate primitive-only string collection.
    Failed snapshots fall back to safe canonical representations (empty dict, empty list, or None)
    and surface serialization defects to fatal_defects, forcing STATUS_BLOCKED and export_eligible=False.
    """
    # Scalar execution flag normalization: strictly validate finite floats
    safe_raw_success = None
    if type(raw_success) is bool:
        safe_raw_success = raw_success
    elif raw_success is None:
        safe_raw_success = None
    elif isinstance(raw_success, (int, float)):
        if isinstance(raw_success, float) and not math.isfinite(raw_success):
            safe_raw_success = None
        else:
            safe_raw_success = raw_success
    elif isinstance(raw_success, str):
        safe_raw_success = _safe_str(raw_success, 100)
    else:
        safe_raw_success = _safe_str(raw_success, 100)

    # Emergency fatal defects accumulator to surface serialization errors (primitive strings only)
    emergency_fatal_defects: List[str] = []

    # Uniform safe serialization producing independent deep copies across all mapping and list fields
    is_tel_ok, safe_telemetry, _ = _safe_json_snapshot(
        telemetry, label="emergency_telemetry", fatal_defects=emergency_fatal_defects
    )
    is_qa_ok, safe_qa_report, _ = _safe_json_snapshot(
        qa_report, label="emergency_qa_report", fatal_defects=emergency_fatal_defects
    )
    is_pre_ok, safe_pre_qa, _ = _safe_json_snapshot(
        pre_adaptation_qa_report,
        label="emergency_pre_qa",
        fatal_defects=emergency_fatal_defects,
    )
    is_rec_ok, safe_reconciliation_meta, _ = _safe_json_snapshot(
        reconciliation_meta,
        label="emergency_reconciliation_meta",
        fatal_defects=emergency_fatal_defects,
    )
    is_fn_ok, safe_factory_notes, _ = _safe_json_snapshot(
        _safe_list_convert(factory_notes),
        label="emergency_factory_notes",
        fatal_defects=emergency_fatal_defects,
    )
    is_ad_ok, safe_adaptation_defects, _ = _safe_json_snapshot(
        _safe_list_convert(adaptation_defects),
        label="emergency_adaptation_defects",
        fatal_defects=emergency_fatal_defects,
    )
    is_br_ok, safe_blocked_reasons, _ = _safe_json_snapshot(
        _safe_list_convert(blocked_reasons),
        label="emergency_blocked_reasons",
        fatal_defects=emergency_fatal_defects,
    )
    safe_cleanup_errors = None
    if cleanup_errors is not None:
        is_ce_ok, safe_cleanup_errors, _ = _safe_json_snapshot(
            _safe_list_convert(cleanup_errors),
            label="emergency_cleanup_errors",
            fatal_defects=emergency_fatal_defects,
        )

    # Field-specific fallback shape enforcement
    final_telemetry = (
        safe_telemetry
        if (is_tel_ok and isinstance(safe_telemetry, dict))
        else (safe_telemetry if isinstance(safe_telemetry, dict) else {})
    )
    final_qa_report = (
        safe_qa_report
        if (is_qa_ok and isinstance(safe_qa_report, dict))
        else (safe_qa_report if isinstance(safe_qa_report, dict) else None)
    )
    final_pre_qa = (
        safe_pre_qa
        if (is_pre_ok and isinstance(safe_pre_qa, dict))
        else (safe_pre_qa if isinstance(safe_pre_qa, dict) else None)
    )
    final_reconciliation = (
        dict(safe_reconciliation_meta)
        if (is_rec_ok and isinstance(safe_reconciliation_meta, dict))
        else (
            safe_reconciliation_meta
            if isinstance(safe_reconciliation_meta, dict)
            else {}
        )
    )
    final_factory_notes = (
        list(safe_factory_notes)
        if (is_fn_ok and isinstance(safe_factory_notes, list))
        else []
    )
    final_adaptation_defects = (
        list(safe_adaptation_defects)
        if (is_ad_ok and isinstance(safe_adaptation_defects, list))
        else []
    )
    final_blocked_reasons = (
        list(safe_blocked_reasons)
        if (is_br_ok and isinstance(safe_blocked_reasons, list))
        else []
    )

    # If serialization defects occurred, surface them as primitive strings into blocked_reasons and reconciliation_meta
    if emergency_fatal_defects:
        for ed in emergency_fatal_defects:
            final_blocked_reasons.append(f"INTEGRITY_DEFECT:{ed}")
        rec_defects = final_reconciliation.get("fatal_defects")
        if isinstance(rec_defects, list):
            final_reconciliation["fatal_defects"] = (
                rec_defects + emergency_fatal_defects
            )
        else:
            final_reconciliation["fatal_defects"] = list(emergency_fatal_defects)

    # Four-tier status calculation under emergency metadata assembly
    effective_paper_status = (
        _safe_str(paper_status, 50) if paper_status else "FAILED"
    )
    effective_export_eligible = bool(export_eligible)
    if emergency_fatal_defects:
        if validated_exec_success:
            effective_paper_status = "BLOCKED"
        effective_export_eligible = False

    m = {
        "success": bool(validated_exec_success),
        "raw_execution_success": safe_raw_success,
        "paper_status": effective_paper_status,
        "export_eligible": effective_export_eligible,
        "failure_code": (
            _safe_str(failure_code, 100) if failure_code is not None else None
        ),
        "failure_detail": (
            _safe_str(failure_detail, 500) if failure_detail is not None else None
        ),
        "telemetry": final_telemetry,
        "factory_notes": final_factory_notes,
        "qa_report": final_qa_report,
        "pre_adaptation_qa_report": final_pre_qa,
        "adaptation_defects": final_adaptation_defects,
        "reconciliation_meta": final_reconciliation,
        "blocked_reasons": final_blocked_reasons,
    }
    if error is not None:
        m["error"] = _safe_str(error, 500)
    if safe_cleanup_errors is not None:
        m["cleanup_errors"] = (
            list(safe_cleanup_errors)
            if isinstance(safe_cleanup_errors, list)
            else []
        )
    return m


# -----------------------------------------------------------------------------
# 3.3 Universal Finalizer Policy (_finalize_reconciliation)
# -----------------------------------------------------------------------------


def _finalize_reconciliation(
    *,
    raw_success: Any,
    exec_succeeded: bool,
    expected_ids: List[str],
    f_post: List[str],
    p_post: List[str],
    b_post: List[str],
    fatal_defects: List[str],
    adaptation_defects: List[str],
    pre_status: Optional[str],
    context: Optional[AgentContext],
    copied_telemetry: dict,
    copied_cost_telemetry: dict,
    copied_cache_telemetry: dict,
    copied_result_telemetry: dict,
    serialized_pre_qa: Optional[dict],
    failure_code: Optional[str],
    validated_pre_blocked: List[str],
    validated_pre_degraded: List[str],
    resolved_population_changed: bool,
    min_ratio: float,
    accounting_complete: bool,
    is_synthetic: bool = False,
    synthetic_reason: Optional[str] = None,
    extra_reconciliation_telemetry: Optional[dict] = None,
) -> Tuple[
    QAReport, List[str], Dict[str, Any], Optional[dict], Optional[dict], dict, bool
]:
    """
    Universal finalization engine used by ALL pipeline paths.
    Enforces 4-tier status precedence, unconditional zero-resolved blocking,
    and synchronized 9-field serialized outputs.
    """
    # Stage 1: Partition Validation Step with Pre-Set Type Checking
    # 0. Validate expected_ids container and element types before set operations
    valid_expected_ids: List[str] = []
    if not isinstance(expected_ids, (list, tuple)) or isinstance(
        expected_ids, (str, bytes)
    ):
        fatal_defects.append(
            f"MALFORMED_EXPECTED_IDS_CONTAINER:type_{_safe_type_name(expected_ids)}"
        )
    else:
        for idx, sid in enumerate(expected_ids):
            if not isinstance(sid, str):
                fatal_defects.append(
                    f"MALFORMED_EXPECTED_SLOT_ID:index_{idx}:type_{_safe_type_name(sid)}"
                )
            elif not sid.strip():
                fatal_defects.append(f"MALFORMED_EXPECTED_SLOT_ID:index_{idx}:empty")
            elif sid != sid.strip():
                fatal_defects.append(
                    f"MALFORMED_EXPECTED_SLOT_ID_WHITESPACE:index_{idx}:{_safe_str(sid, 50)}"
                )
            else:
                valid_expected_ids.append(sid)

    # 1. Uniqueness check on valid expected_ids
    seen_expected: Set[str] = set()
    unique_expected_ids: List[str] = []
    for sid in valid_expected_ids:
        if sid in seen_expected:
            fatal_defects.append(f"DUPLICATE_EXPECTED_SLOT_ID:{sid}")
        else:
            seen_expected.add(sid)
            unique_expected_ids.append(sid)
    expected_set: Set[str] = seen_expected
    expected_count = len(unique_expected_ids)

    # 2. Type-safe partition list validation & per-partition uniqueness
    duplicate_slots: Set[str] = set()

    def _validate_partition_container_and_uniqueness(
        p_raw: Any, p_name: str
    ) -> Tuple[List[str], Set[str]]:
        if not isinstance(p_raw, (list, tuple)) or isinstance(
            p_raw, (str, bytes)
        ):
            fatal_defects.append(
                f"MALFORMED_PARTITION_CONTAINER:{p_name}:type_{_safe_type_name(p_raw)}"
            )
            return [], set()
        seen = set()
        deduped = []
        for idx, item in enumerate(p_raw):
            if not isinstance(item, str):
                fatal_defects.append(
                    f"MALFORMED_SLOT_ID_IN_PARTITION:{p_name}:index_{idx}:type_{_safe_type_name(item)}"
                )
                continue
            if not item.strip() or item != item.strip():
                fatal_defects.append(
                    f"MALFORMED_SLOT_ID_IN_PARTITION_WHITESPACE:{p_name}:index_{idx}:{_safe_str(item, 50)}"
                )
                continue
            if item in seen:
                fatal_defects.append(f"DUPLICATE_SLOT_IN_PARTITION:{p_name}:{item}")
                duplicate_slots.add(item)
            else:
                seen.add(item)
                deduped.append(item)
        return deduped, seen

    u_f, set_f = _validate_partition_container_and_uniqueness(f_post, "f_post")
    u_p, set_p = _validate_partition_container_and_uniqueness(p_post, "p_post")
    u_b, set_b = _validate_partition_container_and_uniqueness(b_post, "b_post")

    # 3. Pairwise disjointness check
    overlap_f_p = set_f & set_p
    overlap_f_b = set_f & set_b
    overlap_p_b = set_p & set_b

    for sid in sorted(overlap_f_p):
        fatal_defects.append(f"PARTITION_OVERLAP_F_AND_P:{sid}")
    for sid in sorted(overlap_f_b):
        fatal_defects.append(f"PARTITION_OVERLAP_F_AND_B:{sid}")
    for sid in sorted(overlap_p_b):
        fatal_defects.append(f"PARTITION_OVERLAP_P_AND_B:{sid}")

    all_overlaps = overlap_f_p | overlap_f_b | overlap_p_b

    # 4. Exhaustive coverage check
    union_parts = set_f | set_p | set_b
    unexpected_slots = union_parts - expected_set
    for sid in sorted(unexpected_slots):
        fatal_defects.append(f"UNEXPECTED_SLOT_IN_PARTITION:{sid}")

    missing_slots = expected_set - union_parts
    for sid in sorted(missing_slots):
        fatal_defects.append(f"UNACCOUNTED_EXPECTED_SLOT:{sid}")

    # 5. Conservative sanitization upholding E = F_hat dot_cup P_hat dot_cup B_hat:
    # Rule 1: Expected IDs involved in duplicates, overlaps, or omissions are moved/added to B_hat
    #         and evicted from F_hat and P_hat.
    # Rule 2: Unexpected IDs (not in E) are COMPLETELY EXCLUDED from F_hat, P_hat, and B_hat.
    #         They appear in fatal_defects diagnostics, but NEVER enter expected partitions.
    problem_expected_slots = ((all_overlaps | duplicate_slots) & expected_set) | missing_slots

    sanitized_f = [
        s for s in u_f if s in expected_set and s not in problem_expected_slots
    ]
    sanitized_p = [
        s for s in u_p if s in expected_set and s not in problem_expected_slots
    ]

    # Sanitized B: expected slots in u_b, plus problem expected slots (overlaps, omitted)
    sanitized_b_set = (set_b & expected_set) | problem_expected_slots
    sanitized_b = [
        s
        for s in unique_expected_ids
        if s in sanitized_b_set and s not in sanitized_f and s not in sanitized_p
    ]

    # Invariant: unique_expected_ids = sanitized_f (disjoint) sanitized_p (disjoint) sanitized_b
    resolved_count = len(sanitized_f)
    unresolved_slots = [
        s for s in unique_expected_ids if s in sanitized_p or s in sanitized_b
    ]

    # Stage 2: Reason assembly & canonical pre-status validation
    CANONICAL_PRE_QA_STATUSES = {
        STATUS_OK,
        STATUS_PASS,
        STATUS_PARTIAL,
        STATUS_DEGRADED,
        STATUS_BLOCKED,
    }
    ALL_VALID_STATUSES = CANONICAL_PRE_QA_STATUSES | {STATUS_FAILED}

    if pre_status is None:
        fatal_defects.append("INVALID_PRE_QA_STATUS:None")
    elif not isinstance(pre_status, str):
        fatal_defects.append(
            f"INVALID_PRE_QA_STATUS_TYPE:type_{_safe_type_name(pre_status)}"
        )
    elif pre_status not in ALL_VALID_STATUSES:
        fatal_defects.append(f"INVALID_PRE_QA_STATUS:{_safe_str(pre_status, 50)}")

    # Execution flag integrity checks
    if type(raw_success) is not bool or type(exec_succeeded) is not bool:
        fatal_defects.append(
            f"INVALID_EXECUTION_FLAG_TYPE:raw_{_safe_type_name(raw_success)}:exec_{_safe_type_name(exec_succeeded)}"
        )
    elif raw_success is True and not exec_succeeded:
        fatal_defects.append("CONTRADICTORY_EXECUTION_FLAGS:raw_true_exec_false")

    blocked_reasons = list(validated_pre_blocked)
    degraded_reasons = list(validated_pre_degraded)

    for defect in fatal_defects:
        blocked_reasons.append(f"INTEGRITY_DEFECT:{defect}")
    for defect in adaptation_defects:
        degraded_reasons.append(defect)

    # 4-Tier Status Precedence Policy
    # Define one authoritative execution-state decision and use it consistently
    if type(raw_success) is not bool or type(exec_succeeded) is not bool:
        # Tier 1: Invalid execution flag type -> BLOCKED
        status = STATUS_BLOCKED
    elif raw_success is False:
        # Tier 2: Genuine execution failure (False) -> FAILED, retaining all fatal defects
        status = STATUS_FAILED
        blocked_reasons.append(
            f"EXECUTION_FAILURE:{_safe_str(failure_code or 'UNKNOWN_ERROR', 100)}"
        )
    elif not exec_succeeded:
        # Tier 3: Inconsistent execution flags (raw_success=True but exec_succeeded=False) -> BLOCKED
        status = STATUS_BLOCKED
    elif fatal_defects:
        # Tier 3: Successful execution with fatal integrity defect(s) -> BLOCKED
        status = STATUS_BLOCKED
    elif pre_status == STATUS_FAILED:
        status = STATUS_FAILED
    elif pre_status == STATUS_BLOCKED:
        # Legitimate pre-adaptation BLOCKED verdict preserved
        status = STATUS_BLOCKED
    elif pre_status not in CANONICAL_PRE_QA_STATUSES:
        # Invalid or missing pre-status fails closed to BLOCKED
        status = STATUS_BLOCKED
    elif resolved_count == 0:
        # Zero resolved slots unconditionally forces STATUS_BLOCKED!
        status = STATUS_BLOCKED
        if BLOCK_REASON_UNRESOLVED_SLOTS not in blocked_reasons:
            blocked_reasons.append(BLOCK_REASON_UNRESOLVED_SLOTS)
    elif not resolved_population_changed and not adaptation_defects:
        status = pre_status
    elif not resolved_population_changed and adaptation_defects:
        status = STATUS_PARTIAL
    else:
        # F population reduced -> ratio evaluation
        ratio = resolved_count / expected_count if expected_count > 0 else 0.0
        is_degraded_audit = (
            context is not None
            and getattr(context, "audit_report", None) is not None
            and getattr(context.audit_report, "degraded_mode", False)
        )
        if ratio < min_ratio:
            if is_degraded_audit and resolved_count > 0:
                if BLOCK_REASON_INSUFFICIENT_QUESTIONS not in degraded_reasons:
                    degraded_reasons.append(BLOCK_REASON_INSUFFICIENT_QUESTIONS)
                status = STATUS_PARTIAL
            else:
                if BLOCK_REASON_INSUFFICIENT_QUESTIONS not in blocked_reasons:
                    blocked_reasons.append(BLOCK_REASON_INSUFFICIENT_QUESTIONS)
                status = STATUS_BLOCKED
        else:
            status = STATUS_PARTIAL

    if status == STATUS_BLOCKED and not blocked_reasons:
        blocked_reasons.append("BLOCKED_REASON_UNSPECIFIED")

    # Stage 3: Construct reconciliation telemetry
    bridge_telemetry = {
        "expected_slots": expected_count,
        "resolved_slots": resolved_count,
        "unresolved_slots": len(unresolved_slots),
        "f_post": list(sanitized_f),
        "p_post": list(sanitized_p),
        "b_post": list(sanitized_b),
        "fatal_defects": list(fatal_defects),
        "adaptation_defects": list(adaptation_defects),
        "accounting_incomplete": not bool(
            accounting_complete
            and expected_count > 0
            and not any(
                d.startswith("INVALID_EXPECTED_SLOT_SPECIFICATION")
                or d.startswith("MALFORMED_EXPECTED_IDS_CONTAINER")
                for d in fatal_defects
            )
            and len(f_post) + len(p_post) + len(b_post) == expected_count
            and set(f_post) | set(p_post) | set(b_post) == expected_set
        ),
        "accounting_completeness": (
            "complete"
            if (
                bool(accounting_complete)
                and expected_count > 0
                and not any(
                    d.startswith("INVALID_EXPECTED_SLOT_SPECIFICATION")
                    or d.startswith("MALFORMED_EXPECTED_IDS_CONTAINER")
                    for d in fatal_defects
                )
                and len(f_post) + len(p_post) + len(b_post) == expected_count
                and set(f_post) | set(p_post) | set(b_post) == expected_set
            )
            else "incomplete"
        ),
    }
    if is_synthetic:
        bridge_telemetry["is_synthetic"] = True
        bridge_telemetry["synthetic_reason"] = synthetic_reason
    RESERVED_BRIDGE_KEYS = {
        "expected_slots",
        "resolved_slots",
        "unresolved_slots",
        "f_post",
        "p_post",
        "b_post",
        "fatal_defects",
        "adaptation_defects",
        "accounting_incomplete",
        "accounting_completeness",
        "is_synthetic",
        "synthetic_reason",
    }
    if extra_reconciliation_telemetry and isinstance(
        extra_reconciliation_telemetry, dict
    ):
        bridge_telemetry["extra_telemetry"] = {
            _safe_str(k, 100): v for k, v in extra_reconciliation_telemetry.items()
        }
        for k, v in extra_reconciliation_telemetry.items():
            if k not in RESERVED_BRIDGE_KEYS and k not in bridge_telemetry:
                bridge_telemetry[k] = v

    copied_telemetry["bridge_reconciliation"] = bridge_telemetry

    # Stage 5: Construct the final QA report
    reconciled_qa = QAReport(
        status=status,
        slot_count=expected_count,
        resolved_slots=resolved_count,
        unresolved_slots=unresolved_slots,
        degraded_reasons=degraded_reasons,
        blocked_reasons=blocked_reasons,
        telemetry=copied_telemetry,
        cost_telemetry=copied_cost_telemetry,
        cache_telemetry=copied_cache_telemetry,
    )

    # Safe serialization with synchronized status update
    is_ser_success, serialized_reconciled_qa, ser_defect = (
        _safe_serialize_qa_report(
            reconciled_qa,
            label="reconciled_qa_report",
            fatal_defects=fatal_defects,
        )
    )

    if not is_ser_success:
        if type(raw_success) is not bool or type(exec_succeeded) is not bool:
            status = STATUS_BLOCKED
        elif raw_success is False:
            status = STATUS_FAILED
        else:
            status = STATUS_BLOCKED

        reconciled_qa.status = status
        if (
            ser_defect
            and f"INTEGRITY_DEFECT:{ser_defect}" not in reconciled_qa.blocked_reasons
        ):
            reconciled_qa.blocked_reasons.append(f"INTEGRITY_DEFECT:{ser_defect}")

        bridge_telemetry["fatal_defects"] = list(fatal_defects)
        copied_telemetry["bridge_reconciliation"] = bridge_telemetry

        # Complete 9-field QAReport fallback schema guaranteed to be JSON-serializable
        safe_fallback_telemetry = _safe_dict_convert(copied_telemetry)
        safe_fallback_telemetry["bridge_reconciliation"] = _safe_dict_convert(bridge_telemetry)
        serialized_reconciled_qa = {
            "status": status,
            "slot_count": expected_count,
            "resolved_slots": resolved_count,
            "unresolved_slots": list(unresolved_slots),
            "degraded_reasons": list(degraded_reasons),
            "blocked_reasons": list(reconciled_qa.blocked_reasons),
            "telemetry": safe_fallback_telemetry,
            "cost_telemetry": _safe_dict_convert(copied_cost_telemetry),
            "cache_telemetry": _safe_dict_convert(copied_cache_telemetry),
            "_is_synthetic_fallback": True,
            "_serialization_error": _safe_str(ser_defect, 200),
        }
        try:
            json.dumps(serialized_reconciled_qa, allow_nan=False)
        except Exception:
            serialized_reconciled_qa["telemetry"] = {
                "bridge_reconciliation": _safe_dict_convert(bridge_telemetry),
                "_telemetry_unserializable": True,
            }
            serialized_reconciled_qa["cost_telemetry"] = {}
            serialized_reconciled_qa["cache_telemetry"] = {}

    # Authoritative export_eligible derivation
    has_fatal = bool(fatal_defects)
    export_eligible = (
        (type(raw_success) is bool and raw_success is True)
        and not has_fatal
        and (status in ("OK", "PASS", "PARTIAL", "DEGRADED"))
    )

    reconciliation_meta = {
        "expected_slots": expected_count,
        "resolved_slots": resolved_count,
        "unresolved_slots": len(unresolved_slots),
        "fatal_defects": list(fatal_defects),
        "adaptation_defects": list(adaptation_defects),
        "f_post": list(sanitized_f),
        "p_post": list(sanitized_p),
        "b_post": list(sanitized_b),
    }

    return (
        reconciled_qa,
        adaptation_defects,
        reconciliation_meta,
        serialized_pre_qa,
        serialized_reconciled_qa,
        copied_result_telemetry,
        export_eligible,
    )


# -----------------------------------------------------------------------------
# 3.4 Early Failure Handlers Delegating to Common Finalizer
# -----------------------------------------------------------------------------


def _handle_context_construction_exception(
    *,
    exc: Exception,
    failure_label: str = "CONTEXT_CONSTRUCTION_ERROR",
) -> Tuple[List[dict], List[dict], dict]:
    """Context-independent initialization failure path. Zero secondary AgentContext construction."""
    exc_type = _safe_type_name(exc)
    exc_msg = _safe_str(exc, 200)
    fatal_msg = f"{failure_label}:{exc_type}:{exc_msg}"
    fatal_defects = [
        fatal_msg,
        "EXPECTED_SLOT_ACCOUNTING_UNESTABLISHED:context_construction_failed",
    ]

    (
        reconciled_qa,
        adaptation_defects,
        reconciliation_meta,
        serialized_pre_qa,
        serialized_reconciled_qa,
        copied_result_telemetry,
        export_eligible,
    ) = _finalize_reconciliation(
        raw_success=False,
        exec_succeeded=False,
        expected_ids=[],
        f_post=[],
        p_post=[],
        b_post=[],
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=None,
        context=None,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code=failure_label,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=False,
        is_synthetic=True,
        synthetic_reason=failure_label,
        extra_reconciliation_telemetry={"context_construction_error": exc_msg},
    )

    meta = _build_pipeline_meta(
        validated_exec_success=False,
        raw_success=False,
        paper_status=reconciled_qa.status,
        export_eligible=export_eligible,
        failure_code=failure_label,
        failure_detail=exc_msg,
        telemetry={},
        factory_notes=[],
        qa_report=serialized_reconciled_qa,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta=reconciliation_meta,
        blocked_reasons=list(getattr(reconciled_qa, "blocked_reasons", []) or []),
        error=exc_msg,
    )
    return [], [], meta


def _handle_dependency_factory_failure(
    *,
    factory: Any,
    context: AgentContext,
) -> Tuple[List[dict], List[dict], dict]:
    """Dependency factory failure routed through common finalizer."""
    code = getattr(factory, "failure_code", None) or "DEPENDENCY_FACTORY_FAILURE"
    detail = (
        getattr(factory, "failure_detail", None) or "Dependency factory build failed"
    )
    fatal_msg = (
        f"DEPENDENCY_FACTORY_FAILURE:{_safe_str(code, 50)}:{_safe_str(detail, 100)}"
    )
    fatal_defects = [fatal_msg]

    try:
        from .checking_agent import CheckingAgent

        agent = CheckingAgent()
        expected_ids = agent._get_validated_expected_slots(context)
    except Exception as slot_exc:
        expected_ids = []
        fatal_defects.append(
            f"INVALID_EXPECTED_SLOT_SPECIFICATION:{_safe_type_name(slot_exc)}:{_safe_str(slot_exc, 200)}"
        )

    (
        reconciled_qa,
        adaptation_defects,
        reconciliation_meta,
        serialized_pre_qa,
        serialized_reconciled_qa,
        copied_result_telemetry,
        export_eligible,
    ) = _finalize_reconciliation(
        raw_success=False,
        exec_succeeded=False,
        expected_ids=expected_ids,
        f_post=[],
        p_post=[],
        b_post=list(expected_ids),
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=None,
        context=context,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code=code,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=False,
        is_synthetic=True,
        synthetic_reason="DEPENDENCY_FACTORY_FAILURE",
    )

    meta = _build_pipeline_meta(
        validated_exec_success=False,
        raw_success=False,
        paper_status=reconciled_qa.status,
        export_eligible=export_eligible,
        failure_code=code,
        failure_detail=detail,
        telemetry={},
        factory_notes=list(getattr(factory, "notes", []) or []),
        qa_report=serialized_reconciled_qa,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta=reconciliation_meta,
        blocked_reasons=list(getattr(reconciled_qa, "blocked_reasons", []) or []),
    )
    return [], [], meta


def _handle_initialization_exception(
    *,
    exc: Exception,
    context: AgentContext,
    failure_label: str,
) -> Tuple[List[dict], List[dict], dict]:
    """Initialization setup failure routed through common finalizer."""
    exc_type = _safe_type_name(exc)
    exc_msg = _safe_str(exc, 200)
    fatal_msg = f"{failure_label}:{exc_type}:{exc_msg}"
    fatal_defects = [fatal_msg]

    try:
        from .checking_agent import CheckingAgent

        agent = CheckingAgent()
        expected_ids = agent._get_validated_expected_slots(context)
    except Exception as slot_exc:
        expected_ids = []
        fatal_defects.append(
            f"INVALID_EXPECTED_SLOT_SPECIFICATION:{_safe_type_name(slot_exc)}:{_safe_str(slot_exc, 200)}"
        )

    (
        reconciled_qa,
        adaptation_defects,
        reconciliation_meta,
        serialized_pre_qa,
        serialized_reconciled_qa,
        copied_result_telemetry,
        export_eligible,
    ) = _finalize_reconciliation(
        raw_success=False,
        exec_succeeded=False,
        expected_ids=expected_ids,
        f_post=[],
        p_post=[],
        b_post=list(expected_ids),
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=None,
        context=context,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code=failure_label,
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=False,
        is_synthetic=True,
        synthetic_reason=failure_label,
        extra_reconciliation_telemetry={"initialization_error": exc_msg},
    )

    meta = _build_pipeline_meta(
        validated_exec_success=False,
        raw_success=False,
        paper_status=reconciled_qa.status,
        export_eligible=export_eligible,
        failure_code=failure_label,
        failure_detail=exc_msg,
        telemetry={},
        factory_notes=[],
        qa_report=serialized_reconciled_qa,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta=reconciliation_meta,
        blocked_reasons=list(getattr(reconciled_qa, "blocked_reasons", []) or []),
        error=exc_msg,
    )
    return [], [], meta


def _handle_orchestrator_exception(
    *,
    orchestrator_exception: Exception,
    context: AgentContext,
    factory: Any,
    cleanup_errors: Optional[List[str]] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """Orchestrator execution exception routed through common finalizer."""
    exc_type = _safe_type_name(orchestrator_exception)
    exc_msg = _safe_str(orchestrator_exception, 200)
    fatal_msg = f"ORCHESTRATOR_EXECUTION_EXCEPTION:{exc_type}:{exc_msg}"
    fatal_defects = [fatal_msg]

    if cleanup_errors:
        for ce in cleanup_errors:
            fatal_defects.append(ce)

    try:
        from .checking_agent import CheckingAgent

        agent = CheckingAgent()
        expected_ids = agent._get_validated_expected_slots(context)
    except Exception as slot_exc:
        expected_ids = []
        fatal_defects.append(
            f"INVALID_EXPECTED_SLOT_SPECIFICATION:{_safe_type_name(slot_exc)}:{_safe_str(slot_exc, 200)}"
        )

    (
        reconciled_qa,
        adaptation_defects,
        reconciliation_meta,
        serialized_pre_qa,
        serialized_reconciled_qa,
        copied_result_telemetry,
        export_eligible,
    ) = _finalize_reconciliation(
        raw_success=False,
        exec_succeeded=False,
        expected_ids=expected_ids,
        f_post=[],
        p_post=[],
        b_post=list(expected_ids),
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=None,
        context=context,
        copied_telemetry={},
        copied_cost_telemetry={},
        copied_cache_telemetry={},
        copied_result_telemetry={},
        serialized_pre_qa=None,
        failure_code="ORCHESTRATOR_EXCEPTION",
        validated_pre_blocked=[],
        validated_pre_degraded=[],
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=False,
        is_synthetic=True,
        synthetic_reason="ORCHESTRATOR_EXECUTION_EXCEPTION",
        extra_reconciliation_telemetry={"orchestrator_exception": exc_msg},
    )

    meta = _build_pipeline_meta(
        validated_exec_success=False,
        raw_success=False,
        paper_status=reconciled_qa.status,
        export_eligible=export_eligible,
        failure_code="ORCHESTRATOR_EXCEPTION",
        failure_detail=exc_msg,
        telemetry={},
        factory_notes=list(getattr(factory, "notes", []) or []),
        qa_report=serialized_reconciled_qa,
        pre_adaptation_qa_report=None,
        adaptation_defects=[],
        reconciliation_meta=reconciliation_meta,
        blocked_reasons=list(getattr(reconciled_qa, "blocked_reasons", []) or []),
        error=exc_msg,
        cleanup_errors=cleanup_errors,
    )
    return [], [], meta


def _handle_adaptation_execution_error(
    *,
    result: OrchestratorResult,
    context: AgentContext,
    adaptation_exception: Exception,
    cleanup_errors: Optional[List[str]] = None,
) -> Tuple[
    QAReport, List[str], Dict[str, Any], Optional[dict], Optional[dict], dict, bool
]:
    """Handles unexpected to_legacy_shape() exceptions via common finalizer."""
    fatal_msg = f"ADAPTATION_EXECUTION_ERROR:{_safe_type_name(adaptation_exception)}:{_safe_str(adaptation_exception, 200)}"
    fatal_defects: List[str] = [fatal_msg]

    if cleanup_errors:
        for ce in cleanup_errors:
            fatal_defects.append(ce)

    raw_success = getattr(result, "success", None)
    if type(raw_success) is not bool:
        fatal_defects.append(
            f"INVALID_EXECUTION_SUCCESS_FLAG:type_{_safe_type_name(raw_success)}"
        )
        exec_succeeded = False
    else:
        exec_succeeded = raw_success

    try:
        from .checking_agent import CheckingAgent

        agent = CheckingAgent()
        expected_ids = agent._get_validated_expected_slots(context)
    except Exception as exc:
        fatal_defects.append(
            f"INVALID_EXPECTED_SLOT_SPECIFICATION:{_safe_type_name(exc)}:{_safe_str(exc, 200)}"
        )
        expected_ids = []

    pre_qa = getattr(result, "qa_report", None)
    copied_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "telemetry", None), fatal_defects, label="pre_qa_telemetry"
    )
    copied_cost_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "cost_telemetry", None),
        fatal_defects,
        label="cost_telemetry",
    )
    copied_cache_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "cache_telemetry", None),
        fatal_defects,
        label="cache_telemetry",
    )
    copied_result_telemetry = _safe_telemetry_copy(
        getattr(result, "telemetry", None), fatal_defects, label="result_telemetry"
    )
    is_pre_ok, serialized_pre_qa, _ = _safe_serialize_qa_report(
        pre_qa, label="pre_adaptation_qa_report", fatal_defects=fatal_defects
    )

    raw_pre_blocked = getattr(pre_qa, "blocked_reasons", None) if pre_qa else None
    validated_pre_blocked = _validate_and_extract_reasons(
        raw_pre_blocked, "blocked_reasons", fatal_defects
    )
    raw_pre_degraded = getattr(pre_qa, "degraded_reasons", None) if pre_qa else None
    validated_pre_degraded = _validate_and_extract_reasons(
        raw_pre_degraded, "degraded_reasons", fatal_defects
    )

    return _finalize_reconciliation(
        raw_success=raw_success,
        exec_succeeded=exec_succeeded,
        expected_ids=expected_ids,
        f_post=[],
        p_post=[],
        b_post=list(expected_ids),
        fatal_defects=fatal_defects,
        adaptation_defects=[],
        pre_status=getattr(pre_qa, "status", None),
        context=context,
        copied_telemetry=copied_telemetry,
        copied_cost_telemetry=copied_cost_telemetry,
        copied_cache_telemetry=copied_cache_telemetry,
        copied_result_telemetry=copied_result_telemetry,
        serialized_pre_qa=serialized_pre_qa,
        failure_code=getattr(result, "failure_code", None),
        validated_pre_blocked=validated_pre_blocked,
        validated_pre_degraded=validated_pre_degraded,
        resolved_population_changed=True,
        min_ratio=0.9,
        accounting_complete=False,
        extra_reconciliation_telemetry={
            "adaptation_execution_error": _safe_str(adaptation_exception, 200)
        },
    )


def _handle_emergency_pipeline_failure(
    *,
    exc: Exception,
    stage: str = "UNHANDLED_PIPELINE_EXCEPTION",
    result: Optional[OrchestratorResult] = None,
    context: Optional[AgentContext] = None,
    pre_qa_snapshot: Optional[dict] = None,
    cleanup_errors: Optional[List[str]] = None,
    accounting_established: bool = False,
    expected_ids: Optional[List[str]] = None,
    raw_success_override: Optional[bool] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Universal emergency containment boundary.
    Uses independent literal dictionary construction (_build_emergency_meta_dict).
    Preserves all established pipeline state, applies four-tier execution status precedence,
    and formats telemetry strictly under 'bridge_reconciliation'.
    Pre-extracts primitive execution state inside the try block so that the terminal fallback
    in outer except returns exact execution-state status (Tier 3 BLOCKED on success) with ZERO helper calls.
    Guaranteed never to raise even under malformed diagnostic state or raising property getters.
    """
    # 0. Pure primitive execution-state defaults ahead of any operations
    validated_exec_success = False
    paper_status = "FAILED"
    safe_raw_success = None
    known_expected_count = 0

    try:
        # Pre-extract primitive execution state safely inside containment
        raw_success = (
            raw_success_override if raw_success_override is not None else None
        )
        if raw_success is None and result is not None:
            try:
                raw_success = getattr(result, "success", None)
            except Exception:
                raw_success = None

        if raw_success is True:
            # Execution succeeded, but emergency failure occurred in reconciliation/meta/dispatcher -> BLOCKED
            validated_exec_success = True
            paper_status = "BLOCKED"
            safe_raw_success = True
        elif result is None or raw_success is False:
            # Pre-execution failure or explicit failure -> FAILED
            validated_exec_success = False
            paper_status = "FAILED"
            safe_raw_success = False if raw_success is False else None
        elif type(raw_success) is not bool:
            # Execution state invalid (non-boolean flag) -> BLOCKED
            validated_exec_success = False
            paper_status = "BLOCKED"
            safe_raw_success = None

        raw_count = 0
        if isinstance(expected_ids, (list, tuple)) and not isinstance(
            expected_ids, (str, bytes)
        ):
            try:
                raw_count = len(expected_ids)
            except Exception:
                raw_count = 0
        if (
            isinstance(raw_count, int)
            and not isinstance(raw_count, bool)
            and raw_count >= 0
        ):
            known_expected_count = raw_count
        else:
            known_expected_count = 0

        exc_type = _safe_type_name(exc)
        exc_msg = _safe_str(exc, 200)
        failure_code = f"EMERGENCY_{stage}"
        failure_detail = f"{exc_type}:{exc_msg}"

        fatal_defects: List[str] = [f"{failure_code}:{failure_detail}"]
        if cleanup_errors and isinstance(cleanup_errors, (list, tuple)):
            for ce in cleanup_errors:
                fatal_defects.append(_safe_str(ce, 200))

        if type(raw_success) is not bool and result is not None:
            fatal_defects.append(
                f"INVALID_EXECUTION_SUCCESS_FLAG:type_{_safe_type_name(raw_success)}"
            )

        export_eligible = False  # Emergency failure is unconditionally export ineligible

        # 2. Extract or preserve expected slot IDs
        known_expected = _safe_list_convert(expected_ids)
        if not known_expected and context is not None:
            try:
                from .checking_agent import CheckingAgent

                known_expected = CheckingAgent()._get_validated_expected_slots(context)
            except Exception:
                known_expected = []

        expected_count = (
            len(known_expected) if known_expected else known_expected_count
        )

        # 3. Preserve pre-QA, cost, and cache telemetry safely without throwing
        safe_pre_qa = None
        if pre_qa_snapshot is not None and isinstance(pre_qa_snapshot, dict):
            safe_pre_qa = _safe_telemetry_copy(pre_qa_snapshot, label="pre_qa_snapshot")
        elif result is not None:
            try:
                pre_qa_obj = getattr(result, "qa_report", None)
                if pre_qa_obj is not None:
                    _, safe_pre_qa, _ = _safe_serialize_qa_report(
                        pre_qa_obj,
                        label="pre_qa_emergency",
                        fatal_defects=fatal_defects,
                    )
            except Exception:
                safe_pre_qa = None

        pre_qa_telemetry = (
            safe_pre_qa.get("telemetry", {})
            if isinstance(safe_pre_qa, dict)
            else {}
        )
        cost_telemetry = (
            safe_pre_qa.get("cost_telemetry", {})
            if isinstance(safe_pre_qa, dict)
            else {}
        )
        cache_telemetry = (
            safe_pre_qa.get("cache_telemetry", {})
            if isinstance(safe_pre_qa, dict)
            else {}
        )

        result_telemetry = {}
        if result is not None:
            try:
                res_tel = getattr(result, "telemetry", None)
                result_telemetry = _safe_telemetry_copy(
                    res_tel, label="result_telemetry"
                )
            except Exception:
                result_telemetry = {}

        # 4. Construct bridge_reconciliation telemetry record
        bridge_telemetry = {
            "expected_slots": expected_count,
            "resolved_slots": 0,
            "unresolved_slots": expected_count,
            "fatal_defects": list(fatal_defects),
            "adaptation_defects": [],
            "accounting_incomplete": not (accounting_established and expected_count > 0),
            "accounting_completeness": (
                "complete"
                if (accounting_established and expected_count > 0)
                else "incomplete"
            ),
            "emergency_failure": True,
            "emergency_stage": stage,
        }

        telemetry_payload = _safe_dict_convert(result_telemetry)
        telemetry_payload["bridge_reconciliation"] = bridge_telemetry

        # 5. Construct emergency QA report preserving 9 canonical fields
        blocked_reasons: List[str] = []
        if safe_pre_qa and isinstance(
            safe_pre_qa.get("blocked_reasons"), (list, tuple)
        ):
            for br in safe_pre_qa["blocked_reasons"]:
                blocked_reasons.append(_safe_str(br, 200))
        for defect in fatal_defects:
            blocked_reasons.append(f"INTEGRITY_DEFECT:{defect}")
        if paper_status == "FAILED":
            blocked_reasons.append(f"EXECUTION_FAILURE:{failure_code}")

        degraded_reasons: List[str] = []
        if safe_pre_qa and isinstance(
            safe_pre_qa.get("degraded_reasons"), (list, tuple)
        ):
            for dr in safe_pre_qa["degraded_reasons"]:
                degraded_reasons.append(_safe_str(dr, 200))

        emergency_qa_dict = {
            "status": paper_status,
            "slot_count": expected_count,
            "resolved_slots": 0,
            "unresolved_slots": list(known_expected),
            "degraded_reasons": degraded_reasons,
            "blocked_reasons": blocked_reasons,
            "telemetry": {
                **_safe_dict_convert(pre_qa_telemetry),
                "bridge_reconciliation": bridge_telemetry,
            },
            "cost_telemetry": _safe_dict_convert(cost_telemetry),
            "cache_telemetry": _safe_dict_convert(cache_telemetry),
            "_is_synthetic_fallback": True,
            "_serialization_error": failure_detail,
        }

        # 6. Build emergency metadata directly using literal constructor
        fallback_meta = _build_emergency_meta_dict(
            validated_exec_success=validated_exec_success,
            raw_success=raw_success,
            paper_status=paper_status,
            export_eligible=export_eligible,
            failure_code=failure_code,
            failure_detail=failure_detail,
            telemetry=telemetry_payload,
            factory_notes=[],
            qa_report=emergency_qa_dict,
            pre_adaptation_qa_report=safe_pre_qa,
            adaptation_defects=[],
            reconciliation_meta={
                "expected_slots": expected_count,
                "resolved_slots": 0,
                "unresolved_slots": expected_count,
                "fatal_defects": list(fatal_defects),
                "adaptation_defects": [],
                "emergency_stage": stage,
            },
            blocked_reasons=blocked_reasons,
            error=failure_detail,
            cleanup_errors=cleanup_errors,
        )

        return [], [], fallback_meta

    except Exception:
        # Infallible terminal fallback returning a fixed literal dictionary.
        # Completely independent with ZERO helper calls, ZERO constructor invocations, and pure primitive values.
        # Preserves pre-extracted primitive execution status (Tier 3 BLOCKED on success, Tier 2 FAILED on failure).
        return (
            [],
            [],
            {
                "success": bool(validated_exec_success),
                "raw_execution_success": safe_raw_success,
                "paper_status": paper_status,
                "export_eligible": False,
                "failure_code": "TERMINAL_EMERGENCY_CONTAINMENT_CRASH",
                "failure_detail": (
                    "Unhandled exception during emergency containment execution"
                ),
                "telemetry": {
                    "bridge_reconciliation": {
                        "expected_slots": known_expected_count,
                        "resolved_slots": 0,
                        "unresolved_slots": known_expected_count,
                        "fatal_defects": [
                            "TERMINAL_EMERGENCY_CONTAINMENT_CRASH"
                        ],
                        "adaptation_defects": [],
                        "accounting_incomplete": True,
                        "accounting_completeness": "incomplete",
                        "emergency_failure": True,
                        "emergency_stage": "TERMINAL_FALLBACK",
                    }
                },
                "factory_notes": [],
                "qa_report": None,
                "pre_adaptation_qa_report": None,
                "adaptation_defects": [],
                "reconciliation_meta": {
                    "expected_slots": known_expected_count,
                    "resolved_slots": 0,
                    "unresolved_slots": known_expected_count,
                    "fatal_defects": [
                        "TERMINAL_EMERGENCY_CONTAINMENT_CRASH"
                    ],
                    "adaptation_defects": [],
                    "emergency_stage": "TERMINAL_FALLBACK",
                },
                "blocked_reasons": ["TERMINAL_EMERGENCY_CONTAINMENT_CRASH"],
                "error": "Terminal emergency containment crash",
            },
        )


# -----------------------------------------------------------------------------
# 3.5 Bilateral Population Reconciliation Engine (_reconcile_post_adaptation)
# -----------------------------------------------------------------------------


def _reconcile_post_adaptation(
    *,
    result: OrchestratorResult,
    context: AgentContext,
    paper_parts: Any,
    full_paper: Any,
    cleanup_errors: Optional[List[str]] = None,
) -> Tuple[
    QAReport, List[str], Dict[str, Any], Optional[dict], Optional[dict], dict, bool
]:
    """
    Authoritative reconciliation engine validating both full_paper and paper_parts
    for complete bilateral bijective consistency.
    """
    pre_qa = getattr(result, "qa_report", None)
    fatal_defects: List[str] = []
    slots_with_fatal_defects: Set[str] = set()
    adaptation_defects: List[str] = []

    if cleanup_errors:
        for ce in cleanup_errors:
            fatal_defects.append(ce)

    raw_success = getattr(result, "success", None)
    if type(raw_success) is not bool:
        fatal_defects.append(
            f"INVALID_EXECUTION_SUCCESS_FLAG:type_{_safe_type_name(raw_success)}"
        )
        exec_succeeded = False
    else:
        exec_succeeded = raw_success

    # Stage 1: Authoritative expected slot IDs
    try:
        from .checking_agent import CheckingAgent

        agent = CheckingAgent()
        expected_ids = agent._get_validated_expected_slots(context)
    except Exception as exc:
        fatal_msg = f"INVALID_EXPECTED_SLOT_SPECIFICATION:{_safe_type_name(exc)}:{_safe_str(exc, 200)}"
        fatal_defects.append(fatal_msg)

        copied_telemetry = _safe_telemetry_copy(
            getattr(pre_qa, "telemetry", None), fatal_defects, label="pre_qa_telemetry"
        )
        copied_cost_telemetry = _safe_telemetry_copy(
            getattr(pre_qa, "cost_telemetry", None),
            fatal_defects,
            label="cost_telemetry",
        )
        copied_cache_telemetry = _safe_telemetry_copy(
            getattr(pre_qa, "cache_telemetry", None),
            fatal_defects,
            label="cache_telemetry",
        )
        copied_result_telemetry = _safe_telemetry_copy(
            getattr(result, "telemetry", None), fatal_defects, label="result_telemetry"
        )
        is_pre_ok, serialized_pre_qa, _ = _safe_serialize_qa_report(
            pre_qa, label="pre_adaptation_qa_report", fatal_defects=fatal_defects
        )

        raw_pre_blocked = getattr(pre_qa, "blocked_reasons", None) if pre_qa else None
        validated_pre_blocked = _validate_and_extract_reasons(
            raw_pre_blocked, "blocked_reasons", fatal_defects
        )
        raw_pre_degraded = (
            getattr(pre_qa, "degraded_reasons", None) if pre_qa else None
        )
        validated_pre_degraded = _validate_and_extract_reasons(
            raw_pre_degraded, "degraded_reasons", fatal_defects
        )

        return _finalize_reconciliation(
            raw_success=raw_success,
            exec_succeeded=exec_succeeded,
            expected_ids=[],
            f_post=[],
            p_post=[],
            b_post=[],
            fatal_defects=fatal_defects,
            adaptation_defects=[],
            pre_status=getattr(pre_qa, "status", None),
            context=context,
            copied_telemetry=copied_telemetry,
            copied_cost_telemetry=copied_cost_telemetry,
            copied_cache_telemetry=copied_cache_telemetry,
            copied_result_telemetry=copied_result_telemetry,
            serialized_pre_qa=serialized_pre_qa,
            failure_code=getattr(result, "failure_code", None),
            validated_pre_blocked=validated_pre_blocked,
            validated_pre_degraded=validated_pre_degraded,
            resolved_population_changed=True,
            min_ratio=0.9,
            accounting_complete=False,
        )

    expected_set = set(expected_ids)
    expected_count = len(expected_ids)

    # Missing pre_qa fail-closed handling
    if pre_qa is None:
        fatal_defects.append("MISSING_PRE_ADAPTATION_QA_REPORT")
        copied_result_telemetry = _safe_telemetry_copy(
            getattr(result, "telemetry", None), fatal_defects, label="result_telemetry"
        )
        return _finalize_reconciliation(
            raw_success=raw_success,
            exec_succeeded=exec_succeeded,
            expected_ids=expected_ids,
            f_post=[],
            p_post=[],
            b_post=list(expected_ids),
            fatal_defects=fatal_defects,
            adaptation_defects=[],
            pre_status=None,
            context=context,
            copied_telemetry={},
            copied_cost_telemetry={},
            copied_cache_telemetry={},
            copied_result_telemetry=copied_result_telemetry,
            serialized_pre_qa=None,
            failure_code=getattr(result, "failure_code", None),
            validated_pre_blocked=[],
            validated_pre_degraded=[],
            resolved_population_changed=True,
            min_ratio=0.9,
            accounting_complete=False,
            is_synthetic=True,
            synthetic_reason="MISSING_PRE_ADAPTATION_QA_REPORT",
        )

    # Stage 2: Copy telemetry & serialize pre-QA
    copied_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "telemetry", None), fatal_defects, label="pre_qa_telemetry"
    )
    copied_cost_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "cost_telemetry", None),
        fatal_defects,
        label="cost_telemetry",
    )
    copied_cache_telemetry = _safe_telemetry_copy(
        getattr(pre_qa, "cache_telemetry", None),
        fatal_defects,
        label="cache_telemetry",
    )
    copied_result_telemetry = _safe_telemetry_copy(
        getattr(result, "telemetry", None), fatal_defects, label="result_telemetry"
    )
    is_pre_ok, serialized_pre_qa, _ = _safe_serialize_qa_report(
        pre_qa, label="pre_adaptation_qa_report", fatal_defects=fatal_defects
    )

    # Pre_qa accounting validation
    valid_statuses = {"OK", "PASS", "PARTIAL", "DEGRADED", "BLOCKED", "FAILED"}
    pre_status = getattr(pre_qa, "status", None)
    if not isinstance(pre_status, str):
        fatal_defects.append(
            f"INVALID_PRE_QA_STATUS_TYPE:{_safe_type_name(pre_status)}"
        )
    elif pre_status not in valid_statuses:
        fatal_defects.append(f"INVALID_PRE_QA_STATUS:{_safe_str(pre_status, 50)}")

    raw_slot_count = getattr(pre_qa, "slot_count", None)
    if type(raw_slot_count) is not int or raw_slot_count != expected_count:
        fatal_defects.append(
            f"MALFORMED_PRE_QA_SLOT_COUNT:{_safe_str(raw_slot_count, 50)}:expected={expected_count}"
        )

    raw_resolved = getattr(pre_qa, "resolved_slots", None)
    if type(raw_resolved) is not int or not (0 <= raw_resolved <= expected_count):
        fatal_defects.append(
            f"MALFORMED_PRE_QA_RESOLVED_SLOTS:{_safe_str(raw_resolved, 50)}"
        )

    raw_unres_slots = getattr(pre_qa, "unresolved_slots", None)
    validated_pre_qa_unres_list: List[str] = []
    seen_unres_strings: Set[str] = set()

    if not isinstance(raw_unres_slots, (list, tuple)) or isinstance(
        raw_unres_slots, (str, bytes)
    ):
        fatal_defects.append(
            f"MALFORMED_PRE_QA_UNRESOLVED_SLOTS_TYPE:{_safe_type_name(raw_unres_slots)}"
        )
    else:
        for idx, u in enumerate(raw_unres_slots):
            if not isinstance(u, str):
                fatal_defects.append(
                    f"MALFORMED_PRE_QA_UNRESOLVED_SLOT_ENTRY:index_{idx}:type_{_safe_type_name(u)}"
                )
                continue
            if not u.strip():
                fatal_defects.append(
                    f"MALFORMED_PRE_QA_UNRESOLVED_SLOT_ID:index_{idx}:empty"
                )
                continue
            if u != u.strip():
                fatal_defects.append(
                    f"MALFORMED_PRE_QA_UNRESOLVED_SLOT_ID_WHITESPACE:{_safe_str(u, 100)}"
                )
                continue
            if u not in expected_set:
                fatal_defects.append(
                    f"UNEXPECTED_PRE_QA_UNRESOLVED_SLOT_ID:{_safe_str(u, 100)}"
                )
                continue
            if u in seen_unres_strings:
                fatal_defects.append(
                    f"DUPLICATE_PRE_QA_UNRESOLVED_SLOTS:{_safe_str(u, 100)}"
                )
                continue
            seen_unres_strings.add(u)
            validated_pre_qa_unres_list.append(u)

    pre_qa_unres_set = set(validated_pre_qa_unres_list)

    total_unres_count = (
        len(raw_unres_slots)
        if isinstance(raw_unres_slots, (list, tuple))
        and not isinstance(raw_unres_slots, (str, bytes))
        else 0
    )
    if (
        type(raw_resolved) is int
        and (raw_resolved + total_unres_count) != expected_count
    ):
        fatal_defects.append(
            f"PRE_QA_ACCOUNTING_MISMATCH:resolved={raw_resolved}+unres={total_unres_count}!=expected={expected_count}"
        )

    # Status-accounting coherence
    if pre_status in ("OK", "PASS"):
        if raw_resolved != expected_count or total_unres_count != 0:
            fatal_defects.append(
                f"STATUS_ACCOUNTING_CONTRADICTION:{pre_status}_has_unresolved_slots"
            )
    elif pre_status == "DEGRADED":
        if raw_resolved != expected_count or total_unres_count != 0:
            fatal_defects.append(
                "STATUS_ACCOUNTING_CONTRADICTION:DEGRADED_has_unresolved_slots"
            )
    elif pre_status == "PARTIAL":
        if type(raw_resolved) is int and (
            raw_resolved <= 0 or total_unres_count == 0
        ):
            fatal_defects.append(
                "STATUS_ACCOUNTING_CONTRADICTION:PARTIAL_invalid_accounting"
            )

    validated_pre_blocked = _validate_and_extract_reasons(
        raw=getattr(pre_qa, "blocked_reasons", None),
        field_name="blocked_reasons",
        fatal_defects=fatal_defects,
    )
    validated_pre_degraded = _validate_and_extract_reasons(
        raw=getattr(pre_qa, "degraded_reasons", None),
        field_name="degraded_reasons",
        fatal_defects=fatal_defects,
    )

    # Question population existence & synchronization
    res_q = getattr(result, "questions", None)
    ctx_q = getattr(context, "questions", None)

    if res_q is None and ctx_q is None:
        fatal_defects.append("MISSING_PRE_ADAPTATION_QUESTIONS_PAYLOAD:both_absent")
        questions_list = []
    elif res_q is not None and ctx_q is None:
        fatal_defects.append(
            "QUESTION_POPULATION_DISAGREEMENT:context_questions_missing"
        )
        questions_list = res_q if isinstance(res_q, list) else []
        for q in questions_list:
            sid = getattr(q, "slot_id", None)
            if isinstance(sid, str) and sid in expected_set:
                slots_with_fatal_defects.add(sid)
    elif res_q is None and ctx_q is not None:
        fatal_defects.append(
            "QUESTION_POPULATION_DISAGREEMENT:result_questions_missing"
        )
        questions_list = ctx_q if isinstance(ctx_q, list) else []
        for q in questions_list:
            sid = getattr(q, "slot_id", None)
            if isinstance(sid, str) and sid in expected_set:
                slots_with_fatal_defects.add(sid)
    else:
        if not isinstance(res_q, list):
            fatal_defects.append(
                f"MALFORMED_PRE_ADAPTATION_QUESTIONS_PAYLOAD:result:type_{_safe_type_name(res_q)}"
            )
        if not isinstance(ctx_q, list):
            fatal_defects.append(
                f"MALFORMED_PRE_ADAPTATION_QUESTIONS_PAYLOAD:context:type_{_safe_type_name(ctx_q)}"
            )

        if isinstance(res_q, list) and isinstance(ctx_q, list):
            min_len = min(len(res_q), len(ctx_q))
            for i in range(min_len):
                rq, cq = res_q[i], ctx_q[i]
                r_sid = getattr(rq, "slot_id", None)
                c_sid = getattr(cq, "slot_id", None)
                r_unres = getattr(rq, "is_unresolved", None)
                c_unres = getattr(cq, "is_unresolved", None)
                r_scope = getattr(rq, "unresolved_scope", None)
                c_scope = getattr(cq, "unresolved_scope", None)
                r_subparts = getattr(rq, "unresolved_subparts", None)
                c_subparts = getattr(cq, "unresolved_subparts", None)
                r_code = getattr(rq, "failure_code", None)
                c_code = getattr(cq, "failure_code", None)
                r_detail = getattr(rq, "failure_detail", None)
                c_detail = getattr(cq, "failure_detail", None)

                if (
                    r_sid != c_sid
                    or r_unres != c_unres
                    or r_scope != c_scope
                    or r_subparts != c_subparts
                    or r_code != c_code
                    or r_detail != c_detail
                ):
                    if r_sid != c_sid:
                        if isinstance(r_sid, str) and r_sid in expected_set:
                            slots_with_fatal_defects.add(r_sid)
                        if isinstance(c_sid, str) and c_sid in expected_set:
                            slots_with_fatal_defects.add(c_sid)
                        fatal_defects.append(
                            f"QUESTION_POPULATION_DISAGREEMENT:index_{i}:res={_safe_str(r_sid, 50)}_vs_ctx={_safe_str(c_sid, 50)}"
                        )
                    else:
                        if isinstance(r_sid, str) and r_sid in expected_set:
                            slots_with_fatal_defects.add(r_sid)
                        fatal_defects.append(
                            f"QUESTION_POPULATION_DISAGREEMENT:{_safe_str(r_sid, 50)}:lifecycle_mismatch"
                        )

            if len(res_q) != len(ctx_q):
                fatal_defects.append(
                    f"QUESTION_POPULATION_DISAGREEMENT:length_mismatch_{len(res_q)}_vs_{len(ctx_q)}"
                )
                longer = res_q if len(res_q) > len(ctx_q) else ctx_q
                for idx in range(min_len, len(longer)):
                    tail_sid = getattr(longer[idx], "slot_id", None)
                    if isinstance(tail_sid, str) and tail_sid in expected_set:
                        slots_with_fatal_defects.add(tail_sid)
                        fatal_defects.append(
                            f"QUESTION_POPULATION_DISAGREEMENT:tail_slot_id:{_safe_str(tail_sid, 50)}"
                        )

            questions_list = res_q
        else:
            questions_list = (
                res_q
                if isinstance(res_q, list)
                else (ctx_q if isinstance(ctx_q, list) else [])
            )

    seen_pre_counts: Dict[str, int] = {}
    q_pre_by_slot: Dict[str, Any] = {}
    for idx, q in enumerate(questions_list):
        sid = getattr(q, "slot_id", None)
        if not isinstance(sid, str) or sid != sid.strip() or not sid.strip():
            fatal_defects.append(
                f"MALFORMED_PRE_ADAPTATION_QUESTION_ID:question[{idx}]"
            )
            continue
        seen_pre_counts[sid] = seen_pre_counts.get(sid, 0) + 1
        if seen_pre_counts[sid] > 1:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"DUPLICATE_PRE_ADAPTATION_QUESTION_ID:{_safe_str(sid, 50)}"
            )
            continue
        if sid not in expected_set:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"UNEXPECTED_PRE_ADAPTATION_QUESTION_ID:{_safe_str(sid, 50)}"
            )
            continue
        q_pre_by_slot[sid] = q

    for sid in expected_ids:
        if sid not in seen_pre_counts:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"MISSING_PRE_ADAPTATION_QUESTION:{_safe_str(sid, 50)}"
            )

    for sid in expected_ids:
        q = q_pre_by_slot.get(sid)
        if q is None:
            continue
        _validate_stage2a_lifecycle_contract(
            sid=sid,
            q=q,
            slots_with_fatal_defects=slots_with_fatal_defects,
            fatal_defects=fatal_defects,
        )
        raw_is_unres = getattr(q, "is_unresolved", False)
        in_qa_unres = sid in pre_qa_unres_set
        if raw_is_unres and not in_qa_unres:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"PRE_ADAPTATION_CONTRADICTION:{_safe_str(sid, 50)}:question_unresolved_but_not_in_qa_report"
            )
        elif not raw_is_unres and in_qa_unres:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"PRE_ADAPTATION_CONTRADICTION:{_safe_str(sid, 50)}:qa_report_unresolved_but_question_clean"
            )

    min_ratio = _resolve_min_ratio(context, pre_qa, fatal_defects)

    # Stage 3A: Inspection of full_paper adapted records
    if not isinstance(full_paper, list):
        fatal_defects.append(
            f"MALFORMED_FULL_PAPER_PAYLOAD:type_{_safe_type_name(full_paper)}"
        )
        full_paper_list = []
    else:
        full_paper_list = full_paper

    seen_adapted_counts: Dict[str, int] = {}
    adapted_by_slot: Dict[str, dict] = {}
    full_paper_slot_ids: List[str] = []

    for idx, q_dict in enumerate(full_paper_list):
        if not isinstance(q_dict, dict):
            fatal_defects.append(
                f"MALFORMED_ADAPTED_RECORD:index_{idx}:type_{_safe_type_name(q_dict)}"
            )
            continue

        sid = q_dict.get("slot_id")
        if not isinstance(sid, str):
            fatal_defects.append(
                f"MALFORMED_ADAPTED_SLOT_ID:index_{idx}:not_a_string"
            )
            continue
        if sid != sid.strip() or not sid.strip():
            fatal_defects.append(
                f"MALFORMED_ADAPTED_SLOT_ID:index_{idx}:whitespace_or_empty"
            )
            continue

        q_mod = q_dict.get("module_id")
        if not isinstance(q_mod, str) or not q_mod.strip():
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"MALFORMED_ADAPTED_QUESTION_MODULE_ID:slot={sid}:type_{_safe_type_name(q_mod)}"
            )
        elif sid.startswith("module_") and not sid.startswith(f"{q_mod.strip()}_"):
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"ADAPTED_QUESTION_SLOT_MODULE_PREFIX_MISMATCH:module={q_mod.strip()}:slot={sid}"
            )

        if sid not in expected_set:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(f"UNEXPECTED_ADAPTED_SLOT_ID:{_safe_str(sid, 50)}")
            continue

        seen_adapted_counts[sid] = seen_adapted_counts.get(sid, 0) + 1
        if seen_adapted_counts[sid] > 1:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(f"DUPLICATE_ADAPTED_SLOT_ID:{_safe_str(sid, 50)}")
            continue

        adapted_by_slot[sid] = q_dict
        full_paper_slot_ids.append(sid)

    for sid in expected_ids:
        if sid not in seen_adapted_counts:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(f"MISSING_ADAPTED_SLOT:{_safe_str(sid, 50)}")

    # Stage 3B: Bijective Validation of paper_parts against full_paper
    if not isinstance(paper_parts, list):
        fatal_defects.append(
            f"MALFORMED_PAPER_PARTS_PAYLOAD:type_{_safe_type_name(paper_parts)}"
        )
        for sid in expected_ids:
            slots_with_fatal_defects.add(sid)
    elif len(full_paper_list) > 0 and len(paper_parts) == 0:
        fatal_defects.append("PAPER_PARTS_EMPTY_WITH_POPULATED_FULL_PAPER")
        for sid in expected_ids:
            slots_with_fatal_defects.add(sid)
    else:
        seen_part_modules: Set[str] = set()
        parts_slot_ids: List[str] = []
        seen_part_slot_counts: Dict[str, int] = {}

        for p_idx, part in enumerate(paper_parts):
            if not isinstance(part, dict):
                fatal_defects.append(
                    f"MALFORMED_PAPER_PART_RECORD:index_{p_idx}:type_{_safe_type_name(part)}"
                )
                continue

            mod_id = part.get("module_id")
            if not isinstance(mod_id, str):
                fatal_defects.append(
                    f"MALFORMED_PAPER_PART_MODULE_ID:index_{p_idx}:not_a_string"
                )
                continue
            if mod_id != mod_id.strip() or not mod_id.strip():
                fatal_defects.append(
                    f"MALFORMED_PAPER_PART_MODULE_ID_WHITESPACE:index_{p_idx}:{_safe_str(mod_id, 50)}"
                )
                continue

            # Register valid module ID immediately to catch duplicates even if questions container is malformed
            is_dup_mod = mod_id in seen_part_modules
            if is_dup_mod:
                fatal_defects.append(
                    f"DUPLICATE_PAPER_PART_MODULE_ID:{_safe_str(mod_id, 50)}"
                )
            else:
                seen_part_modules.add(mod_id)

            part_questions = part.get("questions")
            if not isinstance(part_questions, list):
                fatal_defects.append(
                    f"MALFORMED_PAPER_PART_QUESTIONS:index_{p_idx}:type_{_safe_type_name(part_questions)}"
                )
                continue

            if is_dup_mod:
                for pq in part_questions:
                    if isinstance(pq, dict):
                        pq_sid = pq.get("slot_id")
                        if isinstance(pq_sid, str) and pq_sid in expected_set:
                            slots_with_fatal_defects.add(pq_sid)

            for q_idx, pq in enumerate(part_questions):
                if not isinstance(pq, dict):
                    fatal_defects.append(
                        f"MALFORMED_PAPER_PART_QUESTION_RECORD:part_{p_idx}:q_{q_idx}:not_a_dict"
                    )
                    continue
                pq_sid = pq.get("slot_id")
                if not isinstance(pq_sid, str) or not pq_sid.strip():
                    fatal_defects.append(
                        f"MALFORMED_PAPER_PART_QUESTION_SLOT_ID:part_{p_idx}:q_{q_idx}"
                    )
                    continue

                # Module-slot attribution invariant grounded in legacy_adapter contract
                q_mod = pq.get("module_id")
                if q_mod != mod_id:
                    slots_with_fatal_defects.add(pq_sid)
                    fatal_defects.append(
                        f"QUESTION_MODULE_ATTRIBUTION_MISMATCH:part={mod_id}:slot={pq_sid}:q_mod={_safe_str(q_mod, 50)}"
                    )
                elif pq_sid.startswith("module_") and not pq_sid.startswith(
                    f"{mod_id}_"
                ):
                    slots_with_fatal_defects.add(pq_sid)
                    fatal_defects.append(
                        f"QUESTION_SLOT_MODULE_PREFIX_MISMATCH:part={mod_id}:slot={pq_sid}"
                    )

                # Cross-representation module attribution matching between paper_parts and full_paper
                if pq_sid in adapted_by_slot:
                    full_q = adapted_by_slot[pq_sid]
                    full_mod = full_q.get("module_id")
                    if isinstance(full_mod, str) and full_mod.strip() != mod_id:
                        slots_with_fatal_defects.add(pq_sid)
                        fatal_defects.append(
                            f"QUESTION_MODULE_CROSS_REPRESENTATION_MISMATCH:slot={pq_sid}:part_mod={mod_id}:full_mod={full_mod.strip()}"
                        )

                # Duplicate slot detection within paper_parts
                seen_part_slot_counts[pq_sid] = (
                    seen_part_slot_counts.get(pq_sid, 0) + 1
                )
                if seen_part_slot_counts[pq_sid] > 1:
                    slots_with_fatal_defects.add(pq_sid)
                    fatal_defects.append(
                        f"DUPLICATE_PAPER_PARTS_SLOT_ID:{_safe_str(pq_sid, 50)}"
                    )

                parts_slot_ids.append(pq_sid)

        # Bijective set consistency
        parts_set = set(parts_slot_ids)
        full_set = set(full_paper_slot_ids)

        for sid in parts_set:
            if sid not in full_set:
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"SLOT_MISMATCH_PARTS_EXTRA:{_safe_str(sid, 50)}"
                )

        for sid in full_set:
            if sid not in parts_set:
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"SLOT_MISMATCH_PARTS_MISSING:{_safe_str(sid, 50)}"
                )

        # Order-only disagreement identification
        if len(parts_slot_ids) == len(full_paper_slot_ids) and parts_set == full_set:
            for i in range(len(parts_slot_ids)):
                p_s = parts_slot_ids[i]
                f_s = full_paper_slot_ids[i]
                if p_s != f_s:
                    if p_s in expected_set:
                        slots_with_fatal_defects.add(p_s)
                    if f_s in expected_set:
                        slots_with_fatal_defects.add(f_s)
                    fatal_defects.append(
                        f"PAPER_PARTS_ORDER_MISMATCH:index_{i}:parts={p_s}_vs_full={f_s}"
                    )
        elif parts_slot_ids != full_paper_slot_ids:
            fatal_defects.append(
                f"PAPER_PARTS_FULL_PAPER_DISAGREEMENT:parts_count_{len(parts_slot_ids)}_vs_full_count_{len(full_paper_slot_ids)}"
            )

    # Pre-adaptation partitioning: F_pre, P_pre, B_pre
    f_pre: List[str] = []
    p_pre: List[str] = []
    b_pre: List[str] = []

    for sid in expected_ids:
        q_pre = q_pre_by_slot.get(sid)
        if q_pre is None or sid in slots_with_fatal_defects:
            b_pre.append(sid)
        elif getattr(q_pre, "is_unresolved", False):
            scope = getattr(q_pre, "unresolved_scope", None)
            subparts = getattr(q_pre, "unresolved_subparts", ())
            if (
                scope == "subquestion"
                and isinstance(subparts, (list, tuple))
                and len(subparts) > 0
            ):
                p_pre.append(sid)
            else:
                b_pre.append(sid)
        else:
            f_pre.append(sid)

    # Post-adaptation partitioning: F_post, P_post, B_post
    f_post: List[str] = []
    p_post: List[str] = []
    b_post: List[str] = []

    for sid in expected_ids:
        if (
            sid not in adapted_by_slot
            or seen_adapted_counts.get(sid, 0) != 1
            or sid in slots_with_fatal_defects
        ):
            b_post.append(sid)
            continue

        q_dict = adapted_by_slot[sid]
        slot_has_metadata_defect = False

        is_unres = q_dict.get("is_unresolved")
        if type(is_unres) is not bool:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"MALFORMED_ADAPTED_RECORD_METADATA:{_safe_str(sid, 50)}:is_unresolved_not_boolean"
            )
            slot_has_metadata_defect = True

        adapt_code = q_dict.get("adaptation_failure_code")
        adapt_detail = q_dict.get("adaptation_failure_detail")

        if adapt_code is None and adapt_detail is not None:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"ORPHANED_ADAPTATION_DETAIL:{_safe_str(sid, 50)}:detail_without_failure_code"
            )
            slot_has_metadata_defect = True

        if adapt_detail is not None and not isinstance(adapt_detail, str):
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"MALFORMED_ADAPTATION_METADATA:{_safe_str(sid, 50)}:invalid_detail_type"
            )
            slot_has_metadata_defect = True

        has_adapt_failure = False
        if adapt_code is not None:
            if not isinstance(adapt_code, str):
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"MALFORMED_ADAPTATION_METADATA:{_safe_str(sid, 50)}:not_a_string"
                )
                slot_has_metadata_defect = True
                has_adapt_failure = True
            elif not adapt_code.strip():
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"MALFORMED_ADAPTATION_METADATA:{_safe_str(sid, 50)}:empty_code"
                )
                slot_has_metadata_defect = True
                has_adapt_failure = True
            else:
                has_adapt_failure = True

        if has_adapt_failure and is_unres is False:
            slots_with_fatal_defects.add(sid)
            fatal_defects.append(
                f"INCONSISTENT_ADAPTED_METADATA:{_safe_str(sid, 50)}:adaptation_code_with_is_unresolved_false"
            )
            slot_has_metadata_defect = True

        if slot_has_metadata_defect:
            b_post.append(sid)
            continue

        if has_adapt_failure:
            adaptation_defects.append(
                f"ADAPTATION_FAILURE:{sid}:{_safe_str(adapt_code.strip(), 100)}"
            )

        if sid in b_pre:
            b_post.append(sid)
        elif sid in p_pre:
            if not is_unres:
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"UNRESOLVED_STATE_SILENTLY_CLEARED:{_safe_str(sid, 50)}"
                )
                b_post.append(sid)
            elif has_adapt_failure:
                b_post.append(sid)
            else:
                p_post.append(sid)
        else:  # sid in f_pre
            if is_unres and not has_adapt_failure:
                slots_with_fatal_defects.add(sid)
                fatal_defects.append(
                    f"UNEXPECTED_UNRESOLVED_ADAPTED_SLOT:{_safe_str(sid, 50)}:unresolved_true_without_adaptation_code"
                )
                b_post.append(sid)
            elif has_adapt_failure or is_unres:
                b_post.append(sid)
            else:
                f_post.append(sid)

    resolved_population_changed = len(f_post) < len(f_pre)

    # Derive accounting completeness dynamically from established expected population and complete partition coverage
    accounting_established = bool(expected_ids) and len(expected_ids) > 0 and not any(
        d.startswith("INVALID_EXPECTED_SLOT_SPECIFICATION") for d in fatal_defects
    )
    all_expected_accounted = (
        len(f_post) + len(p_post) + len(b_post) == len(expected_set)
        and set(f_post + p_post + b_post) == expected_set
        if accounting_established
        else False
    )
    accounting_complete = accounting_established and all_expected_accounted

    return _finalize_reconciliation(
        raw_success=raw_success,
        exec_succeeded=exec_succeeded,
        expected_ids=expected_ids,
        f_post=f_post,
        p_post=p_post,
        b_post=b_post,
        fatal_defects=fatal_defects,
        adaptation_defects=adaptation_defects,
        pre_status=pre_status,
        context=context,
        copied_telemetry=copied_telemetry,
        copied_cost_telemetry=copied_cost_telemetry,
        copied_cache_telemetry=copied_cache_telemetry,
        copied_result_telemetry=copied_result_telemetry,
        serialized_pre_qa=serialized_pre_qa,
        failure_code=getattr(result, "failure_code", None),
        validated_pre_blocked=validated_pre_blocked,
        validated_pre_degraded=validated_pre_degraded,
        resolved_population_changed=resolved_population_changed,
        min_ratio=min_ratio,
        accounting_complete=accounting_complete,
    )


# -----------------------------------------------------------------------------
# 3.6 Public Pipeline Entry Points
# -----------------------------------------------------------------------------


class PipelineBridgeResult(tuple):
    """
    Dual-compatible result tuple for run_pipeline_v3_or_legacy.
    Unpacks as (paper_parts, full_paper) for legacy callers,
    while exposing .meta, .qa_report, .paper_status, and .export_eligible.
    Uses strict type-checked boolean assignments rather than arbitrary truthy coercion.
    """

    def __new__(cls, paper_parts: List[dict], full_paper: List[dict], meta: dict):
        obj = super().__new__(cls, (paper_parts, full_paper))
        obj.meta = meta if isinstance(meta, dict) else {}
        obj.paper_status = obj.meta.get("paper_status")
        raw_eligible = obj.meta.get("export_eligible")
        obj.export_eligible = type(raw_eligible) is bool and raw_eligible is True
        raw_success = obj.meta.get("success")
        obj.success = type(raw_success) is bool and raw_success is True
        obj.qa_report = obj.meta.get("qa_report")
        return obj


def _safe_wrap_bridge_result(
    parts: List[dict],
    full: List[dict],
    meta: dict,
) -> Union[PipelineBridgeResult, Tuple[List[dict], List[dict]]]:
    """
    Safely construct PipelineBridgeResult wrapper,
    falling back to a plain 2-element tuple on construction failure while preserving
    the 2-element unpack contract.
    """
    try:
        return PipelineBridgeResult(parts, full, meta)
    except Exception:
        # Fallback to plain tuple strictly preserving 2-element legacy unpacking
        return (parts, full)


def run_v3_pipeline(
    *,
    paper_spec: Any,
    artifact: Any,
    marks_split: List[List[int]],
    knowledge_graph: Any = None,
    request: Optional[dict] = None,
    dependency_factory: Optional[DependencyFactory] = None,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Run the v3 pipeline. Returns (paper_parts, full_paper, meta).
    Outermost exception containment boundary guarantees every exit converges on structured metadata.
    """
    context = None
    result = None
    cleanup_errors: List[str] = []
    expected_ids: List[str] = []
    pre_qa_snapshot: Optional[dict] = None
    accounting_established: bool = False

    try:
        # 1. Protected Context Construction
        try:
            context = AgentContext(
                request=request or {},
                paper_spec=paper_spec,
                artifact=artifact,
                knowledge_graph=knowledge_graph,
                marks_split=list(marks_split or []),
            )
            if not context.plans:
                context.plans = None
            from .checking_agent import CheckingAgent

            expected_ids = CheckingAgent()._get_validated_expected_slots(context)
        except Exception as exc:
            return _handle_context_construction_exception(
                exc=exc,
                failure_label="CONTEXT_CONSTRUCTION_ERROR",
            )

        # 2. Protected Dependency Factory Setup & Build
        if dependency_factory is None:
            try:
                dependency_factory = DependencyFactory(
                    llm_caller_factory=default_llm_caller_factory,
                    vlm_caller_factory=default_vlm_caller_factory,
                    api_caller_factory=default_api_caller_factory,
                    sympy_verifier_factory=default_sympy_verifier_factory,
                )
            except Exception as exc:
                return _handle_initialization_exception(
                    exc=exc,
                    context=context,
                    failure_label="DEPENDENCY_FACTORY_CONSTRUCTION_ERROR",
                )

        try:
            factory = dependency_factory.build()
        except Exception as exc:
            return _handle_initialization_exception(
                exc=exc,
                context=context,
                failure_label="DEPENDENCY_FACTORY_BUILD_ERROR",
            )

        raw_factory_success = getattr(factory, "success", None)
        if type(raw_factory_success) is not bool:
            return _handle_initialization_exception(
                exc=TypeError(
                    f"Malformed factory result: success={_safe_repr(raw_factory_success, 100)}"
                ),
                context=context,
                failure_label="MALFORMED_DEPENDENCY_FACTORY_RESULT",
            )

        if not raw_factory_success:
            return _handle_dependency_factory_failure(
                factory=factory,
                context=context,
            )

        # 3. Protected Knowledge Graph Build
        if knowledge_graph is None and artifact is not None:
            try:
                knowledge_graph = _build_kg_for_artifact(artifact, paper_spec)
                context.knowledge_graph = knowledge_graph
            except Exception as exc:
                return _handle_initialization_exception(
                    exc=exc,
                    context=context,
                    failure_label="KNOWLEDGE_GRAPH_BUILD_ERROR",
                )

        # 4. Protected Orchestrator Construction & Pre-Run Reset
        try:
            planning = getattr(factory, "planning", None)
            writing = getattr(factory, "writing", None)
            evaluation = getattr(factory, "evaluation", None)
            refinement = getattr(factory, "refinement", None)
            checking = getattr(factory, "checking", None)

            orchestrator = AgentOrchestrator(
                planning_agent=planning,
                writing_agent=writing,
                evaluation_agent=evaluation,
                refinement_agent=refinement,
                checking_agent=checking,
            )
        except Exception as exc:
            return _handle_initialization_exception(
                exc=exc,
                context=context,
                failure_label="ORCHESTRATOR_CONSTRUCTION_ERROR",
            )

        pre_run_errors: List[str] = []
        _safe_reset_evaluation(factory, pre_run_errors, stage_label="PRE_RUN")
        if pre_run_errors:
            return _handle_initialization_exception(
                exc=RuntimeError(pre_run_errors[0]),
                context=context,
                failure_label="EVALUATION_RESET_PRE_RUN_ERROR",
            )

        # 5. Protected Orchestrator Run with Non-Masking Cleanup
        orchestrator_exception = None

        try:
            result = orchestrator.run(context)
        except Exception as exc:
            orchestrator_exception = exc
        finally:
            _safe_reset_evaluation(factory, cleanup_errors, stage_label="CLEANUP")
            if cleanup_errors and orchestrator_exception is None:
                orchestrator_exception = RuntimeError(cleanup_errors[0])

        if orchestrator_exception is not None:
            return _handle_orchestrator_exception(
                orchestrator_exception=orchestrator_exception,
                context=context,
                factory=factory,
                cleanup_errors=cleanup_errors,
            )

        # Safely snapshot pre-QA report before adaptation
        if result is not None and getattr(result, "qa_report", None) is not None:
            try:
                _, pre_qa_snapshot, _ = _safe_serialize_qa_report(
                    result.qa_report, label="pre_adaptation_snapshot"
                )
            except Exception:
                pre_qa_snapshot = None

        # 6. Narrow Exception Boundary Around to_legacy_shape()
        adaptation_exception = None
        try:
            paper_parts, full_paper = to_legacy_shape(result, context)
        except Exception as exc:
            paper_parts = []
            full_paper = []
            adaptation_exception = exc

        if adaptation_exception is not None:
            (
                reconciled_qa,
                adaptation_defects,
                reconciliation_meta,
                serialized_pre_qa,
                serialized_reconciled_qa,
                copied_result_telemetry,
                export_eligible,
            ) = _handle_adaptation_execution_error(
                result=result,
                context=context,
                adaptation_exception=adaptation_exception,
                cleanup_errors=cleanup_errors,
            )
        else:
            (
                reconciled_qa,
                adaptation_defects,
                reconciliation_meta,
                serialized_pre_qa,
                serialized_reconciled_qa,
                copied_result_telemetry,
                export_eligible,
            ) = _reconcile_post_adaptation(
                result=result,
                context=context,
                paper_parts=paper_parts,
                full_paper=full_paper,
                cleanup_errors=cleanup_errors,
            )
            accounting_established = True

        # Protected result.qa_report assignment
        try:
            result.qa_report = reconciled_qa
        except Exception:
            pass

        # Update paper_parts with reconciled QA report
        if isinstance(paper_parts, list):
            for part in paper_parts:
                if isinstance(part, dict):
                    part["qa_report"] = serialized_reconciled_qa

        raw_success = getattr(result, "success", None)
        validated_exec_success = type(raw_success) is bool and raw_success is True

        meta = _build_pipeline_meta(
            validated_exec_success=validated_exec_success,
            raw_success=raw_success,
            paper_status=reconciled_qa.status,
            export_eligible=export_eligible,
            failure_code=getattr(result, "failure_code", None),
            failure_detail=getattr(result, "failure_detail", None),
            telemetry=copied_result_telemetry,
            factory_notes=list(getattr(factory, "notes", []) or []),
            qa_report=serialized_reconciled_qa,
            pre_adaptation_qa_report=serialized_pre_qa,
            adaptation_defects=list(adaptation_defects),
            reconciliation_meta=reconciliation_meta,
            blocked_reasons=list(
                getattr(reconciled_qa, "blocked_reasons", []) or []
            ),
            error=(
                _safe_str(adaptation_exception, 200)
                if adaptation_exception is not None
                else None
            ),
            cleanup_errors=cleanup_errors,
        )

        return paper_parts, full_paper, meta

    except Exception as unhandled_pipeline_exc:
        # Outermost containment: captures unhandled exceptions with established pipeline state
        return _handle_emergency_pipeline_failure(
            exc=unhandled_pipeline_exc,
            stage="RUN_V3_PIPELINE",
            result=result,
            context=context,
            pre_qa_snapshot=pre_qa_snapshot,
            cleanup_errors=cleanup_errors,
            accounting_established=accounting_established,
            expected_ids=expected_ids,
        )


def run_pipeline_v3_or_legacy(
    *,
    legacy_runner: Callable[..., Tuple[List[dict], List[dict]]],
    v3_kwargs: dict,
    legacy_kwargs: dict,
    return_meta: bool = False,
    out_meta: Optional[dict] = None,
) -> Union[
    PipelineBridgeResult,
    Tuple[List[dict], List[dict]],
    Tuple[List[dict], List[dict], dict],
]:
    """
    Full public dispatch supporting legacy and v3 pipelines
    with structural legacy validation, bilateral question representation equivalence,
    separation of technical execution from exportability, 2-element tuple unpack compatibility,
    and post-execution dispatcher status preservation.
    """
    v3_meta = None
    try:
        try:
            is_v3 = v3_enabled()
        except Exception:
            is_v3 = True

        if not is_v3:
            # Documented Legacy Mode Pipeline Contract
            legacy_executed_cleanly = False
            leg_err = None
            try:
                legacy_res = legacy_runner(**legacy_kwargs)
                legacy_executed_cleanly = True
            except Exception as leg_exc:
                legacy_res = ([], [])
                leg_err = f"LEGACY_RUNNER_EXCEPTION:{_safe_type_name(leg_exc)}:{_safe_str(leg_exc, 200)}"

            # Structural & Bilateral Representation Validation for legacy output
            has_valid_parts = False
            legacy_malformed = False
            malformation_code = "LEGACY_STRUCTURAL_MALFORMATION"
            malformation_reason = None

            if legacy_executed_cleanly:
                # 1. Tuple Arity Enforcement: strictly 2-element tuple or list
                if not (
                    isinstance(legacy_res, (tuple, list)) and len(legacy_res) == 2
                ):
                    legacy_malformed = True
                    malformation_reason = (
                        f"Legacy output must be strictly a 2-element tuple/list, "
                        f"got {type(legacy_res).__name__} of length "
                        f"{len(legacy_res) if isinstance(legacy_res, (tuple, list)) else 'N/A'}"
                    )
                elif not (
                    isinstance(legacy_res[0], list)
                    and isinstance(legacy_res[1], list)
                ):
                    legacy_malformed = True
                    malformation_reason = (
                        "Legacy output elements must both be lists "
                        "(paper_parts: list, full_paper: list)"
                    )
                else:
                    parts, full = legacy_res[0], legacy_res[1]
                    # 2. Emptiness & Asymmetry Check
                    if len(parts) == 0 and len(full) == 0:
                        # Clean empty output
                        has_valid_parts = False
                    elif (len(parts) == 0 and len(full) > 0) or (
                        len(parts) > 0 and len(full) == 0
                    ):
                        # Asymmetric emptiness
                        legacy_malformed = True
                        malformation_reason = (
                            f"Asymmetric legacy output: paper_parts length {len(parts)} "
                            f"!= full_paper length {len(full)}"
                        )
                    else:
                        # Both non-empty: validate shape and bilateral equivalence
                        parts_qs = []
                        seen_modules = set()
                        parts_valid = True

                        for p_idx, p in enumerate(parts):
                            if not isinstance(p, dict):
                                parts_valid = False
                                malformation_reason = (
                                    f"paper_parts[{p_idx}] not a dict"
                                )
                                break
                            mod_id = (
                                p.get("module_id")
                                or p.get("id")
                                or p.get("module")
                            )
                            if (
                                not mod_id
                                or not isinstance(mod_id, str)
                                or not mod_id.strip()
                            ):
                                parts_valid = False
                                malformation_reason = (
                                    f"paper_parts[{p_idx}] missing or empty module_id"
                                )
                                break
                            mod_id = mod_id.strip()
                            if mod_id in seen_modules:
                                parts_valid = False
                                malformation_reason = f"Duplicate module_id in paper_parts: {mod_id}"
                                break
                            seen_modules.add(mod_id)

                            p_qs = p.get("questions")
                            if not isinstance(p_qs, list) or len(p_qs) == 0:
                                parts_valid = False
                                malformation_reason = (
                                    f"paper_parts[{p_idx}] missing or empty questions list"
                                )
                                break

                            for q_idx, pq in enumerate(p_qs):
                                if not isinstance(pq, dict):
                                    parts_valid = False
                                    malformation_reason = (
                                        f"paper_parts[{p_idx}].questions[{q_idx}] not a dict"
                                    )
                                    break
                                # Field precedence: question_text takes precedence; fallback to text if empty/whitespace
                                valid_text = None
                                qt = pq.get("question_text")
                                if isinstance(qt, str) and qt.strip():
                                    valid_text = qt.strip()
                                else:
                                    t = pq.get("text")
                                    if isinstance(t, str) and t.strip():
                                        valid_text = t.strip()

                                if not valid_text:
                                    parts_valid = False
                                    malformation_reason = (
                                        f"paper_parts[{p_idx}].questions[{q_idx}] empty or whitespace text"
                                    )
                                    break

                                # Module attribution: must be valid non-empty string and match part mod_id
                                q_mod = pq.get("module_id")
                                if not isinstance(q_mod, str) or not q_mod.strip():
                                    parts_valid = False
                                    malformation_reason = (
                                        f"paper_parts[{p_idx}].questions[{q_idx}] missing or non-string module_id"
                                    )
                                    break
                                if q_mod.strip() != mod_id:
                                    parts_valid = False
                                    malformation_reason = (
                                        f"paper_parts[{p_idx}].questions[{q_idx}] module_id '{q_mod}' "
                                        f"does not match part module_id '{mod_id}'"
                                    )
                                    break

                                # Slot ID: must be valid non-empty string
                                pq_sid = pq.get("slot_id")
                                if not isinstance(pq_sid, str) or not pq_sid.strip():
                                    parts_valid = False
                                    malformation_reason = (
                                        f"paper_parts[{p_idx}].questions[{q_idx}] missing or non-string slot_id"
                                    )
                                    break

                                parts_qs.append((pq, valid_text, mod_id, pq_sid.strip()))

                            if not parts_valid:
                                break

                        if not parts_valid:
                            legacy_malformed = True
                        else:
                            # Validate full_paper questions and compare with parts_qs
                            if len(parts_qs) != len(full):
                                legacy_malformed = True
                                malformation_reason = (
                                    f"Question count mismatch: flattened paper_parts has {len(parts_qs)} questions, "
                                    f"full_paper has {len(full)} questions"
                                )
                            else:
                                seen_slot_ids = set()
                                for q_idx, q in enumerate(full):
                                    if not isinstance(q, dict):
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"full_paper[{q_idx}] not a dict"
                                        )
                                        break
                                    valid_text = None
                                    qt = q.get("question_text")
                                    if isinstance(qt, str) and qt.strip():
                                        valid_text = qt.strip()
                                    else:
                                        t = q.get("text")
                                        if isinstance(t, str) and t.strip():
                                            valid_text = t.strip()

                                    if not valid_text:
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"full_paper[{q_idx}] empty or whitespace text"
                                        )
                                        break

                                    pq, pq_text, part_mod_id, expected_pq_sid = parts_qs[q_idx]
                                    if valid_text != pq_text:
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"Question text mismatch at index {q_idx} between "
                                            f"paper_parts and full_paper"
                                        )
                                        break

                                    # Slot ID validation and bilateral equivalence
                                    q_sid = q.get("slot_id")
                                    if not isinstance(q_sid, str) or not q_sid.strip():
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"full_paper[{q_idx}] missing or non-string slot_id"
                                        )
                                        break
                                    clean_q_sid = q_sid.strip()
                                    if clean_q_sid != expected_pq_sid:
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"Slot ID mismatch at index {q_idx}: "
                                            f"paper_parts '{expected_pq_sid}' vs full_paper '{clean_q_sid}'"
                                        )
                                        break
                                    if clean_q_sid in seen_slot_ids:
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"Duplicate slot_id '{clean_q_sid}' at index {q_idx} in legacy output"
                                        )
                                        break
                                    seen_slot_ids.add(clean_q_sid)

                                    # Module ID validation and bilateral attribution
                                    q_mod = q.get("module_id")
                                    if not isinstance(q_mod, str) or not q_mod.strip():
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"full_paper[{q_idx}] missing or non-string module_id"
                                        )
                                        break
                                    if q_mod.strip() != part_mod_id:
                                        legacy_malformed = True
                                        malformation_reason = (
                                            f"full_paper[{q_idx}] module_id '{q_mod}' "
                                            f"does not match paper_parts module_id '{part_mod_id}'"
                                        )
                                        break

                                if not legacy_malformed:
                                    has_valid_parts = True

            if not legacy_executed_cleanly:
                legacy_status = "LEGACY_FAILED"
                legacy_exec_success = False
                legacy_export_eligible = False
                fail_code = "LEGACY_EXECUTION_FAILURE"
                fail_detail = leg_err
            elif legacy_malformed:
                legacy_status = "LEGACY_MALFORMED"
                legacy_exec_success = True  # Technical runner executed without raising!
                legacy_export_eligible = False  # Structural validation failed!
                fail_code = "LEGACY_STRUCTURAL_MALFORMATION"
                fail_detail = (
                    f"Legacy runner output failed dictionary shape validation: {malformation_reason}"
                    if malformation_reason
                    else "Legacy runner output failed dictionary shape validation"
                )
            elif not has_valid_parts:
                legacy_status = "LEGACY_EMPTY"
                legacy_exec_success = True  # Technical runner executed without raising!
                legacy_export_eligible = False  # Empty output cannot be exported!
                fail_code = "LEGACY_EMPTY_OUTPUT"
                fail_detail = "Legacy runner returned empty output"
            else:
                legacy_status = "LEGACY"
                legacy_exec_success = True
                legacy_export_eligible = True
                fail_code = None
                fail_detail = None

            blocked_reasons = []
            if fail_detail:
                blocked_reasons.append(fail_detail)

            legacy_meta = _build_pipeline_meta(
                validated_exec_success=legacy_exec_success,
                raw_success=legacy_exec_success,
                paper_status=legacy_status,
                export_eligible=legacy_export_eligible,
                failure_code=fail_code,
                failure_detail=fail_detail,
                telemetry={},
                factory_notes=[],
                qa_report=None,
                pre_adaptation_qa_report=None,
                adaptation_defects=[],
                reconciliation_meta={
                    "legacy_mode": True,
                    "has_valid_parts": has_valid_parts,
                    "legacy_malformed": legacy_malformed,
                    "legacy_executed_cleanly": legacy_executed_cleanly,
                },
                blocked_reasons=blocked_reasons,
                error=leg_err,
            )
            if out_meta is not None and isinstance(out_meta, dict):
                try:
                    out_meta.update(legacy_meta)
                except Exception:
                    pass
            parts_ret = (
                legacy_res[0]
                if isinstance(legacy_res, (tuple, list)) and len(legacy_res) > 0
                else []
            )
            full_ret = (
                legacy_res[1]
                if isinstance(legacy_res, (tuple, list)) and len(legacy_res) > 1
                else []
            )
            if return_meta:
                return parts_ret, full_ret, legacy_meta
            return _safe_wrap_bridge_result(parts_ret, full_ret, legacy_meta)

        paper_parts, full_paper, meta = run_v3_pipeline(**v3_kwargs)
        v3_meta = meta

        if out_meta is not None and isinstance(out_meta, dict):
            try:
                out_meta.update(meta)
            except Exception:
                pass

        if not meta.get("success"):
            print(
                f"[V3-BRIDGE] v3 failed: {meta.get('failure_code')} — "
                f"{meta.get('failure_detail')}",
                flush=True,
            )
        else:
            status = meta.get("paper_status")
            print(
                f"[V3-BRIDGE] v3 completed: status={status} "
                f"questions={len(full_paper)}",
                flush=True,
            )

        # Attach meta to paper_parts[0] ONLY if paper_parts is non-empty
        if paper_parts:
            paper_parts[0]["_orchestrator_meta"] = meta

        if return_meta:
            return paper_parts, full_paper, meta

        return _safe_wrap_bridge_result(paper_parts, full_paper, meta)

    except Exception as dispatch_exc:
        # Ultimate dispatch containment
        dispatch_exec_succeeded = False
        pre_qa_snap = None
        if isinstance(v3_meta, collections.abc.Mapping):
            raw_v3_success = v3_meta.get("success")
            if type(raw_v3_success) is bool and raw_v3_success is True:
                dispatch_exec_succeeded = True
            pre_qa_snap = v3_meta.get("pre_adaptation_qa_report")

        _, _, emergency_meta = _handle_emergency_pipeline_failure(
            exc=dispatch_exc,
            stage="DISPATCH_EXCEPTION",
            result=None,
            pre_qa_snapshot=pre_qa_snap if isinstance(pre_qa_snap, dict) else None,
            raw_success_override=True if dispatch_exec_succeeded else None,
        )

        # Enforce dispatch status preservation (defense in depth)
        if dispatch_exec_succeeded:
            emergency_meta["success"] = True
            emergency_meta["raw_execution_success"] = True
            emergency_meta["paper_status"] = "BLOCKED"
            emergency_meta["export_eligible"] = False

        if out_meta is not None and isinstance(out_meta, dict):
            try:
                out_meta.update(emergency_meta)
            except Exception:
                pass

        if return_meta:
            return [], [], emergency_meta

        return _safe_wrap_bridge_result([], [], emergency_meta)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _build_kg_for_artifact(artifact: Any, paper_spec: Any) -> Any:
    """
    Build a KnowledgeGraph by partitioning the artifact into modules.

    Uses per-module page ranges derived from the artifact's page count
    and the paper spec's module count.
    """
    try:
        from core.knowledge import KnowledgeGraphBuilder
    except ImportError:
        return None

    max_page = max(
        (getattr(b, "page", 0) for b in getattr(artifact, "text_blocks", []) or []),
        default=0,
    )
    if max_page <= 0 or paper_spec is None:
        return None

    module_count = getattr(paper_spec, "module_count", 0)
    if module_count <= 0:
        return None

    pages_per_module = max(1, max_page // module_count)
    ranges = {}
    for m in range(1, module_count + 1):
        start = (m - 1) * pages_per_module + 1
        end = max_page if m == module_count else m * pages_per_module
        ranges[f"module_{m}"] = (start, end)

    try:
        builder = KnowledgeGraphBuilder()
        return builder.build_all(artifact=artifact, module_page_ranges=ranges)
    except Exception as e:
        print(f"[V3-BRIDGE] KG build failed: {e}", flush=True)
        return None
