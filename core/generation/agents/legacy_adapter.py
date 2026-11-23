"""
Convert v3 OrchestratorResult into the legacy run_pipeline return shape.

The legacy pipeline returns `(paper_parts, full_paper)`:
    paper_parts : List[dict]  — one entry per module, with questions
    full_paper  : List[dict]  — flat list of questions across all modules

Each question is a dict with (at minimum) these keys:
    slot_id, question_text, solution, marking_scheme,
    marks, bloom, co, module_id,
    image_path, figure_caption, table_data, diagram_request

This adapter is the ONLY place that knows about the legacy shape.
If the shape changes, this file changes. Agents do not.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .generated_question import GeneratedQuestion
from .orchestrator_contracts import OrchestratorResult
from .overlap_detector import _specific_overlap, detect_subquestion_overlap

logger = logging.getLogger(__name__)

# Standard Roman numeral pattern covering 1 to 39 (i to xxxix)
_ROMAN_NUMERAL_RE = re.compile(r"^(x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)

# Outer marker regex using a consuming boundary group (no variable-width lookbehinds)
# Matches:
#   (a), (b), (aa)  -> group 'p1'
#   a), b), aa)     -> group 'p2'
#   a., b., aa.     -> group 'p3' (constrained to single letter [a-zA-Z] to avoid matching general words ending with a period)
_OUTER_MARKER_RE = re.compile(
    r"(?P<prefix>^|[\s\n]+)"
    r"(?P<marker>"
    r"\((?P<p1>[a-zA-Z0-9]+)\)"
    r"|(?P<p2>[a-zA-Z0-9]+)\)"
    r"|(?P<p3>[a-zA-Z])\."
    r")\s+",
    re.MULTILINE
)


@dataclass(frozen=True)
class SubpartExtractionResult:
    stem: str
    clauses: Tuple[str, ...]
    success: bool
    raw_text: str


def get_subpart_label(idx: int) -> str:
    """
    Bijective base-26 alphabetic label generator (general utility).
    0 -> 'a', 25 -> 'z', 26 -> 'aa', 27 -> 'ab', 51 -> 'az', 52 -> 'ba', 701 -> 'zz', 702 -> 'aaa'...
    Guaranteed valid and unique for any non-negative integer.
    """
    if idx < 0:
        raise ValueError(f"Index must be non-negative, got {idx}")
    chars = []
    n = idx + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord('a') + rem))
    return "".join(reversed(chars))


def _is_nested_marker(label: str) -> bool:
    """
    Classify whether a label is a nested child marker (Roman numeral or Arabic digit)
    rather than an outer subquestion alphabetic marker.
    Examples of nested markers: 'i', 'ii', 'iii', 'iv', 'v', 'vi', '1', '2', '3'.
    Examples of outer markers for N<=5: 'a', 'b', 'c', 'd', 'e'.
    """
    if not label:
        return False
    lower = label.lower()
    if lower.isdigit():
        return True
    return bool(_ROMAN_NUMERAL_RE.match(lower)) and lower != ""


def _extract_subpart_structure(text: str, n_parts: int) -> SubpartExtractionResult:
    """
    Extract shared context stem and distinct subpart clauses using strict outer markers.

    Outer subquestions in VTU format are strictly sequential alphabetic labels
    matching n_parts: get_subpart_label(0), ..., get_subpart_label(n_parts - 1).

    VTU Bound Policy:
    - Standard university question papers strictly permit at most 5 subparts (a-e).
    - If n_parts > 5, extraction fails closed immediately, eliminating any overlap
      between outer labels and Roman numerals (e.g. outer 'i' at n_parts=9).

    Strict 3-way classification state machine:
    1. Next expected outer marker: advances curr_expected_idx.
    2. Valid nested child marker: accepted ONLY if an outer clause is open (curr_expected_idx >= 1).
    3. Any other marker (unexpected, out-of-order, duplicate, or excess): fails closed immediately.

    Fail-closed policy:
    - If n_parts <= 1, returns the full text as a single clause.
    - If n_parts > 5, returns success=False (exceeds VTU partition maximum).
    - If explicit outer markers matching the exact sequence are absent, returns success=False.
      (Sentence splitting alone is explicitly rejected as evidence of subquestion boundaries).
    - If a nested marker appears before the first outer marker (curr_expected_idx == 0), returns success=False.
    - Nested markers (e.g. (i), (ii)) inside an open outer clause (first, middle, or final) are preserved.
    - If markers are out of order, missing, ambiguous, or in excess of n_parts, returns success=False.
    - NEVER silently slices or concatenates excess clauses.
    - NEVER duplicates question text across partition slots.
    """
    if not text:
        return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text="")

    if n_parts <= 1:
        return SubpartExtractionResult(stem="", clauses=(text.strip(),), success=True, raw_text=text)

    # Enforce VTU partition bound: maximum 5 subparts (a through e)
    if n_parts > 5:
        return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

    expected_labels = [get_subpart_label(i) for i in range(n_parts)]
    matches = list(_OUTER_MARKER_RE.finditer(text))

    if not matches:
        # FAIL-CLOSED: Sentence splitting alone is NOT sufficient structural evidence.
        return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

    matched_markers: List[Tuple[int, re.Match]] = []
    curr_expected_idx = 0

    for match in matches:
        raw_label = match.group("p1") or match.group("p2") or match.group("p3")
        label_lower = raw_label.lower()

        # Branch 1: Next expected outer marker
        if curr_expected_idx < len(expected_labels) and label_lower == expected_labels[curr_expected_idx]:
            matched_markers.append((curr_expected_idx, match))
            curr_expected_idx += 1

        # Branch 2: Nested child marker (must belong to an already open outer clause)
        elif _is_nested_marker(label_lower):
            if curr_expected_idx >= 1:
                # Valid nested item inside open outer clause -> remains part of that clause
                continue
            else:
                # Nested marker appeared before any outer marker was opened -> fail closed!
                return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

        # Branch 3: Invalid, unexpected, out-of-order, duplicate, or excess marker -> fail closed!
        else:
            return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

    # Verify that ALL expected outer markers were found
    if len(matched_markers) != n_parts:
        return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

    # Check trailing text after the last matched outer marker for unclassified excess markers
    last_match = matched_markers[-1][1]
    trailing_text = text[last_match.end():]
    for trailing_m in _OUTER_MARKER_RE.finditer(trailing_text):
        t_label = (trailing_m.group("p1") or trailing_m.group("p2") or trailing_m.group("p3")).lower()
        if not _is_nested_marker(t_label):
            # Excess non-nested marker found after the last expected outer subpart -> fail closed
            return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)

    # Extract stem and clauses using match.start('marker') to accurately exclude the boundary prefix
    first_marker_match = matched_markers[0][1]
    stem = text[:first_marker_match.start("marker")].strip()
    clauses = []

    for i in range(len(matched_markers)):
        start_pos = matched_markers[i][1].end()
        end_pos = matched_markers[i + 1][1].start("marker") if i + 1 < len(matched_markers) else len(text)
        clause = text[start_pos:end_pos].strip()
        if not clause:
            # Empty clause indicates malformed subquestion
            return SubpartExtractionResult(stem="", clauses=(), success=False, raw_text=text)
        clauses.append(clause)

    return SubpartExtractionResult(stem=stem, clauses=tuple(clauses), success=True, raw_text=text)


def _extract_subpart_texts(text: str, n_parts: int) -> List[str]:
    """Backward compatibility wrapper around _extract_subpart_structure."""
    res = _extract_subpart_structure(text, n_parts)
    if res.success and res.clauses:
        stem = res.stem
        clauses = list(res.clauses)
        if stem:
            clauses[0] = f"{stem} {clauses[0]}".strip()
        return clauses
    return [text]


def _strip_shared_prefix(sub_texts: list, min_words: int = 6) -> list:
    """Backward compatibility wrapper."""
    return list(sub_texts)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def to_legacy_shape(
    result: OrchestratorResult,
    context: Any,
) -> Tuple[List[dict], List[dict]]:
    """
    Return `(paper_parts, full_paper)` matching the legacy pipeline.

    paper_parts : one entry per module, with module metadata + questions
    full_paper  : flat list of all question dicts (used by DOCX export)
    """
    if not result.success:
        # Legacy shape on failure: empty outputs.
        return [], []

    questions = list(result.questions or [])
    paper_spec = getattr(context, "paper_spec", None)
    artifact = getattr(context, "artifact", None)

    # Build flat list of question dicts
    full_paper: List[dict] = [_to_question_dict(q) for q in questions]

    # Group by module
    by_module: Dict[str, List[dict]] = defaultdict(list)
    for q_dict in full_paper:
        by_module[q_dict["module_id"]].append(q_dict)

    # Build paper_parts
    paper_parts: List[dict] = []
    if paper_spec is not None and hasattr(paper_spec, "module_count"):
        for module_idx in range(1, paper_spec.module_count + 1):
            module_id = f"module_{module_idx}"
            module_questions = by_module.get(module_id, [])
            paper_parts.append({
                "module_id": module_id,
                "module_index": module_idx,
                "module_title": _module_title(artifact, module_idx),
                "questions": module_questions,
                "question_count": len(module_questions),
            })
    else:
        # No spec: emit one part per module_id we saw.
        for module_id, module_questions in sorted(by_module.items()):
            paper_parts.append({
                "module_id": module_id,
                "module_index": _module_idx_from_id(module_id),
                "module_title": "",
                "questions": module_questions,
                "question_count": len(module_questions),
            })

    # Attach QA report to both for downstream consumers
    qa = _qa_dict(result.qa_report)
    for part in paper_parts:
        part["qa_report"] = qa

    return paper_parts, full_paper


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _clause_references_figure(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return bool(re.search(
        r"\b(?:refer(?:ence)?\s+to\s+(?:the\s+)?(?:given\s+)?(?:figure|fig\.?|diagram)|shown\s+in\s+(?:the\s+)?(?:figure|fig\.?|diagram)|with\s+reference\s+to\s+(?:the\s+)?(?:given\s+)?(?:figure|fig\.?|diagram))\b",
        lower,
    ))


def _clean_topic(concept: str) -> str:
    """Sanitize a topic string: strip bullets, relative clauses, leading participles, etc."""
    if not concept:
        return "the core module concept"
    t = str(concept).strip()
    t = re.sub(r"^[▪•→◆■\*\-\d\.\)\s]+", "", t).strip()
    t = re.sub(r"[▪•→◆■]+", " ", t).strip()
    t = re.split(r",\s*(?:where|which|that|who|when)\b", t, flags=re.IGNORECASE)[0].strip()
    t = t.rstrip(",;:.- ")
    t = re.sub(r"^(?:executed\s+by|where|which|that|by|when|and|or|for|during|using)\s+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+in\s+detail$", "", t, flags=re.IGNORECASE).strip()
    if t.lower().startswith("the "):
        t = t[4:].strip()
    if not t or len(t) < 3:
        return "the core principles"
    return t


def _to_question_dict(q: GeneratedQuestion) -> dict:
    """Convert one GeneratedQuestion to the legacy question dict with synthesized subquestions."""
    # 1. Partition validation
    raw_partition = q.partition if q.partition is not None else ([q.marks] if q.marks else [10])
    if not isinstance(raw_partition, (list, tuple)):
        partition = [10]
        partition_invalid = True
    else:
        partition = []
        partition_invalid = False
        for m in raw_partition:
            if type(m) is not int or isinstance(m, bool) or m <= 0:
                partition_invalid = True
                break
            partition.append(int(m))
        if not partition:
            partition = [10]
            partition_invalid = True

    # Telemetry initialization with deep isolation for mutable lists to prevent shared reference mutation
    telemetry = dict(q.telemetry or {})
    if "warnings" in telemetry and isinstance(telemetry["warnings"], list):
        telemetry["warnings"] = list(telemetry["warnings"])

    is_high_partition = len(partition) > 5
    if is_high_partition:
        logger.warning("Slot %s partition length %d exceeds standard VTU subpart maximum (5)", q.slot_id, len(partition))
        warnings_list = list(telemetry.get("warnings") or [])
        warnings_list.append(f"partition_exceeds_vtu_max:{len(partition)}")
        telemetry["warnings"] = warnings_list

    stem = ""
    sub_questions = []
    overlap_detected = False
    overlap_details = []

    # Preserve pre-existing question-level failure status
    if q.is_unresolved:
        q_is_unresolved = True
        q_unresolved_scope = q.unresolved_scope
        q_unresolved_subparts = list(q.unresolved_subparts)
        q_failure_code = q.failure_code
        q_failure_detail = q.failure_detail
    else:
        q_is_unresolved = False
        q_unresolved_scope = None
        q_unresolved_subparts = []
        q_failure_code = None
        q_failure_detail = None

    q_adaptation_failure_code = None
    q_adaptation_failure_detail = None

    # Fail closed immediately if partition is invalid
    if partition_invalid:
        adaptation_code = "PARTITION_INVALID"
        adaptation_detail = f"Slot has invalid partition specification: {raw_partition!r}."
        sub_questions.append({
            "letter": "a",
            "label": "a",
            "sub_label": "a",
            "text": q.question_text,
            "marks": sum(partition) if partition else 10,
            "co": q.co,
            "bloom": q.bloom,
            "is_unresolved": True,
            "failure_code": q.failure_code or adaptation_code,
            "failure_detail": q.failure_detail or adaptation_detail,
            "adaptation_failure_code": adaptation_code,
            "adaptation_failure_detail": adaptation_detail,
        })
        q_is_unresolved = True
        q_unresolved_scope = "question"
        q_unresolved_subparts = []
        q_failure_code = q.failure_code or adaptation_code
        q_failure_detail = q.failure_detail or adaptation_detail
        q_adaptation_failure_code = adaptation_code
        q_adaptation_failure_detail = adaptation_detail

    # Branch 1 & 2: Structured SubParts Supplied (Stage G4-A)
    elif (
        q.sub_parts is not None
        and (
            bool(q.sub_parts)
            or type(q.sub_parts) is not list
            or getattr(q, "_has_structured_subparts", False)
            or (isinstance(q, dict) and "sub_parts" in q and q["sub_parts"] is not None)
        )
    ):
        structured_fail_code = None
        structured_fail_detail = None

        if type(q.sub_parts) is not list:
            structured_fail_code = "STRUCTURED_SUBPARTS_INVALID_TYPE"
            structured_fail_detail = f"sub_parts must be an exact list, got {type(q.sub_parts).__name__}."
        elif not q.sub_parts:
            # Correction 1: Empty structured subparts must ALWAYS fail closed (including partition=[10])
            structured_fail_code = "STRUCTURED_SUBPARTS_EMPTY"
            structured_fail_detail = f"Slot has partition {partition} but sub_parts is an empty list."
        elif is_high_partition:
            structured_fail_code = "PARTITION_EXCEEDS_VTU_MAX"
            structured_fail_detail = f"Slot partition length {len(partition)} exceeds standard VTU subpart maximum (5)."
        elif len(q.sub_parts) != len(partition):
            structured_fail_code = "STRUCTURED_SUBPARTS_COUNT_MISMATCH"
            structured_fail_detail = f"Partition expected {len(partition)} subparts, got {len(q.sub_parts)}."
        else:
            # Ordered marks check (Correction G4-A: Complete ordered mark partition validation)
            actual_marks = [getattr(sp, "marks", None) for sp in q.sub_parts]
            if actual_marks != partition:
                structured_fail_code = "STRUCTURED_SUBPARTS_MARKS_MISMATCH"
                structured_fail_detail = f"Ordered subpart marks {actual_marks} do not match partition {partition}."
            else:
                # Validation and duplicate label checks
                labels = []
                for sp_idx, sp in enumerate(q.sub_parts):
                    if not hasattr(sp, "validate") or not hasattr(sp, "label"):
                        structured_fail_code = "STRUCTURED_SUBPART_INVALID_OBJECT"
                        structured_fail_detail = f"sub_parts[{sp_idx}] is not a valid SubPart object."
                        break
                    valid, err = sp.validate()
                    if not valid:
                        structured_fail_code = "STRUCTURED_SUBPART_VALIDATION_FAILED"
                        structured_fail_detail = f"SubPart ({getattr(sp, 'label', '?')}) validation failed: {err}"
                        break
                    labels.append(sp.label.lower())

                if not structured_fail_code:
                    # Correction 2: Duplicate-label validation using len(set(labels)) != len(labels)
                    if len(set(labels)) != len(labels):
                        structured_fail_code = "STRUCTURED_SUBPARTS_DUPLICATE_LABELS"
                        structured_fail_detail = f"Duplicate subpart labels detected: {labels}"
                    else:
                        expected_labels = [get_subpart_label(i) for i in range(len(partition))]
                        if labels != expected_labels:
                            structured_fail_code = "STRUCTURED_SUBPARTS_LABEL_MISMATCH"
                            structured_fail_detail = f"Expected subpart labels {expected_labels}, got {labels}."
                        elif q.is_unresolved and q.unresolved_scope == "subquestion":
                            unmapped = [sp_lbl for sp_lbl in q.unresolved_subparts if sp_lbl not in set(labels)]
                            if unmapped:
                                structured_fail_code = "UNRESOLVED_SUBPART_LABEL_MISMATCH"
                                structured_fail_detail = (
                                    f"Stage 2A marked subparts {q.unresolved_subparts} unresolved, "
                                    f"but structured labels are {labels}."
                                )

        if structured_fail_code:
            # FAIL CLOSED: Never silently fall back to text parsing!
            sub_questions.append({
                "letter": "a",
                "label": "a",
                "sub_label": "a",
                "text": q.question_text,
                "marks": sum(partition),
                "co": q.co,
                "bloom": q.bloom,
                "is_unresolved": True,
                "failure_code": q.failure_code or structured_fail_code,
                "failure_detail": q.failure_detail or structured_fail_detail,
                "adaptation_failure_code": structured_fail_code,
                "adaptation_failure_detail": structured_fail_detail,
            })
            q_is_unresolved = True
            q_unresolved_scope = "question"
            q_unresolved_subparts = []
            q_failure_code = q.failure_code or structured_fail_code
            q_failure_detail = q.failure_detail or structured_fail_detail
            q_adaptation_failure_code = structured_fail_code
            q_adaptation_failure_detail = structured_fail_detail
        else:
            # Direct structured projection from authoritative SubPart objects
            for i, sp in enumerate(q.sub_parts):
                letter = sp.label.lower()
                if not q_is_unresolved:
                    sq_unresolved = False
                    sq_code = None
                    sq_detail = None
                elif q_unresolved_scope == "subquestion":
                    sq_unresolved = (letter in q_unresolved_subparts)
                    sq_code = q_failure_code if sq_unresolved else None
                    sq_detail = q_failure_detail if sq_unresolved else None
                else:
                    sq_unresolved = True
                    sq_code = q_failure_code
                    sq_detail = q_failure_detail

                sq_img = sp.image_path
                sq_cap = sp.figure_caption
                sq_fid = sp.figure_id
                sq_sha = sp.image_sha256
                sq_ref = bool(sq_img or sq_fid)

                # If single-part and sp has no image, fall back to parent image if present
                if len(q.sub_parts) == 1 and not sq_img:
                    sq_img = getattr(q, "image_path", None)
                    sq_cap = getattr(q, "figure_caption", None)
                    sq_fid = getattr(q, "figure_id", None)
                    sq_sha = getattr(q, "image_sha256", None)
                    sq_ref = bool(sq_img or sq_fid)

                sub_questions.append({
                    "letter": letter,
                    "label": letter,
                    "sub_label": letter,
                    "text": sp.text,
                    "marks": sp.marks,
                    "co": sp.co or q.co,
                    "bloom": sp.bloom or q.bloom,
                    "is_unresolved": sq_unresolved,
                    "failure_code": sq_code,
                    "failure_detail": sq_detail,
                    "adaptation_failure_code": None,
                    "adaptation_failure_detail": None,
                    "image_path": sq_img,
                    "figure_caption": sq_cap,
                    "figure_id": sq_fid,
                    "image_sha256": sq_sha,
                    "references_image": sq_ref,
                    "topic": sp.topic,
                    "solution": sp.solution,
                    "marking_scheme": list(sp.marking_scheme or []),
                })

            if len(q.sub_parts) > 1:
                overlap_detected, overlap_details = detect_subquestion_overlap([sp.text for sp in q.sub_parts])

    # Branch 3: Genuine Legacy Input Fallback (q.sub_parts is None)
    else:
        extraction = _extract_subpart_structure(q.question_text, len(partition))

        # Determine if subpart label mapping is possible
        extracted_labels = {get_subpart_label(i) for i in range(len(partition))}
        label_mapping_failed = False
        if extraction.success and q.is_unresolved and q.unresolved_scope == "subquestion":
            unmapped = [sp for sp in q.unresolved_subparts if sp not in extracted_labels]
            if unmapped:
                label_mapping_failed = True

        if not extraction.success or label_mapping_failed:
            # Safe fallback: single aggregate entry representing full text (NO text duplication)
            if is_high_partition:
                adaptation_code = "PARTITION_EXCEEDS_VTU_MAX"
                adaptation_detail = f"Slot partition length {len(partition)} exceeds standard VTU subpart maximum (5)."
                if q.is_unresolved:
                    sq_unresolved = True
                    sq_code = q.failure_code
                    sq_detail = q.failure_detail
                    adaptation_detail += f" Preserving pre-existing Stage 2A failure ({q.failure_code})."
                    q_scope = q.unresolved_scope
                    q_subparts = list(q.unresolved_subparts)
                else:
                    sq_unresolved = True
                    sq_code = "PARTITION_EXCEEDS_VTU_MAX"
                    sq_detail = adaptation_detail
                    q_scope = "question"
                    q_subparts = []
            elif label_mapping_failed:
                sq_unresolved = True
                sq_code = q.failure_code
                sq_detail = q.failure_detail
                adaptation_code = "UNRESOLVED_SUBPART_LABEL_MISMATCH"
                adaptation_detail = (
                    f"Stage 2A marked subparts {q.unresolved_subparts} unresolved, "
                    f"but extracted labels are {sorted(extracted_labels)}. "
                    f"Preserving pre-existing failure ({q.failure_code}) on aggregate entry."
                )
                q_scope = q.unresolved_scope
                q_subparts = list(q.unresolved_subparts)
            elif q.is_unresolved:
                sq_unresolved = True
                sq_code = q.failure_code
                sq_detail = q.failure_detail
                adaptation_code = "MULTI_PART_STRUCTURE_MISMATCH"
                adaptation_detail = (
                    f"Question has partition {partition} but lacks parseable subpart structure. "
                    f"Preserving pre-existing Stage 2A failure ({q.failure_code})."
                )
                q_scope = q.unresolved_scope
                q_subparts = list(q.unresolved_subparts)
            else:
                sq_unresolved = True
                sq_code = "MULTI_PART_STRUCTURE_MISMATCH"
                sq_detail = f"Slot has partition {partition} but question text lacks valid structural subpart markers."
                adaptation_code = "MULTI_PART_STRUCTURE_MISMATCH"
                adaptation_detail = sq_detail
                q_scope = "question"
                q_subparts = []

            sub_questions.append({
                "letter": "a",
                "label": "a",
                "sub_label": "a",
                "text": q.question_text,
                "marks": sum(partition),
                "co": q.co,
                "bloom": q.bloom,
                "is_unresolved": sq_unresolved,
                "failure_code": sq_code,
                "failure_detail": sq_detail,
                "adaptation_failure_code": adaptation_code,
                "adaptation_failure_detail": adaptation_detail,
            })

            q_is_unresolved = True
            q_unresolved_scope = q_scope
            q_unresolved_subparts = q_subparts
            q_failure_code = sq_code
            q_failure_detail = sq_detail
            q_adaptation_failure_code = adaptation_code
            q_adaptation_failure_detail = adaptation_detail
            overlap_detected, overlap_details = (False, [])

        else:
            # Clean extraction with verified label mapping (N <= 5)
            stem = extraction.stem
            clauses = extraction.clauses

            # Determine visual asset ownership across subparts (Stage G2-D)
            subpart_visual_map: Dict[str, Dict[str, Any]] = {}
            has_parent_image = bool(getattr(q, "references_image", False) or getattr(q, "image_path", None))
            sub_parts_objs = getattr(q, "sub_parts", None) or []

            # 1. Direct subpart object bindings (figure_id or image_path on SubPart)
            for sp_obj in sub_parts_objs:
                sp_lbl = getattr(sp_obj, "label", "").lower()
                sp_img = getattr(sp_obj, "image_path", None)
                sp_fid = getattr(sp_obj, "figure_id", None)
                sp_cap = getattr(sp_obj, "figure_caption", None)
                sp_sha = getattr(sp_obj, "image_sha256", None)
                if sp_img or sp_fid:
                    subpart_visual_map[sp_lbl] = {
                        "image_path": sp_img or getattr(q, "image_path", None),
                        "figure_caption": sp_cap or getattr(q, "figure_caption", None),
                        "figure_id": sp_fid or getattr(q, "figure_id", None),
                        "image_sha256": sp_sha or getattr(q, "image_sha256", None),
                    }

            # 2. Text reference / single-part fallback if not already mapped
            if has_parent_image and not subpart_visual_map:
                if len(partition) <= 1:
                    # Single part: unambiguous owner is the single subquestion
                    subpart_visual_map[get_subpart_label(0)] = {
                        "image_path": getattr(q, "image_path", None),
                        "figure_caption": getattr(q, "figure_caption", None),
                        "figure_id": getattr(q, "figure_id", None),
                        "image_sha256": getattr(q, "image_sha256", None),
                    }
                else:
                    # Multi-part: check which clauses explicitly reference a figure
                    referencing_indices = [
                        idx for idx, clause in enumerate(clauses)
                        if _clause_references_figure(clause)
                    ]
                    if len(referencing_indices) == 1:
                        owner_lbl = get_subpart_label(referencing_indices[0])
                        subpart_visual_map[owner_lbl] = {
                            "image_path": getattr(q, "image_path", None),
                            "figure_caption": getattr(q, "figure_caption", None),
                            "figure_id": getattr(q, "figure_id", None),
                            "image_sha256": getattr(q, "image_sha256", None),
                        }
                    # Else: ambiguous ownership (0 or >1 referencing subparts) -> DO NOT guess; leave unmapped

            for i in range(len(partition)):
                letter = get_subpart_label(i)

                # Presentation contract: stem attached to part (a) once; clauses clean for subsequent parts
                if i == 0 and stem:
                    sq_text = f"{stem} {clauses[0]}".strip()
                else:
                    sq_text = clauses[i]

                # Granular resolution state determination
                if not q_is_unresolved:
                    sq_unresolved = False
                    sq_code = None
                    sq_detail = None
                elif q_unresolved_scope == "subquestion":
                    sq_unresolved = (letter in q_unresolved_subparts)
                    sq_code = q_failure_code if sq_unresolved else None
                    sq_detail = q_failure_detail if sq_unresolved else None
                else:  # question-level failure
                    sq_unresolved = True
                    sq_code = q_failure_code
                    sq_detail = q_failure_detail

                vis_info = subpart_visual_map.get(letter)
                sq_img = vis_info["image_path"] if vis_info else None
                sq_cap = vis_info["figure_caption"] if vis_info else None
                sq_fid = vis_info["figure_id"] if vis_info else None
                sq_sha = vis_info["image_sha256"] if vis_info else None
                sq_ref = bool(sq_img)

                sub_questions.append({
                    "letter": letter,
                    "label": letter,
                    "sub_label": letter,
                    "text": sq_text,
                    "marks": partition[i],
                    "co": q.co,
                    "bloom": q.bloom,
                    "is_unresolved": sq_unresolved,
                    "failure_code": sq_code,
                    "failure_detail": sq_detail,
                    "adaptation_failure_code": None,
                    "adaptation_failure_detail": None,
                    "image_path": sq_img,
                    "figure_caption": sq_cap,
                    "figure_id": sq_fid,
                    "image_sha256": sq_sha,
                    "references_image": sq_ref,
                })

            # Run overlap detection strictly on clauses if multi-part
            if len(partition) > 1:
                overlap_detected, overlap_details = detect_subquestion_overlap(list(clauses))
            else:
                overlap_detected, overlap_details = (False, [])

    if overlap_detected:
        telemetry["subquestion_overlap"] = True
        telemetry["overlap_details"] = overlap_details
    if q_adaptation_failure_code:
        telemetry["adaptation_failure_code"] = q_adaptation_failure_code

    return {
        "slot_id": q.slot_id,
        "module_id": q.module_id,
        "global_q_idx": q.global_q_idx,
        "text": q.question_text,
        "question_text": q.question_text,
        "stem": stem,
        "sub_questions": sub_questions,
        "subQuestions": sub_questions,
        # Resolution metadata
        "is_unresolved": q_is_unresolved,
        "unresolved_scope": q_unresolved_scope,
        "unresolved_subparts": q_unresolved_subparts,
        "failure_code": q_failure_code,
        "failure_detail": q_failure_detail,
        "adaptation_failure_code": q_adaptation_failure_code,
        "adaptation_failure_detail": q_adaptation_failure_detail,
        # Preserved pedagogical assets
        "solution": q.solution,
        "marking_scheme": list(q.marking_scheme or []),
        "marks": sum(partition),
        "total_marks": sum(partition),
        "partition": partition,
        "bloom": q.bloom,
        "co": q.co,
        "topic": q.topic,
        "image_path": q.image_path,
        "figure_caption": q.figure_caption,
        "references_image": q.references_image,
        "diagram_request": q.diagram_request,
        "generation_source": q.generation_source,
        "attempts": q.attempts,
        "telemetry": telemetry,
        "subquestion_overlap": overlap_detected,
        "overlap_details": overlap_details,
    }


legacy_question_from_generated = _to_question_dict


def _module_title(artifact: Any, module_idx: int) -> str:
    """Extract a module title from the artifact, if available."""
    if artifact is None:
        return f"Module {module_idx}"
    titles = getattr(artifact, "module_titles", None)
    if isinstance(titles, list) and 1 <= module_idx <= len(titles):
        return titles[module_idx - 1]
    return f"Module {module_idx}"


def _module_idx_from_id(module_id: str) -> int:
    try:
        return int(module_id.split("_")[1])
    except (IndexError, ValueError):
        return 0


def _qa_dict(qa_report: Any) -> dict:
    if qa_report is None:
        return {}
    if hasattr(qa_report, "to_dict"):
        return qa_report.to_dict()
    return {}
