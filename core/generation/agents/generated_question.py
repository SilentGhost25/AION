"""
GeneratedQuestion — the output contract of the Writing Agent.

Every downstream agent consumes this structure. The Evaluation Agent
audits it, the Refinement Agent repairs it, the Checking Agent gates it,
the DOCX exporter renders it.

No field is optional except `diagram_request` and `image_path`. The
Writing Agent must produce a question with a solution and marking scheme,
or the slot is a failure.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# Strict Type Validator Helpers
# =============================================================================

def _require_exact_type(val: Any, expected_type: type, field_name: str) -> Any:
    """Enforce exact type equality, preventing bool/subclass confusion."""
    if type(val) is not expected_type:
        raise TypeError(f"{field_name} must be exact {expected_type.__name__}, got {type(val).__name__}: {val!r}")
    return val


def _require_int(val: Any, field_name: str, allow_zero: bool = True) -> int:
    """Strictly validates integers, rejecting booleans and floats."""
    if type(val) is not int or isinstance(val, bool):
        raise TypeError(f"{field_name} must be an integer, got {type(val).__name__}: {val!r}")
    if not allow_zero and val <= 0:
        raise ValueError(f"{field_name} must be strictly positive, got {val!r}")
    if allow_zero and val < 0:
        raise ValueError(f"{field_name} must be non-negative, got {val!r}")
    return val


def _require_str(val: Any, field_name: str, allow_empty: bool = True) -> str:
    """Strictly validates string types and normalizes whitespace."""
    if type(val) is not str:
        raise TypeError(f"{field_name} must be a string, got {type(val).__name__}: {val!r}")
    stripped = val.strip()
    if not allow_empty and not stripped:
        raise ValueError(f"{field_name} must not be empty")
    return stripped


def _require_list(val: Any, field_name: str) -> list:
    """Strictly validates list containers."""
    if type(val) is not list:
        raise TypeError(f"{field_name} must be a list, got {type(val).__name__}: {val!r}")
    return val


def _require_bool(val: Any, field_name: str) -> bool:
    """Strictly validates boolean values."""
    if type(val) is not bool:
        raise TypeError(f"{field_name} must be a boolean, got {type(val).__name__}: {val!r}")
    return val


# =============================================================================
# The SubPart Contract
# =============================================================================

@dataclass
class SubPart:
    """
    Authoritative representation of an individual subquestion part.
    Preserves all planning and generation metadata required for independent
    evaluation, solution provision, and criteria-level marking.
    Mutable state with persistent validation and controlled projection synchronization.
    """
    label: str                                              # Subpart marker: strictly single lowercase ASCII 'a'-'z'
    marks: int                                              # Exact mark allocation: strictly positive integer
    text: str                                               # Self-contained subquestion text
    topic: str = ""                                         # Specific topic allocated from the module KG
    solution: str = ""                                      # Model answer for this subpart
    marking_scheme: List[Dict[str, Any]] = field(default_factory=list)  # Criteria breakdown for this subpart

    def validate(self) -> Tuple[bool, str]:
        """
        Reusable validation of SubPart fields, callable both at construction
        and dynamically after post-construction mutations.
        Strictly non-mutating.
        """
        if type(self.label) is not str:
            return False, f"SubPart.label must be exact str, got {type(self.label).__name__}"
        if type(self.text) is not str:
            return False, f"SubPart.text must be exact str, got {type(self.text).__name__}"
        if type(self.topic) is not str:
            return False, f"SubPart.topic must be exact str, got {type(self.topic).__name__}"
        if type(self.solution) is not str:
            return False, f"SubPart.solution must be exact str, got {type(self.solution).__name__}"
        if type(self.marking_scheme) is not list:
            return False, f"SubPart.marking_scheme must be exact list, got {type(self.marking_scheme).__name__}"

        # Strict label check: single lowercase ASCII character 'a' through 'z'
        # Rejects uppercase ('A'), whitespace-padded (' a '), multi-char ('aa'), digits, non-ascii
        if len(self.label) != 1 or not ('a' <= self.label <= 'z'):
            return False, f"SubPart label must be a single lowercase letter 'a'-'z', got {self.label!r}"

        # Text cannot be empty
        if not self.text.strip():
            return False, "SubPart text must not be empty"

        # Strict integer check for marks (rejecting bool and float)
        if type(self.marks) is not int or isinstance(self.marks, bool):
            return False, f"SubPart.marks must be an exact integer, got {type(self.marks).__name__}: {self.marks!r}"
        if self.marks <= 0:
            return False, f"SubPart.marks must be strictly positive, got {self.marks!r}"

        # Marking scheme criteria validation & criteria-total consistency rule
        if self.marking_scheme:
            crit_sum = 0
            for i, entry in enumerate(self.marking_scheme):
                if type(entry) is not dict:
                    return False, f"SubPart.marking_scheme[{i}] must be exact dict, got {type(entry).__name__}"
                for ck in entry.keys():
                    if type(ck) is not str:
                        return False, f"SubPart.marking_scheme[{i}] key must be exact str, got {type(ck).__name__}: {ck!r}"

                # Check criterion text
                crit_text = entry.get("criterion")
                if type(crit_text) is not str or not crit_text.strip():
                    return False, f"SubPart.marking_scheme[{i}] missing non-empty 'criterion' string"

                # Check criterion marks
                crit_marks = entry.get("marks")
                if type(crit_marks) is not int or isinstance(crit_marks, bool):
                    return False, f"SubPart.marking_scheme[{i}].marks must be exact int, got {type(crit_marks).__name__}"
                if crit_marks <= 0:
                    return False, f"SubPart.marking_scheme[{i}].marks must be strictly positive, got {crit_marks!r}"

                crit_sum += crit_marks

            # Exhaustive criteria sum rule
            if crit_sum != self.marks:
                return False, f"SubPart ({self.label}) criteria sum ({crit_sum}) != subpart marks ({self.marks})"

        return True, ""

    def __post_init__(self) -> None:
        """
        Performs initial validation and one-time whitespace normalization.

        Contract Rule on Mutation & Validation:
        - __post_init__() normalizes permitted text fields (text, topic, solution stripped) once at construction.
        - validate() is strictly read-only: it checks validity without mutating or rewriting fields.
        - If a caller mutates sp.text = '  Explain scheduling.  ' post-construction, validate() verifies
          that sp.text.strip() is non-empty, but does NOT mutate sp.text.
        - Projections and materialization consume the validated values directly as they exist on the object.
        """
        valid, err = self.validate()
        if not valid:
            if "must be exact" in err or "must be an exact integer" in err:
                raise TypeError(err)
            raise ValueError(err)
        # One-time normalization at construction
        self.text = self.text.strip()
        self.topic = self.topic.strip()
        self.solution = self.solution.strip()


# =============================================================================
# The GeneratedQuestion Dataclass
# =============================================================================

@dataclass
class GeneratedQuestion:
    # Identity (fields 1-3)
    slot_id: str
    module_id: str
    global_q_idx: int

    # Content (fields 4-6)
    question_text: str
    solution: str
    marking_scheme: List[Dict[str, Any]]

    # Pedagogical metadata (echoed from the plan for downstream use; fields 7-12)
    marks: int = 0
    partition: List[int] = field(default_factory=list)
    bloom: str = "L2"
    co: str = "CO1"
    topic: str = ""
    evidence_summary: str = ""

    # Visual contract (fields 13-16)
    references_image: bool = False
    image_path: Optional[str] = None
    figure_caption: Optional[str] = None
    diagram_request: Optional[Dict[str, Any]] = None

    # Generation provenance (fields 17-18)
    generation_source: str = "local_llm"     # local_llm | local_vlm | api
    attempts: int = 1

    # Telemetry (populated by agents; field 19)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    # Resolution metadata (Stage 1B contract; fields 20-24)
    is_unresolved: bool = False
    unresolved_scope: Optional[str] = None                  # "subquestion" | "question"
    unresolved_subparts: Tuple[str, ...] = field(default_factory=tuple)  # e.g. ("b",)
    failure_code: Optional[str] = None                      # e.g. "BLOOM_MISMATCH"
    failure_detail: Optional[str] = None                    # Diagnostic summary

    # Canonical structured subparts representation (Field 25 - Stage G1-A)
    sub_parts: List[SubPart] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate container and item types for subparts."""
        _require_exact_type(self.sub_parts, list, "sub_parts")
        for i, sp in enumerate(self.sub_parts):
            if type(sp) is not SubPart:
                raise TypeError(f"sub_parts[{i}] must be exact SubPart, got {type(sp).__name__}")

    @property
    def first_word(self) -> str:
        """First token of the question text (lowercased, stripped of punctuation)."""
        if not self.question_text:
            return ""
        token = self.question_text.strip().split()[0] if self.question_text.strip() else ""
        return token.rstrip(".,:;!?").lower()

    @property
    def is_multi_part(self) -> bool:
        return len(self.partition) > 1

    # =========================================================================
    # Projection & Synchronization Lifecycle
    # =========================================================================

    def materialize_question_text(self) -> str:
        """
        Deterministically synthesize question_text from authoritative sub_parts.
        Format: '(a) text\n\n(b) text' (or 'text' if single part).
        Guarantees 100% regex compatibility with _OUTER_MARKER_RE in legacy_adapter.py
        and _PART_LABEL in local_evaluators.py without embedded mark brackets.
        """
        if not self.sub_parts:
            return self.question_text
        if len(self.sub_parts) == 1 and not self.is_multi_part:
            return self.sub_parts[0].text
        return "\n\n".join(f"({sp.label}) {sp.text}" for sp in self.sub_parts)

    def materialize_solution(self) -> str:
        """
        Deterministically synthesize combined solution from sub_parts.
        When sub_parts are authoritative, does NOT leak stale parent solutions.
        """
        if not self.sub_parts:
            return self.solution
        parts_with_sol = [sp for sp in self.sub_parts if sp.solution]
        if not parts_with_sol:
            return ""  # Authoritative subparts have no solution; do not leak stale parent!
        if len(self.sub_parts) == 1 and not self.is_multi_part:
            return self.sub_parts[0].solution
        return "\n\n".join(f"({sp.label}) {sp.solution}" for sp in parts_with_sol)

    def materialize_marking_scheme(self) -> List[Dict[str, Any]]:
        """
        Deterministically synthesize combined marking_scheme from sub_parts.
        Preserves all extra fields from individual criteria (rubrics, bloom, weights).
        When sub_parts are authoritative, does NOT leak stale parent schemes.
        """
        if not self.sub_parts:
            return self.marking_scheme
        has_any_criteria = any(sp.marking_scheme for sp in self.sub_parts)
        if not has_any_criteria:
            return []  # Authoritative subparts have no criteria; do not leak stale parent!

        combined: List[Dict[str, Any]] = []
        for sp in self.sub_parts:
            for entry in sp.marking_scheme:
                _require_exact_type(entry, dict, f"SubPart({sp.label}) criterion")
                item = dict(entry)  # Preserve all criterion metadata fields
                crit_text = item.get("criterion", "")
                item["criterion"] = f"({sp.label}) {crit_text}".strip()
                item["subpart"] = sp.label
                combined.append(item)
        return combined

    def sync_materialized_projections(self) -> None:
        """Synchronize question_text, solution, and marking_scheme from authoritative sub_parts."""
        if self.sub_parts:
            self.question_text = self.materialize_question_text()
            self.solution = self.materialize_solution()
            self.marking_scheme = self.materialize_marking_scheme()

    def verify_projection_consistency(self) -> Tuple[bool, str]:
        """
        Verify that legacy parent fields exactly match derived projections from authoritative sub_parts.

        Exact Comparison Rule:
        - Compares exact string values without calling .strip(). Since construction-time
          normalization already established standard whitespace on SubPart, consistency checks
          must not mask whitespace divergence between stored projections and authoritative subparts.
        """
        if not self.sub_parts:
            return True, ""
        expected_text = self.materialize_question_text()
        if self.question_text != expected_text:
            return False, "question_text does not match materialized sub_parts projection"
        expected_sol = self.materialize_solution()
        if self.solution != expected_sol:
            return False, "solution does not match materialized sub_parts projection"
        expected_scheme = self.materialize_marking_scheme()
        if self.marking_scheme != expected_scheme:
            return False, "marking_scheme does not match materialized sub_parts projection"
        return True, ""

    def to_dict(self, sync_projections: bool = True) -> Dict[str, Any]:
        """Convert to plain dictionary representation. Synchronizes projections first by default."""
        if sync_projections and self.sub_parts:
            self.sync_materialized_projections()
        return asdict(self)

    # =========================================================================
    # Structural Integrity Validation
    # =========================================================================

    def validate_structure(self, allow_legacy: bool = False, check_projections: bool = True) -> Tuple[bool, str]:
        """
        Unified structural integrity validation of the question and its subparts.

        allow_legacy: ONLY used during controlled historical hydration when allow_legacy=True.
                      Permits ONLY the historical exception: partition len > 1 with missing sub_parts.
                      All other invariants (marks > 0, partition sum == marks, subpart validity if present)
                      are strictly enforced regardless of allow_legacy.
        check_projections: If True and subparts are populated, verifies that legacy parent projections
                           (question_text, solution, marking_scheme) match derived subpart values.
        """
        # Exact boolean validation of control parameters
        if type(allow_legacy) is not bool:
            return False, f"allow_legacy must be an exact bool, got {type(allow_legacy).__name__}: {allow_legacy!r}"
        if type(check_projections) is not bool:
            return False, f"check_projections must be an exact bool, got {type(check_projections).__name__}: {check_projections!r}"

        # 1. Total marks validation (exact int, strictly positive)
        if type(self.marks) is not int or isinstance(self.marks, bool):
            return False, f"marks must be an exact integer, got {type(self.marks).__name__}: {self.marks!r}"
        if self.marks <= 0:
            return False, f"marks must be strictly positive, got {self.marks!r}"

        # 2. Parent mutable fields type validation
        if type(self.question_text) is not str:
            return False, f"question_text must be exact str, got {type(self.question_text).__name__}"
        if type(self.solution) is not str:
            return False, f"solution must be exact str, got {type(self.solution).__name__}"
        if type(self.marking_scheme) is not list:
            return False, f"marking_scheme must be exact list, got {type(self.marking_scheme).__name__}"
        # Parent-level marking scheme elements must be dicts if present with string keys,
        # but do NOT impose criteria-sum rules on parent marking_scheme (preserves legacy compatibility).
        for i, entry in enumerate(self.marking_scheme):
            if type(entry) is not dict:
                return False, f"marking_scheme[{i}] must be exact dict, got {type(entry).__name__}"
            for k in entry.keys():
                if type(k) is not str:
                    return False, f"marking_scheme[{i}] key must be exact str, got {type(k).__name__}: {k!r}"

        # 3. Partition container & element validation
        if type(self.partition) is not list:
            return False, f"partition must be an exact list, got {type(self.partition).__name__}"

        # Enforce upper bound on subparts (maximum 26 subparts 'a' through 'z')
        if len(self.partition) > 26:
            return False, f"Partition count ({len(self.partition)}) exceeds maximum 26 subparts ('a'-'z')"

        for i, m in enumerate(self.partition):
            if type(m) is not int or isinstance(m, bool):
                return False, f"partition[{i}] must be an integer, got {type(m).__name__}: {m!r}"
            if m <= 0:
                return False, f"partition[{i}] must be strictly positive, got {m!r}"

        # 4. Partition marks conservation (Enforced for [M], [6, 4], [5, 3, 2], etc.)
        if self.partition:
            if sum(self.partition) != self.marks:
                return False, f"Sum of partition {self.partition} ({sum(self.partition)}) != total marks ({self.marks})"

        # 5. Subparts container validation (guards post-init mutations)
        if type(self.sub_parts) is not list:
            return False, f"sub_parts must be an exact list, got {type(self.sub_parts).__name__}"

        if len(self.sub_parts) > 26:
            return False, f"sub_parts count ({len(self.sub_parts)}) exceeds maximum 26 subparts ('a'-'z')"

        # Re-validate all mutable subparts internally (strict exact type check)
        for i, sp in enumerate(self.sub_parts):
            if type(sp) is not SubPart:
                return False, f"sub_parts[{i}] must be exact SubPart instance, got {type(sp).__name__}"
            valid, err = sp.validate()
            if not valid:
                return False, f"sub_parts[{i}] validation failed: {err}"

        # 6. Invariant: partition=[] with populated sub_parts is strictly invalid
        if not self.partition and self.sub_parts:
            return False, f"Question has populated sub_parts ({len(self.sub_parts)}) but partition is empty ([])"

        # 7. Multi-part presence requirement with narrow legacy missing-subparts scoping
        legacy_missing_subparts = (
            allow_legacy
            and len(self.partition) > 1
            and not self.sub_parts
        )
        if len(self.partition) > 1 and not self.sub_parts and not legacy_missing_subparts:
            return False, f"Partition {self.partition} requires populated sub_parts in structured generation mode"

        # If subparts are present, enforce all structural subpart invariants
        if self.sub_parts:
            # 8. Count match against partition
            if self.partition and len(self.sub_parts) != len(self.partition):
                return False, f"Subpart count ({len(self.sub_parts)}) does not match partition count ({len(self.partition)})"

            # 9. Sequential label enforcement ('a', 'b', 'c', ...)
            expected_labels = [chr(ord('a') + i) for i in range(len(self.sub_parts))]
            actual_labels = [sp.label for sp in self.sub_parts]
            if actual_labels != expected_labels:
                return False, f"Subpart labels {actual_labels} do not match expected sequential order {expected_labels}"

            # 10. Per-subpart mark match (works for multi-part [6, 4] and single-part [M])
            if self.partition:
                for sp, expected_mark in zip(self.sub_parts, self.partition):
                    if sp.marks != expected_mark:
                        return False, f"Subpart ({sp.label}) marks ({sp.marks}) != planned marks ({expected_mark})"

            # 11. Total subpart sum
            total_sp_marks = sum(sp.marks for sp in self.sub_parts)
            if total_sp_marks != self.marks:
                return False, f"Sum of subpart marks ({total_sp_marks}) != total marks ({self.marks})"

            # 12. Projection consistency check
            if check_projections:
                ok, reason = self.verify_projection_consistency()
                if not ok:
                    return False, f"Projection inconsistency: {reason}"

        return True, ""

    # =========================================================================
    # Strict Deserialization Factory
    # =========================================================================

    @classmethod
    def from_dict(cls, data: Dict[str, Any], allow_legacy: bool = False) -> GeneratedQuestion:
        """
        Authoritative hydration factory from plain dict representation.
        Strictly validates field types, rejecting unexpected types, booleans masquerading as ints,
        unknown top-level fields, and malformed subparts.

        allow_legacy: If True, permits the historical exception where a multi-part
                      partition has missing sub_parts. All other structural rules
                      (marks > 0, partition sum == marks, subpart validation if present)
                      are strictly enforced through validate_structure().
        """
        _require_exact_type(data, dict, "from_dict argument")
        _require_exact_type(allow_legacy, bool, "allow_legacy")

        # 1. Validate that all dictionary keys are strings before sorting
        for k in data.keys():
            if type(k) is not str:
                raise TypeError(f"All dictionary keys must be strings, got {type(k).__name__}: {k!r}")

        # 2. Strict top-level whitelist: reject unknown keys to prevent typos from silently disappearing
        allowed_top_keys = {
            "slot_id", "module_id", "global_q_idx", "question_text", "solution",
            "marking_scheme", "marks", "partition", "bloom", "co", "topic",
            "evidence_summary", "references_image", "image_path", "figure_caption",
            "diagram_request", "generation_source", "attempts", "telemetry",
            "is_unresolved", "unresolved_scope", "unresolved_subparts",
            "failure_code", "failure_detail", "sub_parts"
        }
        unknown_top = set(data.keys()) - allowed_top_keys
        if unknown_top:
            raise ValueError(f"from_dict received unknown top-level fields: {sorted(unknown_top)}")

        slot_id = _require_str(data.get("slot_id", ""), "slot_id")
        module_id = _require_str(data.get("module_id", ""), "module_id")
        global_q_idx = _require_int(data.get("global_q_idx", 0), "global_q_idx", allow_zero=True)
        question_text = _require_str(data.get("question_text", ""), "question_text")
        solution = _require_str(data.get("solution", ""), "solution")

        # Parent marking scheme validation (preserves legacy parent compatibility; no criteria-sum check)
        raw_scheme = data.get("marking_scheme", [])
        marking_scheme = _require_list(raw_scheme, "marking_scheme")
        for i, entry in enumerate(marking_scheme):
            _require_exact_type(entry, dict, f"marking_scheme[{i}]")
            for k in entry.keys():
                if type(k) is not str:
                    raise TypeError(f"marking_scheme[{i}] key must be string, got {type(k).__name__}: {k!r}")

        marks = _require_int(data.get("marks", 0), "marks", allow_zero=True)
        raw_partition = data.get("partition", [])
        partition = _require_list(raw_partition, "partition")
        for i, elem in enumerate(partition):
            _require_int(elem, f"partition[{i}]", allow_zero=False)

        # Optional metadata fields: Omitted fields are intentionally defaulted
        # using standard dataclass defaults for historical backward compatibility.
        # Unknown fields are strictly rejected by the whitelist check above.
        bloom = _require_str(data.get("bloom", "L2"), "bloom")
        co = _require_str(data.get("co", "CO1"), "co")
        topic = _require_str(data.get("topic", ""), "topic")
        evidence_summary = _require_str(data.get("evidence_summary", ""), "evidence_summary")
        references_image = _require_bool(data.get("references_image", False), "references_image")

        raw_image_path = data.get("image_path")
        image_path = _require_str(raw_image_path, "image_path") if raw_image_path is not None else None

        raw_figure_caption = data.get("figure_caption")
        figure_caption = _require_str(raw_figure_caption, "figure_caption") if raw_figure_caption is not None else None

        diagram_request = data.get("diagram_request")
        if diagram_request is not None:
            _require_exact_type(diagram_request, dict, "diagram_request")

        generation_source = _require_str(data.get("generation_source", "local_llm"), "generation_source")
        attempts = _require_int(data.get("attempts", 1), "attempts", allow_zero=False)

        raw_telemetry = data.get("telemetry", {})
        telemetry = _require_exact_type(raw_telemetry, dict, "telemetry")

        is_unresolved = _require_bool(data.get("is_unresolved", False), "is_unresolved")

        raw_unresolved_scope = data.get("unresolved_scope")
        unresolved_scope = _require_str(raw_unresolved_scope, "unresolved_scope") if raw_unresolved_scope is not None else None
        if unresolved_scope is not None and unresolved_scope not in ("question", "subquestion"):
            raise ValueError(f"unresolved_scope must be 'question' or 'subquestion', got {unresolved_scope!r}")

        # Exact tuple or list type check for unresolved_subparts (rejecting arbitrary subclasses)
        raw_subparts_unresolved = data.get("unresolved_subparts", ())
        if type(raw_subparts_unresolved) not in (tuple, list):
            raise TypeError(f"unresolved_subparts must be exact tuple or list, got {type(raw_subparts_unresolved).__name__}")
        unresolved_subparts_list = []
        for j, elem in enumerate(raw_subparts_unresolved):
            unresolved_subparts_list.append(_require_str(elem, f"unresolved_subparts[{j}]"))
        unresolved_subparts = tuple(unresolved_subparts_list)

        raw_failure_code = data.get("failure_code")
        failure_code = _require_str(raw_failure_code, "failure_code") if raw_failure_code is not None else None

        raw_failure_detail = data.get("failure_detail")
        failure_detail = _require_str(raw_failure_detail, "failure_detail") if raw_failure_detail is not None else None

        # Hydrate subparts strictly with defensive copying and precise legacy distinction
        has_subparts_key = "sub_parts" in data
        hydrated_subparts: List[SubPart] = []

        if not has_subparts_key:
            # Historical representation: 'sub_parts' key is entirely absent from dictionary.
            # Eligible for allow_legacy=True if partition is multi-part.
            if not allow_legacy and len(partition) > 1:
                raise ValueError(
                    f"Missing 'sub_parts' key for multi-part question with partition {partition}. "
                    "Pass allow_legacy=True only when loading historical pre-G1-A records."
                )
            hydrated_subparts = []
            validate_allow_legacy = allow_legacy
        else:
            # Structured representation: 'sub_parts' key is explicitly present.
            raw_subparts = data["sub_parts"]
            _require_exact_type(raw_subparts, list, "sub_parts")

            # An explicitly provided empty list [] for a multi-part question represents an
            # incomplete or broken structured question, NOT a historical record.
            # It CANNOT claim the legacy exception.
            if not raw_subparts and len(partition) > 1:
                raise ValueError(
                    f"Explicit 'sub_parts': [] provided for multi-part question with partition {partition}. "
                    "Explicitly empty sub_parts represents an incomplete structured question and cannot claim legacy exception."
                )

            allowed_sp_keys = {"label", "marks", "text", "topic", "solution", "marking_scheme"}
            for i, sp_data in enumerate(raw_subparts):
                if type(sp_data) is SubPart:
                    # Defensive copy to avoid mutating external references
                    sp_copy = copy.deepcopy(sp_data)
                    valid, err = sp_copy.validate()
                    if not valid:
                        raise ValueError(f"sub_parts[{i}] validation failed: {err}")
                    hydrated_subparts.append(sp_copy)
                elif isinstance(sp_data, dict):
                    for sp_k in sp_data.keys():
                        if type(sp_k) is not str:
                            raise TypeError(f"sub_parts[{i}] key must be string, got {type(sp_k).__name__}: {sp_k!r}")
                    extra_keys = set(sp_data.keys()) - allowed_sp_keys
                    if extra_keys:
                        raise ValueError(f"sub_parts[{i}] contains unknown fields: {sorted(extra_keys)}")

                    sp_label = _require_str(sp_data.get("label", ""), f"sub_parts[{i}].label", allow_empty=False)
                    sp_marks = _require_int(sp_data.get("marks"), f"sub_parts[{i}].marks", allow_zero=False)
                    sp_text = _require_str(sp_data.get("text", ""), f"sub_parts[{i}].text", allow_empty=False)
                    sp_topic = _require_str(sp_data.get("topic", ""), f"sub_parts[{i}].topic", allow_empty=True)
                    sp_sol = _require_str(sp_data.get("solution", ""), f"sub_parts[{i}].solution", allow_empty=True)
                    sp_scheme_raw = _require_list(sp_data.get("marking_scheme", []), f"sub_parts[{i}].marking_scheme")

                    # Explicit validation of every criterion entry with exact path reporting before construction
                    sp_scheme_validated: List[Dict[str, Any]] = []
                    for j, crit_entry in enumerate(sp_scheme_raw):
                        _require_exact_type(crit_entry, dict, f"sub_parts[{i}].marking_scheme[{j}]")
                        for ck in crit_entry.keys():
                            if type(ck) is not str:
                                raise TypeError(f"sub_parts[{i}].marking_scheme[{j}] key must be string, got {type(ck).__name__}: {ck!r}")
                        crit_text = crit_entry.get("criterion")
                        if type(crit_text) is not str or not crit_text.strip():
                            raise ValueError(f"sub_parts[{i}].marking_scheme[{j}] missing non-empty 'criterion' string")
                        crit_m = crit_entry.get("marks")
                        if type(crit_m) is not int or isinstance(crit_m, bool):
                            raise TypeError(f"sub_parts[{i}].marking_scheme[{j}].marks must be exact int, got {type(crit_m).__name__}")
                        if crit_m <= 0:
                            raise ValueError(f"sub_parts[{i}].marking_scheme[{j}].marks must be strictly positive, got {crit_m!r}")
                        sp_scheme_validated.append(copy.deepcopy(crit_entry))

                    hydrated_subparts.append(SubPart(
                        label=sp_label,
                        marks=sp_marks,
                        text=sp_text,
                        topic=sp_topic,
                        solution=sp_sol,
                        marking_scheme=sp_scheme_validated,
                    ))
                else:
                    raise TypeError(f"sub_parts[{i}] must be a dict or SubPart, got {type(sp_data).__name__}")

            # Since 'sub_parts' was explicitly provided, validation enforces standard structured rules
            validate_allow_legacy = False

        q = cls(
            slot_id=slot_id,
            module_id=module_id,
            global_q_idx=global_q_idx,
            question_text=question_text,
            solution=solution,
            marking_scheme=marking_scheme,
            marks=marks,
            partition=partition,
            bloom=bloom,
            co=co,
            topic=topic,
            evidence_summary=evidence_summary,
            references_image=references_image,
            image_path=image_path,
            figure_caption=figure_caption,
            diagram_request=diagram_request,
            generation_source=generation_source,
            attempts=attempts,
            telemetry=telemetry,
            is_unresolved=is_unresolved,
            unresolved_scope=unresolved_scope,
            unresolved_subparts=unresolved_subparts,
            failure_code=failure_code,
            failure_detail=failure_detail,
            sub_parts=hydrated_subparts,
        )

        # Complete Structural Validation upon Hydration via the common validation engine
        valid, err = q.validate_structure(allow_legacy=validate_allow_legacy, check_projections=False)
        if not valid:
            raise ValueError(f"Structural validation failed during hydration: {err}")

        return q


# =============================================================================
# SlotFailure and WritingAgentOutput
# =============================================================================

@dataclass
class SlotFailure:
    """A slot that failed during the Writing Agent's run."""
    slot_id: str
    failure_code: str
    failure_detail: str


@dataclass
class WritingAgentOutput:
    """The payload of the Writing Agent's successful AgentResult."""
    questions: List[GeneratedQuestion]
    failures: List[SlotFailure] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return len(self.failures) == 0
