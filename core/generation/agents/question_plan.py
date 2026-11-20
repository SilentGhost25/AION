"""
The QuestionPlan contract.

One plan per main-question slot. The Writing Agent consumes a plan
and produces a GeneratedQuestion. The plan carries everything the
Writing Agent needs and nothing it doesn't.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


# -----------------------------------------------------------------------------
# Plan Status & Failure Enums
# -----------------------------------------------------------------------------

class PlanStatus(str, Enum):
    PLANNED = "PLANNED"                        # Complete structured plan: all subparts fully allocated, grounded, and marks-conserved
    UNRESOLVED = "UNRESOLVED"                  # Planning retries exhausted: sub_part_plans=[], diagnostics retained
    LEGACY_UNVALIDATED = "LEGACY_UNVALIDATED"  # Backward compatibility: unpartitioned or legacy callers without SubPartPlan allocations


class PlanFailureCode(str, Enum):
    INSUFFICIENT_TOPICS = "INSUFFICIENT_TOPICS"        # Retryable: module KG lacks distinct concepts
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"    # Retryable: topic lacks eligible evidence blocks in module segment
    BLOOM_CO_INCOMPATIBLE = "BLOOM_CO_INCOMPATIBLE"    # Retryable: topic/constraints violate CO Bloom band
    SIBLING_COLLISION = "SIBLING_COLLISION"            # Retryable: sibling subparts share identical entities or concept neighborhood overlap
    INVALID_CONFIG = "INVALID_CONFIG"                  # Fatal / Non-retryable: config lookup or structure violation
    UNREGISTERED_BLUEPRINT = "UNREGISTERED_BLUEPRINT"  # Fatal / Non-retryable: unknown blueprint
    MARKS_MISMATCH = "MARKS_MISMATCH"                  # Fatal / Non-retryable: requested marks != config
    MISSING_MODULE_SEGMENT = "MISSING_MODULE_SEGMENT"  # Fatal / Non-retryable: extraction pipeline lacks verified segment


# -----------------------------------------------------------------------------
# Pedagogical Policy & Concept Collision Detection
# -----------------------------------------------------------------------------

# Established AION Educational Mapping (CO1-CO5)
CO_BLOOM_BANDS: Dict[str, Tuple[str, ...]] = {
    "CO1": ("L1", "L2"),   # Understand / Remember
    "CO2": ("L3",),         # Apply
    "CO3": ("L4",),         # Implement / Analyze
    "CO4": ("L5",),         # Evaluate
    "CO5": ("L6",),         # Create
}


def normalize_concept(name: str) -> str:
    s = re.sub(r"^(the|a|an)\s+", "", str(name).strip(), flags=re.IGNORECASE)
    s = re.sub(r"[^a-zA-Z0-9\s\-]", "", s).lower()
    return re.sub(r"\s+", " ", s).strip()


def detect_concept_collision(part_a: SubPartPlan, part_b: SubPartPlan) -> Tuple[bool, Optional[str]]:
    id_a = normalize_concept(part_a.canonical_concept_id or part_a.topic)
    id_b = normalize_concept(part_b.canonical_concept_id or part_b.topic)

    if id_a == id_b:
        return True, f"Identical canonical concept ID '{id_a}'"

    syn_a = {normalize_concept(s) for s in part_a.synonym_ids if s}
    syn_b = {normalize_concept(s) for s in part_b.synonym_ids if s}

    if id_a in syn_b or id_b in syn_a:
        return True, "Direct synonym collision between canonical ID and sibling synonym"

    shared_syn = syn_a.intersection(syn_b)
    shared_syn.discard("")
    if shared_syn:
        return True, f"Shared synonym(s) between subparts: {sorted(list(shared_syn))}"

    footprint_a = {normalize_concept(c) for c in part_a.concept_neighborhood if isinstance(c, str)}
    footprint_b = {normalize_concept(c) for c in part_b.concept_neighborhood if isinstance(c, str)}

    footprint_a.add(id_a)
    footprint_b.add(id_b)

    if id_a in footprint_b:
        return True, f"Concept '{part_a.topic}' collides with sibling '{part_b.label}' concept footprint"
    if id_b in footprint_a:
        return True, f"Concept '{part_b.topic}' collides with sibling '{part_a.label}' concept footprint"

    shared_hood = footprint_a.intersection(footprint_b)
    shared_hood.discard("")
    if shared_hood:
        return True, f"Shared concepts in neighborhood footprints: {sorted(list(shared_hood))}"

    return False, None


# -----------------------------------------------------------------------------
# SubPartPlan Dataclass
# -----------------------------------------------------------------------------

@dataclass
class SubPartPlan:
    label: str                                      # Strictly single lowercase ASCII 'a'-'z'
    marks: int                                      # Exact marks allocated (> 0, not bool)
    topic: str                                      # Dedicated topic allocated from module KG
    bloom: str = "L2"                               # Target Bloom level (L1-L6)
    co: str = "CO1"                                 # Target Course Outcome (CO1-CO5)
    canonical_concept_id: str = ""                  # Normalized canonical ID from KG
    synonym_ids: List[str] = field(default_factory=list) # Persisted direct synonyms from KG
    concept_neighborhood: List[str] = field(default_factory=list)
    evidence_blocks: List[Any] = field(default_factory=list)
    visual_required: bool = False
    figure: Optional[Any] = None

    def validate_fields(self) -> Tuple[bool, str]:
        """Validate internal data types, labels, marks, pedagogy, and syntax."""
        if type(self.label) is not str or len(self.label) != 1 or not ('a' <= self.label <= 'z'):
            return False, f"SubPartPlan label must be single lowercase letter 'a'-'z', got {self.label!r}"
        if type(self.marks) is not int or isinstance(self.marks, bool):
            return False, f"SubPartPlan marks must be exact integer, got {type(self.marks).__name__}: {self.marks!r}"
        if self.marks <= 0:
            return False, f"SubPartPlan marks must be strictly positive, got {self.marks!r}"
        if type(self.topic) is not str or not self.topic.strip():
            return False, "SubPartPlan topic must be non-empty string"
        if type(self.bloom) is not str or self.bloom not in ("L1", "L2", "L3", "L4", "L5", "L6"):
            return False, f"SubPartPlan bloom must be valid Bloom level L1-L6, got {self.bloom!r}"
        if type(self.co) is not str or self.co not in CO_BLOOM_BANDS:
            return False, f"SubPartPlan co must match CO1-CO5, got {self.co!r}"
        if self.bloom not in CO_BLOOM_BANDS[self.co]:
            return False, f"SubPartPlan bloom '{self.bloom}' incompatible with co '{self.co}' band {CO_BLOOM_BANDS[self.co]}"

        if type(self.canonical_concept_id) is not str:
            return False, f"SubPartPlan canonical_concept_id must be str, got {type(self.canonical_concept_id).__name__}"

        if type(self.synonym_ids) is not list:
            return False, f"SubPartPlan synonym_ids must be list, got {type(self.synonym_ids).__name__}"
        for i, s in enumerate(self.synonym_ids):
            if type(s) is not str or not s.strip():
                return False, f"SubPartPlan synonym_ids[{i}] must be non-empty string"

        if type(self.concept_neighborhood) is not list:
            return False, f"SubPartPlan concept_neighborhood must be list, got {type(self.concept_neighborhood).__name__}"
        for i, item in enumerate(self.concept_neighborhood):
            if type(item) is not str or not item.strip():
                return False, f"SubPartPlan concept_neighborhood[{i}] must be non-empty str"

        if type(self.evidence_blocks) is not list:
            return False, f"SubPartPlan evidence_blocks must be list, got {type(self.evidence_blocks).__name__}"

        if type(self.visual_required) is not bool:
            return False, f"SubPartPlan visual_required must be exact bool, got {type(self.visual_required).__name__}"
        if self.visual_required:
            if self.figure is None:
                return False, "SubPartPlan visual_required=True requires a non-None figure reference"
            fig_id = getattr(self.figure, "id", None) if not isinstance(self.figure, dict) else self.figure.get("id")
            if type(fig_id) is not str or not re.match(r"^[a-zA-Z0-9_\-]+$", fig_id):
                return False, f"SubPartPlan figure id must be non-empty alphanumeric string, got {fig_id!r}"
            fig_path = getattr(self.figure, "image_path", None) if not isinstance(self.figure, dict) else self.figure.get("image_path")
            if type(fig_path) is not str or not fig_path.strip():
                return False, "SubPartPlan figure image_path must be non-empty string reference"
        else:
            if self.figure is not None:
                return False, "SubPartPlan visual_required=False must have figure=None"

        return True, ""

    def validate_provenance(self, module_idx: int, page_start: int, page_end: int) -> Tuple[bool, str]:
        """Validate production module evidence and figure provenance. No caller-controlled bypass."""
        if not self.canonical_concept_id.strip():
            return False, "SubPartPlan canonical_concept_id must be non-empty string in production"

        if not self.evidence_blocks:
            return False, "SubPartPlan evidence_blocks must not be empty in production planning"

        if type(module_idx) is not int or isinstance(module_idx, bool) or module_idx <= 0:
            return False, f"Production evidence requires positive exact integer module_idx, got {module_idx!r}"
        if type(page_start) is not int or isinstance(page_start, bool) or page_start <= 0:
            return False, f"Production evidence requires positive exact integer page_start, got {page_start!r}"
        if type(page_end) is not int or isinstance(page_end, bool) or page_end < page_start:
            return False, f"Production evidence requires positive exact integer page_end >= page_start, got {page_end!r} (page_start={page_start!r})"

        expected_mod = f"module_{module_idx}"

        for i, b in enumerate(self.evidence_blocks):
            if b is None or isinstance(b, (int, float, bool)):
                return False, f"SubPartPlan evidence_blocks[{i}] must be a TextBlock or valid evidence dict, got {type(b).__name__}"

            b_id = getattr(b, "id", None) if not isinstance(b, dict) else b.get("id")
            b_text = getattr(b, "text", None) if not isinstance(b, dict) else (b.get("text") or b.get("content"))
            b_page = getattr(b, "page", None) if not isinstance(b, dict) else b.get("page")
            b_mod = getattr(b, "module_id", None) if not isinstance(b, dict) else b.get("module_id")

            if type(b_id) is not str or not b_id.strip():
                return False, f"SubPartPlan evidence_blocks[{i}] missing non-empty block id"
            if type(b_text) is not str or not b_text.strip():
                return False, f"SubPartPlan evidence_blocks[{i}] text must be non-empty string"
            if type(b_page) is not int or isinstance(b_page, bool) or b_page < 1:
                return False, f"SubPartPlan evidence_blocks[{i}] page must be positive int, got {b_page!r}"

            if not (page_start <= b_page <= page_end):
                return False, f"SubPartPlan evidence_blocks[{i}] page {b_page} outside verified module segment [{page_start}, {page_end}]"

            if type(b_mod) is not str or b_mod != expected_mod:
                return False, f"SubPartPlan evidence_blocks[{i}] module_id '{b_mod}' != expected '{expected_mod}'"

        if self.visual_required:
            fig_mod = getattr(self.figure, "module_id", None) if not isinstance(self.figure, dict) else self.figure.get("module_id")
            if fig_mod != expected_mod:
                return False, f"SubPartPlan figure module_id '{fig_mod}' != slot module '{expected_mod}'"

        return True, ""

    def __post_init__(self) -> None:
        valid, err = self.validate_fields()
        if not valid:
            if "must be exact" in err or "must be list" in err:
                raise TypeError(err)
            raise ValueError(err)
        self.topic = self.topic.strip()


# -----------------------------------------------------------------------------
# QuestionPlan Dataclass
# -----------------------------------------------------------------------------

@dataclass
class QuestionPlan:
    # Identity (fields 1-5; mandatory positional)
    slot_id: str                    # "module_1_Q1"
    module_id: str                  # "module_1"
    module_idx: int                 # 1..5
    slot_idx: int                   # 0-indexed within module
    global_q_idx: int               # 1..N (1-10 for IAT1)

    # Marks contract (fields 6-7)
    partition: List[int] = field(default_factory=list)   # [6, 4] or [10]
    total_marks: int = 0

    # Pedagogical contract (fields 8-9)
    co: str = "CO1"
    bloom: str = "L2"

    # Topic grounding (from KG) (fields 10-15; compatibility projections)
    topic: str = ""                         # Primary/legacy topic
    topic_a: str = ""                       # Dedicated topic for sub-part (a)
    topic_b: str = ""                       # Dedicated topic for sub-part (b)
    concept_neighborhood: List = field(default_factory=list)
    concept_neighborhood_a: List = field(default_factory=list)
    concept_neighborhood_b: List = field(default_factory=list)

    # Evidence & Visuals (fields 16-18)
    evidence_blocks: List = field(default_factory=list)
    visual_required: bool = False
    figure: Optional[Any] = None            # FigureArtifact if visual_required

    # Additional constraints & Telemetry (fields 19-20)
    constraints: dict = field(default_factory=dict)
    telemetry: dict = field(default_factory=dict)

    # Appended Stage G1-B Authoritative Fields (fields 21-27)
    sub_part_plans: List[SubPartPlan] = field(default_factory=list)
    status: PlanStatus = PlanStatus.LEGACY_UNVALIDATED
    failure_code: Optional[str] = None
    failure_detail: Optional[str] = None
    module_page_start: Optional[int] = None
    module_page_end: Optional[int] = None
    assessment_identity: Tuple[str, str, str, str] = ("", "", "", "") # (scheme_id, course_type, assessment_type, blueprint_id)

    def __post_init__(self) -> None:
        if self.status == PlanStatus.PLANNED:
            valid, code, err = self.validation_result()
            if not valid:
                raise ValueError(f"QuestionPlan marked PLANNED violates structured plan invariant: {err} ({code})")
        elif self.status == PlanStatus.UNRESOLVED:
            if self.sub_part_plans:
                raise ValueError("QuestionPlan marked UNRESOLVED must not contain sub_part_plans; retain in telemetry only")
            if not self.failure_code or self.failure_code not in PlanFailureCode.__members__:
                raise ValueError(f"QuestionPlan marked UNRESOLVED must have a valid declared PlanFailureCode enum, got {self.failure_code!r}")
            if not self.failure_detail:
                raise ValueError("QuestionPlan marked UNRESOLVED must have non-empty failure_detail")

            attempts = self.telemetry.get("attempts")
            if not isinstance(attempts, list):
                raise ValueError("QuestionPlan marked UNRESOLVED must carry diagnostic 'attempts' list in telemetry")

            is_fatal = self.failure_code in (
                PlanFailureCode.INVALID_CONFIG.value,
                PlanFailureCode.UNREGISTERED_BLUEPRINT.value,
                PlanFailureCode.MARKS_MISMATCH.value,
                PlanFailureCode.MISSING_MODULE_SEGMENT.value,
            )
            if is_fatal and len(attempts) != 0:
                raise ValueError(f"Fatal planning failure {self.failure_code} must have 0 attempts in telemetry, got {len(attempts)}")
            if not is_fatal:
                if not (1 <= len(attempts) <= 3):
                    raise ValueError(f"Retryable failure {self.failure_code} must have 1..3 attempts in telemetry, got {len(attempts)}")
                for att in attempts:
                    if not isinstance(att, dict):
                        raise ValueError("Attempt record must be dict")
                    for req_key in ("attempt_number", "strategy", "failure_code", "failure_detail", "candidate_topics"):
                        if req_key not in att:
                            raise ValueError(f"Attempt record missing required key: {req_key}")

        if not self.topic_a:
            self.topic_a = self.topic
        if not self.topic_b:
            self.topic_b = self.topic
        if not self.concept_neighborhood_a and self.concept_neighborhood:
            self.concept_neighborhood_a = list(self.concept_neighborhood)
        if not self.concept_neighborhood_b and self.concept_neighborhood:
            self.concept_neighborhood_b = list(self.concept_neighborhood)

        self.sync_legacy_projections()

    @property
    def is_multi_part(self) -> bool:
        return len(self.partition) > 1

    @property
    def difficulty_band(self) -> str:
        """Derive target difficulty band from marks and Bloom level."""
        if self.bloom in ("L1", "L2") or self.total_marks <= 4:
            return "easy"
        if self.bloom in ("L3",) or self.total_marks <= 6:
            return "medium"
        return "hard"  # L4, L5, L6 or marks >= 8

    @property
    def is_legacy_representable(self) -> bool:
        """Indicates whether this plan can be safely projected to legacy 2-part format without subpart loss."""
        return len(self.sub_part_plans) <= 2

    def require_legacy_representable(self) -> None:
        """Guard method for legacy consumers that cannot represent more than 2 subparts."""
        if not self.is_legacy_representable:
            raise ValueError(
                f"QuestionPlan has {len(self.sub_part_plans)} subparts and cannot be represented "
                f"safely in legacy 2-part format without subpart loss."
            )

    def is_currently_valid(self) -> bool:
        valid, _, _ = self.validation_result()
        return self.status == PlanStatus.PLANNED and valid

    def validation_result(self) -> Tuple[bool, Optional[PlanFailureCode], str]:
        """
        Defensive revalidation returning (is_valid, failure_code, failure_detail).
        Safely handles mutated or malformed objects without crashing.
        """
        try:
            from ..assessment_config import AssessmentRegistry

            # Defensive type checks on assessment_identity
            if not isinstance(self.assessment_identity, (tuple, list)) or len(self.assessment_identity) != 4:
                return False, PlanFailureCode.INVALID_CONFIG, f"assessment_identity must be 4-tuple, got {type(self.assessment_identity).__name__}"

            scheme_id, course_type, assessment_type, blueprint_id = self.assessment_identity
            if not (isinstance(scheme_id, str) and isinstance(course_type, str) and isinstance(assessment_type, str) and isinstance(blueprint_id, str)):
                return False, PlanFailureCode.INVALID_CONFIG, "All assessment_identity elements must be str"

            if not (scheme_id and course_type and assessment_type and blueprint_id):
                return False, PlanFailureCode.UNREGISTERED_BLUEPRINT, "Missing assessment_identity 4-tuple on QuestionPlan"

            try:
                config = AssessmentRegistry.get(scheme_id, course_type, assessment_type, blueprint_id)
            except (KeyError, ValueError) as e:
                return False, PlanFailureCode.UNREGISTERED_BLUEPRINT, str(e)

            # Defensive type checks on total_marks and partition
            if not isinstance(self.total_marks, int) or isinstance(self.total_marks, bool) or self.total_marks <= 0:
                return False, PlanFailureCode.MARKS_MISMATCH, f"total_marks must be positive exact int, got {self.total_marks!r}"

            if not isinstance(self.partition, list) or not self.partition:
                return False, PlanFailureCode.INVALID_CONFIG, f"partition must be non-empty list, got {self.partition!r}"
            if len(self.partition) > 26:
                return False, PlanFailureCode.INVALID_CONFIG, f"partition length {len(self.partition)} exceeds maximum 26 parts"
            for idx, m in enumerate(self.partition):
                if not isinstance(m, int) or isinstance(m, bool) or m <= 0:
                    return False, PlanFailureCode.MARKS_MISMATCH, f"partition[{idx}] must be positive exact int, got {m!r}"

            # Revalidate assessment policy against registered configuration
            if self.total_marks != config.total_marks_per_question:
                return False, PlanFailureCode.MARKS_MISMATCH, f"total_marks {self.total_marks} != config {config.total_marks_per_question}"
            if tuple(self.partition) not in config.permitted_partitions:
                return False, PlanFailureCode.INVALID_CONFIG, f"partition {self.partition} not permitted by {config.identity}"
            if sum(self.partition) != self.total_marks:
                return False, PlanFailureCode.MARKS_MISMATCH, f"partition sum {sum(self.partition)} != total_marks {self.total_marks}"

            is_synthetic = (config.course_type == "SYNTHETIC_TEST")

            # Validate parent pedagogical contract
            if not isinstance(self.co, str) or self.co not in CO_BLOOM_BANDS:
                return False, PlanFailureCode.BLOOM_CO_INCOMPATIBLE, f"Parent co '{self.co}' not in supported bands CO1-CO5"
            if not isinstance(self.bloom, str) or self.bloom not in CO_BLOOM_BANDS[self.co]:
                return False, PlanFailureCode.BLOOM_CO_INCOMPATIBLE, f"Parent bloom '{self.bloom}' incompatible with co '{self.co}' band {CO_BLOOM_BANDS[self.co]}"

            # Defensive check on constraints and nested dictionaries
            if not isinstance(self.constraints, dict):
                return False, PlanFailureCode.INVALID_CONFIG, f"constraints must be dict, got {type(self.constraints).__name__}"

            subpart_cos = self.constraints.get("subpart_cos")
            subpart_cos_dict = subpart_cos if isinstance(subpart_cos, dict) else {}

            subpart_blooms = self.constraints.get("subpart_blooms")
            subpart_blooms_dict = subpart_blooms if isinstance(subpart_blooms, dict) else {}

            # Subpart plans defensive checks
            if not isinstance(self.sub_part_plans, list):
                return False, PlanFailureCode.INVALID_CONFIG, f"sub_part_plans must be list, got {type(self.sub_part_plans).__name__}"
            if not self.sub_part_plans:
                return False, PlanFailureCode.INVALID_CONFIG, "PLANNED status requires non-empty sub_part_plans"
            if len(self.sub_part_plans) != len(self.partition):
                return False, PlanFailureCode.INVALID_CONFIG, f"sub_part_plans count ({len(self.sub_part_plans)}) != partition count ({len(self.partition)})"

            part_sum = 0
            validated_parts: List[SubPartPlan] = []

            for i, sp in enumerate(self.sub_part_plans):
                if not isinstance(sp, SubPartPlan):
                    return False, PlanFailureCode.INVALID_CONFIG, f"sub_part_plans[{i}] must be SubPartPlan instance, got {type(sp).__name__}"
                expected_label = chr(ord('a') + i)
                if sp.label != expected_label:
                    return False, PlanFailureCode.INVALID_CONFIG, f"sub_part_plans[{i}] label '{sp.label}' != expected sequential '{expected_label}'"
                if sp.marks != self.partition[i]:
                    return False, PlanFailureCode.MARKS_MISMATCH, f"sub_part_plans[{i}] marks {sp.marks} != partition[{i}] {self.partition[i]}"

                # Check override authorization if subpart CO/Bloom differs from parent
                if sp.co != self.co and not (subpart_cos_dict.get(sp.label) == sp.co):
                    return False, PlanFailureCode.BLOOM_CO_INCOMPATIBLE, f"sub_part_plans[{i}] co '{sp.co}' differs from parent '{self.co}' without explicit constraint override"
                if sp.bloom != self.bloom and not (subpart_blooms_dict.get(sp.label) == sp.bloom):
                    return False, PlanFailureCode.BLOOM_CO_INCOMPATIBLE, f"sub_part_plans[{i}] bloom '{sp.bloom}' differs from parent '{self.bloom}' without explicit constraint override"

                valid, err = sp.validate_fields()
                if not valid:
                    return False, PlanFailureCode.INVALID_CONFIG, err

                if not is_synthetic:
                    valid_prov, err_prov = sp.validate_provenance(
                        module_idx=self.module_idx,
                        page_start=self.module_page_start,
                        page_end=self.module_page_end,
                    )
                    if not valid_prov:
                        return False, PlanFailureCode.INSUFFICIENT_EVIDENCE, err_prov

                # Sibling concept set independence check against previously validated parts
                for prev_sp in validated_parts:
                    collided, reason = detect_concept_collision(sp, prev_sp)
                    if collided:
                        return False, PlanFailureCode.SIBLING_COLLISION, f"Sibling collision between '{sp.label}' and '{prev_sp.label}': {reason}"

                validated_parts.append(sp)
                part_sum += sp.marks

            if part_sum != self.total_marks:
                return False, PlanFailureCode.MARKS_MISMATCH, f"Subpart marks sum ({part_sum}) != total_marks ({self.total_marks})"

            return True, None, ""
        except Exception as exc:
            return False, PlanFailureCode.INVALID_CONFIG, f"Unexpected validation exception: {str(exc)}"

    def sync_legacy_projections(self) -> None:
        """
        Deterministically synchronize legacy topic and neighborhood fields from sub_part_plans.
        Idempotent: repeated calls produce identical results.
        When sub_part_plans is empty, explicitly clears ALL legacy projections.
        """
        if not self.sub_part_plans:
            self.topic = ""
            self.topic_a = ""
            self.topic_b = ""
            self.concept_neighborhood = []
            self.concept_neighborhood_a = []
            self.concept_neighborhood_b = []
            return

        self.topic = self.sub_part_plans[0].topic
        self.concept_neighborhood = list(self.sub_part_plans[0].concept_neighborhood)

        self.topic_a = self.sub_part_plans[0].topic
        self.concept_neighborhood_a = list(self.sub_part_plans[0].concept_neighborhood)

        if len(self.sub_part_plans) >= 2:
            self.topic_b = self.sub_part_plans[1].topic
            self.concept_neighborhood_b = list(self.sub_part_plans[1].concept_neighborhood)
        else:
            self.topic_b = ""
            self.concept_neighborhood_b = []

    def __repr__(self) -> str:
        vis = "V" if self.visual_required else "-"
        parts_info = f" parts={len(self.sub_part_plans)}" if self.sub_part_plans else ""
        return (
            f"<Plan {self.slot_id} {self.bloom} {self.total_marks}M "
            f"{self.co} {vis} status={self.status.value}{parts_info} topic={self.topic[:30]!r}>"
        )
