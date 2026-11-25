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

from .question_plan import (
    ConceptFootprint,
    ExpectedResponseContract,
    RubricCriterion,
    VisualTaskContract,
    EvidenceBlockID,
)


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
    label: str = ""                                         # Subpart marker: strictly single lowercase ASCII 'a'-'z'
    marks: int = 0                                          # Exact mark allocation: strictly positive integer
    text: str = ""                                          # Self-contained subquestion text
    topic: str = ""                                         # Specific topic allocated from the module KG
    solution: str = ""                                      # Model answer for this subpart
    marking_scheme: List[Dict[str, Any]] = field(default_factory=list)  # Criteria breakdown for this subpart
    figure_id: Optional[str] = None                         # Bound FigureArtifact.id (Stage G2-D)
    image_path: Optional[str] = None                        # Visual asset path (Stage G2-D)
    figure_caption: Optional[str] = None                    # Figure caption (Stage G2-D)
    image_sha256: Optional[str] = None                      # Cryptographic hash (Stage G2-D)
    bloom: str = ""                                         # Planned Bloom level for this subpart (Stage G4, e.g. "L2", "L3")
    co: str = ""                                            # Planned Course Outcome for this subpart (Stage G4, e.g. "CO1", "CO2")
    evidence_block_ids: Tuple[str, ...] = field(default_factory=tuple)  # Stage G6
    concept_footprint: Optional[ConceptFootprint] = None                 # Stage G6
    expected_response: Optional[ExpectedResponseContract] = None         # Stage G6
    visual_task: Optional[VisualTaskContract] = None                     # Stage G6
    telemetry: Dict[str, Any] = field(default_factory=dict)              # Stage G6
    subpart_index: Optional[int] = None
    question_id: Optional[str] = None
    module_id: Optional[str] = None
    target_cognitive_level: Optional[str] = None
    syllabus_topic: Optional[str] = None
    archetype: Optional[str] = None

    def __init__(
        self,
        *args,
        label: Optional[str] = None,
        marks: Optional[int] = None,
        text: Optional[str] = None,
        topic: Any = "",
        solution: str = "",
        marking_scheme: Optional[List[Dict[str, Any]]] = None,
        figure_id: Optional[str] = None,
        image_path: Optional[str] = None,
        figure_caption: Optional[str] = None,
        image_sha256: Optional[str] = None,
        bloom: str = "",
        co: str = "",
        evidence_block_ids: Tuple[str, ...] = (),
        concept_footprint: Optional[ConceptFootprint] = None,
        expected_response: Optional[ExpectedResponseContract] = None,
        visual_task: Optional[VisualTaskContract] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        subpart_index: Optional[int] = None,
        question_id: Optional[str] = None,
        module_id: Optional[str] = None,
        target_cognitive_level: Optional[str] = None,
        syllabus_topic: Optional[str] = None,
        archetype: Optional[str] = None,
        course_outcome: Optional[str] = None,
        **kwargs,
    ):
        if len(args) > 0:
            if isinstance(args[0], int):
                # Pattern: (subpart_index, label, text, marks, bloom, co, target_cognitive_level, syllabus_topic, archetype, figure_id, image_path, image_sha256)
                subpart_index = args[0]
                if len(args) > 1: label = args[1]
                if len(args) > 2: text = args[2]
                if len(args) > 3: marks = args[3]
                if len(args) > 4: bloom = args[4]
                if len(args) > 5: co = args[5]
                if len(args) > 6: target_cognitive_level = args[6]
                if len(args) > 7: topic = syllabus_topic = args[7]
                if len(args) > 8: archetype = args[8]
                if len(args) > 9: figure_id = args[9]
                if len(args) > 10: image_path = args[10]
                if len(args) > 11: image_sha256 = args[11]
            else:
                # Pattern: (label, marks, text, topic, solution, marking_scheme, figure_id, image_path, figure_caption, image_sha256, bloom, co)
                label = args[0]
                if len(args) > 1: marks = args[1]
                if len(args) > 2: text = args[2]
                if len(args) > 3: topic = args[3]
                if len(args) > 4: solution = args[4]
                if len(args) > 5: marking_scheme = args[5]
                if len(args) > 6: figure_id = args[6]
                if len(args) > 7: image_path = args[7]
                if len(args) > 8: figure_caption = args[8]
                if len(args) > 9: image_sha256 = args[9]
                if len(args) > 10: bloom = args[10]
                if len(args) > 11: co = args[11]

        self.label = label if label is not None else ""
        self.marks = marks if marks is not None else 0
        self.text = text if text is not None else ""
        self.topic = topic
        self.solution = solution if solution is not None else ""
        self.marking_scheme = marking_scheme if marking_scheme is not None else []
        self.figure_id = figure_id
        self.image_path = image_path
        self.figure_caption = figure_caption
        self.image_sha256 = image_sha256
        self.bloom = bloom or ""
        self.co = co or (course_outcome or "")
        self.evidence_block_ids = tuple(evidence_block_ids) if evidence_block_ids else ()
        self.concept_footprint = concept_footprint
        self.expected_response = expected_response
        self.visual_task = visual_task
        self.telemetry = dict(telemetry) if telemetry is not None else {}
        self.subpart_index = subpart_index
        self.question_id = question_id
        self.module_id = module_id
        self.target_cognitive_level = target_cognitive_level
        self.syllabus_topic = syllabus_topic or self.topic
        self.archetype = archetype
        self.__post_init__()

    def materialize_marking_scheme(self, raw_scheme: Any) -> None:
        """Helper to assign and wrap legacy prose or structured scheme."""
        if isinstance(raw_scheme, str):
            self.marking_scheme = [{"criterion": raw_scheme.strip(), "marks": self.marks, "is_legacy_string": True}]
        elif isinstance(raw_scheme, list):
            self.marking_scheme = list(raw_scheme)

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
        if type(self.bloom) is not str:
            return False, f"SubPart.bloom must be exact str, got {type(self.bloom).__name__}"
        if type(self.co) is not str:
            return False, f"SubPart.co must be exact str, got {type(self.co).__name__}"

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

        if self.figure_id is not None and type(self.figure_id) is not str:
            return False, f"SubPart.figure_id must be str or None, got {type(self.figure_id).__name__}"
        if self.image_path is not None and type(self.image_path) is not str:
            return False, f"SubPart.image_path must be str or None, got {type(self.image_path).__name__}"
        if self.image_sha256 is not None and type(self.image_sha256) is not str:
            return False, f"SubPart.image_sha256 must be str or None, got {type(self.image_sha256).__name__}"

        if self.evidence_block_ids is not None:
            if not isinstance(self.evidence_block_ids, (tuple, list)):
                return False, f"SubPart.evidence_block_ids must be tuple, got {type(self.evidence_block_ids).__name__}"
            for b_id in self.evidence_block_ids:
                if not isinstance(b_id, str):
                    return False, f"SubPart.evidence_block_ids element must be str, got {type(b_id).__name__}"

        if self.concept_footprint is not None and not isinstance(self.concept_footprint, ConceptFootprint):
            return False, f"SubPart.concept_footprint must be ConceptFootprint or None, got {type(self.concept_footprint).__name__}"

        if self.expected_response is not None and not isinstance(self.expected_response, ExpectedResponseContract):
            return False, f"SubPart.expected_response must be ExpectedResponseContract or None, got {type(self.expected_response).__name__}"

        if self.visual_task is not None and not isinstance(self.visual_task, VisualTaskContract):
            return False, f"SubPart.visual_task must be VisualTaskContract or None, got {type(self.visual_task).__name__}"

        if not isinstance(self.telemetry, dict):
            return False, f"SubPart.telemetry must be dict, got {type(self.telemetry).__name__}"

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
        if self.evidence_block_ids is not None:
            self.evidence_block_ids = tuple(self.evidence_block_ids)

    def to_dict(self) -> Dict[str, Any]:
        """Convert SubPart to plain dictionary representation."""
        return {
            "label": self.label,
            "marks": self.marks,
            "text": self.text,
            "topic": self.topic,
            "solution": self.solution,
            "marking_scheme": copy.deepcopy(self.marking_scheme),
            "figure_id": self.figure_id,
            "image_path": self.image_path,
            "figure_caption": self.figure_caption,
            "image_sha256": self.image_sha256,
            "bloom": self.bloom,
            "co": self.co,
            "evidence_block_ids": list(self.evidence_block_ids) if self.evidence_block_ids else [],
            "concept_footprint": self.concept_footprint.to_dict() if self.concept_footprint is not None else None,
            "expected_response": self.expected_response.to_dict() if self.expected_response is not None else None,
            "visual_task": self.visual_task.to_dict() if self.visual_task is not None else None,
            "telemetry": copy.deepcopy(self.telemetry),
        }


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

    def __init__(
        self,
        *args,
        slot_id: str = "",
        module_id: str = "",
        global_q_idx: int = 1,
        question_text: str = "",
        solution: str = "",
        marking_scheme: Optional[List[Dict[str, Any]]] = None,
        marks: int = 0,
        partition: Optional[List[int]] = None,
        bloom: str = "L2",
        co: str = "CO1",
        topic: str = "",
        evidence_summary: str = "",
        references_image: bool = False,
        image_path: Optional[str] = None,
        figure_caption: Optional[str] = None,
        diagram_request: Optional[Dict[str, Any]] = None,
        generation_source: str = "local_llm",
        attempts: int = 1,
        telemetry: Optional[Dict[str, Any]] = None,
        is_unresolved: bool = False,
        unresolved_scope: Optional[str] = None,
        unresolved_subparts: Tuple[str, ...] = (),
        failure_code: Optional[str] = None,
        failure_detail: Optional[str] = None,
        sub_parts: Optional[List[SubPart]] = None,
        figure_id: Optional[str] = None,
        image_sha256: Optional[str] = None,
        evidence_block_ids: Tuple[str, ...] = (),
        concept_footprints: Tuple[Dict[str, Any], ...] = (),
        expected_responses: Tuple[Dict[str, Any], ...] = (),
        visual_tasks: Tuple[Dict[str, Any], ...] = (),
        choice_group_id: Optional[str] = None,
        choice_variant: Optional[str] = None,
        primary_archetype: Optional[str] = None,
        question_id: Optional[str] = None,
        subpart_choice_rule: Optional[Dict[str, Any]] = None,
        concept_footprint: Optional[Any] = None,
        **kwargs,
    ):
        if len(args) > 0:
            if len(args) >= 3 and isinstance(args[2], int):
                slot_id = args[0]
                module_id = args[1]
                global_q_idx = args[2]
                if len(args) > 3: question_text = args[3]
                if len(args) > 4: solution = args[4]
                if len(args) > 5: marking_scheme = args[5]
                if len(args) > 6: marks = args[6]
                if len(args) > 7: partition = args[7]
                if len(args) > 8: bloom = args[8]
                if len(args) > 9: co = args[9]
                if len(args) > 10: topic = args[10]
                if len(args) > 11: evidence_summary = args[11]
            else:
                question_id = args[0]
                slot_id = args[1] if len(args) > 1 else args[0]
                if len(args) > 2: question_text = args[2]
                if len(args) > 3: marks = args[3]
                if len(args) > 4: bloom = args[4]

        self._question_id = question_id or (args[0] if len(args) > 0 and not (len(args) >= 3 and isinstance(args[2], int)) else None)
        self.slot_id = slot_id or (self._question_id or "")
        self.module_id = module_id or "module_1"
        self.global_q_idx = global_q_idx
        self.question_text = question_text or ""
        self.solution = solution or ""
        self.marking_scheme = list(marking_scheme) if marking_scheme is not None else []
        self.marks = marks

        if partition is not None:
            self.partition = list(partition)
        else:
            self.partition = []

        self.bloom = bloom or "L2"
        self.co = co or "CO1"
        self.topic = topic or ""
        self.evidence_summary = evidence_summary or ""
        self.references_image = references_image
        self.image_path = image_path
        self.figure_caption = figure_caption
        self.diagram_request = diagram_request
        self.generation_source = generation_source
        self.attempts = attempts
        self.telemetry = dict(telemetry) if telemetry is not None else {}
        self.is_unresolved = is_unresolved
        self.unresolved_scope = unresolved_scope
        self.unresolved_subparts = tuple(unresolved_subparts)
        self.failure_code = failure_code
        self.failure_detail = failure_detail
        self.choice_group_id = choice_group_id
        self.choice_variant = choice_variant
        self.primary_archetype = primary_archetype
        self.subpart_choice_rule = subpart_choice_rule
        self.concept_footprint = concept_footprint

        if sub_parts is not None:
            _require_exact_type(sub_parts, list, "sub_parts")
            for i, sp in enumerate(sub_parts):
                if type(sp) is not SubPart:
                    raise TypeError(f"sub_parts[{i}] must be exact SubPart, got {type(sp).__name__}")
            self.sub_parts = list(sub_parts)
            self._has_structured_subparts = True
            if len(self.sub_parts) == 1:
                if not getattr(self.sub_parts[0], "solution", None) and self.solution:
                    self.sub_parts[0].solution = self.solution
                if not getattr(self.sub_parts[0], "text", None) and self.question_text:
                    self.sub_parts[0].text = self.question_text
            if self.marks == 0:
                self.marks = sum(getattr(sp, "marks", 0) for sp in self.sub_parts)
            if partition is None:
                self.partition = [getattr(sp, "marks", 0) for sp in self.sub_parts]
        else:
            self.sub_parts = []
            self._has_structured_subparts = False

        self._figure_id = figure_id
        self._image_sha256 = image_sha256
        self._question_id = question_id
        self.evidence_block_ids = tuple(evidence_block_ids) if evidence_block_ids else ()
        self.concept_footprints = tuple(concept_footprints) if concept_footprints else ()
        self.expected_responses = tuple(expected_responses) if expected_responses else ()
        self.visual_tasks = tuple(visual_tasks) if visual_tasks else ()
        self.__post_init__()

    @property
    def question_id(self) -> str:
        return self._question_id or self.slot_id

    @question_id.setter
    def question_id(self, val: str) -> None:
        self._question_id = val

    @property
    def figure_id(self) -> Optional[str]:
        if getattr(self, "_figure_id", None) is not None:
            return self._figure_id
        for sp in (self.sub_parts or []):
            f_id = getattr(sp, "figure_id", None) if not isinstance(sp, dict) else sp.get("figure_id")
            if f_id:
                return f_id
        return None

    @figure_id.setter
    def figure_id(self, val: Optional[str]) -> None:
        self._figure_id = val

    @property
    def image_sha256(self) -> Optional[str]:
        if getattr(self, "_image_sha256", None) is not None:
            return self._image_sha256
        for sp in (self.sub_parts or []):
            h = getattr(sp, "image_sha256", None) if not isinstance(sp, dict) else sp.get("image_sha256")
            if h:
                return h
        return None

    @image_sha256.setter
    def image_sha256(self, val: Optional[str]) -> None:
        self._image_sha256 = val

    def __post_init__(self) -> None:
        """Validate container and item types for subparts."""
        if self.sub_parts is not None:
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
                if isinstance(entry, str):
                    # Legacy string entry wrapped
                    item = {"criterion": f"({sp.label}) {entry}".strip(), "marks": sp.marks, "is_legacy_string": True, "subpart": sp.label}
                    combined.append(item)
                    continue
                if not isinstance(entry, dict):
                    sp.telemetry.setdefault("malformed_rubric_entries", []).append(entry)
                    continue
                marks_val = entry.get("marks")
                if type(marks_val) is not int or isinstance(marks_val, bool) or marks_val <= 0:
                    sp.telemetry.setdefault("malformed_rubric_entries", []).append(entry)
                    continue
                crit_text = entry.get("criterion")
                if not isinstance(crit_text, str) or not crit_text.strip():
                    sp.telemetry.setdefault("malformed_rubric_entries", []).append(entry)
                    continue

                item = dict(entry)  # Preserve all criterion metadata fields
                crit_prefix = f"({sp.label}) " if (len(self.sub_parts) > 1 or self.is_multi_part) else ""
                item["criterion"] = f"{crit_prefix}{crit_text}".strip()
                item["subpart"] = sp.label
                combined.append(item)
        return combined

    def sync_materialized_projections(self) -> None:
        """Synchronize question_text, solution, marking_scheme, evidence_block_ids, concept_footprints, expected_responses, visual_tasks from authoritative sub_parts."""
        if self.sub_parts:
            self.question_text = self.materialize_question_text()
            self.solution = self.materialize_solution()
            self.marking_scheme = self.materialize_marking_scheme()

            seen = set()
            ev_ids = []
            for sp in self.sub_parts:
                for b_id in (getattr(sp, "evidence_block_ids", None) or ()):
                    if b_id not in seen:
                        seen.add(b_id)
                        ev_ids.append(b_id)
            self.evidence_block_ids = tuple(ev_ids)

            cf_list = []
            for sp in self.sub_parts:
                cf = getattr(sp, "concept_footprint", None)
                if cf is not None:
                    cf_list.append(cf.to_dict() if hasattr(cf, "to_dict") else dict(cf))
            self.concept_footprints = tuple(cf_list)

            er_list = []
            for sp in self.sub_parts:
                er = getattr(sp, "expected_response", None)
                if er is not None:
                    er_list.append(er.to_dict() if hasattr(er, "to_dict") else dict(er))
            self.expected_responses = tuple(er_list)

            vt_list = []
            for sp in self.sub_parts:
                vt = getattr(sp, "visual_task", None)
                if vt is not None:
                    vt_list.append(vt.to_dict() if hasattr(vt, "to_dict") else dict(vt))
            self.visual_tasks = tuple(vt_list)
        else:
            self.evidence_block_ids = ()
            self.concept_footprints = ()
            self.expected_responses = ()
            self.visual_tasks = ()

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
        d = asdict(self)
        d.pop("_question_id", None)
        if getattr(self, "_figure_id", None) is not None:
            d["figure_id"] = self._figure_id
        if getattr(self, "_image_sha256", None) is not None:
            d["image_sha256"] = self._image_sha256
        d["question_id"] = self.question_id
        if self.sub_parts:
            d["sub_parts"] = [sp.to_dict() for sp in self.sub_parts]
        if getattr(self, "evidence_block_ids", None):
            d["evidence_block_ids"] = list(self.evidence_block_ids)
        if getattr(self, "concept_footprints", None):
            d["concept_footprints"] = list(self.concept_footprints)
        if getattr(self, "expected_responses", None):
            d["expected_responses"] = list(self.expected_responses)
        if getattr(self, "visual_tasks", None):
            d["visual_tasks"] = list(self.visual_tasks)
        if getattr(self, "choice_group_id", None) is not None:
            d["choice_group_id"] = self.choice_group_id
        if getattr(self, "choice_variant", None) is not None:
            d["choice_variant"] = self.choice_variant
        if getattr(self, "primary_archetype", None) is not None:
            d["primary_archetype"] = self.primary_archetype
        if getattr(self, "subpart_choice_rule", None) is not None:
            d["subpart_choice_rule"] = self.subpart_choice_rule
        return d

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
        if self.sub_parts is not None:
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
            "failure_code", "failure_detail", "sub_parts",
            "figure_id", "image_sha256",
            "evidence_block_ids", "concept_footprints", "expected_responses", "visual_tasks",
            "question_id", "choice_group_id", "choice_variant", "primary_archetype", "subpart_choice_rule"
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

        raw_fig_id = data.get("figure_id")
        figure_id = _require_str(raw_fig_id, "figure_id") if raw_fig_id is not None else None

        raw_img_sha = data.get("image_sha256")
        image_sha256 = _require_str(raw_img_sha, "image_sha256") if raw_img_sha is not None else None

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
            hydrated_subparts = None
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

            allowed_sp_keys = {
                "label", "marks", "text", "topic", "solution", "marking_scheme",
                "figure_id", "image_path", "figure_caption", "image_sha256",
                "bloom", "co",
                "evidence_block_ids", "concept_footprint", "expected_response", "visual_task", "telemetry",
                "subpart_index", "question_id", "module_id", "target_cognitive_level", "syllabus_topic", "archetype"
            }
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

                    raw_sp_ev_ids = sp_data.get("evidence_block_ids")
                    if raw_sp_ev_ids is not None:
                        if type(raw_sp_ev_ids) not in (list, tuple):
                            raise TypeError(f"sub_parts[{i}].evidence_block_ids must be list or tuple, got {type(raw_sp_ev_ids).__name__}")
                        sp_ev_ids_list = []
                        for eid in raw_sp_ev_ids:
                            if not isinstance(eid, str):
                                raise TypeError(f"sub_parts[{i}].evidence_block_ids element must be str, got {type(eid).__name__}")
                            sp_ev_ids_list.append(eid)
                        sp_ev_ids = tuple(sp_ev_ids_list)
                    else:
                        sp_ev_ids = ()

                    raw_cf = sp_data.get("concept_footprint")
                    sp_cf = ConceptFootprint.from_dict(raw_cf) if raw_cf is not None else None

                    raw_er = sp_data.get("expected_response")
                    sp_er = ExpectedResponseContract.from_dict(raw_er) if raw_er is not None else None

                    raw_vt = sp_data.get("visual_task")
                    sp_vt = VisualTaskContract.from_dict(raw_vt) if raw_vt is not None else None

                    raw_sp_telem = sp_data.get("telemetry", {})
                    if not isinstance(raw_sp_telem, dict):
                        raise TypeError(f"sub_parts[{i}].telemetry must be dict, got {type(raw_sp_telem).__name__}")
                    sp_telem = dict(raw_sp_telem)

                    hydrated_subparts.append(SubPart(
                        label=sp_label,
                        marks=sp_marks,
                        text=sp_text,
                        topic=sp_topic,
                        solution=sp_sol,
                        marking_scheme=sp_scheme_validated,
                        figure_id=sp_data.get("figure_id"),
                        image_path=sp_data.get("image_path"),
                        figure_caption=sp_data.get("figure_caption"),
                        image_sha256=sp_data.get("image_sha256"),
                        bloom=_require_str(sp_data.get("bloom", ""), f"sub_parts[{i}].bloom", allow_empty=True),
                        co=_require_str(sp_data.get("co", ""), f"sub_parts[{i}].co", allow_empty=True),
                        evidence_block_ids=sp_ev_ids,
                        concept_footprint=sp_cf,
                        expected_response=sp_er,
                        visual_task=sp_vt,
                        telemetry=sp_telem,
                        subpart_index=sp_data.get("subpart_index"),
                        question_id=sp_data.get("question_id"),
                        module_id=sp_data.get("module_id"),
                        target_cognitive_level=sp_data.get("target_cognitive_level"),
                        syllabus_topic=sp_data.get("syllabus_topic"),
                        archetype=sp_data.get("archetype"),
                    ))
                else:
                    raise TypeError(f"sub_parts[{i}] must be a dict or SubPart, got {type(sp_data).__name__}")

            # Since 'sub_parts' was explicitly provided, validation enforces standard structured rules
            validate_allow_legacy = False

        raw_top_ev = data.get("evidence_block_ids", ())
        if type(raw_top_ev) not in (list, tuple):
            raise TypeError("evidence_block_ids must be list or tuple")
        top_ev_ids = tuple(raw_top_ev)

        raw_top_cf = data.get("concept_footprints", ())
        if type(raw_top_cf) not in (list, tuple):
            raise TypeError("concept_footprints must be list or tuple")
        top_cfs = tuple(raw_top_cf)

        raw_top_er = data.get("expected_responses", ())
        if type(raw_top_er) not in (list, tuple):
            raise TypeError("expected_responses must be list or tuple")
        top_ers = tuple(raw_top_er)

        raw_top_vt = data.get("visual_tasks", ())
        if type(raw_top_vt) not in (list, tuple):
            raise TypeError("visual_tasks must be list or tuple")
        top_vts = tuple(raw_top_vt)

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
            figure_id=figure_id,
            image_sha256=image_sha256,
            evidence_block_ids=top_ev_ids,
            concept_footprints=top_cfs,
            expected_responses=top_ers,
            visual_tasks=top_vts,
            question_id=data.get("question_id"),
            choice_group_id=data.get("choice_group_id"),
            choice_variant=data.get("choice_variant"),
            primary_archetype=data.get("primary_archetype"),
            subpart_choice_rule=data.get("subpart_choice_rule"),
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
