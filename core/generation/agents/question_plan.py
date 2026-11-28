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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from core.contracts.assessment_blueprint import QuestionSpecification, SubPartSpecification


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

# Harmonized Stage G6 Educational Mapping (CO1-CO5)
CO_BLOOM_BANDS: Dict[str, Tuple[str, ...]] = {
    "CO1": ("L1", "L2", "L3"),
    "CO2": ("L2", "L3", "L4"),
    "CO3": ("L3", "L4", "L5"),
    "CO4": ("L4", "L5", "L6"),
    "CO5": ("L5", "L6"),
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
# Stage G6 Typed Academic Contracts
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RubricCriterion:
    criterion: str
    marks: int

    def __post_init__(self) -> None:
        if not isinstance(self.criterion, str) or not self.criterion.strip():
            raise TypeError("RubricCriterion 'criterion' must be a non-empty string")
        if type(self.marks) is not int or isinstance(self.marks, bool) or self.marks <= 0:
            raise TypeError(f"RubricCriterion 'marks' must be exact positive int, got {type(self.marks).__name__} ({self.marks!r})")

    def to_dict(self) -> Dict[str, Any]:
        return {"criterion": self.criterion, "marks": self.marks}

    @classmethod
    def from_dict(cls, d: Any) -> "RubricCriterion":
        if not isinstance(d, dict):
            raise TypeError(f"RubricCriterion.from_dict expects dict, got {type(d).__name__}")
        extra = set(d.keys()) - {"criterion", "marks"}
        if extra:
            raise ValueError(f"RubricCriterion contains unexpected keys: {extra}")
        if "criterion" not in d or "marks" not in d:
            raise KeyError("RubricCriterion missing required key 'criterion' or 'marks'")
        crit = d["criterion"]
        marks = d["marks"]
        if not isinstance(crit, str) or not crit.strip():
            raise TypeError("RubricCriterion 'criterion' must be a non-empty string")
        if type(marks) is not int or isinstance(marks, bool) or marks <= 0:
            raise TypeError(f"RubricCriterion 'marks' must be exact positive int, got {type(marks).__name__} ({marks!r})")
        return cls(criterion=crit.strip(), marks=marks)


@dataclass
class ConceptFootprint:
    canonical_concept: str = ""
    syllabus_subtopic: str = "General"
    module_id: Any = 1
    bloom: str = "L2"
    response_type: str = "EXPLANATION"
    operation: str = "EXPLAIN"
    provenance_source: str = "KG_CONCEPT"
    provenance_entity_id: str = ""
    provenance_type: Optional[str] = None
    _primary_operation: Optional[str] = None

    def __init__(
        self,
        *args,
        canonical_concept: str = "",
        syllabus_subtopic: str = "General",
        module_id: Any = 1,
        bloom: str = "L2",
        response_type: str = "EXPLANATION",
        operation: str = "EXPLAIN",
        provenance_source: str = "KG_CONCEPT",
        provenance_entity_id: str = "",
        provenance_type: Optional[str] = None,
        primary_operation: Optional[str] = None,
        **kwargs,
    ):
        if kwargs:
            extra = set(kwargs.keys()) - {"_primary_operation"}
            if extra:
                raise TypeError(f"ConceptFootprint.__init__() got unexpected keyword argument: {extra}")
        if len(args) >= 4 and isinstance(args[1], str) and isinstance(args[2], str) and isinstance(args[3], str):
            canonical_concept = args[0]
            operation = args[1]
            response_type = args[2]
            bloom = args[3]
        elif len(args) == 1:
            canonical_concept = args[0]

        if primary_operation:
            operation = primary_operation
        if provenance_type:
            provenance_source = provenance_type

        # Direct construction type validation
        if type(canonical_concept) is not str or not canonical_concept.strip():
            raise TypeError(f"ConceptFootprint 'canonical_concept' must be non-empty string, got {type(canonical_concept).__name__}: {canonical_concept!r}")
        if type(module_id) not in (int, str) or isinstance(module_id, bool):
            raise TypeError(f"ConceptFootprint 'module_id' must be int or str, got {type(module_id).__name__}: {module_id!r}")
        if type(bloom) is not str or bloom not in ("L1", "L2", "L3", "L4", "L5", "L6"):
            raise ValueError(f"ConceptFootprint 'bloom' must be valid Bloom level L1-L6, got {bloom!r}")

        self.canonical_concept = canonical_concept.strip()
        self.syllabus_subtopic = syllabus_subtopic or "General"
        self.module_id = module_id
        self.bloom = bloom or "L2"
        self.response_type = response_type or "EXPLANATION"
        self.operation = operation or "EXPLAIN"
        self.provenance_source = provenance_source or "KG_CONCEPT"
        self.provenance_entity_id = provenance_entity_id or ""
        self.provenance_type = self.provenance_source
        self._primary_operation = self.operation
        self.__post_init__()

    @property
    def primary_operation(self) -> str:
        return self._primary_operation or self.operation

    @primary_operation.setter
    def primary_operation(self, val: str) -> None:
        self._primary_operation = val
        self.operation = val

    def __post_init__(self) -> None:
        if self.provenance_type and (not self.provenance_source or self.provenance_source == "KG_CONCEPT"):
            self.provenance_source = self.provenance_type
        elif self.provenance_source and not self.provenance_type:
            self.provenance_type = self.provenance_source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_concept": self.canonical_concept,
            "syllabus_subtopic": self.syllabus_subtopic,
            "module_id": self.module_id,
            "bloom": self.bloom,
            "response_type": self.response_type,
            "operation": self.operation,
            "provenance_source": self.provenance_source,
            "provenance_entity_id": self.provenance_entity_id,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "ConceptFootprint":
        if not isinstance(d, dict):
            raise TypeError(f"ConceptFootprint.from_dict expects dict, got {type(d).__name__}")
        allowed_keys = {
            "canonical_concept", "syllabus_subtopic", "module_id", "bloom",
            "response_type", "operation", "provenance_source", "provenance_entity_id",
            "provenance_type", "primary_operation", "_primary_operation",
        }
        extra = set(d.keys()) - allowed_keys
        if extra:
            raise ValueError(f"ConceptFootprint contains unexpected keys: {extra}")

        can_c = d.get("canonical_concept")
        if not isinstance(can_c, str) or not can_c.strip():
            raise TypeError("ConceptFootprint 'canonical_concept' must be non-empty string")

        subtop = d.get("syllabus_subtopic", "General")
        mod_id = d.get("module_id", 1)
        if type(mod_id) not in (int, str) or isinstance(mod_id, bool):
            raise TypeError(f"ConceptFootprint 'module_id' must be int or str, got {mod_id!r}")

        bl = d.get("bloom", "L2")
        rt = d.get("response_type", "EXPLANATION")
        op = d.get("operation") or d.get("primary_operation") or d.get("_primary_operation", "EXPLAIN")
        ps = d.get("provenance_source") or d.get("provenance_type") or "KG_CONCEPT"
        pe = d.get("provenance_entity_id", "")

        return cls(
            canonical_concept=can_c.strip(),
            syllabus_subtopic=str(subtop).strip(),
            module_id=mod_id,
            bloom=str(bl).strip(),
            response_type=str(rt).strip(),
            operation=str(op).strip(),
            provenance_source=str(ps).strip(),
            provenance_entity_id=str(pe).strip(),
            provenance_type=str(ps).strip(),
            primary_operation=str(op).strip(),
        )


@dataclass(frozen=True)
class ExpectedResponseContract:
    response_type: str = "PROSE"
    required_elements: Tuple[str, ...] = ()
    expected_operations: Tuple[str, ...] = ()
    estimated_steps: Optional[int] = None
    solution_key: Optional[str] = None
    estimated_time_minutes: Optional[int] = None
    expected_milestones: Tuple[str, ...] = ()

    def __init__(
        self,
        *args,
        response_type: str = "PROSE",
        required_elements: Tuple[str, ...] = (),
        expected_operations: Tuple[str, ...] = (),
        estimated_steps: Optional[int] = None,
        solution_key: Optional[str] = None,
        estimated_time_minutes: Optional[int] = None,
        expected_milestones: Tuple[str, ...] = (),
        **kwargs,
    ):
        if kwargs:
            unexpected = set(kwargs.keys()) - {"estimated_time_minutes", "expected_milestones"}
            if unexpected:
                raise TypeError(f"ExpectedResponseContract.__init__() got unexpected keyword argument: {unexpected}")
            if "estimated_time_minutes" in kwargs:
                estimated_time_minutes = kwargs["estimated_time_minutes"]
            if "expected_milestones" in kwargs:
                expected_milestones = tuple(kwargs["expected_milestones"])
        if len(args) == 2 and isinstance(args[0], (tuple, list)) and isinstance(args[1], (tuple, list)):
            required_elements = tuple(args[0])
            expected_operations = tuple(args[1])
        elif len(args) >= 3 and isinstance(args[0], str):
            response_type = args[0]
            required_elements = tuple(args[1])
            expected_operations = tuple(args[2])
            if len(args) > 3: estimated_steps = args[3]
            if len(args) > 4: solution_key = args[4]
            if len(args) > 5: estimated_time_minutes = args[5]
            if len(args) > 6: expected_milestones = tuple(args[6])
        elif len(args) == 1 and isinstance(args[0], str):
            response_type = args[0]

        if expected_milestones and not required_elements:
            required_elements = expected_milestones

        # Direct construction type validation
        if type(response_type) is not str or not response_type.strip():
            raise TypeError(f"ExpectedResponseContract 'response_type' must be non-empty string, got {type(response_type).__name__}: {response_type!r}")
        if type(required_elements) not in (tuple, list):
            raise TypeError("ExpectedResponseContract 'required_elements' must be tuple or list")
        for elem in required_elements:
            if type(elem) is not str or not elem.strip():
                raise TypeError(f"required_elements element must be non-empty string, got {type(elem).__name__}: {elem!r}")
        if type(expected_operations) not in (tuple, list):
            raise TypeError("ExpectedResponseContract 'expected_operations' must be tuple or list")
        for op in expected_operations:
            if type(op) is not str or not op.strip():
                raise TypeError(f"expected_operations element must be non-empty string, got {type(op).__name__}: {op!r}")
        if estimated_steps is not None:
            if type(estimated_steps) is not int or isinstance(estimated_steps, bool) or estimated_steps <= 0:
                raise TypeError(f"estimated_steps must be positive int, got {type(estimated_steps).__name__}: {estimated_steps!r}")
        if solution_key is not None:
            if type(solution_key) is not str or not solution_key.strip():
                raise TypeError(f"solution_key must be non-empty string, got {type(solution_key).__name__}: {solution_key!r}")
        if estimated_time_minutes is not None:
            if type(estimated_time_minutes) is not int or isinstance(estimated_time_minutes, bool) or estimated_time_minutes <= 0:
                raise TypeError(f"estimated_time_minutes must be positive int, got {type(estimated_time_minutes).__name__}: {estimated_time_minutes!r}")

        object.__setattr__(self, "response_type", response_type)
        object.__setattr__(self, "required_elements", tuple(required_elements))
        object.__setattr__(self, "expected_operations", tuple(expected_operations))
        object.__setattr__(self, "estimated_steps", estimated_steps)
        object.__setattr__(self, "solution_key", solution_key)
        object.__setattr__(self, "estimated_time_minutes", estimated_time_minutes)
        object.__setattr__(self, "expected_milestones", tuple(expected_milestones))

    def to_dict(self) -> Dict[str, Any]:
        res: Dict[str, Any] = {
            "response_type": self.response_type,
            "required_elements": list(self.required_elements),
            "expected_operations": list(self.expected_operations),
        }
        if self.estimated_steps is not None:
            res["estimated_steps"] = self.estimated_steps
        if self.solution_key is not None:
            res["solution_key"] = self.solution_key
        if self.estimated_time_minutes is not None:
            res["estimated_time_minutes"] = self.estimated_time_minutes
        if self.expected_milestones:
            res["expected_milestones"] = list(self.expected_milestones)
        return res

    @classmethod
    def from_dict(cls, d: Any) -> "ExpectedResponseContract":
        if not isinstance(d, dict):
            raise TypeError(f"ExpectedResponseContract expects dict, got {type(d).__name__}")
        allowed = {"response_type", "required_elements", "expected_operations", "estimated_steps", "solution_key"}
        extra = set(d.keys()) - allowed
        if extra:
            raise ValueError(f"ExpectedResponseContract contains unexpected keys: {extra}")

        # Check container types before existence check so bad types raise TypeError
        if "required_elements" in d and type(d["required_elements"]) not in (tuple, list):
            raise TypeError("ExpectedResponseContract 'required_elements' must be exact tuple or list")
        if "expected_operations" in d and type(d["expected_operations"]) not in (tuple, list):
            raise TypeError("ExpectedResponseContract 'expected_operations' must be exact tuple or list")

        if "response_type" not in d or "required_elements" not in d or "expected_operations" not in d:
            raise KeyError("ExpectedResponseContract missing required fields")

        resp_type = d["response_type"]
        if not isinstance(resp_type, str) or not resp_type.strip():
            raise TypeError("ExpectedResponseContract 'response_type' must be non-empty string")

        raw_req = d["required_elements"]
        req_list = []
        for elem in raw_req:
            if not isinstance(elem, str) or not elem.strip():
                raise TypeError("required_elements elements must be non-empty strings")
            req_list.append(elem.strip())

        raw_ops = d["expected_operations"]
        ops_list = []
        for op in raw_ops:
            if not isinstance(op, str) or not op.strip():
                raise TypeError("expected_operations elements must be non-empty strings")
            ops_list.append(op.strip())

        est_steps = d.get("estimated_steps")
        if est_steps is not None and (type(est_steps) is not int or isinstance(est_steps, bool) or est_steps <= 0):
            raise TypeError("estimated_steps must be positive int")

        sol_key = d.get("solution_key")
        if sol_key is not None and (not isinstance(sol_key, str) or not sol_key.strip()):
            raise TypeError("solution_key must be non-empty string")

        return cls(
            response_type=resp_type.strip(),
            required_elements=tuple(req_list),
            expected_operations=tuple(ops_list),
            estimated_steps=est_steps,
            solution_key=sol_key.strip() if sol_key else None,
        )


@dataclass(frozen=True)
class VisualTaskContract:
    figure_id: str = ""
    is_essential: bool = True
    task_operation: str = ""
    required_features: Tuple[str, ...] = ()
    prohibited_features: Tuple[str, ...] = ()

    def __init__(
        self,
        *args,
        figure_id: str = "",
        is_essential: bool = True,
        task_operation: str = "",
        required_features: Tuple[str, ...] = (),
        prohibited_features: Tuple[str, ...] = (),
        **kwargs,
    ):
        if kwargs:
            unexpected = set(kwargs.keys())
            raise TypeError(f"VisualTaskContract.__init__() got unexpected keyword argument: {unexpected}")
        if args:
            if len(args) > 5:
                raise TypeError(f"VisualTaskContract takes at most 5 positional arguments, got {len(args)}")
            figure_id = args[0]
            if len(args) >= 2:
                is_essential = args[1]
            if len(args) >= 3:
                task_operation = args[2]
            if len(args) >= 4:
                if type(args[3]) not in (tuple, list):
                    raise TypeError("VisualTaskContract positional 'required_features' must be list or tuple")
                required_features = args[3]
            if len(args) >= 5:
                if type(args[4]) not in (tuple, list):
                    raise TypeError("VisualTaskContract positional 'prohibited_features' must be list or tuple")
                prohibited_features = args[4]

        # Direct construction type validation
        if type(figure_id) is not str or not figure_id.strip():
            raise TypeError(f"VisualTaskContract 'figure_id' must be non-empty string, got {type(figure_id).__name__}: {figure_id!r}")
        if type(is_essential) is not bool:
            raise TypeError(f"VisualTaskContract 'is_essential' must be exact bool, got {type(is_essential).__name__}: {is_essential!r}")
        if type(task_operation) is not str or not task_operation.strip():
            raise TypeError(f"VisualTaskContract 'task_operation' must be non-empty string, got {type(task_operation).__name__}: {task_operation!r}")
        if type(required_features) not in (tuple, list):
            raise TypeError("VisualTaskContract 'required_features' must be exact tuple or list")
        for feat in required_features:
            if type(feat) is not str or not feat.strip():
                raise TypeError(f"required_features elements must be non-empty strings, got {type(feat).__name__}: {feat!r}")
        if type(prohibited_features) not in (tuple, list):
            raise TypeError("VisualTaskContract 'prohibited_features' must be exact tuple or list")
        for pfeat in prohibited_features:
            if type(pfeat) is not str or not pfeat.strip():
                raise TypeError(f"prohibited_features elements must be non-empty strings, got {type(pfeat).__name__}: {pfeat!r}")

        object.__setattr__(self, "figure_id", figure_id)
        object.__setattr__(self, "is_essential", is_essential)
        object.__setattr__(self, "task_operation", task_operation)
        object.__setattr__(self, "required_features", tuple(required_features))
        object.__setattr__(self, "prohibited_features", tuple(prohibited_features))

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "figure_id": self.figure_id,
            "is_essential": self.is_essential,
            "task_operation": self.task_operation,
            "required_features": list(self.required_features),
        }
        if self.prohibited_features:
            d["prohibited_features"] = list(self.prohibited_features)
        return d

    @classmethod
    def from_dict(cls, d: Any) -> "VisualTaskContract":
        if not isinstance(d, dict):
            raise TypeError(f"VisualTaskContract expects dict, got {type(d).__name__}")
        allowed = {"figure_id", "is_essential", "task_operation", "required_features", "prohibited_features"}
        extra = set(d.keys()) - allowed
        if extra:
            raise ValueError(f"VisualTaskContract contains unexpected keys: {extra}")
        if not {"figure_id", "is_essential", "task_operation", "required_features"}.issubset(d.keys()):
            raise KeyError("VisualTaskContract missing required fields")

        fig_id = d["figure_id"]
        if not isinstance(fig_id, str) or not fig_id.strip():
            raise TypeError("VisualTaskContract 'figure_id' must be non-empty string")

        is_ess = d["is_essential"]
        if type(is_ess) is not bool:
            raise TypeError(f"VisualTaskContract 'is_essential' must be exact bool, got {type(is_ess).__name__}")

        op = d["task_operation"]
        if not isinstance(op, str) or not op.strip():
            raise TypeError("VisualTaskContract 'task_operation' must be non-empty string")

        raw_feat = d["required_features"]
        if type(raw_feat) not in (tuple, list):
            raise TypeError("VisualTaskContract 'required_features' must be exact tuple or list")
        feat_list = []
        for feat in raw_feat:
            if not isinstance(feat, str) or not feat.strip():
                raise TypeError("required_features elements must be non-empty strings")
            feat_list.append(feat.strip())

        raw_prohib = d.get("prohibited_features", ())
        if type(raw_prohib) not in (tuple, list):
            raise TypeError("VisualTaskContract 'prohibited_features' must be exact tuple or list")
        prohib_list = []
        for pfeat in raw_prohib:
            if not isinstance(pfeat, str) or not pfeat.strip():
                raise TypeError("prohibited_features elements must be non-empty strings")
            prohib_list.append(pfeat.strip())

        return cls(
            figure_id=fig_id.strip(),
            is_essential=is_ess,
            task_operation=op.strip(),
            required_features=tuple(feat_list),
            prohibited_features=tuple(prohib_list),
        )


VALID_ANSWER_LEAKAGE_STATES = frozenset({"NONE", "POSSIBLE", "CONFIRMED"})
VALID_VISUAL_AUDIT_DECISIONS = frozenset({"PASS", "FAIL", "INDETERMINATE"})


@dataclass(frozen=True)
class VisualAssetAudit:
    """Structured audit contract for Visual Asset Correctness & Answer-Leakage Gate."""
    figure_id: str
    source_document_id: str
    source_revision: str
    image_sha256: str
    module_id: str
    page_number: int
    required_features_verified: bool
    question_alignment_verified: bool
    detected_text: Tuple[str, ...] = ()
    detected_values: Tuple[str, ...] = ()
    detected_equations: Tuple[str, ...] = ()
    detected_annotations: Tuple[str, ...] = ()
    answer_leakage: str = "NONE"
    leakage_evidence: Tuple[str, ...] = ()
    decision: str = "INDETERMINATE"

    def __init__(
        self,
        figure_id: str,
        source_document_id: str,
        source_revision: str,
        image_sha256: str,
        module_id: str,
        page_number: int,
        required_features_verified: bool,
        question_alignment_verified: bool,
        detected_text: Any = (),
        detected_values: Any = (),
        detected_equations: Any = (),
        detected_annotations: Any = (),
        answer_leakage: str = "NONE",
        leakage_evidence: Any = (),
        decision: str = "INDETERMINATE",
    ):
        for fname, fval in (
            ("figure_id", figure_id),
            ("source_document_id", source_document_id),
            ("source_revision", source_revision),
            ("image_sha256", image_sha256),
            ("module_id", module_id),
        ):
            if type(fval) is not str or not fval.strip():
                raise TypeError(f"VisualAssetAudit '{fname}' must be non-empty string, got {type(fval).__name__}: {fval!r}")
        if type(page_number) is not int or isinstance(page_number, bool) or page_number < 1:
            raise TypeError(f"VisualAssetAudit 'page_number' must be positive int, got {type(page_number).__name__}: {page_number!r}")
        if type(required_features_verified) is not bool:
            raise TypeError("VisualAssetAudit 'required_features_verified' must be exact bool")
        if type(question_alignment_verified) is not bool:
            raise TypeError("VisualAssetAudit 'question_alignment_verified' must be exact bool")
        if type(answer_leakage) is not str or answer_leakage not in VALID_ANSWER_LEAKAGE_STATES:
            raise ValueError(f"VisualAssetAudit 'answer_leakage' must be one of {sorted(VALID_ANSWER_LEAKAGE_STATES)}, got {answer_leakage!r}")
        if type(decision) is not str or decision not in VALID_VISUAL_AUDIT_DECISIONS:
            raise ValueError(f"VisualAssetAudit 'decision' must be one of {sorted(VALID_VISUAL_AUDIT_DECISIONS)}, got {decision!r}")

        def _norm_seq(val: Any, name: str) -> Tuple[str, ...]:
            if not isinstance(val, (tuple, list)):
                raise TypeError(f"VisualAssetAudit '{name}' must be tuple or list")
            return tuple(str(x) for x in val)

        object.__setattr__(self, "figure_id", figure_id.strip())
        object.__setattr__(self, "source_document_id", source_document_id.strip())
        object.__setattr__(self, "source_revision", source_revision.strip())
        object.__setattr__(self, "image_sha256", image_sha256.strip())
        object.__setattr__(self, "module_id", module_id.strip())
        object.__setattr__(self, "page_number", page_number)
        object.__setattr__(self, "required_features_verified", required_features_verified)
        object.__setattr__(self, "question_alignment_verified", question_alignment_verified)
        object.__setattr__(self, "detected_text", _norm_seq(detected_text, "detected_text"))
        object.__setattr__(self, "detected_values", _norm_seq(detected_values, "detected_values"))
        object.__setattr__(self, "detected_equations", _norm_seq(detected_equations, "detected_equations"))
        object.__setattr__(self, "detected_annotations", _norm_seq(detected_annotations, "detected_annotations"))
        object.__setattr__(self, "answer_leakage", answer_leakage)
        object.__setattr__(self, "leakage_evidence", _norm_seq(leakage_evidence, "leakage_evidence"))
        object.__setattr__(self, "decision", decision)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "source_document_id": self.source_document_id,
            "source_revision": self.source_revision,
            "image_sha256": self.image_sha256,
            "module_id": self.module_id,
            "page_number": self.page_number,
            "required_features_verified": self.required_features_verified,
            "question_alignment_verified": self.question_alignment_verified,
            "detected_text": list(self.detected_text),
            "detected_values": list(self.detected_values),
            "detected_equations": list(self.detected_equations),
            "detected_annotations": list(self.detected_annotations),
            "answer_leakage": self.answer_leakage,
            "leakage_evidence": list(self.leakage_evidence),
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, d: Any) -> "VisualAssetAudit":
        if not isinstance(d, dict):
            raise TypeError(f"VisualAssetAudit expects dict, got {type(d).__name__}")
        return cls(
            figure_id=d["figure_id"],
            source_document_id=d["source_document_id"],
            source_revision=d["source_revision"],
            image_sha256=d["image_sha256"],
            module_id=d["module_id"],
            page_number=d["page_number"],
            required_features_verified=d["required_features_verified"],
            question_alignment_verified=d["question_alignment_verified"],
            detected_text=d.get("detected_text", ()),
            detected_values=d.get("detected_values", ()),
            detected_equations=d.get("detected_equations", ()),
            detected_annotations=d.get("detected_annotations", ()),
            answer_leakage=d.get("answer_leakage", "NONE"),
            leakage_evidence=d.get("leakage_evidence", ()),
            decision=d.get("decision", "INDETERMINATE"),
        )


@dataclass(frozen=True)
class EvidenceBlockID:
    artifact_id: str
    artifact_rev_sha256: str
    module_id: int
    page: int
    block_index: int
    block_sha256: str

    def to_string(self) -> str:
        return f"{self.artifact_id}::rev:{self.artifact_rev_sha256}::m{self.module_id}::p{self.page}::b{self.block_index}::sha256:{self.block_sha256}"

    @classmethod
    def parse(cls, s: str) -> "EvidenceBlockID":
        parts = s.split("::")
        if len(parts) != 6:
            raise ValueError(f"Invalid EvidenceBlockID format: {s!r}")
        artifact_id = parts[0]
        if not parts[1].startswith("rev:"):
            raise ValueError(f"Invalid revision part in EvidenceBlockID: {parts[1]!r}")
        artifact_rev_sha256 = parts[1][4:]
        if not parts[2].startswith("m"):
            raise ValueError(f"Invalid module part in EvidenceBlockID: {parts[2]!r}")
        module_id = int(parts[2][1:])
        if not parts[3].startswith("p"):
            raise ValueError(f"Invalid page part in EvidenceBlockID: {parts[3]!r}")
        page = int(parts[3][1:])
        if not parts[4].startswith("b"):
            raise ValueError(f"Invalid block part in EvidenceBlockID: {parts[4]!r}")
        block_index = int(parts[4][1:])
        if not parts[5].startswith("sha256:"):
            raise ValueError(f"Invalid sha256 part in EvidenceBlockID: {parts[5]!r}")
        block_sha256 = parts[5][7:]
        return cls(
            artifact_id=artifact_id,
            artifact_rev_sha256=artifact_rev_sha256,
            module_id=module_id,
            page=page,
            block_index=block_index,
            block_sha256=block_sha256
        )


# -----------------------------------------------------------------------------
# SubPartPlan Dataclass
# -----------------------------------------------------------------------------

@dataclass
class SubPartPlan:
    label: str                                      # Strictly single lowercase ASCII 'a'-'z'
    marks: int                                      # Exact marks allocated (> 0, not bool)
    topic: str = ""                                 # Dedicated topic allocated from module KG
    bloom: str = "L2"                               # Target Bloom level (L1-L6)
    co: str = "CO1"                                 # Target Course Outcome (CO1-CO5)
    canonical_concept_id: str = ""                  # Normalized canonical ID from KG
    synonym_ids: List[str] = field(default_factory=list) # Persisted direct synonyms from KG
    concept_neighborhood: List[str] = field(default_factory=list)
    evidence_blocks: List[Any] = field(default_factory=list)
    visual_required: bool = False
    figure: Optional[Any] = None
    specification: Optional[SubPartSpecification] = None  # Stage G2-C: Optional[SubPartSpecification]
    concept_footprint: Optional[ConceptFootprint] = None  # Stage G6
    expected_response: Optional[ExpectedResponseContract] = None  # Stage G6
    visual_task: Optional[VisualTaskContract] = None  # Stage G6
    rubric_criteria: List[RubricCriterion] = field(default_factory=list)  # Stage G6
    candidate_blooms: Optional[List[str]] = None
    course_outcome: Optional[str] = None
    syllabus_topic: Optional[str] = None

    def __post_init__(self) -> None:
        if self.course_outcome and (not self.co or self.co == "CO1"):
            self.co = self.course_outcome
        if self.syllabus_topic and not self.topic:
            self.topic = self.syllabus_topic
        if self.bloom in ("CO1", "CO2", "CO3", "CO4", "CO5") and self.topic in ("L1", "L2", "L3", "L4", "L5", "L6"):
            real_topic = self.co
            real_bloom = self.topic
            real_co = self.bloom
            self.topic = real_topic
            self.bloom = real_bloom
            self.co = real_co
        if not self.topic:
            self.topic = "General"
        if self.specification is not None:
            from core.contracts.assessment_blueprint import SubPartSpecification
            if not isinstance(self.specification, SubPartSpecification):
                raise TypeError(
                    f"SubPartPlan specification must be SubPartSpecification or None, got {type(self.specification).__name__}"
                )
        valid, err = self.validate_fields()
        if not valid:
            if "must be exact" in err or "must be list" in err or "must be SubPartSpecification" in err:
                raise TypeError(err)
            raise ValueError(err)
        self.topic = self.topic.strip()

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
        if type(self.co) is not str or not (self.co in CO_BLOOM_BANDS or self.co.startswith("CO")):
            return False, f"SubPartPlan co must match CO1-CO5, got {self.co!r}"

        if not self.candidate_blooms and self.co in CO_BLOOM_BANDS:
            if self.bloom not in CO_BLOOM_BANDS[self.co]:
                return False, f"SubPartPlan bloom {self.bloom!r} not in CO band for {self.co}: {CO_BLOOM_BANDS[self.co]}"

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

        if self.specification is not None:
            from core.contracts.assessment_blueprint import SubPartSpecification
            if not isinstance(self.specification, SubPartSpecification):
                return False, f"SubPartPlan specification must be SubPartSpecification, got {type(self.specification).__name__}"

        if self.concept_footprint is not None and not isinstance(self.concept_footprint, ConceptFootprint):
            return False, f"SubPartPlan concept_footprint must be ConceptFootprint or None, got {type(self.concept_footprint).__name__}"
        if self.expected_response is not None and not isinstance(self.expected_response, ExpectedResponseContract):
            return False, f"SubPartPlan expected_response must be ExpectedResponseContract or None, got {type(self.expected_response).__name__}"
        if self.visual_task is not None and not isinstance(self.visual_task, VisualTaskContract):
            return False, f"SubPartPlan visual_task must be VisualTaskContract or None, got {type(self.visual_task).__name__}"
        if type(self.rubric_criteria) is not list:
            return False, f"SubPartPlan rubric_criteria must be list, got {type(self.rubric_criteria).__name__}"
        for i, rc in enumerate(self.rubric_criteria):
            if not isinstance(rc, RubricCriterion):
                return False, f"SubPartPlan rubric_criteria[{i}] must be RubricCriterion, got {type(rc).__name__}"

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

            if isinstance(b, str):
                b_id = b
                try:
                    ev_parsed = EvidenceBlockID.parse(b)
                    b_page = ev_parsed.page
                    b_mod = f"module_{ev_parsed.module_id}"
                    b_text = b
                except Exception:
                    b_page = page_start
                    b_mod = expected_mod
                    b_text = b
            else:
                b_id = getattr(b, "id", None) if not isinstance(b, dict) else b.get("id")
                b_text = getattr(b, "text", None) if not isinstance(b, dict) else (b.get("text") or b.get("content"))
                b_page = getattr(b, "page", None) if not isinstance(b, dict) else b.get("page")
                b_mod = getattr(b, "module_id", None) if not isinstance(b, dict) else b.get("module_id")

            if b_mod is None and isinstance(b_id, str) and "::rev:" in b_id:
                try:
                    ev_parsed = EvidenceBlockID.parse(b_id)
                    b_mod = f"module_{ev_parsed.module_id}"
                    if b_page is None:
                        b_page = ev_parsed.page
                except Exception:
                    pass

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


# -----------------------------------------------------------------------------
# QuestionPlan Dataclass
# -----------------------------------------------------------------------------

@dataclass
class QuestionPlan:
    # Identity (fields 1-5; defaultable)
    slot_id: str = ""                    # "module_1_Q1"
    module_id: str = ""                  # "module_1"
    module_idx: int = 1                  # 1..5
    slot_idx: int = 0                    # 0-indexed within module
    global_q_idx: int = 1                # 1..N (1-10 for IAT1)

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
    specification: Optional[QuestionSpecification] = None  # Stage G2-C: Optional[QuestionSpecification]
    choice_group_id: Optional[str] = None                  # Stage G6: internal OR choice block ID
    choice_variant: Optional[str] = None                   # Stage G6: alternative identifier ("Q1", "Q2")

    def __post_init__(self) -> None:
        if self.specification is not None:
            from core.contracts.assessment_blueprint import QuestionSpecification
            if not isinstance(self.specification, QuestionSpecification):
                raise TypeError(
                    f"QuestionPlan specification must be QuestionSpecification or None, got {type(self.specification).__name__}"
                )
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

            if self.specification is not None:
                from core.contracts.assessment_blueprint import QuestionSpecification
                if not isinstance(self.specification, QuestionSpecification):
                    return False, PlanFailureCode.INVALID_CONFIG, f"QuestionPlan specification must be QuestionSpecification, got {type(self.specification).__name__}"

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
