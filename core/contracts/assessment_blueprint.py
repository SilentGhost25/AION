"""
Assessment Blueprint & Task Contracts for AION v3.
Stage G2-C: Canonical Assessment Blueprint, Composable Question Archetypes & Task Contracts.

Provides:
  - UniversalArchetype & CognitiveCapability enums (composable assessment patterns).
  - ArtifactModality & ParameterState enums.
  - ParameterRequirement with traceable provenance.
  - InputArtifactRequirement (problem input data) vs ExpectedResponseContract (student response).
  - MarkingRubricContract with arithmetic reconciliation and assessment validity.
  - SubPartSpecification & QuestionSpecification with hard mark reconciliation.
  - AlternativeGroup & AssessmentSlot capturing OR alternatives without flattening.
  - AssessmentBlueprint defining complete paper structure.
  - evaluate_task_completeness() providing a task-specific answerability gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union


# -----------------------------------------------------------------------------
# 1. Core Enumerations
# -----------------------------------------------------------------------------


class UniversalArchetype(str, Enum):
    """Universal engineering question archetypes (composable patterns)."""
    CONCEPTUAL_EXPLANATION = "CONCEPTUAL_EXPLANATION"
    WORKED_PROBLEM_CALCULATION = "WORKED_PROBLEM_CALCULATION"
    ALGORITHM_PROCEDURAL_TRACE = "ALGORITHM_PROCEDURAL_TRACE"
    COMPARATIVE_ANALYSIS = "COMPARATIVE_ANALYSIS"
    SCHEMATIC_SYSTEM_SYNTHESIS = "SCHEMATIC_SYSTEM_SYNTHESIS"
    EMPIRICAL_MATERIAL_DESIGN = "EMPIRICAL_MATERIAL_DESIGN"
    LOGICAL_FORMULATION_INFERENCE = "LOGICAL_FORMULATION_INFERENCE"
    TECHNO_ECONOMIC_EVALUATION = "TECHNO_ECONOMIC_EVALUATION"


class CognitiveCapability(str, Enum):
    """Specific intellectual operations required by an assessment task."""
    NUMERICAL_SOLVING = "NUMERICAL_SOLVING"
    DERIVATION = "DERIVATION"
    SCHEMATIC_ANALYSIS = "SCHEMATIC_ANALYSIS"
    ALGORITHM_TRACE = "ALGORITHM_TRACE"
    TRADE_OFF_ANALYSIS = "TRADE_OFF_ANALYSIS"
    CODE_COMPLIANCE = "CODE_COMPLIANCE"
    OPTIMIZATION = "OPTIMIZATION"
    DIAGRAMMATIC_SYNTHESIS = "DIAGRAMMATIC_SYNTHESIS"


class ArtifactModality(str, Enum):
    """Representation modality of supplied input problem material."""
    VISUAL_IMAGE = "VISUAL_IMAGE"          # Bitmap/vector figure in ArtifactRegistry
    STRUCTURED_TABLE = "STRUCTURED_TABLE"  # Native tabular matrix/dataset (DOCX table)
    TEXTUAL_INSTANCE = "TEXTUAL_INSTANCE"  # Discrete strings, rules, grammar, facts


class ParameterState(str, Enum):
    """Epistemic state of an engineering parameter."""
    GIVEN = "GIVEN"                        # Explicitly supplied in prompt or table
    DERIVABLE = "DERIVABLE"                # Calculable from given data via governing equations
    STUDENT_SELECTED = "STUDENT_SELECTED"  # Intentionally selected and justified under constraints
    STANDARD_REFERENCED = "STANDARD_REFERENCED" # Looked up from engineering code/standard
    UNSPECIFIED = "UNSPECIFIED"            # Parameter omitted or unassigned


class TaskCompleteness(str, Enum):
    """Result of task-specific answerability evaluation."""
    COMPLETE = "COMPLETE"                  # Sufficient information/constraints exist for solution
    INCOMPLETE = "INCOMPLETE"              # Essential parameter or input demonstrably missing
    INDETERMINATE = "INDETERMINATE"        # Insufficient evidence to establish completeness


# -----------------------------------------------------------------------------
# 2. Parameter & Input Artifact Contracts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParameterRequirement:
    """A single domain parameter with explicit provenance and state."""
    name: str
    state: ParameterState
    value: Optional[Any] = None
    unit: Optional[str] = None
    source_reference: Optional[str] = None
    selection_constraints: Optional[str] = None
    governing_relationship: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ParameterRequirement: name must be a non-empty string")
        if not isinstance(self.state, ParameterState):
            raise TypeError(f"ParameterRequirement: invalid ParameterState {self.state!r}")

        if self.state == ParameterState.GIVEN:
            if self.value is None:
                raise ValueError(f"Parameter {self.name!r} marked GIVEN must have an explicit value, not None")
        elif self.state == ParameterState.STUDENT_SELECTED:
            if not self.selection_constraints or not self.selection_constraints.strip():
                raise ValueError(f"Parameter {self.name!r} marked STUDENT_SELECTED must declare selection_constraints")
        elif self.state == ParameterState.DERIVABLE:
            if not (self.governing_relationship and self.governing_relationship.strip()) and not (self.source_reference and self.source_reference.strip()):
                raise ValueError(f"Parameter {self.name!r} marked DERIVABLE must declare governing_relationship or source_reference")


@dataclass(frozen=True)
class InputArtifactRequirement:
    """Material supplied to the student as part of the problem statement."""
    artifact_id: str
    artifact_type: str                     # e.g., "CIRCUIT_SCHEMATIC", "BEAM_LOADING", "DATASET_TABLE"
    modality: ArtifactModality
    parameters: Tuple[ParameterRequirement, ...] = ()
    required_for_solution: bool = True
    caption_or_reference: Optional[str] = None
    structured_content: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if not self.artifact_id or not isinstance(self.artifact_id, str) or not self.artifact_id.strip():
            raise ValueError("InputArtifactRequirement: artifact_id must be a non-empty string")
        if not self.artifact_type or not isinstance(self.artifact_type, str) or not self.artifact_type.strip():
            raise ValueError("InputArtifactRequirement: artifact_type must be a non-empty string")
        if not isinstance(self.modality, ArtifactModality):
            raise TypeError(f"InputArtifactRequirement: invalid modality {self.modality!r}")
        if type(self.parameters) is not tuple:
            object.__setattr__(self, "parameters", tuple(self.parameters))
        for p in self.parameters:
            if not isinstance(p, ParameterRequirement):
                raise TypeError(f"InputArtifactRequirement: parameter must be ParameterRequirement, got {type(p).__name__}")


# -----------------------------------------------------------------------------
# 3. Rubric & Expected Response Contracts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class RubricCriterion:
    """A distinct scoring milestone in a question or subpart rubric."""
    criterion: str
    marks: int
    required_demonstration: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.criterion or not isinstance(self.criterion, str) or not self.criterion.strip():
            raise ValueError("RubricCriterion: criterion must be a non-empty string")
        if type(self.marks) is not int or isinstance(self.marks, bool) or self.marks <= 0:
            raise ValueError(f"RubricCriterion: marks must be strictly positive int, got {self.marks!r}")


@dataclass(frozen=True)
class MarkingRubricContract:
    """Marking rubric allocating points to demonstrated intellectual steps."""
    criteria: Tuple[RubricCriterion, ...]
    is_assessment_validated: bool = False

    def __post_init__(self) -> None:
        if type(self.criteria) is not tuple:
            object.__setattr__(self, "criteria", tuple(self.criteria))
        if not self.criteria:
            raise ValueError("MarkingRubricContract: criteria must be a non-empty tuple")
        for c in self.criteria:
            if not isinstance(c, RubricCriterion):
                raise TypeError(f"MarkingRubricContract: element must be RubricCriterion, got {type(c).__name__}")

    @property
    def total_marks(self) -> int:
        return sum(c.marks for c in self.criteria)


@dataclass(frozen=True)
class ExpectedResponseContract:
    """Specification of what the student is expected to produce."""
    response_type: str                     # e.g., "STEPWISE_NUMERICAL", "DIAGRAM_WITH_EXPLANATION"
    expected_milestones: Tuple[str, ...] = ()
    requested_artifacts: Tuple[str, ...] = () # e.g. ("STUDENT_DIAGRAM", "SHEAR_FORCE_SKETCH")
    estimated_time_minutes: int = 15

    def __post_init__(self) -> None:
        if not self.response_type or not isinstance(self.response_type, str) or not self.response_type.strip():
            raise ValueError("ExpectedResponseContract: response_type must be a non-empty string")
        if type(self.estimated_time_minutes) is not int or isinstance(self.estimated_time_minutes, bool) or self.estimated_time_minutes <= 0:
            raise ValueError(f"ExpectedResponseContract: estimated_time_minutes must be positive int, got {self.estimated_time_minutes!r}")
        if type(self.expected_milestones) is not tuple:
            object.__setattr__(self, "expected_milestones", tuple(self.expected_milestones))
        if type(self.requested_artifacts) is not tuple:
            object.__setattr__(self, "requested_artifacts", tuple(self.requested_artifacts))


# -----------------------------------------------------------------------------
# 4. Specification Contracts (SubPart & Question)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class SubPartSpecification:
    """Complete assessment task contract for an individual subpart."""
    label: str                             # "a", "b", "c"
    marks: int
    bloom_level: str                       # "L1".."L6"
    primary_archetype: UniversalArchetype
    secondary_archetypes: Tuple[UniversalArchetype, ...] = ()
    required_capabilities: Tuple[CognitiveCapability, ...] = ()
    input_artifacts: Tuple[InputArtifactRequirement, ...] = ()
    expected_response: Optional[ExpectedResponseContract] = None
    marking_rubric: Optional[MarkingRubricContract] = None

    def __post_init__(self) -> None:
        if type(self.label) is not str or len(self.label) != 1 or not ('a' <= self.label <= 'z'):
            raise ValueError(f"SubPartSpecification: label must be single lowercase letter 'a'-'z', got {self.label!r}")
        if type(self.marks) is not int or isinstance(self.marks, bool) or self.marks <= 0:
            raise ValueError(f"SubPartSpecification: marks must be strictly positive int, got {self.marks!r}")
        if self.bloom_level not in ("L1", "L2", "L3", "L4", "L5", "L6"):
            raise ValueError(f"SubPartSpecification: invalid bloom_level {self.bloom_level!r}")
        if not isinstance(self.primary_archetype, UniversalArchetype):
            raise TypeError(f"SubPartSpecification: invalid primary_archetype {self.primary_archetype!r}")

        if type(self.secondary_archetypes) is not tuple:
            object.__setattr__(self, "secondary_archetypes", tuple(self.secondary_archetypes))
        for sa in self.secondary_archetypes:
            if not isinstance(sa, UniversalArchetype):
                raise TypeError(f"SubPartSpecification: secondary_archetype must be UniversalArchetype, got {type(sa).__name__}")

        if type(self.required_capabilities) is not tuple:
            object.__setattr__(self, "required_capabilities", tuple(self.required_capabilities))
        for rc in self.required_capabilities:
            if not isinstance(rc, CognitiveCapability):
                raise TypeError(f"SubPartSpecification: capability must be CognitiveCapability, got {type(rc).__name__}")

        if type(self.input_artifacts) is not tuple:
            object.__setattr__(self, "input_artifacts", tuple(self.input_artifacts))
        for ia in self.input_artifacts:
            if not isinstance(ia, InputArtifactRequirement):
                raise TypeError(f"SubPartSpecification: input_artifact must be InputArtifactRequirement, got {type(ia).__name__}")

        if self.expected_response is not None and not isinstance(self.expected_response, ExpectedResponseContract):
            raise TypeError("SubPartSpecification: expected_response must be ExpectedResponseContract")

        if self.marking_rubric is not None:
            if not isinstance(self.marking_rubric, MarkingRubricContract):
                raise TypeError("SubPartSpecification: marking_rubric must be MarkingRubricContract")
            if self.marking_rubric.total_marks != self.marks:
                raise ValueError(
                    f"SubPartSpecification ({self.label}): marking_rubric marks sum ({self.marking_rubric.total_marks}) "
                    f"!= subpart marks ({self.marks})"
                )


@dataclass(frozen=True)
class QuestionSpecification:
    """Assessment task contract for a full question (or alternative option)."""
    question_id: str
    syllabus_topic: str
    course_outcome: str                    # "CO1".."CO5"
    bloom_level: str                       # "L1".."L6"
    marks: int
    subparts: Tuple[SubPartSpecification, ...]
    primary_archetype: Optional[UniversalArchetype] = None
    secondary_archetypes: Tuple[UniversalArchetype, ...] = ()
    required_capabilities: Tuple[CognitiveCapability, ...] = ()
    input_artifacts: Tuple[InputArtifactRequirement, ...] = ()
    expected_response: Optional[ExpectedResponseContract] = None
    marking_rubric: Optional[MarkingRubricContract] = None
    alternative_group_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.question_id or not isinstance(self.question_id, str) or not self.question_id.strip():
            raise ValueError("QuestionSpecification: question_id must be a non-empty string")
        if not self.syllabus_topic or not isinstance(self.syllabus_topic, str) or not self.syllabus_topic.strip():
            raise ValueError("QuestionSpecification: syllabus_topic must be a non-empty string")
        if not self.course_outcome or not isinstance(self.course_outcome, str) or not self.course_outcome.strip():
            raise ValueError("QuestionSpecification: course_outcome must be a non-empty string")
        if self.bloom_level not in ("L1", "L2", "L3", "L4", "L5", "L6"):
            raise ValueError(f"QuestionSpecification: invalid bloom_level {self.bloom_level!r}")
        if type(self.marks) is not int or isinstance(self.marks, bool) or self.marks <= 0:
            raise ValueError(f"QuestionSpecification: marks must be strictly positive int, got {self.marks!r}")

        if type(self.subparts) is not tuple:
            object.__setattr__(self, "subparts", tuple(self.subparts))
        if not self.subparts:
            raise ValueError("QuestionSpecification: subparts tuple must not be empty")
        for sp in self.subparts:
            if not isinstance(sp, SubPartSpecification):
                raise TypeError(f"QuestionSpecification: subpart must be SubPartSpecification, got {type(sp).__name__}")

        # Invariant 1: Subpart marks sum == question marks
        subparts_sum = sum(sp.marks for sp in self.subparts)
        if subparts_sum != self.marks:
            raise ValueError(
                f"QuestionSpecification ({self.question_id}): subparts marks sum ({subparts_sum}) "
                f"!= question marks ({self.marks})"
            )

        # Invariant 2: Question-level rubric reconciliation if supplied
        if self.marking_rubric is not None:
            if not isinstance(self.marking_rubric, MarkingRubricContract):
                raise TypeError("QuestionSpecification: marking_rubric must be MarkingRubricContract")
            if self.marking_rubric.total_marks != self.marks:
                raise ValueError(
                    f"QuestionSpecification ({self.question_id}): question rubric sum ({self.marking_rubric.total_marks}) "
                    f"!= question marks ({self.marks})"
                )


# -----------------------------------------------------------------------------
# 5. Blueprint & Slot Contracts
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class AlternativeGroup:
    """A set of mutually exclusive alternative questions under a selection rule."""
    group_id: str
    alternatives: Tuple[QuestionSpecification, ...]
    selection_rule: str = "SELECT_ONE"

    def __post_init__(self) -> None:
        if not self.group_id or not isinstance(self.group_id, str) or not self.group_id.strip():
            raise ValueError("AlternativeGroup: group_id must be a non-empty string")
        if self.selection_rule not in ("SELECT_ONE", "COMPULSORY"):
            raise ValueError(f"AlternativeGroup: invalid selection_rule {self.selection_rule!r}")

        if type(self.alternatives) is not tuple:
            object.__setattr__(self, "alternatives", tuple(self.alternatives))
        if not self.alternatives:
            raise ValueError("AlternativeGroup: alternatives must not be empty")

        for alt in self.alternatives:
            if not isinstance(alt, QuestionSpecification):
                raise TypeError(f"AlternativeGroup: alternative must be QuestionSpecification, got {type(alt).__name__}")

        # Invariant: All alternatives in an AlternativeGroup must have identical marks
        first_marks = self.alternatives[0].marks
        for alt in self.alternatives[1:]:
            if alt.marks != first_marks:
                raise ValueError(
                    f"AlternativeGroup ({self.group_id}): mark mismatch between alternatives "
                    f"({alt.question_id} has {alt.marks}M vs {first_marks}M)"
                )


@dataclass(frozen=True)
class AssessmentSlot:
    """An assessment slot representing a main question (with its OR alternatives)."""
    slot_id: str
    marks: int
    course_outcome: str
    bloom_level: str
    groups: Tuple[AlternativeGroup, ...]

    def __post_init__(self) -> None:
        if not self.slot_id or not isinstance(self.slot_id, str) or not self.slot_id.strip():
            raise ValueError("AssessmentSlot: slot_id must be a non-empty string")
        if type(self.marks) is not int or isinstance(self.marks, bool) or self.marks <= 0:
            raise ValueError(f"AssessmentSlot: marks must be strictly positive int, got {self.marks!r}")
        if not self.course_outcome or not isinstance(self.course_outcome, str) or not self.course_outcome.strip():
            raise ValueError("AssessmentSlot: course_outcome must be non-empty string")
        if self.bloom_level not in ("L1", "L2", "L3", "L4", "L5", "L6"):
            raise ValueError(f"AssessmentSlot: invalid bloom_level {self.bloom_level!r}")

        if type(self.groups) is not tuple:
            object.__setattr__(self, "groups", tuple(self.groups))
        if not self.groups:
            raise ValueError("AssessmentSlot: groups must not be empty")

        for grp in self.groups:
            if not isinstance(grp, AlternativeGroup):
                raise TypeError(f"AssessmentSlot: group must be AlternativeGroup, got {type(grp).__name__}")
            for alt in grp.alternatives:
                if alt.marks != self.marks:
                    raise ValueError(
                        f"AssessmentSlot ({self.slot_id}): alternative {alt.question_id} marks ({alt.marks}) "
                        f"!= slot marks ({self.marks})"
                    )
                if alt.course_outcome != self.course_outcome:
                    raise ValueError(
                        f"AssessmentSlot ({self.slot_id}): alternative {alt.question_id} CO ({alt.course_outcome}) "
                        f"!= slot CO ({self.course_outcome})"
                    )
                if alt.bloom_level != self.bloom_level:
                    raise ValueError(
                        f"AssessmentSlot ({self.slot_id}): alternative {alt.question_id} Bloom ({alt.bloom_level}) "
                        f"!= slot Bloom ({self.bloom_level})"
                    )


@dataclass(frozen=True)
class AssessmentBlueprint:
    """Authoritative structural blueprint of the examination paper."""
    blueprint_id: str
    title: str
    course_code: str
    total_marks: int
    duration_minutes: int
    instructions: str
    slots: Tuple[AssessmentSlot, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.blueprint_id or not isinstance(self.blueprint_id, str) or not self.blueprint_id.strip():
            raise ValueError("AssessmentBlueprint: blueprint_id must be a non-empty string")
        if not self.title or not isinstance(self.title, str) or not self.title.strip():
            raise ValueError("AssessmentBlueprint: title must be a non-empty string")
        if not self.course_code or not isinstance(self.course_code, str) or not self.course_code.strip():
            raise ValueError("AssessmentBlueprint: course_code must be a non-empty string")
        if type(self.total_marks) is not int or isinstance(self.total_marks, bool) or self.total_marks <= 0:
            raise ValueError(f"AssessmentBlueprint: total_marks must be positive int, got {self.total_marks!r}")
        if type(self.duration_minutes) is not int or isinstance(self.duration_minutes, bool) or self.duration_minutes <= 0:
            raise ValueError(f"AssessmentBlueprint: duration_minutes must be positive int, got {self.duration_minutes!r}")
        if not self.instructions or not isinstance(self.instructions, str) or not self.instructions.strip():
            raise ValueError("AssessmentBlueprint: instructions must be a non-empty string")

        if type(self.slots) is not tuple:
            object.__setattr__(self, "slots", tuple(self.slots))
        if not self.slots:
            raise ValueError("AssessmentBlueprint: slots must not be empty")

        for s in self.slots:
            if not isinstance(s, AssessmentSlot):
                raise TypeError(f"AssessmentBlueprint: slot must be AssessmentSlot, got {type(s).__name__}")

        # Invariant: Slot marks sum == blueprint total marks
        slot_sum = sum(s.marks for s in self.slots)
        if slot_sum != self.total_marks:
            raise ValueError(
                f"AssessmentBlueprint ({self.blueprint_id}): slot marks sum ({slot_sum}) "
                f"!= total marks ({self.total_marks})"
            )


# -----------------------------------------------------------------------------
# 6. Task Completeness Evaluation (Answerability Gate)
# -----------------------------------------------------------------------------


def evaluate_task_completeness(
    spec: Union[QuestionSpecification, SubPartSpecification]
) -> Tuple[TaskCompleteness, Optional[str]]:
    """
    Task-specific answerability evaluation.
    Determines if sufficient information and constraints exist for a valid response.
    Returns (TaskCompleteness, reason).
    """
    archetype = getattr(spec, "primary_archetype", None)
    if archetype is None and hasattr(spec, "subparts") and spec.subparts:
        # For a parent question, verify all subparts
        for sp in spec.subparts:
            status, reason = evaluate_task_completeness(sp)
            if status != TaskCompleteness.COMPLETE:
                return status, f"Subpart ({sp.label}): {reason}"
        return TaskCompleteness.COMPLETE, None

    if archetype in (UniversalArchetype.WORKED_PROBLEM_CALCULATION, UniversalArchetype.ALGORITHM_PROCEDURAL_TRACE):
        if not spec.input_artifacts:
            return TaskCompleteness.INCOMPLETE, f"{archetype.value} requires input artifacts or data"
        for artifact in spec.input_artifacts:
            for param in artifact.parameters:
                if param.state == ParameterState.UNSPECIFIED and artifact.required_for_solution:
                    return TaskCompleteness.INCOMPLETE, f"Critical parameter {param.name!r} is UNSPECIFIED"
                if param.state == ParameterState.GIVEN and param.value is None:
                    return TaskCompleteness.INCOMPLETE, f"Parameter {param.name!r} marked GIVEN lacks explicit value"
        return TaskCompleteness.COMPLETE, None

    elif archetype == UniversalArchetype.COMPARATIVE_ANALYSIS:
        if spec.expected_response and len(spec.expected_response.expected_milestones) < 2:
            return TaskCompleteness.INCOMPLETE, "Comparative analysis requires at least two comparison dimensions"
        return TaskCompleteness.COMPLETE, None

    elif archetype == UniversalArchetype.CONCEPTUAL_EXPLANATION:
        if not getattr(spec, "syllabus_topic", None) and not hasattr(spec, "label"):
            return TaskCompleteness.INCOMPLETE, "Conceptual explanation missing syllabus topic"
        return TaskCompleteness.COMPLETE, None

    elif archetype in (UniversalArchetype.SCHEMATIC_SYSTEM_SYNTHESIS, UniversalArchetype.EMPIRICAL_MATERIAL_DESIGN):
        if not spec.input_artifacts:
            return TaskCompleteness.INDETERMINATE, f"Evidence cannot establish completeness for {archetype.value} without input artifacts"
        return TaskCompleteness.COMPLETE, None

    return TaskCompleteness.INDETERMINATE, f"Available evidence cannot establish completeness for archetype {archetype!r}"
