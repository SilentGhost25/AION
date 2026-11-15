"""
AION v2 Evidence Planning & Grounding Contracts
================================================
Defines the authoritative structures for:
- EvidenceRelationStrength: Explicit hierarchy of relationship evidential weight.
- EvidenceRole: Strict distinction between PRIMARY evidence and CONTEXTUAL metadata.
- EvidenceItemRef: Typed artifact reference with explicit role, strength, and provenance.
- EvidenceRequirements: Hard constraints vs soft preferences per archetype.
- EvidenceBundle: Structured artifact selection (NEVER a flattened prompt string).
- GroundingViolation & GroundingReport: 6-level firewall diagnostics.
- InsufficientEvidenceError: Fast-fail exception when evidence cannot be satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from aion.core.dom.artifacts import ArtifactType, BaseArtifact
from aion.core.dom.evidence import EvidenceBudget, QuestionArchetype
from aion.core.question.contracts import QuestionSpec

if TYPE_CHECKING:
    from aion.core.dom.document_dom import DocumentDOM


class EvidenceRelationStrength(IntEnum):
    """
    Evidential strength ordering.
    Higher values represent stronger semantic/structural grounding.
    PROXIMAL_TO is NEVER treated as equivalent to an explicit reference.
    """
    EXPLICIT_REFERENCE = 5   # Direct textual citation, e.g. "Figure 3.4", "Eq. (4.2)"
    CAPTION = 4              # Formal artifact caption or label
    CONTAINMENT = 3          # Enclosed within the same SectionNode
    SEMANTIC_BINDING = 2     # FigureContextBinding (slide title / neighboring bullets)
    PROXIMAL = 1             # Spatial adjacency on the same page heuristic


class EvidenceRole(str, Enum):
    """
    Role of an evidence item in supporting question generation.
    """
    PRIMARY = "PRIMARY"          # Core evidence to be directly assessed (formula, figure, table)
    CONTEXTUAL = "CONTEXTUAL"    # Supporting context (slide header, bullet points explaining figure)


@dataclass
class EvidenceItemRef:
    """
    Typed pointer representing a selected evidence artifact with its evidentiary role and provenance.
    """
    artifact_id: str
    artifact_type: ArtifactType
    evidence_role: EvidenceRole = EvidenceRole.PRIMARY
    relation_strength: EvidenceRelationStrength = EvidenceRelationStrength.CONTAINMENT
    provenance_page: int = 1
    provenance_bbox: Optional[Tuple[float, float, float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "evidence_role": self.evidence_role.value,
            "relation_strength": int(self.relation_strength),
            "provenance_page": self.provenance_page,
            "provenance_bbox": list(self.provenance_bbox) if self.provenance_bbox else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceItemRef:
        bbox = tuple(data["provenance_bbox"]) if data.get("provenance_bbox") else None
        return cls(
            artifact_id=data["artifact_id"],
            artifact_type=ArtifactType(data["artifact_type"]),
            evidence_role=EvidenceRole(data.get("evidence_role", EvidenceRole.PRIMARY.value)),
            relation_strength=EvidenceRelationStrength(data.get("relation_strength", EvidenceRelationStrength.CONTAINMENT)),
            provenance_page=int(data.get("provenance_page", 1)),
            provenance_bbox=bbox,
        )


@dataclass
class EvidenceRequirements:
    """
    Archetype-driven evidentiary constraints.
    Distinguishes hard required modalities from soft preferred modalities.
    """
    archetype: QuestionArchetype
    required_artifact_types: Set[ArtifactType] = field(default_factory=set)
    preferred_artifact_types: Set[ArtifactType] = field(default_factory=set)
    budget: EvidenceBudget = field(default_factory=lambda: EvidenceBudget(archetype=QuestionArchetype.CONCEPTUAL))


@dataclass
class EvidenceBundle:
    """
    Structured, boundary-locked artifact selection assembled for one QuestionSpec.
    INVARIANT: Contains typed artifact IDs and provenance references, NOT serialized prompt strings.
    """
    bundle_id: str
    spec_id: str
    primary_ku_id: str
    secondary_ku_ids: List[str] = field(default_factory=list)

    # Typed Artifact References
    item_refs: List[EvidenceItemRef] = field(default_factory=list)
    text_artifact_ids: List[str] = field(default_factory=list)
    equation_artifact_ids: List[str] = field(default_factory=list)
    figure_artifact_ids: List[str] = field(default_factory=list)
    table_artifact_ids: List[str] = field(default_factory=list)
    algorithm_artifact_ids: List[str] = field(default_factory=list)

    budget_used: Dict[str, int] = field(default_factory=dict)

    @property
    def all_artifact_ids(self) -> List[str]:
        return [ref.artifact_id for ref in self.item_refs]

    def resolve_artifacts(self, dom: DocumentDOM) -> List[BaseArtifact]:
        """Resolve concrete artifacts from the DOM registry in deterministic order."""
        artifacts = []
        for ref in self.item_refs:
            art = dom.registry.get(ref.artifact_id)
            if art is not None:
                artifacts.append(art)
        return artifacts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "spec_id": self.spec_id,
            "primary_ku_id": self.primary_ku_id,
            "secondary_ku_ids": self.secondary_ku_ids,
            "item_refs": [ref.to_dict() for ref in self.item_refs],
            "text_artifact_ids": self.text_artifact_ids,
            "equation_artifact_ids": self.equation_artifact_ids,
            "figure_artifact_ids": self.figure_artifact_ids,
            "table_artifact_ids": self.table_artifact_ids,
            "algorithm_artifact_ids": self.algorithm_artifact_ids,
            "budget_used": self.budget_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceBundle:
        return cls(
            bundle_id=data["bundle_id"],
            spec_id=data["spec_id"],
            primary_ku_id=data["primary_ku_id"],
            secondary_ku_ids=list(data.get("secondary_ku_ids", [])),
            item_refs=[EvidenceItemRef.from_dict(r) for r in data.get("item_refs", [])],
            text_artifact_ids=list(data.get("text_artifact_ids", [])),
            equation_artifact_ids=list(data.get("equation_artifact_ids", [])),
            figure_artifact_ids=list(data.get("figure_artifact_ids", [])),
            table_artifact_ids=list(data.get("table_artifact_ids", [])),
            algorithm_artifact_ids=list(data.get("algorithm_artifact_ids", [])),
            budget_used=dict(data.get("budget_used", {})),
        )


class GroundingViolationType(str, Enum):
    DOCUMENT_MISMATCH = "DOCUMENT_MISMATCH"
    CROSS_MODULE_CONTAMINATION = "CROSS_MODULE_CONTAMINATION"
    OUT_OF_SYLLABUS = "OUT_OF_SYLLABUS"
    CROSS_KU_LEAKAGE = "CROSS_KU_LEAKAGE"
    PROVENANCE_MISSING = "PROVENANCE_MISSING"
    BUDGET_OVERFLOW = "BUDGET_OVERFLOW"
    UNKNOWN_SOURCE_ENTITY_IN_DRAFT = "UNKNOWN_SOURCE_ENTITY_IN_DRAFT"


@dataclass
class GroundingViolation:
    violation_type: GroundingViolationType
    severity: str                         # "CRITICAL" | "WARNING"
    message: str
    artifact_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_type": self.violation_type.value,
            "severity": self.severity,
            "message": self.message,
            "artifact_id": self.artifact_id,
        }


@dataclass
class GroundingReport:
    """
    Results of the 6-level grounding firewall audit.
    """
    is_valid: bool
    violations: List[GroundingViolation] = field(default_factory=list)
    levels_passed: Dict[str, bool] = field(default_factory=dict)
    provenance_chain_intact: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "violations": [v.to_dict() for v in self.violations],
            "levels_passed": self.levels_passed,
            "provenance_chain_intact": self.provenance_chain_intact,
        }


class InsufficientEvidenceError(Exception):
    """Raised when a KnowledgeUnit lacks the required artifact modalities for an archetype."""
    pass
