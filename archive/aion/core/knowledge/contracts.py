"""
AION v2 Knowledge Compiler Contracts & Primitives
==================================================
Defines the authoritative semantic index structures:
- KnowledgeUnit (pure index over immutable DOM artifacts, NO duplicate text/formulas)
- SemanticCandidate & SemanticType (probabilistic candidate signals, not hardcoded facts)
- FigureContextBinding (relationship-layer contextual bindings for uncaptioned figures)
- ImportanceScore (explainable mathematical score breakdown)
- SyllabusScopeMask (fail-closed syllabus boundaries)
- CognitiveDemand (Bloom's Revised Taxonomy levels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from aion.core.dom.document_dom import DocumentDOM
    from aion.core.dom.artifacts import (
        BaseArtifact,
        EquationArtifact,
        FigureArtifact,
        TextArtifact,
        TableArtifact,
        AlgorithmArtifact,
        ExampleArtifact,
        ProcedureArtifact,
    )


class CognitiveDemand(str, Enum):
    """Bloom's Revised Taxonomy cognitive demand levels."""
    L1_REMEMBER = "L1_REMEMBER"
    L2_UNDERSTAND = "L2_UNDERSTAND"
    L3_APPLY = "L3_APPLY"
    L4_ANALYZE = "L4_ANALYZE"
    L5_EVALUATE = "L5_EVALUATE"
    L6_CREATE = "L6_CREATE"


class SemanticType(str, Enum):
    """Academic semantic classifications for artifacts and concepts."""
    CONCEPTUAL = "CONCEPTUAL"
    DEFINITION = "DEFINITION"
    PROCEDURE = "PROCEDURE"
    ALGORITHM = "ALGORITHM"
    WORKED_EXAMPLE = "WORKED_EXAMPLE"
    DERIVATION = "DERIVATION"
    COMPARISON = "COMPARISON"
    SYSTEM_ARCHITECTURE = "SYSTEM_ARCHITECTURE"


@dataclass
class SemanticCandidate:
    """
    Probabilistic candidate classification produced by deterministic pattern detectors.
    Deterministic systems propose candidates; validated compilation decides.
    """
    artifact_id: str
    candidate_type: SemanticType
    confidence: float                     # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list) # e.g. ["copula_is_a", "bold_term", "heading_context"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "candidate_type": self.candidate_type.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SemanticCandidate:
        return cls(
            artifact_id=data["artifact_id"],
            candidate_type=SemanticType(data["candidate_type"]),
            confidence=float(data["confidence"]),
            evidence=list(data.get("evidence", [])),
        )


class BindingType(str, Enum):
    CAPTION_EXPLICIT = "CAPTION_EXPLICIT"  # Formal figure label match, e.g. 'Figure 3.1'
    SLIDE_CONTEXT = "SLIDE_CONTEXT"        # Slide title + page bullets enclosure
    SECTION_CONTEXT = "SECTION_CONTEXT"    # Section title enclosure
    PROXIMITY = "PROXIMITY"                # Same page spatial adjacency


@dataclass
class FigureContextBinding:
    """
    Contextual binding connecting an uncaptioned or visual figure to its surrounding
    semantic context (slide title, adjacent bullets) on the relationship layer.
    The original FigureArtifact remains completely immutable.
    """
    figure_id: str
    section_id: str
    page_number: int
    context_title: str                     # e.g. "Autoencoders: Bottleneck Architecture"
    context_artifact_ids: List[str] = field(default_factory=list) # Adjacent text/bullet IDs
    binding_type: BindingType = BindingType.SLIDE_CONTEXT
    confidence: float = 0.85

    def to_dict(self) -> Dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "section_id": self.section_id,
            "page_number": self.page_number,
            "context_title": self.context_title,
            "context_artifact_ids": self.context_artifact_ids,
            "binding_type": self.binding_type.value,
            "confidence": round(self.confidence, 3),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> FigureContextBinding:
        return cls(
            figure_id=data["figure_id"],
            section_id=data["section_id"],
            page_number=data.get("page_number", 1),
            context_title=data.get("context_title", ""),
            context_artifact_ids=list(data.get("context_artifact_ids", [])),
            binding_type=BindingType(data.get("binding_type", BindingType.SLIDE_CONTEXT.value)),
            confidence=float(data.get("confidence", 0.85)),
        )


@dataclass
class ImportanceScore:
    """
    Transparent, explainable pedagogical importance score breakdown.
    Every factor contributing to topic importance is preserved for observability.
    """
    total: float                           # 0.0 to 1.0 composite
    heading_weight: float = 0.0            # Based on heading hierarchy (H1=1.0, H2=0.8, H3=0.6)
    equation_density: float = 0.0          # Presence of governing mathematical formulations
    text_density: float = 0.0              # Depth of descriptive prose
    syllabus_priority: float = 0.0         # Mandatory topic boost from syllabus
    structural_prominence: float = 0.0     # Presence of diagrams, algorithms, worked examples
    rationale: str = ""                    # Human-readable explanation of the score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": round(self.total, 3),
            "heading_weight": round(self.heading_weight, 3),
            "equation_density": round(self.equation_density, 3),
            "text_density": round(self.text_density, 3),
            "syllabus_priority": round(self.syllabus_priority, 3),
            "structural_prominence": round(self.structural_prominence, 3),
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ImportanceScore:
        return cls(
            total=float(data["total"]),
            heading_weight=float(data.get("heading_weight", 0.0)),
            equation_density=float(data.get("equation_density", 0.0)),
            text_density=float(data.get("text_density", 0.0)),
            syllabus_priority=float(data.get("syllabus_priority", 0.0)),
            structural_prominence=float(data.get("structural_prominence", 0.0)),
            rationale=data.get("rationale", ""),
        )


@dataclass
class SyllabusScopeMask:
    """
    Authoritative boundary mask defining the allowed assessment scope for a module.
    Enforces a strict fail-closed policy: content outside allowed ranges/topics
    is marked in_syllabus=False and blocked from examination generation.
    """
    module_index: int                      # 1 to 5
    allowed_page_ranges: List[Tuple[int, int]] = field(default_factory=list) # e.g. [(1, 40)]
    mandatory_topics: List[str] = field(default_factory=list) # Topics with syllabus_priority boost
    excluded_topics: List[str] = field(default_factory=list)  # Explicitly omitted topics
    strict_enforcement: bool = True

    def is_page_allowed(self, page_number: int) -> bool:
        """Check if page falls within any allowed range."""
        if not self.allowed_page_ranges:
            # If no page ranges specified, all pages in module scope are tentatively accepted
            return True
        for start, end in self.allowed_page_ranges:
            if start <= page_number <= end:
                return True
        return False

    def is_topic_excluded(self, topic_title: str) -> bool:
        """Check if topic matches any excluded keyword."""
        title_lower = topic_title.lower()
        for excl in self.excluded_topics:
            if excl.lower() in title_lower:
                return True
        return False

    def is_topic_mandatory(self, topic_title: str) -> bool:
        """Check if topic matches any mandatory keyword."""
        title_lower = topic_title.lower()
        for mand in self.mandatory_topics:
            if mand.lower() in title_lower:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module_index": self.module_index,
            "allowed_page_ranges": [list(r) for r in self.allowed_page_ranges],
            "mandatory_topics": self.mandatory_topics,
            "excluded_topics": self.excluded_topics,
            "strict_enforcement": self.strict_enforcement,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyllabusScopeMask:
        return cls(
            module_index=int(data["module_index"]),
            allowed_page_ranges=[tuple(r) for r in data.get("allowed_page_ranges", [])],
            mandatory_topics=list(data.get("mandatory_topics", [])),
            excluded_topics=list(data.get("excluded_topics", [])),
            strict_enforcement=bool(data.get("strict_enforcement", True)),
        )


@dataclass
class KnowledgeUnit:
    """
    Authoritative index over the immutable DocumentDOM.
    ARCHITECTURAL INVARIANT: Contains ONLY typed artifact IDs and pointer references.
    NEVER stores copied text blocks, duplicated formula strings, or raw source bodies.
    """
    ku_id: str                             # Canonical ID, e.g. "KU-M1-SEC003-001"
    module_index: int                      # 1 to 5
    section_id: str                        # Parent SectionNode ID
    concept_title: str                     # Clean concept or slide topic name
    page_span: Tuple[int, int]             # (start_page, end_page)

    # -----------------------------------------------------------------------
    # Pure Index: Typed Artifact ID Collections (NO DUPLICATE BLOBS)
    # -----------------------------------------------------------------------
    source_artifact_ids: List[str] = field(default_factory=list) # All artifact IDs composing this KU
    text_artifact_ids: List[str] = field(default_factory=list)
    equation_artifact_ids: List[str] = field(default_factory=list)
    figure_artifact_ids: List[str] = field(default_factory=list)
    table_artifact_ids: List[str] = field(default_factory=list)
    example_artifact_ids: List[str] = field(default_factory=list)
    procedure_artifact_ids: List[str] = field(default_factory=list)
    algorithm_artifact_ids: List[str] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Pedagogical Metadata
    # -----------------------------------------------------------------------
    primary_semantic_type: SemanticType = SemanticType.CONCEPTUAL
    semantic_candidates: List[SemanticCandidate] = field(default_factory=list)
    importance: ImportanceScore = field(default_factory=lambda: ImportanceScore(total=0.5))
    bloom_affinities: List[CognitiveDemand] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list) # KU IDs that precede this concept
    in_syllabus: bool = True
    scope_uncertain: bool = False          # True if compiled without a syllabus scope mask
    assessed_count: int = 0                # Primary assessment frequency
    secondary_assessed_count: int = 0      # Secondary assessment frequency

    # -----------------------------------------------------------------------
    # Dynamic DOM Resolver Helpers (Convenience Accessors)
    # -----------------------------------------------------------------------
    def resolve_equations(self, dom: DocumentDOM) -> List[EquationArtifact]:
        """Resolve governing equation artifacts dynamically from the DOM vault."""
        return [
            art for aid in self.equation_artifact_ids
            if (art := dom.registry.get(aid)) is not None
        ]  # type: ignore

    def resolve_figures(self, dom: DocumentDOM) -> List[FigureArtifact]:
        """Resolve illustrative figure artifacts dynamically from the DOM vault."""
        return [
            art for aid in self.figure_artifact_ids
            if (art := dom.registry.get(aid)) is not None
        ]  # type: ignore

    def resolve_texts(self, dom: DocumentDOM) -> List[TextArtifact]:
        """Resolve body text artifacts dynamically from the DOM vault."""
        return [
            art for aid in self.text_artifact_ids
            if (art := dom.registry.get(aid)) is not None
        ]  # type: ignore

    def resolve_all_artifacts(self, dom: DocumentDOM) -> List[BaseArtifact]:
        """Resolve all constituent artifacts in reading order."""
        return [
            art for aid in self.source_artifact_ids
            if (art := dom.registry.get(aid)) is not None
        ]

    # -----------------------------------------------------------------------
    # Serialization
    # -----------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the index structure. Guarantees ZERO duplicate source text."""
        return {
            "ku_id": self.ku_id,
            "module_index": self.module_index,
            "section_id": self.section_id,
            "concept_title": self.concept_title,
            "page_span": list(self.page_span),
            "source_artifact_ids": self.source_artifact_ids,
            "text_artifact_ids": self.text_artifact_ids,
            "equation_artifact_ids": self.equation_artifact_ids,
            "figure_artifact_ids": self.figure_artifact_ids,
            "table_artifact_ids": self.table_artifact_ids,
            "example_artifact_ids": self.example_artifact_ids,
            "procedure_artifact_ids": self.procedure_artifact_ids,
            "algorithm_artifact_ids": self.algorithm_artifact_ids,
            "primary_semantic_type": self.primary_semantic_type.value,
            "semantic_candidates": [c.to_dict() for c in self.semantic_candidates],
            "importance": self.importance.to_dict(),
            "bloom_affinities": [b.value for b in self.bloom_affinities],
            "prerequisites": self.prerequisites,
            "in_syllabus": self.in_syllabus,
            "scope_uncertain": self.scope_uncertain,
            "assessed_count": self.assessed_count,
            "secondary_assessed_count": self.secondary_assessed_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> KnowledgeUnit:
        return cls(
            ku_id=data["ku_id"],
            module_index=int(data["module_index"]),
            section_id=data["section_id"],
            concept_title=data["concept_title"],
            page_span=tuple(data.get("page_span", [1, 1])),
            source_artifact_ids=list(data.get("source_artifact_ids", [])),
            text_artifact_ids=list(data.get("text_artifact_ids", [])),
            equation_artifact_ids=list(data.get("equation_artifact_ids", [])),
            figure_artifact_ids=list(data.get("figure_artifact_ids", [])),
            table_artifact_ids=list(data.get("table_artifact_ids", [])),
            example_artifact_ids=list(data.get("example_artifact_ids", [])),
            procedure_artifact_ids=list(data.get("procedure_artifact_ids", [])),
            algorithm_artifact_ids=list(data.get("algorithm_artifact_ids", [])),
            primary_semantic_type=SemanticType(data.get("primary_semantic_type", SemanticType.CONCEPTUAL.value)),
            semantic_candidates=[SemanticCandidate.from_dict(c) for c in data.get("semantic_candidates", [])],
            importance=ImportanceScore.from_dict(data.get("importance", {"total": 0.5})),
            bloom_affinities=[CognitiveDemand(b) for b in data.get("bloom_affinities", [])],
            prerequisites=list(data.get("prerequisites", [])),
            in_syllabus=bool(data.get("in_syllabus", True)),
            scope_uncertain=bool(data.get("scope_uncertain", False)),
            assessed_count=int(data.get("assessed_count", 0)),
            secondary_assessed_count=int(data.get("secondary_assessed_count", 0)),
        )
