"""
AION VRE Contracts
==================
Typed contract definitions for the Visual Reasoning Engine (VRE).
Strictly prevents untyped dict passing between VRE modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class VREDecisionState(str, Enum):
    """Explicit decision state for Visual Reasoning Engine."""
    IMAGE_NOT_NEEDED = "IMAGE_NOT_NEEDED"
    IMAGE_NEEDED_AND_VALID = "IMAGE_NEEDED_AND_VALID"
    IMAGE_NEEDED_BUT_INVALID = "IMAGE_NEEDED_BUT_INVALID"
    IMAGE_UNSUPPORTED = "IMAGE_UNSUPPORTED"


class BloomLevel(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"


class QuantityType(str, Enum):
    INTEGER = "INTEGER"
    FLOAT = "FLOAT"
    RESISTANCE = "RESISTANCE"
    VOLTAGE = "VOLTAGE"
    CURRENT = "CURRENT"
    FORCE = "FORCE"
    LENGTH = "LENGTH"
    GENERIC = "GENERIC"


@dataclass
class FigureInput:
    """Input payload for a candidate figure."""
    image_path: str
    page_number: int = 1
    bbox: Optional[Tuple[int, int, int, int]] = None
    source_text: str = ""
    module: str = "module_1"
    concept: str = ""
    confidence: float = 1.0


@dataclass
class FigureExtractionResult:
    """Output of Figure Extraction & Quality Gate."""
    status: str
    image_path: Optional[str]
    page_number: int
    bbox: Optional[Tuple[int, int, int, int]]
    confidence: float
    extraction_method: str
    width: int = 0
    height: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.status == "PASS" and self.image_path is not None


@dataclass
class ConfidenceMetrics:
    """Multi-source evidence confidence breakdown."""
    domain_confidence: float = 0.0
    class_confidence: float = 0.0
    topology_confidence: float = 0.0
    ocr_confidence: float = 0.0
    semantic_confidence: float = 0.0

    @property
    def composite_confidence(self) -> float:
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        scores = [
            self.domain_confidence,
            self.class_confidence,
            self.topology_confidence,
            self.ocr_confidence,
            self.semantic_confidence,
        ]
        return sum(w * s for w, s in zip(weights, scores))


@dataclass
class FigureClassification:
    """Output of FSC (Figure Semantic Classifier)."""
    domain: str
    figure_class: str
    operations: List[str]
    confidence: ConfidenceMetrics = field(default_factory=ConfidenceMetrics)
    supported: bool = True
    reason: str = ""


@dataclass
class Node:
    id: str
    position: Tuple[int, int] = (0, 0)
    shape: str = "circle"
    is_source: bool = False
    is_sink: bool = False
    node_type: str = "default"


@dataclass
class Edge:
    id: str
    from_node: str
    to_node: str
    directed: bool = False
    weight: Optional[float] = None
    from_terminal: str = ""
    to_terminal: str = ""


@dataclass
class TopologyGraph:
    nodes: List[Node] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    is_directed: bool = False
    is_weighted: bool = False
    is_cyclic: bool = False
    is_connected: bool = True
    loops: List[List[str]] = field(default_factory=list)


@dataclass
class LabelMap:
    node_labels: Dict[str, str] = field(default_factory=dict)
    edge_labels: Dict[str, str] = field(default_factory=dict)
    component_labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class QuantityMap:
    edge_weights: Dict[str, float] = field(default_factory=dict)
    node_values: Dict[str, float] = field(default_factory=dict)
    component_values: Dict[str, Tuple[float, str]] = field(default_factory=dict)
    span_length: Optional[float] = None


@dataclass
class ConstraintSet:
    academic_laws: List[str] = field(default_factory=list)
    valid_operations: List[str] = field(default_factory=list)
    difficulty_ceiling: str = "L4"


@dataclass
class MutationRule:
    target: str
    constraint: str
    value_range: Tuple[float, float]
    quantity_type: QuantityType = QuantityType.GENERIC
    filter_expr: str = ""


@dataclass
class MutabilityProfile:
    topology_mutable: bool = False
    labels_mutable: bool = True
    quantities_mutable: bool = True
    mutation_rules: List[MutationRule] = field(default_factory=list)


@dataclass
class VKO:
    """Visual Knowledge Object."""
    id: str
    source_image: str
    figure_class: str
    domain: str
    topology: TopologyGraph = field(default_factory=TopologyGraph)
    labels: LabelMap = field(default_factory=LabelMap)
    quantities: QuantityMap = field(default_factory=QuantityMap)
    constraints: ConstraintSet = field(default_factory=ConstraintSet)
    mutability: MutabilityProfile = field(default_factory=MutabilityProfile)

    def clone(self) -> VKO:
        """Deep copy VKO for numerical parameter mutation."""
        import copy
        return copy.deepcopy(self)


@dataclass
class OperationStep:
    step_number: int
    operation: str
    input_type: str
    output_type: str
    verifiable: bool = True


@dataclass
class OperationChain:
    chain_id: str
    bloom_level: str
    steps: List[OperationStep] = field(default_factory=list)
    expected_output_type: str = "NUMERIC"
    marks_estimate: int = 5
    score: float = 0.0

    @property
    def operations(self) -> List[str]:
        return [s.operation for s in self.steps]


@dataclass
class VQG:
    """Visual Question Graph representation."""
    vko_id: str
    operation_chains: List[OperationChain] = field(default_factory=list)
    bloom_mapping: Dict[str, List[OperationChain]] = field(default_factory=dict)


@dataclass
class VRERequest:
    """Inbound request payload for the VRE."""
    request_id: str
    subject: str
    department: str
    module: str
    topic: str
    bloom_level: str
    marks: int
    question_type: str = "analytical"
    figure_candidates: List[FigureInput] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)


class RenderMode(str, Enum):
    SVG = "SVG"
    ANNOTATED_SOURCE = "ANNOTATED_SOURCE"
    ORIGINAL_SOURCE = "ORIGINAL_SOURCE"


@dataclass
class VREDecision:
    """Decision produced by QPVDE."""
    state: VREDecisionState
    reason: str
    confidence: float
    image_dependency_score: float = 0.0
    vko: Optional[VKO] = None
    selected_chain: Optional[OperationChain] = None
    mandatory: bool = False

    @property
    def use_image(self) -> bool:
        return self.state == VREDecisionState.IMAGE_NEEDED_AND_VALID


@dataclass
class QuestionPlan:
    """Deterministic question plan produced before LLM text expansion."""
    plan_id: str
    operation: str
    vko_id: str
    bloom_level: str
    marks: int
    source_element: str
    destination_element: str
    expected_output: str
    question_plan_hash: str = ""
    anchors: Dict[str, Any] = field(default_factory=dict)
    reference_solution: Any = None


@dataclass
class ProvenanceRecord:
    """Provenance tracking record."""
    source_document: str
    page: int
    figure_id: str
    module: str
    concept: str
    question_plan_hash: str = ""
    source_bbox: Optional[Tuple[int, int, int, int]] = None
    vko_id: str = ""
    operation_chain_id: str = ""
    generated: bool = True
    trace: List[str] = field(default_factory=list)


@dataclass
class VREOutput:
    """Final output payload of VRE execution."""
    success: bool
    text: str
    figure_svg: Optional[str] = None
    figure_caption: str = ""
    render_mode: RenderMode = RenderMode.SVG
    bloom: str = "L3"
    marks: int = 5
    image_dependency_score: float = 0.0
    question_plan_hash: str = ""
    decision_state: VREDecisionState = VREDecisionState.IMAGE_NOT_NEEDED
    provenance: Optional[ProvenanceRecord] = None
    reference_solution: Any = None
    reason: str = ""
