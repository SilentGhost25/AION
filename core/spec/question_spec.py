"""
Question Specification — Shared Loosely-Coupled Object
Each pipeline stage enriches this object rather than rewriting free-form text.
Makes system easy to extend across VTU subjects without if subject branches.

Expanded per audit: never let composer infer anything, everything already planned.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class QuestionSpec:
    # Subject context
    subject: str  # e.g., "Data Structures", "Automotive"
    subject_code: str  # e.g., "CSE", "AU"
    module: str  # e.g., "Module 3: Trees"
    
    # Knowledge
    knowledge_unit: str  # canonical concept
    knowledge_unit_id: str
    assessment_objective: str  # e.g., "Analyse why heap gives O(log n) insertion"
    student_ability: str  # e.g., "can trace heap insertion and compare with array"
    question_type: str  # Algorithmic, Analytical, Conceptual, Numerical, Diagram, Case Study, Proof, Design, Optimization, Debugging
    assessment_type: str  # e.g., "Numerical tracing", "Procedure sequencing"
    expected_reasoning: str  # e.g., "Stepwise execution"
    bloom: str  # L1-L6
    bloom_level: int
    marks: int
    requires_diagram: bool
    requires_numerical: bool
    formula_required: bool
    expected_answer_type: str  # Stepwise procedure, Diagram, Calculation, etc.
    expected_answer_schema: Dict[str, Any] = field(default_factory=dict)
    difficulty: str = "medium"  # easy, medium, hard
    
    # Grounding - strict
    grounding_evidence: List[str] = field(default_factory=list)
    allowed_entities: List[str] = field(default_factory=list)
    forbidden_entities: List[str] = field(default_factory=list)
    source_hash: str = ""
    confidence: float = 0.0
    
    # Constraints
    constraints: Dict[str, Any] = field(default_factory=dict)
    numerical_constraints: Dict[str, Any] = field(default_factory=dict)
    scenario: Optional[str] = None
    required_operations: List[str] = field(default_factory=list)
    numerical_payload: Optional[Dict[str, Any]] = None
    
    # Enrichment by stages
    reasoning_ops: List[str] = field(default_factory=list)
    examiner_style: str = "standard"
    diagram_spec: Optional[str] = None
    
    # Variation
    archetype: str = ""  # e.g., "BST insertion - balanced", "Queue circular - overflow"
    variation_key: str = ""
    
    # Composition
    draft_question: Optional[str] = None
    final_question: Optional[str] = None
    audit_result: Optional[Dict] = None
    
    # Strict gate
    grounding_coverage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "module": self.module,
            "knowledge_unit": self.knowledge_unit,
            "assessment_objective": self.assessment_objective,
            "student_ability": self.student_ability,
            "question_type": self.question_type,
            "assessment_type": self.assessment_type,
            "expected_reasoning": self.expected_reasoning,
            "required_operations": self.required_operations,
            "bloom": self.bloom,
            "marks": self.marks,
            "requires_diagram": self.requires_diagram,
            "requires_numerical": self.requires_numerical,
            "formula_required": self.formula_required,
            "expected_answer_type": self.expected_answer_type,
            "expected_answer_schema": self.expected_answer_schema,
            "grounding_evidence": self.grounding_evidence[:2],
            "allowed_entities": self.allowed_entities,
            "forbidden_entities": self.forbidden_entities,
            "scenario": self.scenario,
            "numerical_constraints": self.numerical_constraints,
            "constraints": self.constraints,
            "archetype": self.archetype,
        }
