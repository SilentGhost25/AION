"""
AION Module: Data Contracts
Maturity:    v0.1 — IMMUTABLE PIPELINE CONTRACTS
Upgrades to: Schema-enforced AOM Protobuf / Pydantic Validators
Contract:    This file is the single source of truth for objects flowing between stages.
             Every future module upgrade (rule-based -> neural) MUST honor these contracts.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

try:
    from core.sdk.aom import KnowledgeGene, KnowledgeObject
    HAS_AOM = True
except ImportError:
    HAS_AOM = False


@dataclass
class Document:
    """Output of uploader + extractor"""
    doc_id: str
    source_path: str
    raw_text: str
    file_type: str
    uploaded_at: str = field(default_factory=lambda: datetime.now().isoformat())
    formulas: List[str] = field(default_factory=list)


@dataclass
class CleanedDocument:
    """Output of cleaner"""
    doc_id: str
    clean_text: str
    removed_line_count: int          # quality metric
    original_line_count: int


@dataclass
class Concept:
    """
    Output of learner — SEED of the Academic Genome.
    Every field here maps to a future DNA strand.
    """
    concept_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    confidence: float = 0.5

    # DNA strands (empty in v0.1, reserved for trained modules)
    definition_dna: Optional[str] = None
    relationship_dna: List[Dict[str, str]] = field(default_factory=list)
    bloom_dna: Optional[int] = None
    source_dna: Optional[str] = None
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_aom_gene(self) -> Optional[Any]:
        """Bridge v0.1 Concept to core.sdk.aom.KnowledgeGene if available."""
        if not HAS_AOM:
            return None
        return KnowledgeGene(
            gene_id=self.concept_id,
            knowledge_id=f"KNO-{self.concept_id}",
            sequence_order=1,
            gene_type="concept_seed",
            raw_content=self.content,
            concept_name=self.content[:40],
            canonical_definition=self.definition_dna or self.content,
            confidence_score=self.confidence,
            relationships={r.get("target", ""): r.get("relation", "") for r in self.relationship_dna if "target" in r},
            expected_answer=None,
        )


@dataclass
class GeneratedQuestion:
    """
    Output of generator.
    Enforces RAG^2 (Reverse Assessment Generation): ideal_answer is populated FIRST.
    """
    question_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    concept_id: str = ""
    ideal_answer: str = ""           # generated BEFORE question_text
    marking_scheme: Optional[str] = None
    question_text: str = ""
    marks: int = 5
    bloom_level: Optional[int] = None
