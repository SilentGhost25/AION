"""
AION v2 Canonical Question Contracts
====================================
The single source of truth for question specifications across the examination pipeline.
Harmonizes with VTU assessment constraints (marks, Bloom's cognitive taxonomy, archetypes).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aion.core.dom.evidence import ModelCapability, QuestionArchetype
from aion.core.knowledge.contracts import CognitiveDemand


@dataclass
class QuestionSpec:
    """
    Authoritative specification for one question slot in an examination.
    Consistently consumed by EvidencePlanner, GroundingVerifier, and generative routers.
    """
    spec_id: str                          # e.g. "SPEC-M1-Q01A"
    module_index: int                     # 1 to 5
    q_number: int                         # Question number, e.g. 1 to 10 in VTU
    part_letter: str                      # Sub-question label, e.g. "a", "b", "c"
    marks: int                            # VTU marks (e.g. 4, 6, 8, 10, 20)
    bloom_level: CognitiveDemand          # L1_REMEMBER .. L6_CREATE
    archetype: QuestionArchetype          # NUMERICAL, DERIVATION, CIRCUIT_SYSTEM, etc.
    primary_ku_id: str                    # Target concept KnowledgeUnit ID
    secondary_ku_ids: List[str] = field(default_factory=list) # Optional compound KUs
    co: str = "CO1"                       # Course outcome (e.g. "CO1", "CO2")
    is_or: bool = False                   # True if part of an OR pair in VTU schema
    topic: str = ""                       # Human-readable concept/topic label
    target_model_capability: ModelCapability = ModelCapability.TEXT_ONLY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "module_index": self.module_index,
            "q_number": self.q_number,
            "part_letter": self.part_letter,
            "marks": self.marks,
            "bloom_level": self.bloom_level.value,
            "archetype": self.archetype.value,
            "primary_ku_id": self.primary_ku_id,
            "secondary_ku_ids": self.secondary_ku_ids,
            "co": self.co,
            "is_or": self.is_or,
            "topic": self.topic,
            "target_model_capability": self.target_model_capability.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QuestionSpec:
        return cls(
            spec_id=data["spec_id"],
            module_index=int(data["module_index"]),
            q_number=int(data["q_number"]),
            part_letter=data["part_letter"],
            marks=int(data["marks"]),
            bloom_level=CognitiveDemand(data["bloom_level"]),
            archetype=QuestionArchetype(data["archetype"]),
            primary_ku_id=data["primary_ku_id"],
            secondary_ku_ids=list(data.get("secondary_ku_ids", [])),
            co=data.get("co", "CO1"),
            is_or=bool(data.get("is_or", False)),
            topic=data.get("topic", ""),
            target_model_capability=ModelCapability(
                data.get("target_model_capability", ModelCapability.TEXT_ONLY.value)
            ),
        )
