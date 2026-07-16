# AION-Trainer/ese/question_metadata.py
"""
Question Metadata — records the complete internal reasoning audit trail
for a generated question, from planning intent through ranking,
realization, and validation checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional


@dataclass
class CandidateRecord:
    text: str
    source: str
    scores: Dict[str, float] = field(default_factory=dict)
    disqualified: bool = False
    disqualification_reason: str = ""


@dataclass
class QuestionMetadata:
    metadata_id: str
    slot_id: str
    concept_id: str
    concept_name: str
    bloom_level: str
    marks: int
    question_type: str
    
    # Audit trail
    planner_intent: Dict[str, Any] = field(default_factory=dict)
    candidates: List[CandidateRecord] = field(default_factory=list)
    selected_text: str = ""
    realized_text: str = ""
    
    # Validation results
    grammar_issues: List[Dict[str, Any]] = field(default_factory=list)
    vtu_issues: List[Dict[str, Any]] = field(default_factory=list)
    
    status: str = "draft"  # "draft" | "verified" | "flagged"
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionMetadata":
        return cls(**data)
