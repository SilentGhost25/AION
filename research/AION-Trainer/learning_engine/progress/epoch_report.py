# learning_engine/progress/epoch_report.py
"""
Epoch Report — Data structure representing the report card generated after
every epoch of study.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any


@dataclass
class EpochReport:
    epoch: int
    concept_understanding: float  # average understanding percentage (0-100)
    relationship_strength: float  # relationship coverage / mapping strength (0-100)
    question_quality: float       # question acceptance rate percentage (0-100)
    answer_quality: float         # average answer quality score percentage (0-100)
    examiner_similarity: float    # match rate to examiner preference distribution (0-100)
    grammar: float                # average question grammar score (0-100)
    coverage: float               # subject concepts covered (0-100)
    weak_concepts_count: int
    strong_concepts_count: int
    weak_concepts: List[str] = field(default_factory=list)
    strong_concepts: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EpochReport:
        return cls(**data)
