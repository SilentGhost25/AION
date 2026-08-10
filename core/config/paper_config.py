"""
AION VTU Question Paper Structure Configuration
===============================================
Defines VTU exam structure, mark allocation, and OR-pair parity rules.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class ExamFormatSpec:
    exam_type: str
    total_marks: int
    marks_per_question: int
    subquestion_breakdowns: List[List[int]]
    module_count: int


@dataclass(frozen=True)
class PaperConfig:
    formats: Dict[str, ExamFormatSpec] = field(
        default_factory=lambda: {
            "IA": ExamFormatSpec(
                exam_type="IA",
                total_marks=50,
                marks_per_question=10,
                subquestion_breakdowns=[[6, 4], [5, 5], [10]],
                module_count=5,
            ),
            "SEE": ExamFormatSpec(
                exam_type="SEE",
                total_marks=100,
                marks_per_question=20,
                subquestion_breakdowns=[[8, 6, 6], [10, 10], [7, 7, 6]],
                module_count=5,
            ),
        }
    )


PAPER_CONFIG = PaperConfig()
