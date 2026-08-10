"""
AION Core Contracts — FinalPaper Intermediate Representation
============================================================
Defines QuestionSegment, FinalQuestion, ORPair, and FinalPaper
as specified in Part IX of the Production Hardening Specification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class QuestionSegment:
    """One segment of a question — text, math, figure, or table."""
    segment_type : str                  # "text" | "math" | "figure" | "table"
    value        : Optional[str] = None # for text
    latex        : Optional[str] = None # for math
    display_mode : bool = False         # for math: block vs inline
    figure_id    : Optional[str] = None # for figure
    table_id     : Optional[str] = None # for table
    alt_text     : Optional[str] = None # for figure accessibility


@dataclass
class FinalQuestion:
    """
    Immutable representation of one validated question.
    DOCX/PDF/JSON renderers receive this — never raw strings.
    """
    question_id        : str
    question_no        : int
    sub_label          : str
    module_id          : int
    marks              : int             # LOCKED
    bloom              : str             # LOCKED
    co                 : str             # LOCKED
    question_type      : str
    status             : str             # "APPROVED" only
    segments           : List[QuestionSegment] = field(default_factory=list)
    evidence_refs      : List[str] = field(default_factory=list)
    source_pages       : List[int] = field(default_factory=list)
    grounding_score    : float = 1.0
    qa_score           : float = 1.0
    bloom_validated    : bool = True
    evidence_validated : bool = True
    math_validated     : bool = True

    def is_exportable(self) -> bool:
        return (
            self.status == "APPROVED" and
            self.bloom_validated and
            self.evidence_validated and
            self.math_validated
        )


@dataclass
class ORPair:
    module_id         : int
    alt_a             : FinalQuestion
    alt_b             : FinalQuestion
    mark_distribution : Tuple[int, ...]  # LOCKED

    def parity_valid(self) -> bool:
        return self.alt_a.marks == self.alt_b.marks


@dataclass
class FinalPaperIR:
    """
    Authoritative intermediate representation before rendering.
    Renderers (DOCX, PDF, JSON) treat this as their source of truth.
    """
    paper_id     : str
    request_id   : str
    created_at   : str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    subject      : str = "Engineering"
    department   : str = "CSE"
    exam_type    : str = "IAT1"
    total_marks  : int = 50
    or_pairs     : List[ORPair] = field(default_factory=list)
    qa_score     : float = 1.0
    qa_status    : str = "PASS"          # "PASS" | "PASS_WITH_WARNINGS"
    plan_id      : str = "plan_001"

    def is_exportable(self) -> bool:
        return (
            self.qa_status in {"PASS", "PASS_WITH_WARNINGS"} and
            all(
                alt.is_exportable()
                for pair in self.or_pairs
                for alt in [pair.alt_a, pair.alt_b]
            )
        )
