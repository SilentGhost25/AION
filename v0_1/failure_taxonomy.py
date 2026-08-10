"""
AION Empirical Failure Taxonomy
===============================
Standardized failure taxonomy classifying any pipeline anomaly across 12 distinct categories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class FailureCategory(str, Enum):
    EXTRACTION = "EXTRACTION"
    ENCODING = "ENCODING"
    EQUATION = "EQUATION"
    RETRIEVAL = "RETRIEVAL"
    GROUNDING = "GROUNDING"
    PLANNING = "PLANNING"
    LLM = "LLM"
    VRE = "VRE"
    SOLVER = "SOLVER"
    STRUCTURE = "STRUCTURE"
    RENDERING = "RENDERING"
    FRONTEND = "FRONTEND"


@dataclass
class FailureRecord:
    """Represents a classified failure event."""
    category: FailureCategory
    error_code: str
    message: str
    stage: str
    doc_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class FailureClassifier:
    """Classifies raw errors into AION Failure Taxonomy categories."""

    @classmethod
    def classify(cls, error_msg: str, stage: str = "unknown") -> FailureRecord:
        msg_low = error_msg.lower()

        if "extract" in msg_low or "read_text" in msg_low or "pdf" in msg_low:
            cat = FailureCategory.EXTRACTION
        elif "encoding" in msg_low or "utf-8" in msg_low or "latin" in msg_low:
            cat = FailureCategory.ENCODING
        elif "equation" in msg_low or "math" in msg_low or "symbol" in msg_low:
            cat = FailureCategory.EQUATION
        elif "retriev" in msg_low or "chunk" in msg_low:
            cat = FailureCategory.RETRIEVAL
        elif "grounding" in msg_low or "context" in msg_low:
            cat = FailureCategory.GROUNDING
        elif "plan" in msg_low or "slot" in msg_low:
            cat = FailureCategory.PLANNING
        elif "vre" in msg_low or "fsc" in msg_low or "vko" in msg_low:
            cat = FailureCategory.VRE
        elif "solver" in msg_low or "dijkstra" in msg_low or "circuit" in msg_low:
            cat = FailureCategory.SOLVER
        elif "structure" in msg_low or "mark" in msg_low or "or" in msg_low:
            cat = FailureCategory.STRUCTURE
        elif "render" in msg_low or "svg" in msg_low or "pdf" in msg_low or "docx" in msg_low:
            cat = FailureCategory.RENDERING
        elif "frontend" in msg_low or "cors" in msg_low or "payload" in msg_low:
            cat = FailureCategory.FRONTEND
        else:
            cat = FailureCategory.LLM

        return FailureRecord(
            category=cat,
            error_code=f"ERR_{cat.value}_001",
            message=error_msg,
            stage=stage,
        )
