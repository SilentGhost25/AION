"""
AION Generation — Question Evidence Contract
==============================================
Immutable bridge contract between evidence retrieval and Qwen question realization.
Passes validated evidence references, mathematical artifacts, and structural bounds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class EvidenceReference:
    evidence_id : str
    page        : int
    support     : float = 1.0


@dataclass(frozen=True)
class QuestionEvidenceContract:
    """
    Locked contract containing all context Qwen needs for realization.
    Qwen receives this contract — Qwen never receives raw unvalidated documents.
    """
    question_id     : str
    slot_id         : str
    module_id       : int
    topic           : str
    bloom_level     : str
    bloom_verb      : str
    marks           : int
    co              : str

    evidence_refs   : Tuple[EvidenceReference, ...] = field(default_factory=tuple)
    evidence_texts  : Tuple[str, ...] = field(default_factory=tuple)

    math_artifacts  : Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    table_artifacts : Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
