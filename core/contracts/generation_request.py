"""
AION Master Production Specification — Generation Request Contract
===================================================================
The single authoritative request object entering the backend from the frontend.
Every field is validated before any pipeline stage begins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


class GenerationRequestValidationError(Exception):
    """Raised when a GenerationRequest fails validation."""
    pass


@dataclass
class GenerationRequest:
    """
    The single authoritative request object entering the backend from the frontend.
    Every field is validated before any pipeline stage begins.
    """

    # ── IDENTITY ──────────────────────────────────────────────────────────────
    request_id      : str           # UUID, assigned by backend on receipt
    received_at     : str           # ISO-8601 string

    # ── ACADEMIC CONFIG ───────────────────────────────────────────────────────
    subject         : str
    department      : str
    semester        : int           # 1–8
    exam_type       : str           # "IAT1" | "IAT2" | "SEE"
    modules         : List[int]     # e.g. [1, 2, 3, 4, 5]
    total_marks     : int           # e.g. 50

    # ── STRUCTURE CONFIG ──────────────────────────────────────────────────────
    subquestion_count     : int = 2     # 1–4, default=2
    distribution_policy   : str = "BALANCED" # "BALANCED" | "PRIMARY_HEAVY" | "PROGRESSIVE" | "CUSTOM"
    custom_distribution   : Optional[List[int]] = None   # only if CUSTOM

    # ── ACADEMIC MAPPING ──────────────────────────────────────────────────────
    bloom_levels    : List[str] = field(default_factory=lambda: ["L2", "L3"])
    co_mapping      : Dict[int, str] = field(default_factory=dict)
    question_types  : List[str] = field(default_factory=lambda: ["descriptive", "numerical"])

    # ── DOCUMENT ──────────────────────────────────────────────────────────────
    document_id     : str = "doc_001"
    model           : str = "qwen2.5:14b"

    # ── REPRODUCIBILITY ───────────────────────────────────────────────────────
    seed            : Optional[int] = None    # None = generate fresh seed

    # ── VALIDATION RESULT ─────────────────────────────────────────────────────
    validated       : bool = False
    validation_errors : List[str] = field(default_factory=list)

    ALLOWED_EXAM_TYPES : Set[str] = field(default_factory=lambda: {"IAT1", "IAT2", "SEE", "IA"}, init=False)
    ALLOWED_SUBQ_COUNTS : Set[int] = field(default_factory=lambda: {1, 2, 3, 4}, init=False)
    ALLOWED_BLOOM_LEVELS : Set[str] = field(default_factory=lambda: {"L1", "L2", "L3", "L4", "L5", "L6"}, init=False)
    ALLOWED_POLICIES : Set[str] = field(default_factory=lambda: {"BALANCED", "PRIMARY_HEAVY", "PROGRESSIVE", "CUSTOM"}, init=False)

    def validate(self) -> GenerationRequest:
        errors = []

        if not self.subject or not self.subject.strip():
            errors.append("subject is required")
        if not self.department or not self.department.strip():
            errors.append("department is required")
        if self.semester not in range(1, 9):
            errors.append("semester must be 1–8")
        if self.exam_type not in {"IAT1", "IAT2", "SEE", "IA"}:
            errors.append("exam_type must be one of IAT1, IAT2, SEE, IA")
        if not self.modules:
            errors.append("modules list is required")
        if self.total_marks <= 0:
            errors.append("total_marks must be positive")
        if self.modules and self.total_marks % len(self.modules) != 0:
            errors.append(
                f"total_marks ({self.total_marks}) must be divisible "
                f"by module count ({len(self.modules)})"
            )
        if self.subquestion_count not in {1, 2, 3, 4}:
            errors.append("subquestion_count must be 1, 2, 3, or 4")
        if self.distribution_policy not in {"BALANCED", "PRIMARY_HEAVY", "PROGRESSIVE", "CUSTOM"}:
            errors.append("distribution_policy must be BALANCED, PRIMARY_HEAVY, PROGRESSIVE, or CUSTOM")
        if self.distribution_policy == "CUSTOM":
            if not self.custom_distribution:
                errors.append("custom_distribution required when policy=CUSTOM")
            elif len(self.custom_distribution) != self.subquestion_count:
                errors.append("custom_distribution length must equal subquestion_count")
            elif self.modules and sum(self.custom_distribution) != self.total_marks // len(self.modules):
                errors.append("custom_distribution must sum to marks_per_module")

        for b in self.bloom_levels:
            if b not in {"L1", "L2", "L3", "L4", "L5", "L6"}:
                errors.append(f"Invalid bloom level: {b}")
        if not self.document_id:
            errors.append("document_id is required")

        self.validation_errors = errors
        self.validated = (len(errors) == 0)

        if not self.validated:
            raise GenerationRequestValidationError(f"GenerationRequest validation failed: {errors}")

        return self

    def log_trace(self) -> Dict[str, Any]:
        """Returns the full request as a loggable dict for pipeline tracing."""
        return {
            "request_id": self.request_id,
            "subject": self.subject,
            "department": self.department,
            "exam_type": self.exam_type,
            "modules": self.modules,
            "total_marks": self.total_marks,
            "subquestion_count": self.subquestion_count,
            "distribution_policy": self.distribution_policy,
            "custom_distribution": self.custom_distribution,
            "bloom_levels": self.bloom_levels,
            "document_id": self.document_id,
            "model": self.model,
            "seed": self.seed,
        }
