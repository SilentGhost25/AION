"""
Assessment Configuration Registry for AION v3.

Defines the AssessmentConfig dataclass and the thread-safe AssessmentRegistry.
Enforces strict 4-tuple identity: (scheme_id, course_type, assessment_type, blueprint_id).
All configurations are immutable once frozen.
"""

from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

ID_PATTERN = re.compile(r"^[A-Z0-9_]+$")


@dataclass(frozen=True)
class CognitiveTargetDistribution:
    l1_percentage: float = 0.0
    l2_percentage: float = 0.0
    l3_percentage: float = 0.0
    l4_percentage: float = 0.0
    l5_percentage: float = 0.0
    l6_percentage: float = 0.0
    tolerance_percentage: float = 5.0

    def __post_init__(self) -> None:
        targets = [self.l1_percentage, self.l2_percentage, self.l3_percentage,
                   self.l4_percentage, self.l5_percentage, self.l6_percentage]
        for idx, t in enumerate(targets, 1):
            if type(t) not in (int, float) or type(t) is bool:
                raise TypeError(f"Target L{idx} must be int or float, got {type(t).__name__}")
            if math.isnan(t) or math.isinf(t) or t < 0.0:
                raise ValueError(f"Target L{idx} must be non-negative real, got {t!r}")
        tol = self.tolerance_percentage
        if type(tol) not in (int, float) or type(tol) is bool or tol < 0.0:
            raise TypeError("tolerance_percentage must be non-negative real")
        total = sum(targets)
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Cognitive targets must sum to 100%, got {total:.2f}%")


@dataclass(frozen=True)
class AssessmentConfig:
    scheme_id: str = "VTU_2022_NEP"                          # e.g., "VTU_2022_NEP"
    course_type: str = "ENGINEERING_THEORY"                  # "AEC_GENERAL_THEORY" | "ENGINEERING_THEORY" | "SYNTHETIC_TEST"
    assessment_type: str = "CIE_IAT"                         # "CIE_IAT" | "SEE" | "TEST_MODE" | "IA"
    blueprint_id: str = "DEFAULT_BLUEPRINT"                  # e.g., "VTU_AEC_ANNEXURE_II" | "TEST_BP_3PART"
    total_marks_per_question: int = 10                       # e.g., 10 for CIE, 20 for SEE
    permitted_partitions: Tuple[Tuple[int, ...], ...] = ((5, 5), (6, 4), (10,))  # e.g., ((5, 5), (4, 6), (3, 7), (10,))
    max_subparts: int = 2                                   # e.g., 2 for CIE, 3 for SEE (strictly 1 <= max_subparts <= 26)
    reference_source: str = "VTU 2022 Scheme Regulations and Model Question Paper Blueprint" # Mandatory auditable citation
    allow_choice: Optional[bool] = None                     # Optional; omitted if not established by cited provision
    questions_per_module: Optional[int] = None              # Optional; omitted if not established by cited provision
    cognitive_target: Optional[CognitiveTargetDistribution] = None # Stage G6
    mandatory_course_outcomes: Optional[Any] = None         # Stage G6: tuple, list, or set of str
    bloom_preference: Optional[str] = None                  # Stage G6: None | "HIGHER" | "LOWER"
    allow_synthetic: bool = False                           # Stage G6: Synthetic testing bypass
    generate_solutions: bool = True                         # Stage G6: Solution generation requested
    generate_rubrics: bool = True                           # Stage G6: Rubric generation requested

    def __post_init__(self) -> None:
        for attr_name in ("scheme_id", "course_type", "assessment_type", "blueprint_id"):
            val = getattr(self, attr_name)
            if type(val) is not str or not ID_PATTERN.match(val):
                raise ValueError(f"{attr_name} must match {ID_PATTERN.pattern}, got {val!r}")

        if type(self.total_marks_per_question) is not int or isinstance(self.total_marks_per_question, bool) or self.total_marks_per_question <= 0:
            raise ValueError(f"total_marks_per_question must be positive int, got {self.total_marks_per_question!r}")

        if type(self.max_subparts) is not int or isinstance(self.max_subparts, bool) or not (1 <= self.max_subparts <= 26):
            raise ValueError(f"max_subparts must be int between 1 and 26, got {self.max_subparts!r}")

        if type(self.permitted_partitions) is not tuple or not self.permitted_partitions:
            raise ValueError("permitted_partitions must be a non-empty tuple of tuples")

        if len(self.permitted_partitions) != len(set(self.permitted_partitions)):
            raise ValueError(f"permitted_partitions contains duplicate entries: {self.permitted_partitions}")

        for idx, part in enumerate(self.permitted_partitions):
            if type(part) is not tuple or not part:
                raise ValueError(f"permitted_partitions[{idx}] must be a non-empty tuple")
            if len(part) > self.max_subparts:
                raise ValueError(f"permitted_partitions[{idx}] length {len(part)} > max_subparts {self.max_subparts}")
            for m in part:
                if type(m) is not int or isinstance(m, bool) or m <= 0:
                    raise ValueError(f"Partition marks must be positive exact int, got {m!r} in {part}")
            if sum(part) != self.total_marks_per_question:
                raise ValueError(
                    f"Partition {part} sum ({sum(part)}) != total_marks_per_question ({self.total_marks_per_question})"
                )

        if type(self.reference_source) is not str or len(self.reference_source.strip()) < 10:
            raise ValueError("reference_source must be an explicit, auditable citation string (>= 10 chars)")

        if self.allow_choice is not None and type(self.allow_choice) is not bool:
            raise TypeError(f"allow_choice must be bool or None, got {type(self.allow_choice).__name__}")

        if self.questions_per_module is not None:
            if type(self.questions_per_module) is not int or isinstance(self.questions_per_module, bool) or self.questions_per_module <= 0:
                raise ValueError(f"questions_per_module must be positive exact int or None, got {self.questions_per_module!r}")

        if self.cognitive_target is not None and not isinstance(self.cognitive_target, CognitiveTargetDistribution):
            raise TypeError(f"cognitive_target must be CognitiveTargetDistribution or None, got {type(self.cognitive_target).__name__}")

        if self.mandatory_course_outcomes is not None:
            if type(self.mandatory_course_outcomes) not in (tuple, list, set, frozenset):
                raise TypeError(f"mandatory_course_outcomes must be tuple, list, or set or None, got {type(self.mandatory_course_outcomes).__name__}")
            for co_name in self.mandatory_course_outcomes:
                if type(co_name) is not str or not co_name.strip():
                    raise TypeError("mandatory_course_outcomes items must be non-empty strings")

        if self.bloom_preference is not None and self.bloom_preference not in ("HIGHER", "LOWER"):
            raise ValueError(f"bloom_preference must be 'HIGHER', 'LOWER', or None, got {self.bloom_preference!r}")

        if type(self.allow_synthetic) is not bool:
            raise TypeError(f"allow_synthetic must be bool, got {type(self.allow_synthetic).__name__}")
        if type(self.generate_solutions) is not bool:
            raise TypeError(f"generate_solutions must be bool, got {type(self.generate_solutions).__name__}")
        if type(self.generate_rubrics) is not bool:
            raise TypeError(f"generate_rubrics must be bool, got {type(self.generate_rubrics).__name__}")

    @property
    def identity(self) -> Tuple[str, str, str, str]:
        return (self.scheme_id, self.course_type, self.assessment_type, self.blueprint_id)

    @property
    def identity_key(self) -> str:
        return f"{self.scheme_id}:{self.course_type}:{self.assessment_type}:{self.blueprint_id}"


class AssessmentRegistry:
    """Registry of verified university assessment configurations."""
    _REGISTRY: Dict[str, AssessmentConfig] = {}
    _FROZEN: bool = False
    _LOCK: threading.RLock = threading.RLock()

    @classmethod
    def _make_key(cls, scheme_id: str, course_type: str, assessment_type: str, blueprint_id: str) -> str:
        s = str(scheme_id).strip().upper()
        c = str(course_type).strip().upper()
        a = str(assessment_type).strip().upper()
        b = str(blueprint_id).strip().upper()
        for name, val in (("scheme_id", s), ("course_type", c), ("assessment_type", a), ("blueprint_id", b)):
            if not ID_PATTERN.match(val):
                raise ValueError(f"Invalid identifier {name}={val!r}; must match {ID_PATTERN.pattern}")
        return f"{s}:{c}:{a}:{b}"

    @classmethod
    def register(cls, config: AssessmentConfig) -> None:
        with cls._LOCK:
            if cls._FROZEN:
                raise RuntimeError("AssessmentRegistry is frozen; no new configurations may be registered.")
            key = cls._make_key(config.scheme_id, config.course_type, config.assessment_type, config.blueprint_id)
            if key in cls._REGISTRY:
                if cls._REGISTRY[key] == config:
                    return
                raise ValueError(f"Assessment configuration already registered for key: '{key}' with different configuration")
            cls._REGISTRY[key] = config

    @classmethod
    def get(cls, scheme_id: str, course_type: str, assessment_type: str, blueprint_id: str) -> AssessmentConfig:
        with cls._LOCK:
            key = cls._make_key(scheme_id, course_type, assessment_type, blueprint_id)
            if key not in cls._REGISTRY:
                raise KeyError(f"No assessment configuration registered for '{key}'. Automatic fallback is strictly prohibited.")
            return cls._REGISTRY[key]

    @classmethod
    def freeze(cls) -> None:
        with cls._LOCK:
            cls._FROZEN = True

    @classmethod
    def is_frozen(cls) -> bool:
        with cls._LOCK:
            return cls._FROZEN

    @classmethod
    def _reset_for_tests(cls) -> None:
        if not os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("AION_TEST_MODE"):
            raise RuntimeError("_reset_for_tests() is an internal test helper strictly restricted to test environments.")
        with cls._LOCK:
            cls._REGISTRY.clear()
            cls._FROZEN = False


# Verified AEC Baseline Registration
AssessmentRegistry.register(AssessmentConfig(
    scheme_id="VTU_2022_NEP",
    course_type="AEC_GENERAL_THEORY",
    assessment_type="CIE_IAT",
    blueprint_id="VTU_AEC_ANNEXURE_II",
    total_marks_per_question=10,
    permitted_partitions=((5, 5), (4, 6), (3, 7), (10,)),
    max_subparts=2,
    reference_source="VTU 2022 Scheme AEC Regulations (2024), Annexure-II, Clause 4(a) (10M questions, max 2 subquestions: 5+5, 4+6, 3+7, 10)",
    allow_choice=None,
    questions_per_module=None,
))

# Automatically lock registry post-initialization during bootstrap
AssessmentRegistry.freeze()
