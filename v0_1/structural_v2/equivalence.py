"""
AION Structural Architecture v2 — Equivalence Builder & Question Type Matrix
=============================================================================
OR-Pair Academic Equivalence Builder, Difficulty Resolver, and Domain Question Type Matrix.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .contracts import (
    AlternativeEquivalenceProfile,
    BloomLevel,
    DifficultyBand,
    StructuralSignature,
    VisualPrior,
)


class QuestionTypeError(Exception):
    """Raised when no matching domain question type is found."""
    pass


class DifficultyResolver:
    """Resolves numerical difficulty band from marks and Bloom level."""

    @classmethod
    def resolve(cls, marks: int, bloom: BloomLevel) -> DifficultyBand:
        score = (marks / 10.0) * 0.5 + (bloom.value / 6.0) * 0.5
        if score < 0.35:
            return DifficultyBand.EASY
        elif score < 0.65:
            return DifficultyBand.MEDIUM
        else:
            return DifficultyBand.HARD


@dataclass
class TypeRecord:
    name: str
    blooms: List[BloomLevel]
    marks_range: Tuple[int, int]
    prior: VisualPrior


class DomainQuestionTypeMatrix:
    """Domain-specific question type matrix for ECE/EEE, CSE/AIML, MECH, and CIVIL."""

    REGISTRY: Dict[str, List[TypeRecord]] = {
        "ECE": [
            TypeRecord("CIRCUIT_NUMERICAL", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.OPTIONAL),
            TypeRecord("KVL_ANALYSIS", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("KCL_ANALYSIS", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("WAVEFORM_ANALYSIS", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("BLOCK_DIAGRAM_ANALYSIS", [BloomLevel.L2, BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.PREFERRED),
            TypeRecord("SIGNAL_CALCULATION", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.OPTIONAL),
            TypeRecord("PHASOR_CALCULATION", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.OPTIONAL),
            TypeRecord("TRANSFORMER_CALCULATION", [BloomLevel.L3, BloomLevel.L4, BloomLevel.L5], (5, 10), VisualPrior.OPTIONAL),
            TypeRecord("CONCEPTUAL_ECE", [BloomLevel.L1, BloomLevel.L2, BloomLevel.L4, BloomLevel.L5], (2, 10), VisualPrior.FORBIDDEN),
            TypeRecord("SYSTEM_DESCRIPTION", [BloomLevel.L2, BloomLevel.L4, BloomLevel.L5], (4, 10), VisualPrior.OPTIONAL),
        ],
        "CSE": [
            TypeRecord("ALGORITHM_TRACE", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.FORBIDDEN),
            TypeRecord("GRAPH_SHORTEST_PATH", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("MST_CONSTRUCTION", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("AVL_ROTATION", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.OPTIONAL),
            TypeRecord("TREE_TRAVERSAL", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.OPTIONAL),
            TypeRecord("HASHING_ANALYSIS", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.FORBIDDEN),
            TypeRecord("SORT_TRACE", [BloomLevel.L3, BloomLevel.L4], (4, 10), VisualPrior.FORBIDDEN),
            TypeRecord("CODE_DESIGN", [BloomLevel.L5, BloomLevel.L6], (6, 10), VisualPrior.FORBIDDEN),
            TypeRecord("CONCEPTUAL_CS", [BloomLevel.L1, BloomLevel.L2, BloomLevel.L3, BloomLevel.L4, BloomLevel.L5], (1, 10), VisualPrior.FORBIDDEN),
        ],
        "MECH": [
            TypeRecord("BEAM_REACTION", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("SFD_BMD", [BloomLevel.L3, BloomLevel.L4], (6, 10), VisualPrior.PREFERRED),
            TypeRecord("THERMODYNAMIC_CALC", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.OPTIONAL),
            TypeRecord("HEAT_TRANSFER", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.OPTIONAL),
            TypeRecord("MECHANISM_ANALYSIS", [BloomLevel.L4, BloomLevel.L5], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("CONCEPTUAL_MECH", [BloomLevel.L1, BloomLevel.L2, BloomLevel.L4, BloomLevel.L5], (2, 10), VisualPrior.OPTIONAL),
        ],
        "CIVIL": [
            TypeRecord("STRUCTURAL_ANALYSIS", [BloomLevel.L3, BloomLevel.L4, BloomLevel.L5], (5, 10), VisualPrior.PREFERRED),
            TypeRecord("SFD_BMD", [BloomLevel.L3, BloomLevel.L4], (6, 10), VisualPrior.PREFERRED),
            TypeRecord("CONCRETE_DESIGN", [BloomLevel.L4, BloomLevel.L5, BloomLevel.L6], (6, 10), VisualPrior.OPTIONAL),
            TypeRecord("SURVEY_CALCULATION", [BloomLevel.L3, BloomLevel.L4], (5, 10), VisualPrior.OPTIONAL),
            TypeRecord("CONCEPTUAL_CIVIL", [BloomLevel.L1, BloomLevel.L2, BloomLevel.L4, BloomLevel.L5], (2, 10), VisualPrior.OPTIONAL),
        ],
    }

    # Normalize aliases
    ALIAS_MAP = {
        "EEE": "ECE",
        "AIML": "CSE",
        "CS": "CSE",
        "COMPUTER SCIENCE": "CSE",
        "MECHANICAL": "MECH",
    }

    @classmethod
    def get(cls, domain: str, bloom: BloomLevel, marks: int) -> List[str]:
        dom = domain.upper()
        dom = cls.ALIAS_MAP.get(dom, dom)

        type_set = cls.REGISTRY.get(dom, cls.REGISTRY["CSE"])
        candidates = [
            t for t in type_set
            if bloom in t.blooms and t.marks_range[0] <= marks <= t.marks_range[1]
        ]

        if not candidates:
            # Fallback relaxation
            candidates = [t for t in type_set if t.marks_range[0] <= marks <= t.marks_range[1]]

        if not candidates:
            raise QuestionTypeError(f"No type candidates for domain '{domain}', bloom {bloom}, marks {marks}")

        return [t.name for t in candidates]

    @classmethod
    def get_visual_prior(cls, domain: str, question_type: str) -> VisualPrior:
        dom = domain.upper()
        dom = cls.ALIAS_MAP.get(dom, dom)
        type_set = cls.REGISTRY.get(dom, cls.REGISTRY["CSE"])
        for t in type_set:
            if t.name == question_type:
                return t.prior
        return VisualPrior.OPTIONAL


class ORPairEquivalenceBuilder:
    """Builds identical AlternativeEquivalenceProfile for both alternatives in an OR pair."""

    @classmethod
    def build(
        cls,
        σ: StructuralSignature,
        P: Tuple[BloomLevel, ...],
        domain: str,
        rng: random.Random,
    ) -> AlternativeEquivalenceProfile:

        # Step 1 — Difficulty Profile
        diff_list = [DifficultyResolver.resolve(m, b) for m, b in zip(σ.mark_distribution, P)]
        difficulty_profile = tuple(diff_list)

        # Step 2 — Question Type Profile
        type_list = []
        for m, b in zip(σ.mark_distribution, P):
            candidates = DomainQuestionTypeMatrix.get(domain, b, m)
            chosen = rng.choice(candidates)
            type_list.append(chosen)
        question_type_profile = tuple(type_list)

        # Step 3 — Cognitive Weights
        cognitive_weights = tuple(m / float(σ.total_marks) for m in σ.mark_distribution)

        # Step 4 — Profile Assembly
        profile = AlternativeEquivalenceProfile(
            bloom_profile=P,
            difficulty_profile=difficulty_profile,
            question_type_profile=question_type_profile,
            cognitive_weights=cognitive_weights,
        )

        return profile
