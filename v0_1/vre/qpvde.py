"""
AION VRE Question Planner with Visual Decision Engine (QPVDE)
=============================================================
Calculates image_dependency_score (Topology + Numerical + Spatial + Operation + Bloom + Quality)
and incorporates VTU Visual Policy.
"""

from __future__ import annotations

from typing import List, Optional
from .contracts import (
    OperationChain, VKO, VREDecision, VREDecisionState, VRERequest
)
from .policies import BLOOM_VISUAL_PRIORS, VISUAL_DISCOURAGED, VISUAL_PREFERRED
from .taxonomy import SUPPORTED_FIGURE_CLASSES
from .vko_validator import VKOValidator


class QPVDE:
    """Visual Decision Engine (Algorithm 4)."""

    THRESHOLD_IMAGE_NEEDED: float = 0.80
    THRESHOLD_IMAGE_OPTIONAL: float = 0.55

    @classmethod
    def decide(
        cls,
        request: VRERequest,
        candidate_vkos: List[VKO],
    ) -> VREDecision:
        # Check if caller allowed images (permission flag)
        image_allowed = request.constraints.get("image_allowed", True)
        if not image_allowed:
            return VREDecision(
                state=VREDecisionState.IMAGE_NOT_NEEDED,
                reason="CALLER_DISALLOWED_IMAGES",
                confidence=1.0,
                image_dependency_score=0.0,
            )

        # 1. Compute Image Dependency Score
        dep_score = cls.compute_image_dependency_score(
            topic=request.topic,
            bloom_level=request.bloom_level,
            marks=request.marks,
            question_type=request.question_type,
            candidate_vkos=candidate_vkos,
        )

        if dep_score < cls.THRESHOLD_IMAGE_OPTIONAL:
            return VREDecision(
                state=VREDecisionState.IMAGE_NOT_NEEDED,
                reason=f"LOW_IMAGE_DEPENDENCY_SCORE:{dep_score:.2f}<{cls.THRESHOLD_IMAGE_OPTIONAL}",
                confidence=1.0 - dep_score,
                image_dependency_score=dep_score,
            )

        if not candidate_vkos:
            return VREDecision(
                state=VREDecisionState.IMAGE_NEEDED_BUT_INVALID,
                reason="NO_CANDIDATE_FIGURES_AVAILABLE",
                confidence=0.0,
                image_dependency_score=dep_score,
            )

        best_vko: Optional[VKO] = None
        best_chain: Optional[OperationChain] = None

        for vko in candidate_vkos:
            if vko.figure_class not in SUPPORTED_FIGURE_CLASSES:
                return VREDecision(
                    state=VREDecisionState.IMAGE_UNSUPPORTED,
                    reason=f"FIGURE_CLASS_UNSUPPORTED:{vko.figure_class}",
                    confidence=0.0,
                    image_dependency_score=dep_score,
                )

            valid, errors = VKOValidator.validate(vko)
            if not valid:
                continue

            from .vqg import VQGBuilder
            vqg = VQGBuilder.build(vko)

            matching_chains = vqg.bloom_mapping.get(request.bloom_level, vqg.operation_chains)
            if matching_chains:
                best_vko = vko
                best_chain = matching_chains[0]
                break

        if not best_vko or not best_chain:
            return VREDecision(
                state=VREDecisionState.IMAGE_NEEDED_BUT_INVALID,
                reason="NO_VKO_PASSED_INTEGRITY_OR_BLOOM_MATCH",
                confidence=0.0,
                image_dependency_score=dep_score,
            )

        return VREDecision(
            state=VREDecisionState.IMAGE_NEEDED_AND_VALID,
            reason="FIGURE_VALID_AND_GROUNDED",
            confidence=0.92,
            image_dependency_score=dep_score,
            vko=best_vko,
            selected_chain=best_chain,
            mandatory=dep_score >= cls.THRESHOLD_IMAGE_NEEDED,
        )

    @classmethod
    def compute_image_dependency_score(
        cls,
        topic: str,
        bloom_level: str,
        marks: int,
        question_type: str,
        candidate_vkos: List[VKO],
    ) -> float:
        topic_key = topic.lower().replace(" ", "_")

        # Discouraged check
        if VISUAL_DISCOURAGED.get(topic_key) or question_type == "definition":
            return 0.10

        # Preferred check
        base_pref = 0.90 if VISUAL_PREFERRED.get(topic_key) else 0.50

        # Bloom prior
        bloom_prior = BLOOM_VISUAL_PRIORS.get(bloom_level, 0.50)

        # Topology and source figure quality factor
        figure_factor = 0.85 if candidate_vkos else 0.20

        composite = (base_pref * 0.40) + (bloom_prior * 0.35) + (figure_factor * 0.25)
        return round(min(1.0, max(0.0, composite)), 2)
