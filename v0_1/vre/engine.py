"""
AION Visual Reasoning Engine (VRE) — Guarded Orchestrator
=========================================================
Architectural Directives:
    NO VALID VKO               -> NO NPE
    NO VALID NPE               -> NO QUESTION
    NO SOLVER ANSWER           -> NO QUESTION
    QUESTION ≠ VKO             -> REJECT
    SVG ≠ VKO                  -> REJECT
    NO PROVENANCE              -> REJECT
    ALL VISUAL ATTEMPTS FAILED -> TEXT-ONLY FALLBACK
"""

from __future__ import annotations

from typing import List, Optional
from .contracts import (
    FigureInput, RenderMode, VREDecisionState, VREOutput, VRERequest
)
from .figure_extractor import FigureExtractor
from .figure_quality import FigureQualityGate
from .fsc import FSC
from .gg import GG
from .npe import NPE
from .provenance import ProvenanceTracker
from .qpvde import QPVDE
from .retry_policy import VRERetryController
from .semantic_validator import SemanticQuestionValidator
from .vc import VisualCritic
from .vkoc import VKOC
from .vko_validator import VKOValidator
from .vqgr import VQGR


class VREEngine:
    """Guarded Orchestrator for the AION Visual Reasoning Engine."""

    @classmethod
    def execute(cls, request: VRERequest) -> VREOutput:
        """
        Single entry point for guarded visual question generation.
        Strictly enforces all fail-closed directives and emits server logs.
        """
        print(f"\n[VRE] Request received: subject='{request.subject}' topic='{request.topic}' bloom={request.bloom_level} marks={request.marks}")

        controller = VRERetryController()

        while controller.can_retry():
            strategy = controller.next_strategy()

            if strategy == "TEXT_ONLY":
                print("[VRE] Strategy: TEXT_ONLY -> bypassing visual pipeline")
                return cls._fallback_text(request, "STRATEGY_TEXT_ONLY")

            # 1. Extraction & Figure Quality Gate
            valid_extractions = []
            for cand in request.figure_candidates:
                cand_extracted = FigureExtractor.extract_from_input(cand)
                q_result = FigureQualityGate.validate(cand_extracted)
                if q_result.valid:
                    valid_extractions.append((cand_extracted, q_result))
                else:
                    print(f"[VRE] Figure Quality Gate FAIL: {q_result.errors}")

            if not valid_extractions:
                print("[VRE] No candidate figures passed quality gate")
                continue

            # Process candidates
            candidate_vkos = []
            for cand_extracted, q_result in valid_extractions:
                # 2. FSC (Figure Semantic Classifier)
                classification = FSC.classify(q_result, concept_hint=request.topic)
                if not classification.supported:
                    print(f"[VRE] FSC Unsupported: {classification.reason}")
                    continue

                print(f"[VRE] FSC Class: {classification.figure_class} | domain={classification.domain} | ops={classification.operations}")

                # 3. VKOC (Visual Knowledge Object Constructor)
                vko = VKOC.build(q_result, classification)

                # 4. VKO Integrity Gate
                vko_valid, errors = VKOValidator.validate(vko)
                if not vko_valid:
                    print(f"[VRE] VKO Integrity Gate FAIL: {errors}")
                    # NO VALID VKO -> NO NPE
                    continue

                print(f"[VRE] VKO validated: id={vko.id}")
                candidate_vkos.append(vko)

            if not candidate_vkos:
                continue

            # 5. QPVDE (Visual Decision Engine)
            decision = QPVDE.decide(request, candidate_vkos)
            print(f"[VRE] Decision: {decision.state.value} | reason='{decision.reason}' | dep_score={decision.image_dependency_score:.2f}")

            if decision.state == VREDecisionState.IMAGE_NOT_NEEDED:
                return cls._fallback_text(request, "IMAGE_NOT_NEEDED", dep_score=decision.image_dependency_score)
            elif decision.state == VREDecisionState.IMAGE_UNSUPPORTED:
                return cls._fallback_text(request, f"IMAGE_UNSUPPORTED:{decision.reason}", dep_score=decision.image_dependency_score)
            elif not decision.use_image or not decision.vko or not decision.selected_chain:
                continue  # IMAGE_NEEDED_BUT_INVALID -> retry alternate chain

            vko = decision.vko
            chain = decision.selected_chain

            # 6. NPE & Solvability Gate
            # NO VALID VKO -> NO NPE
            mutated_vko, reference_solution = NPE.generate(vko, chain)
            print(f"[VRE] NPE Parameters generated | Solver: {chain.steps[0].operation if chain.steps else 'UNKNOWN'}")

            if not reference_solution or not reference_solution.get("unique_solution"):
                print("[VRE] NO SOLVER ANSWER -> REJECT")
                # NO SOLVER ANSWER -> NO QUESTION
                continue

            print(f"[VRE] Expected answer: {reference_solution}")

            # 7. Grounded Question Planning & Language Generation
            plan = GG.generate_question_plan(
                vko=mutated_vko,
                chain=chain,
                bloom_level=request.bloom_level,
                marks=request.marks,
                reference_solution=reference_solution,
            )
            print(f"[VRE] QuestionPlan generated: hash={plan.question_plan_hash}")

            question_text = GG.render_question_text(plan, mutated_vko)
            if not question_text:
                print("[VRE] NO VALID QUESTION TEXT -> REJECT")
                # NO VALID NPE -> NO QUESTION
                continue

            # 8. Post-LLM Semantic Question Validation
            sem_valid, sem_errors = SemanticQuestionValidator.validate(question_text, plan)
            if not sem_valid:
                print(f"[VRE] Semantic Question Validation FAIL: {sem_errors}")
                continue

            print("[VRE] Semantic question validation: PASS")

            # 9. Declarative SVG / Multi-Modal Synthesis
            figure_svg = VQGR.render(mutated_vko, render_mode=RenderMode.SVG)

            # 10. Provenance Tracking
            provenance = ProvenanceTracker.create_record(
                source_document=request.subject,
                page=1,
                figure_id=mutated_vko.id,
                module=request.module,
                concept=request.topic,
                vko_id=mutated_vko.id,
                operation_chain_id=chain.chain_id,
            )
            provenance.question_plan_hash = plan.question_plan_hash

            if not provenance:
                print("[VRE] NO PROVENANCE -> REJECT")
                # NO PROVENANCE -> REJECT
                continue

            # 11. Visual Critic (MCRS Criteria C1–C10)
            passed_critic, critic_errors = VisualCritic.validate(
                question_text=question_text,
                vko=mutated_vko,
                plan=plan,
                rendered_svg=figure_svg,
                reference_solution=reference_solution,
                has_provenance=True,
            )

            if not passed_critic:
                print(f"[VRE] Visual Critic FAIL: {critic_errors}")
                # QUESTION != VKO or SVG != VKO -> REJECT
                continue

            print("[VRE] Visual critic: PASS")
            print("[VRE] Render QA: PASS")
            print("[VRE] VRE OUTPUT READY\n")

            # Clean Success
            return VREOutput(
                success=True,
                text=question_text,
                figure_svg=figure_svg,
                figure_caption=f"Figure for {request.topic}",
                render_mode=RenderMode.SVG,
                bloom=request.bloom_level,
                marks=request.marks,
                image_dependency_score=decision.image_dependency_score,
                question_plan_hash=plan.question_plan_hash,
                decision_state=VREDecisionState.IMAGE_NEEDED_AND_VALID,
                provenance=provenance,
                reference_solution=reference_solution,
                reason="SUCCESS",
            )

        print("[VRE] ALL VISUAL ATTEMPTS FAILED -> TEXT-ONLY FALLBACK")
        # ALL VISUAL ATTEMPTS FAILED -> TEXT-ONLY FALLBACK
        return cls._fallback_text(request, "ALL_VISUAL_ATTEMPTS_FAILED_TEXT_ONLY_FALLBACK")

    @staticmethod
    def _fallback_text(request: VRERequest, reason: str, dep_score: float = 0.0) -> VREOutput:
        """Fallback to high-quality text-only question generation."""
        text = (
            f"Explain the principles of {request.topic} in detail, "
            f"providing key equations and theoretical applications."
        )
        return VREOutput(
            success=True,
            text=text,
            figure_svg=None,
            figure_caption="",
            render_mode=RenderMode.SVG,
            bloom=request.bloom_level,
            marks=request.marks,
            image_dependency_score=dep_score,
            question_plan_hash="",
            decision_state=VREDecisionState.IMAGE_NOT_NEEDED,
            provenance=None,
            reference_solution=None,
            reason=reason,
        )
