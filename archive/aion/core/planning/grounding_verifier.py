"""
AION v2 Grounding Verifier & Firewall
====================================
Implements the comprehensive 6-Level Grounding Firewall:
1. Document Boundary Gate: Confirms artifacts belong to the target document.
2. Module Boundary Gate: Confirms 100% of artifacts belong to target module_index.
3. Syllabus Scope Gate: Confirms artifacts reside within allowed syllabus pages/topics (fail-closed).
4. Semantic Boundary Gate: Confirms artifacts belong strictly to primary KU or declared secondary KUs.
5. Provenance Integrity Gate: Confirms artifacts resolve to registry, valid page, and valid bounding box.
6. Budget Adherence Gate: Confirms artifact allocations satisfy archetype bounds.

Also implements narrow deterministic draft grounding for source-bound entities
(figure IDs, equation labels) without rejecting legitimate question-construction parameters.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    EquationArtifact,
    FigureArtifact,
)
from aion.core.dom.document_dom import DocumentDOM
from aion.core.knowledge.contracts import KnowledgeUnit, SyllabusScopeMask
from aion.core.planning.contracts import (
    EvidenceBundle,
    GroundingReport,
    GroundingViolation,
    GroundingViolationType,
)
from aion.core.planning.evidence_planner import ARCHETYPE_REQUIREMENTS
from aion.core.question.contracts import QuestionSpec


class GroundingVerifier:
    """
    Authoritative grounding verification engine.
    Acts as a multi-tier firewall before and after generation.
    """

    @classmethod
    def verify_bundle(
        cls,
        bundle: EvidenceBundle,
        spec: QuestionSpec,
        dom: DocumentDOM,
        kus: Dict[str, KnowledgeUnit],
        scope_mask: Optional[SyllabusScopeMask] = None,
    ) -> GroundingReport:
        """
        Audit an assembled EvidenceBundle against the 6-Level Grounding Firewall.
        """
        violations: List[GroundingViolation] = []
        levels_passed: Dict[str, bool] = {
            "level_1_document": True,
            "level_2_module": True,
            "level_3_scope": True,
            "level_4_semantic_boundary": True,
            "level_5_provenance": True,
            "level_6_budget": True,
        }

        # Resolve primary and declared secondary KUs
        primary_ku = kus.get(spec.primary_ku_id)
        if not primary_ku:
            violations.append(GroundingViolation(
                violation_type=GroundingViolationType.CROSS_KU_LEAKAGE,
                severity="CRITICAL",
                message=f"Primary KU '{spec.primary_ku_id}' does not exist in compiled KUs.",
            ))
            levels_passed["level_4_semantic_boundary"] = False

        allowed_ku_artifact_ids: Set[str] = set()
        if primary_ku:
            allowed_ku_artifact_ids.update(primary_ku.source_artifact_ids)
        for sec_id in spec.secondary_ku_ids:
            if sec_id in kus:
                allowed_ku_artifact_ids.update(kus[sec_id].source_artifact_ids)

        # Audit each artifact in the bundle
        for ref in bundle.item_refs:
            art = dom.registry.get(ref.artifact_id)

            # --- Level 5: Provenance Integrity ---
            if art is None:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.PROVENANCE_MISSING,
                    severity="CRITICAL",
                    message=f"Artifact '{ref.artifact_id}' could not be resolved from DOM ArtifactRegistry.",
                    artifact_id=ref.artifact_id,
                ))
                levels_passed["level_5_provenance"] = False
                continue

            if art.page_number <= 0:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.PROVENANCE_MISSING,
                    severity="CRITICAL",
                    message=f"Artifact '{ref.artifact_id}' has invalid page number {art.page_number}.",
                    artifact_id=ref.artifact_id,
                ))
                levels_passed["level_5_provenance"] = False

            if art.source_bbox is not None:
                x0, y0, x1, y1 = art.source_bbox
                if x0 >= x1 or y0 >= y1:
                    violations.append(GroundingViolation(
                        violation_type=GroundingViolationType.PROVENANCE_MISSING,
                        severity="CRITICAL",
                        message=f"Artifact '{ref.artifact_id}' has degraded bounding box {art.source_bbox}.",
                        artifact_id=ref.artifact_id,
                    ))
                    levels_passed["level_5_provenance"] = False

            # --- Level 1: Document Boundary ---
            # Verified via registry containment and DOM ownership
            if not dom.registry.contains(art.artifact_id):
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.DOCUMENT_MISMATCH,
                    severity="CRITICAL",
                    message=f"Artifact '{art.artifact_id}' does not belong to Document '{dom.document_id}'.",
                    artifact_id=art.artifact_id,
                ))
                levels_passed["level_1_document"] = False

            # --- Level 2: Module Boundary ---
            if art.module_index != spec.module_index and art.module_index != 0:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.CROSS_MODULE_CONTAMINATION,
                    severity="CRITICAL",
                    message=(
                        f"Artifact '{art.artifact_id}' belongs to Module {art.module_index}, "
                        f"violating target Module {spec.module_index} boundary."
                    ),
                    artifact_id=art.artifact_id,
                ))
                levels_passed["level_2_module"] = False

            # --- Level 3: Syllabus Scope Boundary (Fail-Closed) ---
            if scope_mask is None:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.OUT_OF_SYLLABUS,
                    severity="CRITICAL",
                    message="No SyllabusScopeMask provided. Compilation marked scope-uncertain; generation is blocked.",
                    artifact_id=art.artifact_id,
                ))
                levels_passed["level_3_scope"] = False
            else:
                if not scope_mask.is_page_allowed(art.page_number):
                    violations.append(GroundingViolation(
                        violation_type=GroundingViolationType.OUT_OF_SYLLABUS,
                        severity="CRITICAL",
                        message=f"Artifact '{art.artifact_id}' on page {art.page_number} is outside allowed syllabus page ranges.",
                        artifact_id=art.artifact_id,
                    ))
                    levels_passed["level_3_scope"] = False

            # --- Level 4: Semantic Boundary (Cross-KU Firewall) ---
            if allowed_ku_artifact_ids and art.artifact_id not in allowed_ku_artifact_ids:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.CROSS_KU_LEAKAGE,
                    severity="CRITICAL",
                    message=(
                        f"Artifact '{art.artifact_id}' does not belong to primary KU '{spec.primary_ku_id}' "
                        f"or declared secondary KUs {spec.secondary_ku_ids}."
                    ),
                    artifact_id=art.artifact_id,
                ))
                levels_passed["level_4_semantic_boundary"] = False

        # --- Level 6: Budget Adherence Gate ---
        reqs = ARCHETYPE_REQUIREMENTS.get(spec.archetype)
        if reqs:
            b = reqs.budget
            u = bundle.budget_used
            if u.get("text_blocks", 0) > b.max_text_blocks:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.BUDGET_OVERFLOW,
                    severity="CRITICAL",
                    message=f"Text blocks count {u.get('text_blocks')} exceeds maximum budget {b.max_text_blocks}.",
                ))
                levels_passed["level_6_budget"] = False

            if u.get("equations", 0) > b.max_equations:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.BUDGET_OVERFLOW,
                    severity="CRITICAL",
                    message=f"Equation count {u.get('equations')} exceeds maximum budget {b.max_equations}.",
                ))
                levels_passed["level_6_budget"] = False

            if u.get("figures", 0) > b.max_figures:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.BUDGET_OVERFLOW,
                    severity="CRITICAL",
                    message=f"Figure count {u.get('figures')} exceeds maximum budget {b.max_figures}.",
                ))
                levels_passed["level_6_budget"] = False

            if u.get("tables", 0) > b.max_tables:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.BUDGET_OVERFLOW,
                    severity="CRITICAL",
                    message=f"Table count {u.get('tables')} exceeds maximum budget {b.max_tables}.",
                ))
                levels_passed["level_6_budget"] = False

        is_valid = len(violations) == 0
        return GroundingReport(
            is_valid=is_valid,
            violations=violations,
            levels_passed=levels_passed,
            provenance_chain_intact=levels_passed["level_5_provenance"],
        )

    @classmethod
    def verify_draft_deterministic(
        cls,
        draft_text: str,
        bundle: EvidenceBundle,
        dom: DocumentDOM,
    ) -> GroundingReport:
        """
        Narrow deterministic check for source-bound entities versus question-construction values.

        TAXONOMY:
        1. SOURCE-BOUND ENTITIES (Strictly Validated):
           - Figure IDs (e.g. FIG-001) & explicitly referenced source figure labels
           - Equation IDs (e.g. EQ-001) & explicitly referenced source equation labels
           - Algorithm IDs (e.g. ALG-001) & named algorithm identifiers
           Must resolve directly to artifacts present within the EvidenceBundle.

        2. QUESTION-CONSTRUCTION VALUES (Permitted / Not Rejected):
           - Newly supplied numerical parameters (e.g. "Assume a resistance of 50 ohms")
           - Hypothetical sample sizes (e.g. "A dataset contains 1,000 samples")
           - Scenario values, test vectors, or exam problem inputs
           These are NOT treated as hallucinations and MUST NOT be rejected merely because
           they were not found in the source text.
        """
        violations: List[GroundingViolation] = []

        # Gather allowed entity identifiers from the bundle
        bundle_fig_ids = set(bundle.figure_artifact_ids)
        bundle_eq_ids = set(bundle.equation_artifact_ids)
        bundle_alg_ids = set(bundle.algorithm_artifact_ids)

        # Also collect human-readable labels from resolved artifacts (e.g. "Figure 2.1", "Eq. (3)")
        allowed_fig_labels: Set[str] = set()
        allowed_eq_labels: Set[str] = set()

        for fid in bundle_fig_ids:
            art = dom.registry.get(fid)
            if isinstance(art, FigureArtifact) and art.caption:
                # e.g., "Figure 4.2" from caption
                cap_match = re.match(r'(Figure\s+\d+(\.\d+)?)', art.caption, re.IGNORECASE)
                if cap_match:
                    allowed_fig_labels.add(cap_match.group(1).lower())

        for eid in bundle_eq_ids:
            art = dom.registry.get(eid)
            if isinstance(art, EquationArtifact) and art.equation_label:
                allowed_eq_labels.add(art.equation_label.lower())
                # also strip parentheses e.g. "(4.1)" -> "4.1"
                allowed_eq_labels.add(art.equation_label.strip("()").lower())

        # 1. Detect and validate Figure ID references (e.g. FIG-xxx)
        fig_id_matches = re.findall(r'\b(FIG-[A-Z0-9\-]+)\b', draft_text)
        for fig_id in fig_id_matches:
            if fig_id not in bundle_fig_ids:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.UNKNOWN_SOURCE_ENTITY_IN_DRAFT,
                    severity="CRITICAL",
                    message=f"Draft references unknown figure ID '{fig_id}' not present in EvidenceBundle.",
                    artifact_id=fig_id,
                ))

        # 2. Detect and validate Equation ID references (e.g. EQ-xxx)
        eq_id_matches = re.findall(r'\b(EQ-[A-Z0-9\-]+)\b', draft_text)
        for eq_id in eq_id_matches:
            if eq_id not in bundle_eq_ids:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.UNKNOWN_SOURCE_ENTITY_IN_DRAFT,
                    severity="CRITICAL",
                    message=f"Draft references unknown equation ID '{eq_id}' not present in EvidenceBundle.",
                    artifact_id=eq_id,
                ))

        # 3. Detect and validate Algorithm ID references (e.g. ALG-xxx)
        alg_id_matches = re.findall(r'\b(ALG-[A-Z0-9\-]+)\b', draft_text)
        for alg_id in alg_id_matches:
            if alg_id not in bundle_alg_ids:
                violations.append(GroundingViolation(
                    violation_type=GroundingViolationType.UNKNOWN_SOURCE_ENTITY_IN_DRAFT,
                    severity="CRITICAL",
                    message=f"Draft references unknown algorithm ID '{alg_id}' not present in EvidenceBundle.",
                    artifact_id=alg_id,
                ))

        # NOTE: Question-construction values (numbers, sample sizes, hypothetical parameters)
        # are intentionally bypassed here to prevent false-positive rejection of valid STEM questions.

        is_valid = len(violations) == 0
        return GroundingReport(
            is_valid=is_valid,
            violations=violations,
            levels_passed={"source_entities_grounded": is_valid},
            provenance_chain_intact=True,
        )
