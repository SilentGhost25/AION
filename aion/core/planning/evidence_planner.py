"""
AION v2 Evidence Planner Engine
===============================
Assembles bounded, boundary-locked EvidenceBundles for specific QuestionSpecs.
- Enforces archetype requirements and budgets (prevents both context dilution and prompt starvation).
- Observes strict relationship strength ranking (EXPLICIT_REFERENCE > CAPTION > CONTAINMENT > SEMANTIC_BINDING > PROXIMAL).
- Distinguishes PRIMARY evidence from CONTEXTUAL figure bindings.
- Strictly confines evidence to the primary KU and explicitly declared secondary KUs.
- Operates 100% deterministically.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    TextArtifact,
)
from aion.core.dom.document_dom import DocumentDOM
from aion.core.dom.evidence import EvidenceBudget, QuestionArchetype
from aion.core.dom.relationships import RelationshipType
from aion.core.knowledge.contracts import (
    FigureContextBinding,
    KnowledgeUnit,
    SyllabusScopeMask,
)
from aion.core.planning.contracts import (
    EvidenceBundle,
    EvidenceItemRef,
    EvidenceRelationStrength,
    EvidenceRequirements,
    EvidenceRole,
    InsufficientEvidenceError,
)
from aion.core.question.contracts import QuestionSpec


# ---------------------------------------------------------------------------
# Archetype Evidentiary Policies
# ---------------------------------------------------------------------------

ARCHETYPE_REQUIREMENTS: Dict[QuestionArchetype, EvidenceRequirements] = {
    QuestionArchetype.NUMERICAL: EvidenceRequirements(
        archetype=QuestionArchetype.NUMERICAL,
        required_artifact_types={ArtifactType.EQUATION},
        preferred_artifact_types={ArtifactType.EXAMPLE, ArtifactType.TEXT},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.NUMERICAL,
            max_text_blocks=2,
            min_equations=1,
            max_equations=3,
            min_figures=0,
            max_figures=1,
            min_tables=0,
            max_tables=1,
        ),
    ),
    QuestionArchetype.DERIVATION: EvidenceRequirements(
        archetype=QuestionArchetype.DERIVATION,
        required_artifact_types={ArtifactType.EQUATION},
        preferred_artifact_types={ArtifactType.TEXT},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.DERIVATION,
            max_text_blocks=1,
            min_equations=2,
            max_equations=5,
            min_figures=0,
            max_figures=1,
            min_tables=0,
            max_tables=0,
        ),
    ),
    QuestionArchetype.CIRCUIT_SYSTEM: EvidenceRequirements(
        archetype=QuestionArchetype.CIRCUIT_SYSTEM,
        required_artifact_types={ArtifactType.FIGURE},
        preferred_artifact_types={ArtifactType.TEXT, ArtifactType.EQUATION},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.CIRCUIT_SYSTEM,
            max_text_blocks=2,
            min_equations=0,
            max_equations=2,
            min_figures=1,
            max_figures=1,
            min_tables=0,
            max_tables=0,
        ),
    ),
    QuestionArchetype.COMPARISON: EvidenceRequirements(
        archetype=QuestionArchetype.COMPARISON,
        required_artifact_types=set(),  # May be supported by Table OR multiple text blocks
        preferred_artifact_types={ArtifactType.TABLE, ArtifactType.TEXT},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.COMPARISON,
            max_text_blocks=3,
            min_equations=0,
            max_equations=0,
            min_figures=0,
            max_figures=0,
            min_tables=0,
            max_tables=2,
        ),
    ),
    QuestionArchetype.ALGORITHMIC: EvidenceRequirements(
        archetype=QuestionArchetype.ALGORITHMIC,
        required_artifact_types={ArtifactType.ALGORITHM},
        preferred_artifact_types={ArtifactType.CODE, ArtifactType.TEXT},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.ALGORITHMIC,
            max_text_blocks=2,
            min_equations=0,
            max_equations=1,
            min_figures=0,
            max_figures=1,
            min_algorithms=1,
            max_algorithms=1,
        ),
    ),
    QuestionArchetype.CONCEPTUAL: EvidenceRequirements(
        archetype=QuestionArchetype.CONCEPTUAL,
        required_artifact_types={ArtifactType.TEXT},
        preferred_artifact_types={ArtifactType.DEFINITION, ArtifactType.FIGURE},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.CONCEPTUAL,
            max_text_blocks=3,
            min_equations=0,
            max_equations=1,
            min_figures=0,
            max_figures=1,
            min_tables=0,
            max_tables=1,
        ),
    ),
    QuestionArchetype.PROCEDURAL: EvidenceRequirements(
        archetype=QuestionArchetype.PROCEDURAL,
        required_artifact_types={ArtifactType.TEXT},
        preferred_artifact_types={ArtifactType.PROCEDURE, ArtifactType.LIST},
        budget=EvidenceBudget(
            archetype=QuestionArchetype.PROCEDURAL,
            max_text_blocks=2,
            min_equations=0,
            max_equations=0,
            min_figures=0,
            max_figures=1,
            min_tables=0,
            max_tables=1,
        ),
    ),
}


class EvidencePlanner:
    """
    Deterministic evidence planning engine.
    Curates minimal, high-affinity multimodal evidence for a QuestionSpec.
    """

    def __init__(
        self,
        dom: DocumentDOM,
        kus: Dict[str, KnowledgeUnit],
        figure_bindings: Optional[Dict[str, FigureContextBinding]] = None,
        scope_mask: Optional[SyllabusScopeMask] = None,
    ):
        self.dom = dom
        self.kus = kus
        self.figure_bindings = figure_bindings or {}
        self.scope_mask = scope_mask

    def plan(self, spec: QuestionSpec) -> EvidenceBundle:
        """
        Assemble the bounded EvidenceBundle for a given QuestionSpec.
        Guarantees:
        1. Fail-fast validation of required modalities.
        2. Relationship-strength ranked artifact selection.
        3. Strict primary vs contextual figure binding tagging.
        4. Cross-KU boundary enforcement.
        5. 100% deterministic output.
        """
        if spec.primary_ku_id not in self.kus:
            raise KeyError(f"Primary KU '{spec.primary_ku_id}' not found in registered KnowledgeUnits.")

        primary_ku = self.kus[spec.primary_ku_id]
        requirements = ARCHETYPE_REQUIREMENTS.get(
            spec.archetype,
            EvidenceRequirements(
                archetype=spec.archetype,
                required_artifact_types={ArtifactType.TEXT},
                budget=EvidenceBudget(archetype=spec.archetype, max_text_blocks=3),
            )
        )
        budget = requirements.budget

        # Step 1: Validate hard requirements (Fail-Fast Starvation Gate)
        self._validate_hard_requirements(primary_ku, requirements, spec)

        # Step 2: Select primary multimodal artifacts according to budget and archetype affinity
        wants_equations = (
            ArtifactType.EQUATION in requirements.required_artifact_types
            or ArtifactType.EQUATION in requirements.preferred_artifact_types
            or budget.min_equations > 0
        )
        selected_eqs = self._select_equations(primary_ku, budget.max_equations) if wants_equations else []

        wants_figures = (
            ArtifactType.FIGURE in requirements.required_artifact_types
            or ArtifactType.FIGURE in requirements.preferred_artifact_types
            or budget.min_figures > 0
        )
        selected_figs, contextual_texts = (
            self._select_figures_with_context(primary_ku, budget.max_figures) if wants_figures else ([], [])
        )

        wants_tables = (
            ArtifactType.TABLE in requirements.required_artifact_types
            or ArtifactType.TABLE in requirements.preferred_artifact_types
            or budget.min_tables > 0
        )
        selected_tables = self._select_tables(primary_ku, budget.max_tables) if wants_tables else []

        wants_algorithms = (
            ArtifactType.ALGORITHM in requirements.required_artifact_types
            or ArtifactType.ALGORITHM in requirements.preferred_artifact_types
            or budget.min_algorithms > 0
        )
        selected_algs = self._select_algorithms(primary_ku, budget.max_algorithms) if wants_algorithms else []

        # Step 3: Select text paragraphs using explicit relationship strength hierarchy
        primary_artifact_ids: Set[str] = set()
        for eq, _ in selected_eqs:
            primary_artifact_ids.add(eq.artifact_id)
        for fig, _ in selected_figs:
            primary_artifact_ids.add(fig.artifact_id)
        for tbl in selected_tables:
            primary_artifact_ids.add(tbl.artifact_id)
        for alg in selected_algs:
            primary_artifact_ids.add(alg.artifact_id)
        selected_texts = self._select_texts(
            primary_ku=primary_ku,
            primary_artifact_ids=primary_artifact_ids,
            max_text_blocks=budget.max_text_blocks,
            excluded_text_ids={c.artifact_id for c in contextual_texts},
        )

        # Step 4: Handle Secondary KUs (if specified in QuestionSpec)
        secondary_ku_artifacts: List[Tuple[BaseArtifact, EvidenceRelationStrength, EvidenceRole]] = []
        if spec.secondary_ku_ids:
            secondary_ku_artifacts = self._plan_secondary_kus(
                spec.secondary_ku_ids,
                budget=budget,
                current_counts={
                    "equations": len(selected_eqs),
                    "figures": len(selected_figs),
                    "tables": len(selected_tables),
                    "texts": len(selected_texts),
                },
            )

        # Step 5: Build structured EvidenceItemRefs
        item_refs: List[EvidenceItemRef] = []

        # Equations
        for eq, strength in selected_eqs:
            item_refs.append(EvidenceItemRef(
                artifact_id=eq.artifact_id,
                artifact_type=ArtifactType.EQUATION,
                evidence_role=EvidenceRole.PRIMARY,
                relation_strength=strength,
                provenance_page=eq.page_number,
                provenance_bbox=eq.source_bbox,
            ))

        # Figures (PRIMARY)
        for fig, strength in selected_figs:
            item_refs.append(EvidenceItemRef(
                artifact_id=fig.artifact_id,
                artifact_type=ArtifactType.FIGURE,
                evidence_role=EvidenceRole.PRIMARY,
                relation_strength=strength,
                provenance_page=fig.page_number,
                provenance_bbox=fig.source_bbox,
            ))

        # Figure Context Texts (CONTEXTUAL)
        for ctx_text in contextual_texts:
            item_refs.append(EvidenceItemRef(
                artifact_id=ctx_text.artifact_id,
                artifact_type=ArtifactType.TEXT,
                evidence_role=EvidenceRole.CONTEXTUAL,
                relation_strength=EvidenceRelationStrength.SEMANTIC_BINDING,
                provenance_page=ctx_text.page_number,
                provenance_bbox=ctx_text.source_bbox,
            ))

        # Tables (PRIMARY)
        for tbl in selected_tables:
            item_refs.append(EvidenceItemRef(
                artifact_id=tbl.artifact_id,
                artifact_type=ArtifactType.TABLE,
                evidence_role=EvidenceRole.PRIMARY,
                relation_strength=EvidenceRelationStrength.CONTAINMENT,
                provenance_page=tbl.page_number,
                provenance_bbox=tbl.source_bbox,
            ))

        # Algorithms (PRIMARY)
        for alg in selected_algs:
            item_refs.append(EvidenceItemRef(
                artifact_id=alg.artifact_id,
                artifact_type=ArtifactType.ALGORITHM,
                evidence_role=EvidenceRole.PRIMARY,
                relation_strength=EvidenceRelationStrength.CONTAINMENT,
                provenance_page=alg.page_number,
                provenance_bbox=alg.source_bbox,
            ))

        # Primary Texts (PRIMARY)
        for txt, strength in selected_texts:
            item_refs.append(EvidenceItemRef(
                artifact_id=txt.artifact_id,
                artifact_type=ArtifactType.TEXT,
                evidence_role=EvidenceRole.PRIMARY,
                relation_strength=strength,
                provenance_page=txt.page_number,
                provenance_bbox=txt.source_bbox,
            ))

        # Secondary KU items
        for art, strength, role in secondary_ku_artifacts:
            item_refs.append(EvidenceItemRef(
                artifact_id=art.artifact_id,
                artifact_type=art.artifact_type,
                evidence_role=role,
                relation_strength=strength,
                provenance_page=art.page_number,
                provenance_bbox=art.source_bbox,
            ))

        # Deterministic sorting: by artifact type, relation strength descending, then artifact ID
        item_refs.sort(key=lambda r: (r.artifact_type.value, -int(r.relation_strength), r.artifact_id))

        # Categorized ID lists
        text_ids = [r.artifact_id for r in item_refs if r.artifact_type == ArtifactType.TEXT and r.evidence_role == EvidenceRole.PRIMARY]
        eq_ids = [r.artifact_id for r in item_refs if r.artifact_type == ArtifactType.EQUATION]
        fig_ids = [r.artifact_id for r in item_refs if r.artifact_type == ArtifactType.FIGURE]
        tbl_ids = [r.artifact_id for r in item_refs if r.artifact_type == ArtifactType.TABLE]
        alg_ids = [r.artifact_id for r in item_refs if r.artifact_type == ArtifactType.ALGORITHM]

        budget_used = {
            "text_blocks": len(text_ids),
            "equations": len(eq_ids),
            "figures": len(fig_ids),
            "tables": len(tbl_ids),
            "algorithms": len(alg_ids),
            "contextual_items": len(contextual_texts),
        }

        bundle_id = f"BNDL-{spec.spec_id}"

        return EvidenceBundle(
            bundle_id=bundle_id,
            spec_id=spec.spec_id,
            primary_ku_id=spec.primary_ku_id,
            secondary_ku_ids=list(spec.secondary_ku_ids),
            item_refs=item_refs,
            text_artifact_ids=text_ids,
            equation_artifact_ids=eq_ids,
            figure_artifact_ids=fig_ids,
            table_artifact_ids=tbl_ids,
            algorithm_artifact_ids=alg_ids,
            budget_used=budget_used,
        )

    # -----------------------------------------------------------------------
    # Internal Planning Selection & Ranking
    # -----------------------------------------------------------------------

    def _validate_hard_requirements(
        self,
        ku: KnowledgeUnit,
        reqs: EvidenceRequirements,
        spec: QuestionSpec,
    ) -> None:
        """Enforces hard required modalities for an archetype. Fails fast if starved."""
        if ArtifactType.EQUATION in reqs.required_artifact_types:
            if len(ku.equation_artifact_ids) < reqs.budget.min_equations:
                raise InsufficientEvidenceError(
                    f"KU '{ku.ku_id}' has {len(ku.equation_artifact_ids)} equation(s), "
                    f"but archetype '{spec.archetype.value}' requires at least {reqs.budget.min_equations}."
                )

        if ArtifactType.FIGURE in reqs.required_artifact_types:
            if len(ku.figure_artifact_ids) < reqs.budget.min_figures:
                raise InsufficientEvidenceError(
                    f"KU '{ku.ku_id}' has {len(ku.figure_artifact_ids)} figure(s), "
                    f"but archetype '{spec.archetype.value}' requires at least {reqs.budget.min_figures}."
                )

        if ArtifactType.ALGORITHM in reqs.required_artifact_types:
            alg_count = len(ku.algorithm_artifact_ids) + len(ku.procedure_artifact_ids)
            if alg_count < reqs.budget.min_algorithms:
                raise InsufficientEvidenceError(
                    f"KU '{ku.ku_id}' has {alg_count} algorithm/procedure(s), "
                    f"but archetype '{spec.archetype.value}' requires at least {reqs.budget.min_algorithms}."
                )

    def _select_equations(
        self,
        ku: KnowledgeUnit,
        max_equations: int,
    ) -> List[Tuple[EquationArtifact, EvidenceRelationStrength]]:
        if max_equations <= 0 or not ku.equation_artifact_ids:
            return []

        candidates: List[EquationArtifact] = []
        for eq_id in ku.equation_artifact_ids:
            art = self.dom.registry.get(eq_id)
            if isinstance(art, EquationArtifact):
                candidates.append(art)

        # Ranking score:
        # 1. Verification status: VERIFIED (10) > CANDIDATE (2) > FAILED (-100)
        # 2. Reference count: number of incoming explicit REFERENCES
        def score_eq(e: EquationArtifact) -> Tuple[int, int, str]:
            status_score = 10 if e.verification_status == "VERIFIED" else (2 if e.verification_status == "CANDIDATE" else -100)
            incoming_refs = len(self.dom.relationships.get_incoming(e.artifact_id, RelationshipType.REFERENCES))
            # Negative artifact_id tie-breaker ensures pure determinism
            return (status_score, incoming_refs, e.artifact_id)

        candidates.sort(key=score_eq, reverse=True)
        selected = candidates[:max_equations]

        result = []
        for eq in selected:
            incoming_refs = self.dom.relationships.get_incoming(eq.artifact_id, RelationshipType.REFERENCES)
            strength = EvidenceRelationStrength.EXPLICIT_REFERENCE if incoming_refs else EvidenceRelationStrength.CONTAINMENT
            result.append((eq, strength))

        return result

    def _select_figures_with_context(
        self,
        ku: KnowledgeUnit,
        max_figures: int,
    ) -> Tuple[List[Tuple[FigureArtifact, EvidenceRelationStrength]], List[TextArtifact]]:
        if max_figures <= 0 or not ku.figure_artifact_ids:
            return [], []

        candidates: List[FigureArtifact] = []
        for fig_id in ku.figure_artifact_ids:
            art = self.dom.registry.get(fig_id)
            if isinstance(art, FigureArtifact):
                candidates.append(art)

        # Deterministic sort
        candidates.sort(key=lambda f: f.artifact_id)
        selected_figs = candidates[:max_figures]

        figures_with_strength: List[Tuple[FigureArtifact, EvidenceRelationStrength]] = []
        contextual_texts: List[TextArtifact] = []

        for fig in selected_figs:
            strength = EvidenceRelationStrength.CONTAINMENT
            figures_with_strength.append((fig, strength))

            # Retrieve FigureContextBinding if present
            binding = self.figure_bindings.get(fig.artifact_id)
            if binding:
                for txt_id in binding.context_artifact_ids:
                    txt_art = self.dom.registry.get(txt_id)
                    if isinstance(txt_art, TextArtifact) and txt_art not in contextual_texts:
                        contextual_texts.append(txt_art)

        return figures_with_strength, contextual_texts

    def _select_tables(
        self,
        ku: KnowledgeUnit,
        max_tables: int,
    ) -> List[TableArtifact]:
        if max_tables <= 0 or not ku.table_artifact_ids:
            return []

        candidates: List[TableArtifact] = []
        for tbl_id in ku.table_artifact_ids:
            art = self.dom.registry.get(tbl_id)
            if isinstance(art, TableArtifact):
                candidates.append(art)

        candidates.sort(key=lambda t: t.artifact_id)
        return candidates[:max_tables]

    def _select_algorithms(
        self,
        ku: KnowledgeUnit,
        max_algorithms: int,
    ) -> List[AlgorithmArtifact]:
        if max_algorithms <= 0 or not ku.algorithm_artifact_ids:
            return []

        candidates: List[AlgorithmArtifact] = []
        for alg_id in ku.algorithm_artifact_ids:
            art = self.dom.registry.get(alg_id)
            if isinstance(art, AlgorithmArtifact):
                candidates.append(art)

        candidates.sort(key=lambda a: a.artifact_id)
        return candidates[:max_algorithms]

    def _select_texts(
        self,
        primary_ku: KnowledgeUnit,
        primary_artifact_ids: Set[str],
        max_text_blocks: int,
        excluded_text_ids: Set[str],
    ) -> List[Tuple[TextArtifact, EvidenceRelationStrength]]:
        if max_text_blocks <= 0 or not primary_ku.text_artifact_ids:
            return []

        candidates: List[Tuple[TextArtifact, EvidenceRelationStrength]] = []

        for txt_id in primary_ku.text_artifact_ids:
            if txt_id in excluded_text_ids:
                continue

            art = self.dom.registry.get(txt_id)
            if not isinstance(art, TextArtifact):
                continue

            # Evaluate evidentiary relation strength to selected primary artifacts
            outgoing_edges = self.dom.relationships.get_outgoing(art.artifact_id)
            references_primary = any(
                e.target_id in primary_artifact_ids and e.rel_type == RelationshipType.REFERENCES
                for e in outgoing_edges
            )
            proximal_primary = any(
                e.target_id in primary_artifact_ids and e.rel_type == RelationshipType.PROXIMAL_TO
                for e in outgoing_edges
            )

            if references_primary:
                strength = EvidenceRelationStrength.EXPLICIT_REFERENCE
            elif proximal_primary:
                strength = EvidenceRelationStrength.PROXIMAL
            else:
                strength = EvidenceRelationStrength.CONTAINMENT

            candidates.append((art, strength))

        # Sort candidate text blocks strictly by relationship strength descending,
        # then reading order, then artifact ID for pure determinism
        candidates.sort(
            key=lambda item: (int(item[1]), -item[0].reading_order if item[0].reading_order else 0, item[0].artifact_id),
            reverse=True,
        )

        return candidates[:max_text_blocks]

    def _plan_secondary_kus(
        self,
        secondary_ku_ids: List[str],
        budget: EvidenceBudget,
        current_counts: Dict[str, int],
    ) -> List[Tuple[BaseArtifact, EvidenceRelationStrength, EvidenceRole]]:
        """
        Retrieves complementary artifacts from secondary KUs without exceeding remaining budget.
        Guarantees cross-KU isolation: only artifacts belonging to declared secondary KUs are selected.
        """
        secondary_items: List[Tuple[BaseArtifact, EvidenceRelationStrength, EvidenceRole]] = []

        remaining_eqs = max(0, budget.max_equations - current_counts.get("equations", 0))
        remaining_texts = max(0, budget.max_text_blocks - current_counts.get("texts", 0))

        for s_id in sorted(secondary_ku_ids):
            s_ku = self.kus.get(s_id)
            if not s_ku:
                continue

            # Complementary equations
            if remaining_eqs > 0 and s_ku.equation_artifact_ids:
                for eq_id in sorted(s_ku.equation_artifact_ids):
                    eq_art = self.dom.registry.get(eq_id)
                    if isinstance(eq_art, EquationArtifact):
                        secondary_items.append((eq_art, EvidenceRelationStrength.CONTAINMENT, EvidenceRole.PRIMARY))
                        remaining_eqs -= 1
                        if remaining_eqs <= 0:
                            break

            # Complementary text
            if remaining_texts > 0 and s_ku.text_artifact_ids:
                for txt_id in sorted(s_ku.text_artifact_ids):
                    txt_art = self.dom.registry.get(txt_id)
                    if isinstance(txt_art, TextArtifact):
                        secondary_items.append((txt_art, EvidenceRelationStrength.CONTAINMENT, EvidenceRole.PRIMARY))
                        remaining_texts -= 1
                        if remaining_texts <= 0:
                            break

        return secondary_items
