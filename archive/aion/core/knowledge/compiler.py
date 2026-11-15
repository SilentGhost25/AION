"""
AION v2 Knowledge Compiler
==========================
Compiles the immutable DocumentDOM into an authoritative semantic index:
- Groups section artifacts into discrete KnowledgeUnits (KUs).
- Populates typed artifact pointer lists (source, equation, figure, table, text, etc.).
- Proposes probabilistic SemanticCandidates (never treats heuristics as final truth).
- Creates FigureContextBindings on the relationship layer for uncaptioned figures.
- Enforces strict SyllabusScopeMask boundaries (fail-closed when absent or out-of-range).
- Computes transparent, explainable ImportanceScores.

ARCHITECTURAL INVARIANT:
The Knowledge Layer references content; it does NOT become another content store.
Zero duplicate source text or formula strings are stored in KnowledgeUnits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    HeadingArtifact,
    TextArtifact,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    ExampleArtifact,
    ProcedureArtifact,
    DefinitionArtifact,
)
from aion.core.dom.document_dom import DocumentDOM, SectionNode
from aion.core.dom.relationships import Relationship, RelationshipType
from aion.core.knowledge.contracts import (
    CognitiveDemand,
    FigureContextBinding,
    BindingType,
    ImportanceScore,
    KnowledgeUnit,
    SemanticCandidate,
    SemanticType,
    SyllabusScopeMask,
)
from aion.core.knowledge.topology import (
    DocumentTopology,
    DocumentTopologyClassifier,
    TopologyAnalysis,
)


@dataclass
class KnowledgeCompilationResult:
    """
    Observable output of the Knowledge Compilation pass.
    """
    kus: Dict[str, KnowledgeUnit] = field(default_factory=dict)
    figure_bindings: Dict[str, FigureContextBinding] = field(default_factory=dict)
    topology_analysis: Optional[TopologyAnalysis] = None
    scope_mask_applied: bool = False
    total_kus: int = 0
    in_syllabus_kus: int = 0
    scope_uncertain_kus: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_kus": self.total_kus,
            "in_syllabus_kus": self.in_syllabus_kus,
            "scope_uncertain_kus": self.scope_uncertain_kus,
            "scope_mask_applied": self.scope_mask_applied,
            "topology": self.topology_analysis.to_dict() if self.topology_analysis else None,
            "kus": {k: v.to_dict() for k, v in self.kus.items()},
            "figure_bindings": {k: v.to_dict() for k, v in self.figure_bindings.items()},
        }


class SemanticPatternDetector:
    """
    Deterministic pattern detector that generates probabilistic SemanticCandidates.
    Candidate signals carry confidence scores and evidence traces.
    """

    # Definition heuristics
    COPULA_DEF_PATTERN = re.compile(
        r'^(?:[A-Z][\w\s-]{2,35})\s+(?:is\s+defined\s+as|refers\s+to|is\s+a|are\s+defined\s+as|is\s+called|denotes)\b',
        re.IGNORECASE
    )
    # Negative patterns that look like definitions but are not
    DEF_EXCLUSION_PATTERN = re.compile(
        r'^(?:consider|suppose|assume|let|note\s+that|we\s+have|in\s+order\s+to|as\s+shown|recall\s+that)\b',
        re.IGNORECASE
    )

    # Procedure heuristics
    NUMBERED_STEP_PATTERN = re.compile(r'^\s*(?:\d+[\.\)]|\([a-z\d]+\))\s+[A-Z]', re.MULTILINE)
    IMPERATIVE_VERB_PATTERN = re.compile(
        r'^\s*(?:compute|calculate|initialize|determine|select|execute|repeat|update|substitute|verify)\b',
        re.IGNORECASE | re.MULTILINE
    )

    # Comparison heuristics
    COMPARISON_PATTERN = re.compile(
        r'\b(?:difference\s+between|comparison\s+between|versus|vs\.?|contrast|compared\s+to)\b',
        re.IGNORECASE
    )

    # Worked example heuristics
    EXAMPLE_PATTERN = re.compile(
        r'\b(?:example\s+\d+|problem\s+\d+|worked\s+example|numerical\s+problem)\b',
        re.IGNORECASE
    )

    # Derivation heuristics
    DERIVATION_PATTERN = re.compile(
        r'\b(?:derivation\s+of|deriving|substituting\s+eq|integrating\s+both|from\s+equation|differentiating)\b',
        re.IGNORECASE
    )

    @classmethod
    def detect(cls, text_artifact: TextArtifact) -> List[SemanticCandidate]:
        candidates: List[SemanticCandidate] = []
        text = text_artifact.normalized_text or text_artifact.raw_text
        if not text.strip():
            return candidates

        # 1. Definition check
        if cls.COPULA_DEF_PATTERN.search(text):
            if not cls.DEF_EXCLUSION_PATTERN.search(text):
                evidence = ["copula_pattern"]
                confidence = 0.85
                if text_artifact.metadata.get("is_bold_start", False):
                    confidence = 0.92
                    evidence.append("bold_term_start")
                candidates.append(SemanticCandidate(
                    artifact_id=text_artifact.artifact_id,
                    candidate_type=SemanticType.DEFINITION,
                    confidence=confidence,
                    evidence=evidence,
                ))

        # 2. Procedure / Algorithm check
        steps = cls.NUMBERED_STEP_PATTERN.findall(text)
        imperatives = cls.IMPERATIVE_VERB_PATTERN.findall(text)
        if len(steps) >= 2 or len(imperatives) >= 2:
            evidence = []
            if len(steps) >= 2:
                evidence.append(f"numbered_steps_{len(steps)}")
            if len(imperatives) >= 2:
                evidence.append(f"imperative_verbs_{len(imperatives)}")
            candidates.append(SemanticCandidate(
                artifact_id=text_artifact.artifact_id,
                candidate_type=SemanticType.PROCEDURE,
                confidence=min(0.90, 0.70 + 0.05 * (len(steps) + len(imperatives))),
                evidence=evidence,
            ))

        # 3. Comparison check
        if cls.COMPARISON_PATTERN.search(text):
            candidates.append(SemanticCandidate(
                artifact_id=text_artifact.artifact_id,
                candidate_type=SemanticType.COMPARISON,
                confidence=0.85,
                evidence=["comparison_keywords"],
            ))

        # 4. Worked Example check
        if cls.EXAMPLE_PATTERN.search(text):
            candidates.append(SemanticCandidate(
                artifact_id=text_artifact.artifact_id,
                candidate_type=SemanticType.WORKED_EXAMPLE,
                confidence=0.88,
                evidence=["example_keywords"],
            ))

        # 5. Derivation check
        if cls.DERIVATION_PATTERN.search(text):
            candidates.append(SemanticCandidate(
                artifact_id=text_artifact.artifact_id,
                candidate_type=SemanticType.DERIVATION,
                confidence=0.86,
                evidence=["derivation_keywords"],
            ))

        return candidates


class KnowledgeCompiler:
    """
    Compiles DocumentDOM + SyllabusScopeMask into a structured KnowledgeUnit index.
    """

    def __init__(
        self,
        dom: DocumentDOM,
        scope_mask: Optional[SyllabusScopeMask] = None,
        topology: Optional[DocumentTopology] = None,
    ):
        self.dom = dom
        self.scope_mask = scope_mask
        self.topology_analysis = (
            DocumentTopologyClassifier.classify(dom) if topology is None
            else TopologyAnalysis(
                topology=topology,
                confidence=1.0,
                recommended_sectioning="SLIDE_BASED" if topology == DocumentTopology.SLIDE_DECK else "HIERARCHICAL",
                avg_words_per_page=0.0,
                heading_to_page_ratio=0.0,
                figure_to_page_ratio=0.0,
                total_pages=dom.total_pages,
                total_headings=len(dom.sections),
                total_figures=0,
                rationale="Explicitly supplied topology",
            )
        )
        self.topology = self.topology_analysis.topology

    def compile(self) -> KnowledgeCompilationResult:
        """
        Execute full knowledge compilation:
        1. Contextual figure binding on the relationship layer.
        2. Section-to-KU clustering with typed artifact ID partitioning.
        3. Semantic candidate detection and aggregation.
        4. Syllabus scope mask enforcement.
        5. Explainable importance calculation.
        """
        # Step 1: Bind uncaptioned figures contextually
        figure_bindings = self._bind_figures()

        # Step 2: Cluster sections into KnowledgeUnits
        kus: Dict[str, KnowledgeUnit] = {}
        ku_counter = 0

        # Sort sections by start page then section ID for stable compilation
        sorted_sections = sorted(
            self.dom.sections.values(),
            key=lambda s: (s.start_page, s.section_id)
        )

        for sec in sorted_sections:
            ku_counter += 1
            ku = self._compile_section_to_ku(sec, ku_counter, figure_bindings)
            kus[ku.ku_id] = ku

        # Calculate summary metrics
        in_syllabus_count = sum(1 for k in kus.values() if k.in_syllabus)
        scope_uncertain_count = sum(1 for k in kus.values() if k.scope_uncertain)

        return KnowledgeCompilationResult(
            kus=kus,
            figure_bindings=figure_bindings,
            topology_analysis=self.topology_analysis,
            scope_mask_applied=self.scope_mask is not None,
            total_kus=len(kus),
            in_syllabus_kus=in_syllabus_count,
            scope_uncertain_kus=scope_uncertain_count,
        )

    # -----------------------------------------------------------------------
    # Internal Compilation Steps
    # -----------------------------------------------------------------------

    def _bind_figures(self) -> Dict[str, FigureContextBinding]:
        """
        Creates contextual bindings for figures on the relationship layer.
        Keeps FigureArtifact immutable.
        """
        bindings: Dict[str, FigureContextBinding] = {}

        for art in self.dom.registry.all():
            if not isinstance(art, FigureArtifact):
                continue

            fig = art
            sec = self.dom.get_section(fig.section_id)
            context_title = sec.title if sec else f"Page {fig.page_number}"

            # Collect adjacent text blocks on the same page
            page_artifacts = [
                a for a in self.dom.registry.all()
                if a.page_number == fig.page_number and a.artifact_id != fig.artifact_id
            ]
            adjacent_text_ids = [
                a.artifact_id for a in page_artifacts if isinstance(a, TextArtifact)
            ]

            b_type = (
                BindingType.SLIDE_CONTEXT
                if self.topology == DocumentTopology.SLIDE_DECK
                else (BindingType.SECTION_CONTEXT if sec else BindingType.PROXIMITY)
            )

            binding = FigureContextBinding(
                figure_id=fig.artifact_id,
                section_id=fig.section_id,
                page_number=fig.page_number,
                context_title=context_title,
                context_artifact_ids=adjacent_text_ids,
                binding_type=b_type,
                confidence=0.88 if sec else 0.70,
            )
            bindings[fig.artifact_id] = binding

            # Register relationship edges in the DOM relationship graph
            if sec:
                self.dom.relationships.add_relationship(Relationship(
                    source_id=sec.section_id,
                    target_id=fig.artifact_id,
                    rel_type=RelationshipType.ILLUSTRATES,
                    confidence=binding.confidence,
                    metadata={"strategy": "contextual_section_binding", "title": context_title}
                ))

            for text_id in adjacent_text_ids:
                self.dom.relationships.add_relationship(Relationship(
                    source_id=text_id,
                    target_id=fig.artifact_id,
                    rel_type=RelationshipType.PROXIMAL_TO,
                    confidence=0.80,
                    metadata={"strategy": "same_page_proximity", "page": fig.page_number}
                ))

        return bindings

    def _compile_section_to_ku(
        self,
        sec: SectionNode,
        counter: int,
        figure_bindings: Dict[str, FigureContextBinding],
    ) -> KnowledgeUnit:
        effective_mod_idx = sec.module_index
        if self.scope_mask is not None and self.scope_mask.module_index > 0:
            if sec.module_index in (0, 1):
                effective_mod_idx = self.scope_mask.module_index

        ku_id = f"KU-M{effective_mod_idx}-{sec.section_id}-{counter:03d}"

        # Fetch artifacts attached to this section
        artifacts = self.dom.get_artifacts_for_section(sec.section_id)

        # Categorize artifact IDs by type (STRICT INDEX ONLY - NO SOURCE BLOBS)
        source_ids: List[str] = []
        text_ids: List[str] = []
        equation_ids: List[str] = []
        figure_ids: List[str] = []
        table_ids: List[str] = []
        example_ids: List[str] = []
        procedure_ids: List[str] = []
        algorithm_ids: List[str] = []

        all_candidates: List[SemanticCandidate] = []

        min_p = sec.start_page
        max_p = sec.end_page

        for a in artifacts:
            aid = a.artifact_id
            source_ids.append(aid)
            min_p = min(min_p, a.page_number)
            max_p = max(max_p, a.page_number)

            if isinstance(a, TextArtifact):
                text_ids.append(aid)
                # Propose semantic candidates
                cands = SemanticPatternDetector.detect(a)
                all_candidates.extend(cands)
            elif isinstance(a, EquationArtifact):
                equation_ids.append(aid)
            elif isinstance(a, FigureArtifact):
                figure_ids.append(aid)
            elif isinstance(a, TableArtifact):
                table_ids.append(aid)
            elif isinstance(a, AlgorithmArtifact):
                algorithm_ids.append(aid)
            elif isinstance(a, ExampleArtifact):
                example_ids.append(aid)
            elif isinstance(a, ProcedureArtifact):
                procedure_ids.append(aid)
            elif isinstance(a, DefinitionArtifact):
                all_candidates.append(SemanticCandidate(
                    artifact_id=aid,
                    candidate_type=SemanticType.DEFINITION,
                    confidence=0.98,
                    evidence=["native_definition_artifact"]
                ))

        # Decide primary semantic type based on candidate consensus & artifact presence
        primary_type = self._determine_primary_semantic_type(
            all_candidates, len(equation_ids), len(algorithm_ids), len(example_ids), len(table_ids)
        )

        # Enforce SyllabusScopeMask with fail-closed invariant
        in_syllabus, scope_uncertain = self._evaluate_scope(sec.title, min_p, max_p)

        # Compute explainable importance score
        importance = self._compute_importance(
            sec, len(equation_ids), len(text_ids), len(figure_ids),
            len(table_ids), len(example_ids), in_syllabus
        )

        # Infer Bloom taxonomy affinities
        bloom_levels = self._infer_bloom_affinities(primary_type, len(equation_ids), len(example_ids))

        return KnowledgeUnit(
            ku_id=ku_id,
            module_index=effective_mod_idx,
            section_id=sec.section_id,
            concept_title=sec.title,
            page_span=(min_p, max_p),
            source_artifact_ids=source_ids,
            text_artifact_ids=text_ids,
            equation_artifact_ids=equation_ids,
            figure_artifact_ids=figure_ids,
            table_artifact_ids=table_ids,
            example_artifact_ids=example_ids,
            procedure_artifact_ids=procedure_ids,
            algorithm_artifact_ids=algorithm_ids,
            primary_semantic_type=primary_type,
            semantic_candidates=all_candidates,
            importance=importance,
            bloom_affinities=bloom_levels,
            prerequisites=[],
            in_syllabus=in_syllabus,
            scope_uncertain=scope_uncertain,
            assessed_count=0,
            secondary_assessed_count=0,
        )

    def _determine_primary_semantic_type(
        self,
        candidates: List[SemanticCandidate],
        equation_count: int,
        algorithm_count: int,
        example_count: int,
        table_count: int,
    ) -> SemanticType:
        if algorithm_count > 0:
            return SemanticType.ALGORITHM
        if example_count > 0:
            return SemanticType.WORKED_EXAMPLE
        if equation_count >= 2:
            return SemanticType.DERIVATION

        # Evaluate candidate proposals
        if candidates:
            # Pick candidate with highest confidence
            best = max(candidates, key=lambda c: c.confidence)
            if best.confidence >= 0.70:
                return best.candidate_type

        if table_count > 0:
            return SemanticType.COMPARISON

        return SemanticType.CONCEPTUAL

    def _evaluate_scope(self, title: str, start_page: int, end_page: int) -> Tuple[bool, bool]:
        """
        Strict fail-closed scope evaluation.
        - If scope_mask is None: in_syllabus=False, scope_uncertain=True.
        - If out of allowed page range: in_syllabus=False, scope_uncertain=False.
        - If excluded topic: in_syllabus=False, scope_uncertain=False.
        - Otherwise: in_syllabus=True, scope_uncertain=False.
        """
        if self.scope_mask is None:
            # FAIL-CLOSED: No scope mask means scope cannot be safely validated
            return False, True

        # Excluded topic check
        if self.scope_mask.is_topic_excluded(title):
            return False, False

        # Page range check
        if self.scope_mask.allowed_page_ranges:
            in_range = any(
                self.scope_mask.is_page_allowed(p)
                for p in range(start_page, end_page + 1)
            )
            if not in_range:
                return False, False

        return True, False

    def _compute_importance(
        self,
        sec: SectionNode,
        eq_count: int,
        text_count: int,
        fig_count: int,
        table_count: int,
        ex_count: int,
        in_syllabus: bool,
    ) -> ImportanceScore:
        # 1. Heading weight (H1=1.0, H2=0.8, H3=0.6)
        h_weight = 1.0 if sec.level == 1 else (0.8 if sec.level == 2 else 0.6)

        # 2. Equation density
        eq_density = min(1.0, eq_count * 0.25)

        # 3. Text density
        t_density = min(1.0, text_count * 0.15)

        # 4. Structural prominence
        structural = 0.0
        if fig_count > 0:
            structural += 0.35
        if table_count > 0:
            structural += 0.25
        if ex_count > 0:
            structural += 0.40
        structural = min(1.0, structural)

        # 5. Syllabus priority
        s_priority = 0.0
        if self.scope_mask and in_syllabus:
            if self.scope_mask.is_topic_mandatory(sec.title):
                s_priority = 1.0
            else:
                s_priority = 0.5
        elif not in_syllabus:
            s_priority = 0.0

        # Composite total (0.0 to 1.0)
        total = (
            0.30 * h_weight +
            0.25 * eq_density +
            0.15 * t_density +
            0.15 * structural +
            0.15 * s_priority
        )
        total = round(min(1.0, max(0.1, total)), 3)

        rationale_parts = []
        if h_weight >= 0.8:
            rationale_parts.append(f"Major section (L{sec.level})")
        if eq_count > 0:
            rationale_parts.append(f"{eq_count} equation(s)")
        if fig_count > 0:
            rationale_parts.append(f"{fig_count} figure(s)")
        if s_priority == 1.0:
            rationale_parts.append("Mandatory syllabus topic")
        elif not in_syllabus:
            rationale_parts.append("Outside active syllabus scope")

        rationale = ", ".join(rationale_parts) if rationale_parts else "Standard conceptual section"

        return ImportanceScore(
            total=total,
            heading_weight=h_weight,
            equation_density=eq_density,
            text_density=t_density,
            syllabus_priority=s_priority,
            structural_prominence=structural,
            rationale=rationale,
        )

    def _infer_bloom_affinities(
        self,
        sem_type: SemanticType,
        eq_count: int,
        ex_count: int,
    ) -> List[CognitiveDemand]:
        affinities: List[CognitiveDemand] = []

        if sem_type in (SemanticType.DEFINITION, SemanticType.CONCEPTUAL):
            affinities.extend([CognitiveDemand.L1_REMEMBER, CognitiveDemand.L2_UNDERSTAND])
        elif sem_type in (SemanticType.PROCEDURE, SemanticType.ALGORITHM):
            affinities.extend([CognitiveDemand.L2_UNDERSTAND, CognitiveDemand.L3_APPLY])
        elif sem_type in (SemanticType.DERIVATION, SemanticType.WORKED_EXAMPLE):
            affinities.extend([CognitiveDemand.L3_APPLY, CognitiveDemand.L4_ANALYZE])
        elif sem_type == SemanticType.COMPARISON:
            affinities.extend([CognitiveDemand.L2_UNDERSTAND, CognitiveDemand.L4_ANALYZE])
        else:
            affinities.extend([CognitiveDemand.L2_UNDERSTAND, CognitiveDemand.L3_APPLY])

        if eq_count > 0 and CognitiveDemand.L3_APPLY not in affinities:
            affinities.append(CognitiveDemand.L3_APPLY)

        return affinities
