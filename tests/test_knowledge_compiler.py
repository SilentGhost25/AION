"""
AION v2 Knowledge Compiler & Coverage Ledger Test Suite
========================================================
Validates the Phase 3 authoritative semantic index:
- Test A: Textbook H1/H2/H3 clustering into KUs on real PDF (037de9d9-3f7.pdf).
- Test B: Slide deck topology classification (01cad428-1c0.pdf: 56 slide sections, NOT 56 modules).
- Test C: Contextual figure binding for uncaptioned figures on the relationship layer.
- Test D: Fail-closed SyllabusScopeMask enforcement (missing mask blocks selection; page ranges exclude out-of-scope KUs).
- Test E: Complete provenance chain traceability (KU -> artifact_id -> page -> bbox -> DOM artifact).
- Test F: Zero content duplication invariant (KU payload contains pointers only; zero raw text or formula strings).
- Test G: Semantic candidate detection and multi-KU assessment tracking in CoverageLedger.
"""

import json
import tempfile
from pathlib import Path
import pytest

from aion.core.dom.artifacts import (
    ArtifactType,
    TextArtifact,
    EquationArtifact,
    FigureArtifact,
)
from aion.core.dom.document_dom import DocumentDOM, SectionNode
from aion.core.extraction.pdf_extractor import extract_pdf_to_dom
from aion.core.knowledge.contracts import (
    CognitiveDemand,
    FigureContextBinding,
    ImportanceScore,
    KnowledgeUnit,
    SemanticCandidate,
    SemanticType,
    SyllabusScopeMask,
)
from aion.core.knowledge.topology import (
    DocumentTopology,
    DocumentTopologyClassifier,
)
from aion.core.knowledge.compiler import (
    KnowledgeCompiler,
    SemanticPatternDetector,
)
from aion.core.knowledge.ledger import (
    CoverageLedger,
    KUAssessmentState,
)


@pytest.fixture(scope="module")
def assets_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ---------------------------------------------------------------------------
# Test A: Textbook Compilation
# ---------------------------------------------------------------------------

def test_case_a_textbook_compilation(assets_dir):
    """
    Test A: Compiles a real textbook / notes PDF (037de9d9-3f7.pdf, 40 pages).
    Verifies that section boundaries from the DOM cluster artifacts into discrete KUs.
    """
    pdf_path = Path("workspace/uploads/037de9d9-3f7.pdf")
    assert pdf_path.exists(), "Sample PDF does not exist"

    dom = extract_pdf_to_dom(str(pdf_path), module_mapping={4: (1, 40)}, assets_root_dir=assets_dir)
    assert dom.total_pages == 40
    assert len(dom.sections) > 0

    mask = SyllabusScopeMask(
        module_index=4,
        allowed_page_ranges=[(1, 40)],
        mandatory_topics=["File System", "Directory"],
    )

    compiler = KnowledgeCompiler(dom=dom, scope_mask=mask)
    result = compiler.compile()

    assert result.total_kus > 0
    assert result.in_syllabus_kus == result.total_kus
    assert result.scope_uncertain_kus == 0

    # Verify each KU maps to a valid section and contains artifact IDs
    for ku_id, ku in result.kus.items():
        assert ku.section_id in dom.sections
        assert len(ku.source_artifact_ids) > 0
        assert ku.in_syllabus is True
        # Verify explainable importance
        assert 0.0 <= ku.importance.total <= 1.0
        assert ku.importance.rationale != ""

    # Check that mandatory topic gets syllabus priority boost
    mandatory_kus = [k for k in result.kus.values() if "file system" in k.concept_title.lower()]
    if mandatory_kus:
        assert mandatory_kus[0].importance.syllabus_priority == 1.0


# ---------------------------------------------------------------------------
# Test B: Slide Deck Compilation (NO Module Hallucination)
# ---------------------------------------------------------------------------

def test_case_b_slide_deck_no_module_hallucination(assets_dir):
    """
    Test B: 82-page slide deck (01cad428-1c0.pdf).
    Verifies that topology classifier identifies SLIDE_DECK.
    Slide titles become SectionNodes, strictly NOT 56 or 82 ModuleNodes!
    """
    pdf_path = Path("workspace/uploads/01cad428-1c0.pdf")
    assert pdf_path.exists(), "Sample PDF does not exist"

    dom = extract_pdf_to_dom(str(pdf_path), assets_root_dir=assets_dir)
    assert dom.total_pages == 82

    # Verify DOM module invariant
    assert len(dom.modules) <= 2, f"Expected at most 2 modules, found {len(dom.modules)}"
    assert len(dom.sections) >= 50, f"Expected ~56 slide sections, found {len(dom.sections)}"

    topology_analysis = DocumentTopologyClassifier.classify(dom)
    assert topology_analysis.topology == DocumentTopology.SLIDE_DECK
    assert topology_analysis.recommended_sectioning == "SLIDE_BASED"

    mask = SyllabusScopeMask(
        module_index=1,
        allowed_page_ranges=[(1, 82)],
    )

    compiler = KnowledgeCompiler(dom=dom, scope_mask=mask, topology=DocumentTopology.SLIDE_DECK)
    result = compiler.compile()

    # Verify that KUs correspond to slide sections, NOT modules
    assert result.total_kus >= 50
    for ku in result.kus.values():
        assert ku.module_index == 1  # Module index preserved from section/mask
        sec = dom.get_section(ku.section_id)
        assert sec is not None


# ---------------------------------------------------------------------------
# Test C: Uncaptioned Figure Context Binding
# ---------------------------------------------------------------------------

def test_case_c_uncaptioned_figure_context_binding(assets_dir):
    """
    Test C: Slide deck contains uncaptioned figures.
    Verifies that FigureContextBinding connects figures to slide titles and adjacent bullets
    on the relationship layer, leaving FigureArtifact completely immutable.
    """
    pdf_path = Path("workspace/uploads/01cad428-1c0.pdf")
    assert pdf_path.exists(), "Sample PDF does not exist"

    dom = extract_pdf_to_dom(str(pdf_path), assets_root_dir=assets_dir)
    figure_artifacts = [a for a in dom.registry.all() if isinstance(a, FigureArtifact)]
    assert len(figure_artifacts) >= 10, "Expected at least 10 figures in the slide deck"

    # Capture initial figure state to test immutability
    first_fig = figure_artifacts[0]
    initial_caption = first_fig.caption
    initial_desc = first_fig.semantic_description

    compiler = KnowledgeCompiler(dom=dom, topology=DocumentTopology.SLIDE_DECK)
    result = compiler.compile()

    assert len(result.figure_bindings) >= 10

    # Check binding details
    binding = result.figure_bindings.get(first_fig.artifact_id)
    assert binding is not None
    assert binding.figure_id == first_fig.artifact_id
    assert binding.page_number == first_fig.page_number
    assert binding.context_title != ""  # Contains slide title
    assert binding.confidence >= 0.70

    # Invariant: Original FigureArtifact must remain IMMUTABLE
    assert first_fig.caption == initial_caption
    assert first_fig.semantic_description == initial_desc

    # Invariant: RelationshipGraph must contain ILLUSTRATES edge for contextual binding
    illustrates_edges = dom.relationships.get_incoming(first_fig.artifact_id)
    assert len(illustrates_edges) > 0


# ---------------------------------------------------------------------------
# Test D: Syllabus Scope Mask Fail-Closed Invariant
# ---------------------------------------------------------------------------

def test_case_d_syllabus_scope_mask_fail_closed(assets_dir):
    """
    Test D: Fail-closed invariant.
    Case 1: No scope mask -> KUs marked scope_uncertain, coverage selection BLOCKED.
    Case 2: Scope mask restricting to pages 1-15 -> pages 16-40 marked in_syllabus=False and EXCLUDED.
    """
    pdf_path = Path("workspace/uploads/037de9d9-3f7.pdf")
    dom = extract_pdf_to_dom(str(pdf_path), assets_root_dir=assets_dir)

    # --- Case 1: No Scope Mask (Fail-Closed) ---
    compiler_no_mask = KnowledgeCompiler(dom=dom, scope_mask=None)
    result_no_mask = compiler_no_mask.compile()

    assert result_no_mask.scope_mask_applied is False
    assert result_no_mask.in_syllabus_kus == 0
    assert result_no_mask.scope_uncertain_kus == result_no_mask.total_kus

    ledger_no_mask = CoverageLedger()
    ledger_no_mask.register_kus(result_no_mask.kus)

    # Coverage candidate selection MUST BE STRICTLY BLOCKED!
    candidate, diag = ledger_no_mask.select_candidate_ku(module_index=1)
    assert candidate is None
    assert "No eligible in-syllabus KUs found" in diag["reason"]

    # --- Case 2: Restricted Scope Mask (Pages 29–40: Unit 4 Mobile OS) ---
    restricted_mask = SyllabusScopeMask(
        module_index=1,
        allowed_page_ranges=[(29, 40)],
    )
    compiler_scoped = KnowledgeCompiler(dom=dom, scope_mask=restricted_mask)
    result_scoped = compiler_scoped.compile()

    assert result_scoped.in_syllabus_kus > 0
    assert result_scoped.in_syllabus_kus < result_scoped.total_kus

    ledger_scoped = CoverageLedger()
    ledger_scoped.register_kus(result_scoped.kus)

    # Unit 3 File Systems (pages 1–28) must be strictly EXCLUDED
    for ku_id, ku in result_scoped.kus.items():
        if ku.page_span[1] < 29:
            assert ku.in_syllabus is False
            assert ledger_scoped.states[ku_id] == KUAssessmentState.EXCLUDED
        elif ku.page_span[0] >= 29:
            assert ku.in_syllabus is True
            assert ledger_scoped.states[ku_id] == KUAssessmentState.UNASSESSED

    # Candidate selection should succeed only for in-range KUs (pages 29-40)
    cand_ku, cand_diag = ledger_scoped.select_candidate_ku(module_index=1)
    assert cand_ku is not None
    assert cand_ku.in_syllabus is True
    assert cand_ku.page_span[0] >= 29
    assert cand_ku.page_span[1] <= 40


# ---------------------------------------------------------------------------
# Test E: Provenance Traceability
# ---------------------------------------------------------------------------

def test_case_e_provenance_traceability(assets_dir):
    """
    Test E: Every KnowledgeUnit must be strictly traceable back to
    artifact IDs -> page -> bbox -> original DOM artifact.
    """
    pdf_path = Path("workspace/uploads/074c9920-6b0.pdf")
    dom = extract_pdf_to_dom(str(pdf_path), assets_root_dir=assets_dir)

    mask = SyllabusScopeMask(module_index=1, allowed_page_ranges=[(1, 10)])
    compiler = KnowledgeCompiler(dom=dom, scope_mask=mask)
    result = compiler.compile()

    for ku_id, ku in result.kus.items():
        assert len(ku.source_artifact_ids) > 0

        # Resolve all artifacts dynamically
        artifacts = ku.resolve_all_artifacts(dom)
        assert len(artifacts) == len(ku.source_artifact_ids)

        for art in artifacts:
            # Provenance checks
            assert art.artifact_id in ku.source_artifact_ids
            assert ku.page_span[0] <= art.page_number <= ku.page_span[1]
            if art.source_bbox is not None:
                assert len(art.source_bbox) == 4
                x0, y0, x1, y1 = art.source_bbox
                assert x0 < x1
                assert y0 < y1


# ---------------------------------------------------------------------------
# Test F: Zero Content Duplication Invariant
# ---------------------------------------------------------------------------

def test_case_f_zero_content_duplication_invariant(assets_dir):
    """
    Test F: Strictly validates that KnowledgeUnit serialized payload contains
    NO duplicated text strings, raw source blobs, or formula strings.
    The Knowledge Layer is a pure index over the DOM.
    """
    pdf_path = Path("workspace/uploads/074c9920-6b0.pdf")
    dom = extract_pdf_to_dom(str(pdf_path), assets_root_dir=assets_dir)

    mask = SyllabusScopeMask(module_index=1, allowed_page_ranges=[(1, 10)])
    compiler = KnowledgeCompiler(dom=dom, scope_mask=mask)
    result = compiler.compile()

    allowed_ku_keys = {
        "ku_id",
        "module_index",
        "section_id",
        "concept_title",
        "page_span",
        "source_artifact_ids",
        "text_artifact_ids",
        "equation_artifact_ids",
        "figure_artifact_ids",
        "table_artifact_ids",
        "example_artifact_ids",
        "procedure_artifact_ids",
        "algorithm_artifact_ids",
        "primary_semantic_type",
        "semantic_candidates",
        "importance",
        "bloom_affinities",
        "prerequisites",
        "in_syllabus",
        "scope_uncertain",
        "assessed_count",
        "secondary_assessed_count",
    }

    forbidden_content_keys = {
        "raw_text",
        "normalized_text",
        "governing_formulas",
        "latex",
        "image_bytes",
        "markdown_repr",
        "rows",
        "pseudocode",
    }

    for ku_id, ku in result.kus.items():
        ku_dict = ku.to_dict()

        # 1. Assert no forbidden content keys
        for key in ku_dict.keys():
            assert key not in forbidden_content_keys, f"Forbidden key '{key}' found in KnowledgeUnit {ku_id}"
            assert key in allowed_ku_keys, f"Unexpected key '{key}' found in KnowledgeUnit {ku_id}"

        # 2. Assert serialized JSON contains NO raw content payload
        json_str = json.dumps(ku_dict)
        assert "governing_formulas" not in json_str

        # 3. Verify convenience property accessors work dynamically without mutating KU
        resolved_eqs = ku.resolve_equations(dom)
        assert isinstance(resolved_eqs, list)
        for eq in resolved_eqs:
            assert isinstance(eq, EquationArtifact)

        resolved_figs = ku.resolve_figures(dom)
        assert isinstance(resolved_figs, list)
        for fig in resolved_figs:
            assert isinstance(fig, FigureArtifact)

        resolved_texts = ku.resolve_texts(dom)
        assert isinstance(resolved_texts, list)
        for txt in resolved_texts:
            assert isinstance(txt, TextArtifact)


# ---------------------------------------------------------------------------
# Test G: Semantic Candidates & Multi-KU Coverage Ledger
# ---------------------------------------------------------------------------

def test_case_g_semantic_candidates_and_multi_ku_ledger():
    """
    Test G: Validates candidate semantic detection and multi-KU assessment
    lifecycle tracking in CoverageLedger.
    """
    # 1. Semantic Pattern Detector Candidates
    txt_def = TextArtifact(
        artifact_id="TXT-M1-P01-001",
        raw_text="Ensemble learning is defined as a technique that combines several base models to produce an optimal predictive model.",
        normalized_text="Ensemble learning is defined as a technique that combines several base models to produce an optimal predictive model.",
        metadata={"is_bold_start": True},
    )
    cands_def = SemanticPatternDetector.detect(txt_def)
    assert len(cands_def) >= 1
    assert cands_def[0].candidate_type == SemanticType.DEFINITION
    assert cands_def[0].confidence >= 0.85
    assert "copula_pattern" in cands_def[0].evidence

    # False positive suppression: "Consider the algorithm..."
    txt_consider = TextArtifact(
        artifact_id="TXT-M1-P01-002",
        raw_text="Consider the following algorithm which is a standard implementation.",
    )
    cands_consider = SemanticPatternDetector.detect(txt_consider)
    def_cands = [c for c in cands_consider if c.candidate_type == SemanticType.DEFINITION]
    assert len(def_cands) == 0, "Expected 'Consider ...' to NOT be flagged as a definition"

    # 2. Multi-KU Coverage Ledger Tracking
    ku1 = KnowledgeUnit(
        ku_id="KU-M1-SEC01-001",
        module_index=1,
        section_id="SEC-01",
        concept_title="Bagging and Bootstrap Aggregating",
        page_span=(1, 3),
        importance=ImportanceScore(total=0.85, syllabus_priority=1.0),
        bloom_affinities=[CognitiveDemand.L3_APPLY],
        in_syllabus=True,
    )
    ku2 = KnowledgeUnit(
        ku_id="KU-M1-SEC02-002",
        module_index=1,
        section_id="SEC-02",
        concept_title="Random Forests Out-of-Bag Error",
        page_span=(4, 6),
        importance=ImportanceScore(total=0.75, syllabus_priority=0.5),
        bloom_affinities=[CognitiveDemand.L4_ANALYZE],
        in_syllabus=True,
    )

    ledger = CoverageLedger()
    ledger.register_kus({ku1.ku_id: ku1, ku2.ku_id: ku2})

    # Initial state
    assert ledger.states[ku1.ku_id] == KUAssessmentState.UNASSESSED
    assert ledger.states[ku2.ku_id] == KUAssessmentState.UNASSESSED

    # Record multi-KU assessment for Question 1 (10 marks VTU question combining both concepts)
    record = ledger.record_assessment(
        question_id="Q1A",
        module_index=1,
        primary_ku_id=ku1.ku_id,
        secondary_ku_ids=[ku2.ku_id],
        concept="Ensemble Bagging & Out-of-Bag Analysis",
        cognitive_demand=CognitiveDemand.L3_APPLY,
    )

    assert record.primary_ku_id == ku1.ku_id
    assert record.secondary_ku_ids == [ku2.ku_id]

    # Primary KU transitioned to ASSESSED, Secondary KU transitioned to ALLOCATED
    assert ledger.states[ku1.ku_id] == KUAssessmentState.ASSESSED
    assert ku1.assessed_count == 1
    assert ledger.states[ku2.ku_id] == KUAssessmentState.ALLOCATED
    assert ku2.secondary_assessed_count == 1

    # Verify coverage audit report
    report = ledger.get_coverage_report()
    assert report["total_kus"] == 2
    assert report["primary_assessed_kus"] == 1
    assert report["total_touched_kus"] == 2
    assert report["primary_coverage_pct"] == 50.0
    assert report["total_touch_coverage_pct"] == 100.0
    assert report["module_breakdown"][1]["assessed_kus"] == 1
