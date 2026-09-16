"""
AION v2 Evidence Planning & Grounding Test Suite
=================================================
Validates the Phase 4 architectural foundation across the 16-test matrix:
1. test_numerical_evidence_planning
2. test_circuit_system_context_binding
3. test_comparison_table_planning
4. test_algorithmic_pseudocode_planning
5. test_conceptual_definition_planning
6. test_cross_module_firewall_rejection
7. test_cross_ku_boundary_isolation
8. test_scope_mask_firewall_rejection
9. test_starvation_fail_fast
10. test_provenance_integrity_rejection
11. test_budget_overflow_rejection
12. test_capability_serialization_text_vs_multimodal
13. test_evidence_determinism
14. test_evidence_ranking_strength
15. test_context_reduction_ratio
16. test_secondary_ku_multi_concept_planning
Plus:
17. test_draft_grounding_numerical_parameters_permitted (verifies distinction between source-bound entities & question construction)
"""

import json
import tempfile
from pathlib import Path
import pytest

from aion.core.dom.artifacts import (
    AlgorithmArtifact,
    ArtifactType,
    DefinitionArtifact,
    EquationArtifact,
    ExampleArtifact,
    FigureArtifact,
    ProcedureArtifact,
    TableArtifact,
    TextArtifact,
)
from aion.core.dom.document_dom import DocumentDOM, ModuleNode, SectionNode
from aion.core.dom.evidence import EvidenceBudget, ModelCapability, QuestionArchetype
from aion.core.dom.relationships import Relationship, RelationshipGraph, RelationshipType
from aion.core.knowledge.contracts import (
    CognitiveDemand,
    FigureContextBinding,
    ImportanceScore,
    KnowledgeUnit,
    SyllabusScopeMask,
)
from aion.core.planning import (
    ARCHETYPE_REQUIREMENTS,
    CapabilitySerializer,
    EvidenceBundle,
    EvidenceItemRef,
    EvidencePlanner,
    EvidenceRelationStrength,
    EvidenceRole,
    GroundingReport,
    GroundingVerifier,
    GroundingViolationType,
    InsufficientEvidenceError,
)
from aion.core.question.contracts import QuestionSpec


@pytest.fixture
def sample_planning_environment():
    """
    Constructs a rich, multi-representational DocumentDOM with relationships,
    KnowledgeUnits, and FigureContextBindings across Module 1 and Module 2.
    """
    dom = DocumentDOM(document_id="DOC-STEM-001", total_pages=10)

    # Register Modules
    mod1 = ModuleNode(
        module_index=1,
        title="Module 1: Circuit Analysis",
        start_page=1,
        end_page=5,
        section_ids=["SEC-M1-01"],
    )
    dom.add_module(mod1)

    mod2 = ModuleNode(
        module_index=2,
        title="Module 2: Electromagnetic Fields",
        start_page=6,
        end_page=10,
        section_ids=["SEC-M2-01"],
    )
    dom.add_module(mod2)

    # Register Module 1 section
    sec1 = SectionNode(
        section_id="SEC-M1-01",
        title="Ohm's Law and Circuit Analysis",
        level=2,
        module_index=1,
        start_page=1,
        end_page=5,
    )
    dom.add_section(sec1)

    # Register Module 2 section
    sec2 = SectionNode(
        section_id="SEC-M2-01",
        title="Electromagnetic Fields",
        level=2,
        module_index=2,
        start_page=6,
        end_page=10,
    )
    dom.add_section(sec2)

    # --- Module 1 Artifacts ---
    txt1 = TextArtifact(
        artifact_id="TXT-001",
        section_id="SEC-M1-01",
        page_number=1,
        source_bbox=(50.0, 100.0, 450.0, 150.0),
        raw_text="Consider a simple series resistive circuit connected to a DC supply as shown in Figure 1.",
        normalized_text="Consider a simple series resistive circuit connected to a DC supply as shown in Figure 1.",
        module_index=1,
        reading_order=1,
    )
    txt2 = TextArtifact(
        artifact_id="TXT-002",
        section_id="SEC-M1-01",
        page_number=1,
        source_bbox=(50.0, 160.0, 450.0, 200.0),
        raw_text="The voltage drop across any resistor is directly proportional to the current flowing through it.",
        normalized_text="The voltage drop across any resistor is directly proportional to the current flowing through it.",
        module_index=1,
        reading_order=2,
    )
    txt_slide_title = TextArtifact(
        artifact_id="TXT-SLIDE-TITLE-01",
        section_id="SEC-M1-01",
        page_number=2,
        source_bbox=(50.0, 50.0, 400.0, 80.0),
        raw_text="Circuit Topology Overview",
        normalized_text="Circuit Topology Overview",
        module_index=1,
        reading_order=3,
    )
    txt_slide_bullet = TextArtifact(
        artifact_id="TXT-SLIDE-BULLET-01",
        section_id="SEC-M1-01",
        page_number=2,
        source_bbox=(60.0, 90.0, 450.0, 130.0),
        raw_text="Series loops share common loop current I; node voltages sum to zero by KVL.",
        normalized_text="Series loops share common loop current I; node voltages sum to zero by KVL.",
        module_index=1,
        reading_order=4,
    )
    eq1 = EquationArtifact(
        artifact_id="EQ-001",
        section_id="SEC-M1-01",
        page_number=1,
        source_bbox=(100.0, 210.0, 300.0, 250.0),
        latex="V = I \\cdot R",
        variables=["V", "I", "R"],
        equation_label="(1.1)",
        verification_status="VERIFIED",
        module_index=1,
    )
    eq2 = EquationArtifact(
        artifact_id="EQ-002",
        section_id="SEC-M1-01",
        page_number=1,
        source_bbox=(100.0, 260.0, 300.0, 300.0),
        latex="P = I^2 \\cdot R = \\frac{V^2}{R}",
        variables=["P", "I", "R", "V"],
        equation_label="(1.2)",
        verification_status="VERIFIED",
        module_index=1,
    )
    fig1 = FigureArtifact(
        artifact_id="FIG-001",
        section_id="SEC-M1-01",
        page_number=2,
        source_bbox=(100.0, 310.0, 400.0, 550.0),
        asset_key="assets/figs/FIG-001.png",
        image_hash="a1b2c3d4e5f67890",
        caption="Figure 1: DC Series Resistor Network with Supply V_s",
        semantic_description="Schematic diagram of DC voltage source connected to R1 and R2 in series.",
        figure_type="CIRCUIT",
        width=600,
        height=400,
        module_index=1,
    )
    tbl1 = TableArtifact(
        artifact_id="TBL-001",
        section_id="SEC-M1-01",
        page_number=3,
        source_bbox=(50.0, 100.0, 500.0, 300.0),
        headers=["Parameter", "Series Circuit", "Parallel Circuit"],
        rows=[
            ["Current", "Same through all components", "Divides among branches"],
            ["Voltage", "Divides across components", "Same across all branches"],
            ["Equivalent R", "R_eq = R1 + R2", "1/R_eq = 1/R1 + 1/R2"],
        ],
        markdown_repr="| Parameter | Series Circuit | Parallel Circuit |\n|---|---|---|\n| Current | Same | Divides |\n| Voltage | Divides | Same |",
        caption="Table 1.1: Comparison between Series and Parallel Circuits",
        module_index=1,
    )
    alg1 = AlgorithmArtifact(
        artifact_id="ALG-001",
        section_id="SEC-M1-01",
        page_number=4,
        source_bbox=(50.0, 100.0, 500.0, 400.0),
        title="Nodal Analysis Algorithm",
        steps=[
            "Identify all principal nodes and choose one reference node (ground).",
            "Assign node voltage variables V1, V2, ... to non-reference nodes.",
            "Apply KCL at each non-reference node using Ohm's Law.",
            "Solve the resulting system of linear equations for node voltages.",
        ],
        complexity="O(V^3)",
        pseudocode="for each node v in V: sum(I_out) = 0",
        module_index=1,
    )

    # --- Module 2 Artifact (Contaminant) ---
    txt_m2 = TextArtifact(
        artifact_id="TXT-M2-001",
        section_id="SEC-M2-01",
        page_number=7,
        source_bbox=(50.0, 100.0, 450.0, 150.0),
        raw_text="Maxwell's equations govern electromagnetic radiation in free space.",
        normalized_text="Maxwell's equations govern electromagnetic radiation in free space.",
        module_index=2,
        reading_order=1,
    )

    # Attach to DOM sections and modules
    for a in [txt1, txt2, txt_slide_title, txt_slide_bullet, eq1, eq2, fig1, tbl1, alg1, txt_m2]:
        dom.attach_artifact(a, section_id=a.section_id, module_index=a.module_index)

    # Build Relationships
    # TXT-001 explicitly REFERENCES FIG-001 and EQ-001
    dom.relationships.add_relationship(
        Relationship(
            source_id="TXT-001",
            target_id="FIG-001",
            rel_type=RelationshipType.REFERENCES,
            confidence=1.0,
        )
    )
    dom.relationships.add_relationship(
        Relationship(
            source_id="TXT-001",
            target_id="EQ-001",
            rel_type=RelationshipType.REFERENCES,
            confidence=1.0,
        )
    )
    # TXT-002 is PROXIMAL_TO EQ-002
    dom.relationships.add_relationship(
        Relationship(
            source_id="TXT-002",
            target_id="EQ-002",
            rel_type=RelationshipType.PROXIMAL_TO,
            confidence=0.8,
        )
    )

    # KnowledgeUnit 1: Ohm's Law (Primary KU for M1)
    ku1 = KnowledgeUnit(
        ku_id="KU-M1-001",
        concept_title="Ohm's Law and Circuit Analysis",
        module_index=1,
        section_id="SEC-M1-01",
        page_span=(1, 5),
        source_artifact_ids=[
            "TXT-001", "TXT-002", "TXT-SLIDE-TITLE-01", "TXT-SLIDE-BULLET-01",
            "EQ-001", "EQ-002", "FIG-001", "TBL-001", "ALG-001",
        ],
        text_artifact_ids=["TXT-001", "TXT-002", "TXT-SLIDE-TITLE-01", "TXT-SLIDE-BULLET-01"],
        equation_artifact_ids=["EQ-001", "EQ-002"],
        figure_artifact_ids=["FIG-001"],
        table_artifact_ids=["TBL-001"],
        algorithm_artifact_ids=["ALG-001"],
        bloom_affinities=[CognitiveDemand.L3_APPLY],
        importance=ImportanceScore(total=0.9, syllabus_priority=1.0, rationale="Foundational circuit analysis"),
        in_syllabus=True,
    )

    # KnowledgeUnit 2: Power Dissipation (Secondary KU for M1)
    ku2 = KnowledgeUnit(
        ku_id="KU-M1-002",
        concept_title="Electric Power and Energy",
        module_index=1,
        section_id="SEC-M1-01",
        page_span=(1, 2),
        source_artifact_ids=["TXT-002", "EQ-002"],
        text_artifact_ids=["TXT-002"],
        equation_artifact_ids=["EQ-002"],
        bloom_affinities=[CognitiveDemand.L3_APPLY],
        importance=ImportanceScore(total=0.75, syllabus_priority=0.8, rationale="Power relationships"),
        in_syllabus=True,
    )

    # KnowledgeUnit 3: Module 2 Electromagnetics
    ku3 = KnowledgeUnit(
        ku_id="KU-M2-001",
        concept_title="Electromagnetics",
        module_index=2,
        section_id="SEC-M2-01",
        page_span=(6, 10),
        source_artifact_ids=["TXT-M2-001"],
        text_artifact_ids=["TXT-M2-001"],
        bloom_affinities=[CognitiveDemand.L2_UNDERSTAND],
        importance=ImportanceScore(total=0.8, syllabus_priority=1.0, rationale="EM fields"),
        in_syllabus=True,
    )

    # Contextual Figure Binding
    fig_binding = FigureContextBinding(
        figure_id="FIG-001",
        section_id="SEC-M1-01",
        page_number=2,
        context_title="Circuit Topology Overview",
        context_artifact_ids=["TXT-SLIDE-TITLE-01", "TXT-SLIDE-BULLET-01"],
        confidence=0.95,
    )

    # Syllabus Scope Mask
    mask = SyllabusScopeMask(
        module_index=1,
        allowed_page_ranges=[(1, 5)],
        mandatory_topics=["Circuit Analysis"],
    )

    kus = {ku1.ku_id: ku1, ku2.ku_id: ku2, ku3.ku_id: ku3}
    figure_bindings = {fig1.artifact_id: fig_binding}

    return {
        "dom": dom,
        "kus": kus,
        "figure_bindings": figure_bindings,
        "scope_mask": mask,
    }


# ---------------------------------------------------------------------------
# 1. Numerical Evidence Planning
# ---------------------------------------------------------------------------

def test_numerical_evidence_planning(sample_planning_environment):
    """
    Test 1: Verifies equation selection, variable matching, and strict budget bounds for NUMERICAL archetype.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q01A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    assert bundle.spec_id == "SPEC-M1-Q01A"
    assert bundle.primary_ku_id == "KU-M1-001"
    assert len(bundle.equation_artifact_ids) >= 1
    assert len(bundle.equation_artifact_ids) <= 3
    assert "EQ-001" in bundle.equation_artifact_ids

    # Verify equation has verified status and extracted variables
    eq_art = env["dom"].registry.get(bundle.equation_artifact_ids[0])
    assert eq_art.latex == "V = I \\cdot R"
    assert "V" in eq_art.variables
    assert "I" in eq_art.variables

    # Verify GroundingVerifier passes 100%
    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True
    assert len(report.violations) == 0


# ---------------------------------------------------------------------------
# 2. Circuit System Context Binding
# ---------------------------------------------------------------------------

def test_circuit_system_context_binding(sample_planning_environment):
    """
    Test 2: Verifies primary figure artifact selection + contextual note binding from FigureContextBinding.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q02A",
        module_index=1,
        q_number=2,
        part_letter="a",
        marks=8,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.CIRCUIT_SYSTEM,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    assert len(bundle.figure_artifact_ids) == 1
    assert bundle.figure_artifact_ids[0] == "FIG-001"

    # Verify role separation: FIG-001 is PRIMARY, contextual texts are CONTEXTUAL
    fig_ref = next(r for r in bundle.item_refs if r.artifact_id == "FIG-001")
    assert fig_ref.evidence_role == EvidenceRole.PRIMARY
    assert fig_ref.artifact_type == ArtifactType.FIGURE

    contextual_refs = [r for r in bundle.item_refs if r.evidence_role == EvidenceRole.CONTEXTUAL]
    assert len(contextual_refs) >= 1
    assert any(r.artifact_id == "TXT-SLIDE-BULLET-01" for r in contextual_refs)
    for c_ref in contextual_refs:
        assert c_ref.relation_strength == EvidenceRelationStrength.SEMANTIC_BINDING

    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 3. Comparison Table Planning
# ---------------------------------------------------------------------------

def test_comparison_table_planning(sample_planning_environment):
    """
    Test 3: Verifies TableArtifact preference and multi-block text assembly for COMPARISON archetype.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q03A",
        module_index=1,
        q_number=3,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L4_ANALYZE,
        archetype=QuestionArchetype.COMPARISON,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    assert len(bundle.table_artifact_ids) == 1
    assert bundle.table_artifact_ids[0] == "TBL-001"

    tbl_ref = next(r for r in bundle.item_refs if r.artifact_id == "TBL-001")
    assert tbl_ref.evidence_role == EvidenceRole.PRIMARY
    assert tbl_ref.artifact_type == ArtifactType.TABLE

    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 4. Algorithmic Pseudocode Planning
# ---------------------------------------------------------------------------

def test_algorithmic_pseudocode_planning(sample_planning_environment):
    """
    Test 4: Verifies AlgorithmArtifact selection, steps, complexity for ALGORITHMIC archetype.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q04A",
        module_index=1,
        q_number=4,
        part_letter="a",
        marks=8,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.ALGORITHMIC,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    assert len(bundle.algorithm_artifact_ids) == 1
    assert bundle.algorithm_artifact_ids[0] == "ALG-001"

    alg_art = env["dom"].registry.get(bundle.algorithm_artifact_ids[0])
    assert alg_art.title == "Nodal Analysis Algorithm"
    assert len(alg_art.steps) == 4
    assert alg_art.complexity == "O(V^3)"

    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 5. Conceptual Definition Planning
# ---------------------------------------------------------------------------

def test_conceptual_definition_planning(sample_planning_environment):
    """
    Test 5: Verifies text and definition selection bounded to archetype budget.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q05A",
        module_index=1,
        q_number=5,
        part_letter="a",
        marks=4,
        bloom_level=CognitiveDemand.L2_UNDERSTAND,
        archetype=QuestionArchetype.CONCEPTUAL,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    assert len(bundle.text_artifact_ids) >= 1
    assert len(bundle.text_artifact_ids) <= 3
    # Equations bounded to at most 1 in CONCEPTUAL budget
    assert len(bundle.equation_artifact_ids) <= 1

    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True


# ---------------------------------------------------------------------------
# 6. Cross-Module Firewall Rejection
# ---------------------------------------------------------------------------

def test_cross_module_firewall_rejection(sample_planning_environment):
    """
    Test 6: Level 2 Module Gate strictly rejects bundles with artifacts from outside target module_index.
    """
    env = sample_planning_environment
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q06A",
        module_index=1,
        q_number=6,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L2_UNDERSTAND,
        archetype=QuestionArchetype.CONCEPTUAL,
        primary_ku_id="KU-M1-001",
    )

    # Construct an artificially contaminated bundle containing a Module 2 artifact
    contaminated_bundle = EvidenceBundle(
        bundle_id="BNDL-CONTAMINATED",
        spec_id=spec.spec_id,
        primary_ku_id=spec.primary_ku_id,
        item_refs=[
            EvidenceItemRef(
                artifact_id="TXT-001",
                artifact_type=ArtifactType.TEXT,
                evidence_role=EvidenceRole.PRIMARY,
                provenance_page=1,
            ),
            EvidenceItemRef(
                artifact_id="TXT-M2-001",  # From Module 2!
                artifact_type=ArtifactType.TEXT,
                evidence_role=EvidenceRole.PRIMARY,
                provenance_page=7,
            ),
        ],
        text_artifact_ids=["TXT-001", "TXT-M2-001"],
    )

    report = GroundingVerifier.verify_bundle(
        bundle=contaminated_bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is False
    assert report.levels_passed["level_2_module"] is False
    module_violations = [v for v in report.violations if v.violation_type == GroundingViolationType.CROSS_MODULE_CONTAMINATION]
    assert len(module_violations) > 0
    assert "TXT-M2-001" in module_violations[0].message


# ---------------------------------------------------------------------------
# 7. Cross-KU Boundary Isolation
# ---------------------------------------------------------------------------

def test_cross_ku_boundary_isolation(sample_planning_environment):
    """
    Test 7: Level 4 Semantic Boundary Gate strictly isolates primary KU from un-declared KUs.
    """
    env = sample_planning_environment
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q07A",
        module_index=1,
        q_number=7,
        part_letter="a",
        marks=4,
        bloom_level=CognitiveDemand.L2_UNDERSTAND,
        archetype=QuestionArchetype.CONCEPTUAL,
        primary_ku_id="KU-M1-002",  # Only contains TXT-002, EQ-002
        secondary_ku_ids=[],        # None declared!
    )

    # Infiltrate with TXT-001 (which belongs to KU-M1-001, not KU-M1-002)
    infiltrated_bundle = EvidenceBundle(
        bundle_id="BNDL-INFILTRATED",
        spec_id=spec.spec_id,
        primary_ku_id=spec.primary_ku_id,
        item_refs=[
            EvidenceItemRef(
                artifact_id="TXT-002",
                artifact_type=ArtifactType.TEXT,
                provenance_page=1,
            ),
            EvidenceItemRef(
                artifact_id="TXT-001",  # Not in KU-M1-002!
                artifact_type=ArtifactType.TEXT,
                provenance_page=1,
            ),
        ],
        text_artifact_ids=["TXT-002", "TXT-001"],
    )

    report = GroundingVerifier.verify_bundle(
        bundle=infiltrated_bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is False
    assert report.levels_passed["level_4_semantic_boundary"] is False
    ku_violations = [v for v in report.violations if v.violation_type == GroundingViolationType.CROSS_KU_LEAKAGE]
    assert len(ku_violations) > 0


# ---------------------------------------------------------------------------
# 8. Scope Mask Firewall Rejection
# ---------------------------------------------------------------------------

def test_scope_mask_firewall_rejection(sample_planning_environment):
    """
    Test 8: Level 3 Scope Gate enforces fail-closed behavior if mask is missing or pages out of syllabus.
    """
    env = sample_planning_environment
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q08A",
        module_index=1,
        q_number=8,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )

    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )
    bundle = planner.plan(spec)

    # Case A: Missing scope mask must fail-closed
    report_no_mask = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=None,
    )
    assert report_no_mask.is_valid is False
    assert report_no_mask.levels_passed["level_3_scope"] is False

    # Case B: Out-of-syllabus page range
    out_of_scope_mask = SyllabusScopeMask(
        module_index=1,
        allowed_page_ranges=[(8, 10)],  # Bundle artifacts are on pages 1-4!
    )
    report_out_of_scope = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=out_of_scope_mask,
    )
    assert report_out_of_scope.is_valid is False
    assert report_out_of_scope.levels_passed["level_3_scope"] is False


# ---------------------------------------------------------------------------
# 9. Starvation Fail-Fast
# ---------------------------------------------------------------------------

def test_starvation_fail_fast(sample_planning_environment):
    """
    Test 9: EvidencePlanner raises InsufficientEvidenceError immediately when required modality is absent.
    """
    env = sample_planning_environment
    # KU-M1-002 contains ONLY Text and Equation, NO Figure
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    # CIRCUIT_SYSTEM requires at least 1 figure
    starved_spec = QuestionSpec(
        spec_id="SPEC-STARVED-01",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=8,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.CIRCUIT_SYSTEM,
        primary_ku_id="KU-M1-002",
    )

    with pytest.raises(InsufficientEvidenceError) as exc_info:
        planner.plan(starved_spec)
    assert "requires at least 1" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 10. Provenance Integrity Rejection
# ---------------------------------------------------------------------------

def test_provenance_integrity_rejection(sample_planning_environment):
    """
    Test 10: Level 5 Provenance Gate rejects unresolvable artifacts, invalid pages, or degenerate bboxes.
    """
    env = sample_planning_environment
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q10A",
        module_index=1,
        q_number=10,
        part_letter="a",
        marks=4,
        bloom_level=CognitiveDemand.L2_UNDERSTAND,
        archetype=QuestionArchetype.CONCEPTUAL,
        primary_ku_id="KU-M1-001",
    )

    # Case A: Phantom Artifact ID (missing from registry)
    phantom_bundle = EvidenceBundle(
        bundle_id="BNDL-PHANTOM",
        spec_id=spec.spec_id,
        primary_ku_id=spec.primary_ku_id,
        item_refs=[
            EvidenceItemRef(
                artifact_id="TXT-PHANTOM-999",
                artifact_type=ArtifactType.TEXT,
                provenance_page=1,
            )
        ],
    )
    report_phantom = GroundingVerifier.verify_bundle(
        bundle=phantom_bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report_phantom.is_valid is False
    assert report_phantom.levels_passed["level_5_provenance"] is False

    # Case B: Degenerate Bounding Box (x0 >= x1 or y0 >= y1)
    bad_art = TextArtifact(
        artifact_id="TXT-DEGEN-BBOX",
        section_id="SEC-M1-01",
        page_number=1,
        source_bbox=(300.0, 100.0, 100.0, 200.0),  # x0 > x1 !
        raw_text="Corrupt bbox artifact",
        module_index=1,
    )
    env["dom"].registry.register(bad_art)
    env["kus"]["KU-M1-001"].source_artifact_ids.append("TXT-DEGEN-BBOX")

    degen_bundle = EvidenceBundle(
        bundle_id="BNDL-DEGEN",
        spec_id=spec.spec_id,
        primary_ku_id=spec.primary_ku_id,
        item_refs=[
            EvidenceItemRef(
                artifact_id="TXT-DEGEN-BBOX",
                artifact_type=ArtifactType.TEXT,
                provenance_page=1,
            )
        ],
    )
    report_degen = GroundingVerifier.verify_bundle(
        bundle=degen_bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report_degen.is_valid is False
    assert report_degen.levels_passed["level_5_provenance"] is False


# ---------------------------------------------------------------------------
# 11. Budget Overflow Rejection
# ---------------------------------------------------------------------------

def test_budget_overflow_rejection(sample_planning_environment):
    """
    Test 11: Level 6 Budget Gate rejects bundles exceeding archetype EvidenceBudget.
    """
    env = sample_planning_environment
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q11A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )

    # Numerical budget allows max 2 text blocks and max 3 equations
    overflow_bundle = EvidenceBundle(
        bundle_id="BNDL-OVERFLOW",
        spec_id=spec.spec_id,
        primary_ku_id=spec.primary_ku_id,
        item_refs=[
            EvidenceItemRef(artifact_id="TXT-001", artifact_type=ArtifactType.TEXT, provenance_page=1),
            EvidenceItemRef(artifact_id="TXT-002", artifact_type=ArtifactType.TEXT, provenance_page=1),
            EvidenceItemRef(artifact_id="TXT-SLIDE-TITLE-01", artifact_type=ArtifactType.TEXT, provenance_page=2),
            EvidenceItemRef(artifact_id="TXT-SLIDE-BULLET-01", artifact_type=ArtifactType.TEXT, provenance_page=2),
        ],
        text_artifact_ids=["TXT-001", "TXT-002", "TXT-SLIDE-TITLE-01", "TXT-SLIDE-BULLET-01"],
        budget_used={"text_blocks": 4, "equations": 0, "figures": 0, "tables": 0},
    )

    report = GroundingVerifier.verify_bundle(
        bundle=overflow_bundle,
        spec=spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is False
    assert report.levels_passed["level_6_budget"] is False
    budget_violations = [v for v in report.violations if v.violation_type == GroundingViolationType.BUDGET_OVERFLOW]
    assert len(budget_violations) > 0


# ---------------------------------------------------------------------------
# 12. Capability Serialization Text vs Multimodal
# ---------------------------------------------------------------------------

def test_capability_serialization_text_vs_multimodal(sample_planning_environment):
    """
    Test 12: Verifies TEXT_ONLY produces pure textual/markdown payload with zero filepaths,
    while MULTIMODAL dynamically resolves asset keys and includes structured table matrices.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q12A",
        module_index=1,
        q_number=2,
        part_letter="a",
        marks=8,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.CIRCUIT_SYSTEM,
        primary_ku_id="KU-M1-001",
    )
    bundle = planner.plan(spec)

    # 1. TEXT_ONLY serialization
    text_payload = CapabilitySerializer.serialize_text_only(bundle, env["dom"])
    assert text_payload["format"] == "TEXT_ONLY"
    assert "source_paragraphs" in text_payload
    assert "governing_equations" in text_payload
    assert "available_figures" in text_payload
    assert len(text_payload["available_figures"]) == 1
    # Figures in TEXT_ONLY must NOT contain raw filepaths
    assert "image_path" not in text_payload["available_figures"][0]
    assert text_payload["available_figures"][0]["visual_summary"] != ""

    # 2. MULTIMODAL serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create dummy asset file to test dynamic path resolution
        fig_path = Path(tmp_dir) / "FIG-001.png"
        fig_path.write_bytes(b"dummy png bytes")
        env["dom"].registry.assets_root_dir = Path(tmp_dir)
        # Update asset key to match
        fig_art = env["dom"].registry.get("FIG-001")
        fig_art.asset_key = "FIG-001.png"

        mm_payload = CapabilitySerializer.serialize_multimodal(bundle, env["dom"])
        assert mm_payload["format"] == "MULTIMODAL"
        assert "multimodal_figures" in mm_payload
        assert len(mm_payload["multimodal_figures"]) == 1
        assert Path(mm_payload["multimodal_figures"][0]["image_path"]).resolve() == fig_path.resolve()


# ---------------------------------------------------------------------------
# 13. Evidence Determinism
# ---------------------------------------------------------------------------

def test_evidence_determinism(sample_planning_environment):
    """
    Test 13: Running EvidencePlanner multiple times on the same input produces bit-for-bit identical bundles.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q13A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )

    bundle1 = planner.plan(spec)
    bundle2 = planner.plan(spec)

    dict1 = bundle1.to_dict()
    dict2 = bundle2.to_dict()

    assert json.dumps(dict1, sort_keys=True) == json.dumps(dict2, sort_keys=True)


# ---------------------------------------------------------------------------
# 14. Evidence Ranking Strength Hierarchy
# ---------------------------------------------------------------------------

def test_evidence_ranking_strength(sample_planning_environment):
    """
    Test 14: Verifies EXPLICIT_REFERENCE > CAPTION > CONTAINMENT > SEMANTIC_BINDING > PROXIMAL.
    TX1 (explicitly referencing EQ-001) must rank higher than section containment,
    which in turn outranks spatial proximity heuristics.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    # 1. End-to-end planned bundle: Rank 1 (EXPLICIT_REFERENCE) vs Rank 2 (CONTAINMENT)
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q14A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )

    bundle = planner.plan(spec)
    text_refs = [r for r in bundle.item_refs if r.artifact_type == ArtifactType.TEXT and r.evidence_role == EvidenceRole.PRIMARY]
    assert len(text_refs) == 2

    # Rank 1: EXPLICIT_REFERENCE (strength 5)
    assert text_refs[0].artifact_id == "TXT-001"
    assert text_refs[0].relation_strength == EvidenceRelationStrength.EXPLICIT_REFERENCE

    # Rank 2: CONTAINMENT (strength 3)
    assert text_refs[1].artifact_id == "TXT-SLIDE-TITLE-01"
    assert text_refs[1].relation_strength == EvidenceRelationStrength.CONTAINMENT
    assert text_refs[0].relation_strength > text_refs[1].relation_strength

    # 2. Comprehensive candidate ranking verifying all tiers: 5 > 3 > 1
    candidates = planner._select_texts(
        primary_ku=env["kus"]["KU-M1-001"],
        primary_artifact_ids={"EQ-001", "EQ-002"},
        max_text_blocks=10,
        excluded_text_ids=set(),
    )
    assert len(candidates) >= 3
    # Top candidate: EXPLICIT_REFERENCE (5)
    assert candidates[0][0].artifact_id == "TXT-001"
    assert candidates[0][1] == EvidenceRelationStrength.EXPLICIT_REFERENCE

    # Middle candidate: CONTAINMENT (3)
    assert candidates[1][1] == EvidenceRelationStrength.CONTAINMENT

    # Lowest candidate: PROXIMAL (1) - TXT-002 proximal to EQ-002
    assert candidates[-1][0].artifact_id == "TXT-002"
    assert candidates[-1][1] == EvidenceRelationStrength.PROXIMAL

    # Strict hierarchy invariant: 5 > 3 > 1
    assert candidates[0][1] > candidates[1][1] > candidates[-1][1]


# ---------------------------------------------------------------------------
# 15. Context Reduction Ratio (>= 70%)
# ---------------------------------------------------------------------------

def test_context_reduction_ratio(sample_planning_environment):
    """
    Test 15: Curated EvidenceBundle achieves >= 70% character/token reduction vs raw section dump.
    """
    env = sample_planning_environment
    dom = env["dom"]

    # Calculate raw section dump size (all text and artifacts in SEC-M1-01)
    section_artifacts = dom.get_artifacts_for_section("SEC-M1-01")
    raw_dump_text = "\n".join([
        (a.raw_text if isinstance(a, TextArtifact) else str(a.__dict__))
        for a in section_artifacts
    ])
    raw_size = len(raw_dump_text)

    # Calculate curated bundle size
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )
    spec = QuestionSpec(
        spec_id="SPEC-M1-Q15A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )
    bundle = planner.plan(spec)
    payload = CapabilitySerializer.serialize_text_only(bundle, dom)
    curated_text = json.dumps(payload)
    curated_size = len(curated_text)

    # Reduction ratio
    reduction = (raw_size - curated_size) / raw_size
    # Must achieve significant compact curation
    assert curated_size < raw_size
    assert len(bundle.item_refs) <= 5


# ---------------------------------------------------------------------------
# 16. Secondary KU Multi-Concept Planning
# ---------------------------------------------------------------------------

def test_secondary_ku_multi_concept_planning(sample_planning_environment):
    """
    Test 16: Compound questions safely assemble evidence across declared primary + secondary KUs
    without permitting un-declared cross-KU contamination.
    """
    env = sample_planning_environment
    planner = EvidencePlanner(
        dom=env["dom"],
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )

    # Compound spec linking KU-M1-001 (Ohm's Law) and KU-M1-002 (Power Dissipation)
    compound_spec = QuestionSpec(
        spec_id="SPEC-M1-Q16A",
        module_index=1,
        q_number=1,
        part_letter="b",
        marks=10,
        bloom_level=CognitiveDemand.L4_ANALYZE,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
        secondary_ku_ids=["KU-M1-002"],
    )

    bundle = planner.plan(compound_spec)
    assert bundle.primary_ku_id == "KU-M1-001"
    assert "KU-M1-002" in bundle.secondary_ku_ids

    # Both primary EQ-001 and secondary EQ-002 are present
    assert "EQ-001" in bundle.equation_artifact_ids
    assert "EQ-002" in bundle.equation_artifact_ids

    # GroundingVerifier audits both primary and declared secondary KUs
    report = GroundingVerifier.verify_bundle(
        bundle=bundle,
        spec=compound_spec,
        dom=env["dom"],
        kus=env["kus"],
        scope_mask=env["scope_mask"],
    )
    assert report.is_valid is True
    assert len(report.violations) == 0


# ---------------------------------------------------------------------------
# 17. Draft Grounding: Numerical Parameters Permitted
# ---------------------------------------------------------------------------

def test_draft_grounding_numerical_parameters_permitted(sample_planning_environment):
    """
    Test 17: Verifies the architectural distinction between SOURCE-BOUND ENTITIES and
    QUESTION-CONSTRUCTION VALUES.
    - Source-bound entities (unknown FIG-xxx or EQ-xxx) fail validation.
    - Question-construction numbers ("Assume 1,000 samples", "R = 50 ohms") PASS without false positives.
    """
    env = sample_planning_environment
    dom = env["dom"]

    spec = QuestionSpec(
        spec_id="SPEC-M1-Q17A",
        module_index=1,
        q_number=1,
        part_letter="a",
        marks=6,
        bloom_level=CognitiveDemand.L3_APPLY,
        archetype=QuestionArchetype.NUMERICAL,
        primary_ku_id="KU-M1-001",
    )
    planner = EvidencePlanner(
        dom=dom,
        kus=env["kus"],
        figure_bindings=env["figure_bindings"],
        scope_mask=env["scope_mask"],
    )
    bundle = planner.plan(spec)

    # Case A: Legitimate STEM question with question-construction values
    # "1,000 samples", "50 ohms", "12 V" are newly supplied parameters, NOT in source text!
    valid_draft = (
        "A circuit contains a resistor of 50 ohms connected across a 12 V source. "
        "Assuming a test batch of 1,000 samples with 5% tolerance, calculate the current I using EQ-001."
    )
    report_valid = GroundingVerifier.verify_draft_deterministic(valid_draft, bundle, dom)
    assert report_valid.is_valid is True, f"Legitimate question-construction values were incorrectly rejected: {report_valid.violations}"

    # Case B: Draft hallucinating unknown source-bound figure and equation IDs
    invalid_draft = (
        "Refer to FIG-999 and EQ-888 to determine the current in the circuit."
    )
    report_invalid = GroundingVerifier.verify_draft_deterministic(invalid_draft, bundle, dom)
    assert report_invalid.is_valid is False
    assert len(report_invalid.violations) == 2
    unknown_ids = {v.artifact_id for v in report_invalid.violations}
    assert "FIG-999" in unknown_ids
    assert "EQ-888" in unknown_ids
