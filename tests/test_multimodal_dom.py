"""
Unit & Integration Tests for AION v2 Multimodal Document DOM
============================================================
Verifies all 5 core architectural constraints:
1. No flat-text extraction / no monolithic chunks
2. Extended artifact typing (Definitions, Examples, Procedures, Algorithms)
3. Portable asset keys (no absolute machine paths on artifacts)
4. Deterministic multi-strategy reference linking
5. Strict page accounting (zero silent page loss)
Also validates EvidenceBundle model-capability-aware serialization and budgets.
"""

import tempfile
from pathlib import Path
import pytest

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    TextArtifact,
    HeadingArtifact,
    EquationArtifact,
    FigureArtifact,
    TableArtifact,
    AlgorithmArtifact,
    DefinitionArtifact,
    ExampleArtifact,
    ProcedureArtifact,
    create_artifact_from_dict,
)
from aion.core.dom.artifact_registry import ArtifactRegistry
from aion.core.dom.relationships import (
    RelationshipType,
    Relationship,
    RelationshipGraph,
    ExplicitReferenceLinker,
    SectionContainmentLinker,
    SpatialProximityLinker,
    DeterministicReferenceLinker,
)
from aion.core.dom.document_dom import (
    DocumentDOM,
    ModuleNode,
    SectionNode,
    PageState,
    PageAuditRecord,
)
from aion.core.dom.evidence import (
    ModelCapability,
    QuestionArchetype,
    EvidenceBudget,
    ARCHETYPE_BUDGETS,
    EvidenceBundle,
)
from aion.core.extraction.contracts import (
    PageExtractionResult,
    PageExtractionStatus,
    RawTextBlock,
    RawHeading,
    RawEquation,
    RawFigure,
    RawTable,
)
from aion.core.extraction.fusion import ArtifactFusionEngine


def test_constraint_1_and_2_extended_typed_artifacts():
    """Verify artifacts retain discrete identity and extended academic typing."""
    txt = TextArtifact(
        artifact_id="TXT-M1-P01-0001",
        raw_text="The transport layer provides logical communication.",
        normalized_text="The transport layer provides logical communication.",
        reading_order=1,
    )
    assert txt.artifact_type == ArtifactType.TEXT
    assert "logical communication" in txt.normalized_text

    defn = DefinitionArtifact(
        artifact_id="DEF-M1-P01-0002",
        term="Throughput",
        definition="The rate at which data is successfully transmitted.",
    )
    assert defn.artifact_type == ArtifactType.DEFINITION
    assert defn.term == "Throughput"

    ex = ExampleArtifact(
        artifact_id="EX-M1-P02-0003",
        title="Example 1.1",
        problem_statement="Calculate propagation delay for 1000km fiber.",
        numerical_values={"distance_km": 1000, "speed": 2e8},
    )
    assert ex.artifact_type == ArtifactType.EXAMPLE
    assert ex.numerical_values["distance_km"] == 1000

    # Test round-trip factory reconstruction
    d = ex.to_dict()
    reconstructed = create_artifact_from_dict(d)
    assert isinstance(reconstructed, ExampleArtifact)
    assert reconstructed.artifact_id == ex.artifact_id
    assert reconstructed.numerical_values == ex.numerical_values


def test_constraint_3_portable_asset_resolution(tmp_path):
    """Verify figure paths are portable and resolved via registry, not hardcoded."""
    reg = ArtifactRegistry(assets_root_dir=tmp_path / "assets")

    # Store binary image data
    mock_png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDRtest_image"
    asset_key, sha, local_path = reg.store_asset_bytes(
        mock_png_bytes, relative_subpath="figures/m3", filename_hint="FIG-001.png"
    )

    fig = FigureArtifact(
        artifact_id="FIG-M3-P42-0001",
        module_index=3,
        section_id="SEC-M3-001",
        page_number=42,
        asset_key=asset_key,
        image_hash=sha,
        caption="Figure 3.4: Three-Phase Transformer",
        width=640,
        height=480,
    )
    reg.register(fig)

    # Invariant: Artifact stores portable asset_key, NOT an absolute machine path
    assert not Path(fig.asset_key).is_absolute()
    assert fig.asset_key.startswith("figures/m3/")

    # Invariant: Registry resolves canonical figure_id to local filesystem Path
    resolved_path = reg.resolve_figure_path("FIG-M3-P42-0001")
    assert resolved_path is not None
    assert resolved_path.exists()
    assert resolved_path.read_bytes() == mock_png_bytes


def test_constraint_4_deterministic_reference_linking():
    """Verify multi-tier reference linker resolves explicit refs and spatial containment."""
    graph = RelationshipGraph()

    fig = FigureArtifact(
        artifact_id="FIG-M2-P10-0001",
        page_number=10,
        caption="Figure 2.4: Finite State Machine for TCP",
        source_bbox=(50.0, 200.0, 400.0, 400.0),
    )
    eq = EquationArtifact(
        artifact_id="EQ-M2-P10-0002",
        page_number=10,
        latex=r"R = \frac{C}{1 - \rho}",
        equation_label="(2.1)",
        source_bbox=(50.0, 450.0, 300.0, 480.0),
    )
    txt = TextArtifact(
        artifact_id="TXT-M2-P10-0003",
        page_number=10,
        raw_text="As illustrated in Figure 2.4, the connection transitions. Furthermore, Eq. (2.1) defines response time.",
        normalized_text="As illustrated in Figure 2.4, the connection transitions. Furthermore, Eq. (2.1) defines response time.",
        source_bbox=(50.0, 100.0, 500.0, 180.0),
    )

    artifacts = [fig, eq, txt]
    linker = DeterministicReferenceLinker()
    links_added = linker.link(artifacts, graph)

    assert links_added >= 2

    # Check that txt references fig
    fig_edges = graph.get_outgoing(txt.artifact_id, RelationshipType.REFERENCES)
    target_ids = {e.target_id for e in fig_edges}
    assert fig.artifact_id in target_ids
    assert eq.artifact_id in target_ids

    # Check bidirectional lookup
    sources_to_fig = graph.get_incoming(fig.artifact_id, RelationshipType.REFERENCES)
    assert len(sources_to_fig) == 1
    assert sources_to_fig[0].source_id == txt.artifact_id


def test_constraint_5_strict_page_accounting():
    """Verify every single page is audited and missing pages are immediately caught."""
    dom = DocumentDOM(document_id="DOC-VTU-001", title="Computer Networks", total_pages=5)

    dom.record_page_audit(PageAuditRecord(page_number=1, state=PageState.EXTRACTED, char_count=1200))
    dom.record_page_audit(PageAuditRecord(page_number=2, state=PageState.EXTRACTED, char_count=950))
    dom.record_page_audit(PageAuditRecord(page_number=3, state=PageState.OCR_REQUIRED, char_count=300, ocr_applied=True))
    dom.record_page_audit(PageAuditRecord(page_number=4, state=PageState.EMPTY_EXPECTED, char_count=0))
    # Page 5 is intentionally omitted to verify accounting check

    summary = dom.get_page_audit_summary()
    assert summary["total_pages"] == 5
    assert summary["recorded_pages"] == 4
    assert summary["is_complete"] is False
    assert summary["missing_pages"] == [5]
    assert summary["state_breakdown"]["EXTRACTED"] == 2
    assert summary["state_breakdown"]["OCR_REQUIRED"] == 1
    assert summary["state_breakdown"]["EMPTY_EXPECTED"] == 1

    # Now record page 5
    dom.record_page_audit(PageAuditRecord(page_number=5, state=PageState.EXTRACTED, char_count=1100))
    updated_summary = dom.get_page_audit_summary()
    assert updated_summary["is_complete"] is True
    assert updated_summary["missing_pages"] == []


def test_evidence_bundle_model_capability_aware_payload(tmp_path):
    """Verify EvidenceBundle serializes differently for text-only vs multimodal models."""
    reg = ArtifactRegistry(assets_root_dir=tmp_path / "assets")

    # Store real mock asset
    asset_key, sha, local_p = reg.store_asset_bytes(b"dummy_image_data", relative_subpath="figures")
    fig = FigureArtifact(
        artifact_id="FIG-M3-P15-001",
        asset_key=asset_key,
        caption="Figure 3.2: TCP Window Growth",
        semantic_description="X-axis shows RTT, Y-axis shows Congestion Window, depicts slow start and linear growth.",
        width=500,
        height=300,
    )
    reg.register(fig)

    eq = EquationArtifact(
        artifact_id="EQ-M3-P15-002",
        latex=r"W_{new} = W_{current} + \frac{1}{W_{current}}",
        variables=["W_{new}", "W_{current}"],
        equation_label="(3.4)",
    )
    reg.register(eq)

    txt = TextArtifact(
        artifact_id="TXT-M3-P15-003",
        normalized_text="During congestion avoidance, the window increases linearly.",
        referenced_artifact_ids=[fig.artifact_id, eq.artifact_id],
    )
    reg.register(txt)

    bundle = EvidenceBundle(
        bundle_id="BNDL-M3-Q05",
        module_index=3,
        section_id="SEC-M3-002",
        section_title="3.4 Congestion Control",
        primary_concept="TCP Congestion Avoidance",
        archetype=QuestionArchetype.NUMERICAL,
        texts=[txt],
        equations=[eq],
        figures=[fig],
    )

    # 1. Test TEXT_ONLY Payload (Qwen2.5-32B text mode)
    text_payload = bundle.to_llm_payload(ModelCapability.TEXT_ONLY, registry=reg)
    assert "available_figures" in text_payload
    assert "multimodal_figures" not in text_payload
    fig_summary = text_payload["available_figures"][0]
    assert fig_summary["figure_id"] == "FIG-M3-P15-001"
    assert "linear growth" in fig_summary["visual_summary"]
    # Verify no machine file paths leak to text model
    assert "image_path" not in fig_summary

    # 2. Test MULTIMODAL Payload (Qwen2.5-VL mode)
    mm_payload = bundle.to_llm_payload(ModelCapability.MULTIMODAL, registry=reg)
    assert "multimodal_figures" in mm_payload
    mm_fig = mm_payload["multimodal_figures"][0]
    assert mm_fig["figure_id"] == "FIG-M3-P15-001"
    assert mm_fig["image_path"] == str(local_p)
    assert mm_fig["dimensions"] == [500, 300]


def test_evidence_budget_archetype_enforcement():
    """Verify archetype budgets prevent prompt starvation and context dilution."""
    num_budget = ARCHETYPE_BUDGETS[QuestionArchetype.NUMERICAL]
    assert num_budget.min_equations >= 1
    assert num_budget.max_equations <= 3
    assert num_budget.max_text_blocks == 2

    circuit_budget = ARCHETYPE_BUDGETS[QuestionArchetype.CIRCUIT_SYSTEM]
    assert circuit_budget.min_figures == 1  # Mandatory figure for circuit questions!

    comp_budget = ARCHETYPE_BUDGETS[QuestionArchetype.COMPARISON]
    assert comp_budget.min_tables == 1   # Mandatory table for comparison questions!


def test_artifact_fusion_engine_end_to_end(tmp_path):
    """Test full extraction-to-DOM fusion pipeline with multi-page stream."""
    dom = DocumentDOM(document_id="DOC-TEST-001", title="Operating Systems", total_pages=2)
    dom.registry.assets_root_dir = tmp_path / "assets"
    fusion = ArtifactFusionEngine(dom)

    # Page 1: Chapter Heading, Definition, Equation
    p1 = PageExtractionResult(
        page_number=1,
        status=PageExtractionStatus.EXTRACTED,
        char_count=500,
        headings=[RawHeading(title="CPU Scheduling", level=2, numbering="3.2")],
        text_blocks=[
            RawTextBlock(text="Definition: CPU scheduling is the basis of multiprogrammed operating systems."),
            RawTextBlock(text="The scheduler selects a process from the ready queue."),
        ],
        equations=[RawEquation(latex=r"T_{wait} = T_{turnaround} - T_{burst}", label="(3.1)")],
    )
    fusion.fuse_page(p1, module_mapping={1: (1, 2)})

    # Page 2: Algorithm, Figure, Example
    mock_fig_bytes = b"\x89PNG\r\n\x1a\n_mock_queue_diagram"
    p2 = PageExtractionResult(
        page_number=2,
        status=PageExtractionStatus.EXTRACTED,
        char_count=600,
        text_blocks=[
            RawTextBlock(text="Algorithm 3.1: Round Robin\n1. Select first process\n2. Run for time quantum q\n3. Preempt"),
            RawTextBlock(text="Example 3.1: Given 3 processes with burst times 10, 4, 2, calculate average turnaround time."),
        ],
        figures=[RawFigure(image_bytes=mock_fig_bytes, caption_hint="Figure 3.2: Round Robin Queue")],
    )
    fusion.fuse_page(p2, module_mapping={1: (1, 2)})
    fusion.finalize()

    # Verify DOM structure
    assert 1 in dom.modules
    mod1 = dom.modules[1]
    assert mod1.start_page == 1
    assert mod1.end_page == 2

    # Verify semantic artifact classification
    definitions = dom.registry.get_by_type(ArtifactType.DEFINITION)
    assert len(definitions) == 1
    assert "CPU scheduling" in definitions[0].definition or "CPU scheduling" in definitions[0].term

    algorithms = dom.registry.get_by_type(ArtifactType.ALGORITHM)
    assert len(algorithms) == 1
    assert "Round Robin" in algorithms[0].title

    examples = dom.registry.get_by_type(ArtifactType.EXAMPLE)
    assert len(examples) == 1
    assert "burst times" in examples[0].problem_statement

    figures = dom.registry.get_by_type(ArtifactType.FIGURE)
    assert len(figures) == 1
    assert "Round Robin Queue" in figures[0].caption

    # Verify page accounting
    summary = dom.get_page_audit_summary()
    assert summary["is_complete"] is True
    assert summary["total_pages"] == 2
    assert summary["recorded_pages"] == 2
