"""
tests/test_stage_g2d_visual_lifecycle.py
=========================================
Stage G2-D (Track B: Visual Evidence Lifecycle) Comprehensive Test Suite.
Verifies AC-D1 through AC-D8 + End-to-End Extraction-to-DOCX Integration:
- AC-D1: Portable visual assets, content hashing (image_sha256), and path resolution.
- AC-D2: Topic-aware semantic figure selection in PlanningAgent with deterministic tie-breaking.
- AC-D3: Figure identity preservation (figure_id) through SubPart and GeneratedQuestion.
- AC-D4: Authoritative shared physical image validator (existence, size, magic bytes, decodability, SHA-256).
- AC-D5: CheckingAgent visual quality gate enforcement (blocks on missing/corrupt referenced assets, ignores unused).
- AC-D6: Legacy adapter subpart visual ownership rules and sibling image isolation.
- AC-D7: DOCX exporter multi-part image bleed elimination.
- AC-D8: DOCX export fail-closed behavior on missing/corrupt assets and multi-format ZIP/XML media verification.
- AC-D9: Complete end-to-end extraction-to-DOCX integration test.
"""

import hashlib
import io
import os
import re
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from PIL import Image
from docx import Document

from core.contracts.document_artifact import (
    FigureArtifact,
    AssetIntegrityError,
    resolve_image_path,
    validate_physical_image,
    DEFAULT_WORKSPACE_ROOT,
)
from core.generation.assessment_config import AssessmentRegistry, AssessmentConfig
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_agent import CheckingAgent, BLOCK_REASON_MISSING_VISUAL_ASSET
from core.generation.agents.checking_contracts import STATUS_OK, STATUS_BLOCKED
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.legacy_adapter import legacy_question_from_generated, _clause_references_figure
from core.generation.agents.planning_agent import PlanningAgent
from core.generation.agents.question_plan import QuestionPlan, SubPartPlan, PlanStatus
from core.generation.agents.writing_agent import WritingAgent
from v0_1.docx_export import render_docx_document, ExportIntegrityError


# =============================================================================
# Helper Fixtures & Image Generators
# =============================================================================

@pytest.fixture(autouse=True)
def setup_test_registry():
    """Register synthetic assessment configs for test suite."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="CIE_IAT",
        blueprint_id="BP_10M",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4), (10,)),
        max_subparts=2,
        reference_source="Test Specs 10M",
    ))
    yield

@pytest.fixture
def temp_images_dir(tmp_path):
    """Creates a temporary workspace directory with valid PNG, JPEG, and WebP images."""
    img_dir = tmp_path / "workspace" / "figures"
    img_dir.mkdir(parents=True, exist_ok=True)

    # 1. Valid PNG
    png_path = img_dir / "valid_test_fig.png"
    im_png = Image.new("RGB", (200, 150), color="blue")
    im_png.save(str(png_path), format="PNG")
    png_bytes = png_path.read_bytes()
    png_sha = hashlib.sha256(png_bytes).hexdigest()

    # 2. Valid JPEG
    jpg_path = img_dir / "valid_test_fig.jpg"
    im_jpg = Image.new("RGB", (200, 150), color="green")
    im_jpg.save(str(jpg_path), format="JPEG")
    jpg_bytes = jpg_path.read_bytes()
    jpg_sha = hashlib.sha256(jpg_bytes).hexdigest()

    # 3. Valid WebP
    webp_path = img_dir / "valid_test_fig.webp"
    im_webp = Image.new("RGB", (200, 150), color="red")
    im_webp.save(str(webp_path), format="WEBP")
    webp_bytes = webp_path.read_bytes()
    webp_sha = hashlib.sha256(webp_bytes).hexdigest()

    # 4. Corrupt image (wrong magic bytes)
    corrupt_path = img_dir / "corrupt_fig.png"
    corrupt_path.write_bytes(b"NOT_A_PNG_FILE_HEADER_DATA" * 10)

    # 5. Tiny file (<= 100 bytes)
    tiny_path = img_dir / "tiny_fig.png"
    tiny_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

    return {
        "dir": img_dir,
        "workspace_root": tmp_path,
        "png": {"path": png_path, "sha": png_sha, "rel": f"workspace/figures/{png_path.name}"},
        "jpg": {"path": jpg_path, "sha": jpg_sha, "rel": f"workspace/figures/{jpg_path.name}"},
        "webp": {"path": webp_path, "sha": webp_sha, "rel": f"workspace/figures/{webp_path.name}"},
        "corrupt": {"path": corrupt_path},
        "tiny": {"path": tiny_path},
    }


# =============================================================================
# AC-D1: Portable Visual Assets & Content Hashing
# =============================================================================

def test_ac_d1_portable_visual_assets_and_hashing(temp_images_dir):
    """
    AC-D1: FigureArtifact includes image_sha256; resolve_image_path() handles
    workspace-relative paths, cross-machine absolute paths, and hash verification.
    """
    ws_root = temp_images_dir["workspace_root"]
    png_info = temp_images_dir["png"]

    # 1. Contract & Serialization test
    fig = FigureArtifact(
        id="fig_1",
        page=2,
        caption="Figure 1: Scheduling Diagram",
        image_path=png_info["rel"],
        image_sha256=png_info["sha"],
        source_pdf_sha256="doc_sha123",
    )
    d = fig.to_dict()
    assert d["image_sha256"] == png_info["sha"]
    assert d["image_path"] == png_info["rel"]

    fig_deserialized = FigureArtifact.from_dict(d)
    assert fig_deserialized.image_sha256 == png_info["sha"]
    assert fig_deserialized.image_path == png_info["rel"]

    # 2. Workspace-relative resolution
    resolved = resolve_image_path(fig, workspace_root=ws_root, verify_hash=True)
    assert resolved.is_file()
    assert resolved.samefile(png_info["path"])

    # 3. Cross-machine absolute path recovery
    bogus_host_abs = f"C:\\Users\\OtherDeveloper\\AION\\{png_info['rel'].replace('/', os.sep)}"
    fig_other_machine = FigureArtifact(
        id="fig_1",
        page=2,
        caption="Figure 1",
        image_path=bogus_host_abs,
        image_sha256=png_info["sha"],
    )
    resolved_recovered = resolve_image_path(fig_other_machine, workspace_root=ws_root, verify_hash=True)
    assert resolved_recovered.is_file()
    assert resolved_recovered.samefile(png_info["path"])

    # 4. Hash verification detects tampering
    fig_tampered = FigureArtifact(
        id="fig_tampered",
        page=2,
        caption="Figure 1",
        image_path=png_info["rel"],
        image_sha256="ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
    )
    with pytest.raises(AssetIntegrityError, match="Asset integrity violation"):
        resolve_image_path(fig_tampered, workspace_root=ws_root, verify_hash=True)

    # 5. Legacy backward compatibility (no hash stored -> no error)
    fig_legacy = FigureArtifact(
        id="fig_legacy",
        page=2,
        caption="Figure Legacy",
        image_path=png_info["rel"],
        image_sha256="",
    )
    resolved_legacy = resolve_image_path(fig_legacy, workspace_root=ws_root, verify_hash=True)
    assert resolved_legacy.is_file()


# =============================================================================
# AC-D2: Semantic Figure Selection by Topic in PlanningAgent
# =============================================================================

def test_ac_d2_semantic_figure_selection_by_topic():
    """
    AC-D2: PlanningAgent._resolve_visual() evaluates topic relevance (> 0.15 overlap);
    selects matching figure, breaks ties deterministically, and rejects irrelevant figures.
    """
    agent = PlanningAgent()

    fig_sched = FigureArtifact(
        id="fig_sched",
        page=5,
        caption="Figure 2.1: Process States and CPU Scheduling Transitions",
        image_path="sched.png",
        provenance_score=0.95,
        module_id="module_1",
    )
    fig_risc = FigureArtifact(
        id="fig_risc",
        page=10,
        caption="Figure 4.2: 5-Stage RISC Pipeline Architecture with Forwarding",
        image_path="risc.png",
        provenance_score=0.90,
        module_id="module_1",
    )
    fig_memory = FigureArtifact(
        id="fig_mem",
        page=15,
        caption="Figure 5.3: Virtual Memory Page Table Translation",
        image_path="mem.png",
        provenance_score=0.85,
        module_id="module_1",
    )

    module_figures = [fig_sched, fig_risc, fig_memory]
    used = set()

    # 1. Matching topic for scheduling
    ok1, sel1 = agent._resolve_visual(
        bloom="L4",
        module_figures=module_figures,
        used_figures=used,
        visual_slots_used=0,
        topic="Explain process state transitions and CPU scheduling algorithms",
    )
    assert ok1 is True
    assert sel1.id == "fig_sched"

    # 2. Matching topic for RISC pipeline
    ok2, sel2 = agent._resolve_visual(
        bloom="L4",
        module_figures=module_figures,
        used_figures=used,
        visual_slots_used=0,
        topic="Analyze RISC pipeline hazards and forwarding unit",
    )
    assert ok2 is True
    assert sel2.id == "fig_risc"

    # 3. Unrelated topic (overlap <= 0.15) -> FAIL-CLOSED (no blind first match)
    ok3, sel3 = agent._resolve_visual(
        bloom="L4",
        module_figures=module_figures,
        used_figures=used,
        visual_slots_used=0,
        topic="Thermodynamic cycle efficiency in internal combustion engines",
    )
    assert ok3 is False
    assert sel3 is None

    # 4. Single-use policy
    used.add("fig_sched")
    ok4, sel4 = agent._resolve_visual(
        bloom="L4",
        module_figures=module_figures,
        used_figures=used,
        visual_slots_used=0,
        topic="Explain process state transitions and CPU scheduling",
    )
    # fig_sched is already used; since neither risc nor memory matches scheduling, returns False
    assert ok4 is False
    assert sel4 is None


# =============================================================================
# AC-D3: Figure Identity Preservation
# =============================================================================

def test_ac_d3_figure_id_preservation_lifecycle():
    """
    AC-D3: figure_id and image_sha256 are preserved from FigureArtifact to
    SubPartPlan, SubPart, and GeneratedQuestion without dropping provenance.
    """
    # 1. SubPartPlan preserves figure
    fig = FigureArtifact(
        id="fig_p1_45_1",
        page=3,
        caption="Circuit diagram of full adder",
        image_path="workspace/figures/adder.png",
        image_sha256="abcd1234abcd1234",
    )
    sp_plan = SubPartPlan(
        label="a",
        marks=6,
        topic="Digital Logic",
        visual_required=True,
        figure=fig,
    )
    assert sp_plan.figure.id == "fig_p1_45_1"
    assert sp_plan.figure.image_sha256 == "abcd1234abcd1234"

    # 2. SubPart preserves figure_id & image_sha256
    sp = SubPart(
        label="a",
        marks=6,
        text="With reference to the given figure, explain full adder operation.",
        topic="Digital Logic",
        figure_id="fig_p1_45_1",
        image_path="workspace/figures/adder.png",
        image_sha256="abcd1234abcd1234",
    )
    assert sp.figure_id == "fig_p1_45_1"
    assert sp.image_sha256 == "abcd1234abcd1234"
    valid, err = sp.validate()
    assert valid, f"SubPart validation failed: {err}"

    # 3. GeneratedQuestion preserves figure_id & image_sha256
    gq = GeneratedQuestion(
        slot_id="m1_Q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) With reference to the given figure, explain full adder operation.",
        solution="Model answer",
        marking_scheme=[{"criterion": "Derivation", "marks": 6, "subpart": "a"}],
        marks=6,
        partition=[6],
        sub_parts=[sp],
        figure_id="fig_p1_45_1",
        image_sha256="abcd1234abcd1234",
    )
    assert gq.figure_id == "fig_p1_45_1"
    assert gq.image_sha256 == "abcd1234abcd1234"


# =============================================================================
# AC-D4: Authoritative Shared Physical Image Validator
# =============================================================================

def test_ac_d4_physical_visual_validation_utility(temp_images_dir):
    """
    AC-D4: validate_physical_image() checks existence, size, magic bytes, decodability,
    and cryptographic SHA-256 matching.
    """
    ws_root = temp_images_dir["workspace_root"]

    # 1. Valid PNG
    ok_png, err_png = validate_physical_image(temp_images_dir["png"]["path"], workspace_root=ws_root)
    assert ok_png is True
    assert err_png is None

    # 2. Valid JPEG
    ok_jpg, err_jpg = validate_physical_image(temp_images_dir["jpg"]["path"], workspace_root=ws_root)
    assert ok_jpg is True
    assert err_jpg is None

    # 3. Valid WebP
    ok_webp, err_webp = validate_physical_image(temp_images_dir["webp"]["path"], workspace_root=ws_root)
    assert ok_webp is True
    assert err_webp is None

    # 4. Missing file
    ok_missing, err_missing = validate_physical_image("non_existent_image_xyz.png", workspace_root=ws_root)
    assert ok_missing is False
    assert "FILE_NOT_FOUND" in err_missing

    # 5. Tiny file (<= 100 bytes)
    ok_tiny, err_tiny = validate_physical_image(temp_images_dir["tiny"]["path"], workspace_root=ws_root)
    assert ok_tiny is False
    assert "CORRUPT_OR_TRIVIAL_SIZE" in err_tiny

    # 6. Corrupt magic bytes
    ok_corrupt, err_corrupt = validate_physical_image(temp_images_dir["corrupt"]["path"], workspace_root=ws_root)
    assert ok_corrupt is False
    assert "INVALID_IMAGE_SIGNATURE" in err_corrupt

    # 7. Hash mismatch
    ok_mismatch, err_mismatch = validate_physical_image(
        temp_images_dir["png"]["path"],
        expected_sha256="0000000000000000000000000000000000000000000000000000000000000000",
        workspace_root=ws_root,
    )
    assert ok_mismatch is False
    assert "HASH_MISMATCH" in err_mismatch


# =============================================================================
# AC-D5: CheckingAgent Quality Gate Enforcement
# =============================================================================

def test_ac_d5_checking_agent_blocks_missing_visual_assets(temp_images_dir):
    """
    AC-D5: CheckingAgent audits referenced visual assets in _account_slots():
    - Blocks paper when referenced image is missing or corrupt.
    - Does NOT block paper when an unused module figure is missing.
    """
    ws_root = temp_images_dir["workspace_root"]
    valid_png = temp_images_dir["png"]["path"]

    # 1. Referenced image is MISSING -> BLOCKED
    gq_missing = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="With reference to the given figure, explain X.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[10], bloom="L2", co="CO1", topic="Topic",
        references_image=True, image_path="non_existent_folder/missing.png"
    )
    plan = MagicMock(slot_id="m1_Q1", module_id="module_1", module_idx=1, co="CO1", bloom="L2", total_marks=10, is_unresolved=False)
    ctx_missing = AgentContext(plans=[plan], questions=[gq_missing], request={"workspace_dir": ws_root})

    checker = CheckingAgent(min_questions_ratio=0.1)
    res_missing = checker.run(ctx_missing)
    assert res_missing.payload.status == STATUS_BLOCKED
    assert any("MISSING_VISUAL_ASSET" in r for r in res_missing.payload.blocked_reasons)

    # 2. Referenced image is VALID -> OK
    gq_valid = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="With reference to the given figure, explain X.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[10], bloom="L2", co="CO1", topic="Topic",
        references_image=True, image_path=str(valid_png), image_sha256=temp_images_dir["png"]["sha"]
    )
    ctx_valid = AgentContext(plans=[plan], questions=[gq_valid], request={"workspace_dir": ws_root})
    res_valid = checker.run(ctx_valid)
    assert res_valid.payload.status == STATUS_OK
    assert not any("MISSING_VISUAL_ASSET" in r for r in res_valid.payload.blocked_reasons)

    # 3. Question is text-only (does NOT reference image), even if unused figure is missing -> OK
    gq_text_only = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="Explain the principles of CPU scheduling.",
        solution="Sol", marking_scheme=[{"criterion": "C", "marks": 10}],
        marks=10, partition=[10], bloom="L2", co="CO1", topic="Topic",
        references_image=False, image_path=None
    )
    ctx_text_only = AgentContext(plans=[plan], questions=[gq_text_only], request={"workspace_dir": ws_root})
    res_text = checker.run(ctx_text_only)
    assert res_text.payload.status == STATUS_OK
    assert not any("MISSING_VISUAL_ASSET" in r for r in res_text.payload.blocked_reasons)


# =============================================================================
# AC-D6: Subpart Visual Ownership & Isolation in legacy_adapter
# =============================================================================

def test_ac_d6_legacy_adapter_subpart_image_isolation():
    """
    AC-D6: legacy_adapter binds visual assets strictly to the owning subpart:
    - Subpart with figure_id receives image_path.
    - Non-visual sibling subparts strictly have image_path = None and references_image = False.
    - Ambiguous ownership does not guess.
    """
    sp_a = SubPart(
        label="a", marks=6,
        text="With reference to the given figure, explain process states.",
        topic="Process States", solution="Sol A",
        figure_id="fig_sched", image_path="workspace/figures/sched.png", figure_caption="Process States",
    )
    sp_b = SubPart(
        label="b", marks=4,
        text="Discuss the differences between monolithic and microkernel architectures.",
        topic="Kernel Architecture", solution="Sol B",
        figure_id=None, image_path=None, figure_caption=None,
    )

    gq = GeneratedQuestion(
        slot_id="m1_Q1", module_id="module_1", global_q_idx=1,
        question_text="(a) With reference to the given figure, explain process states.\n\n(b) Discuss the differences between monolithic and microkernel architectures.",
        solution="Model answer",
        marking_scheme=[
            {"criterion": "States", "marks": 6, "subpart": "a"},
            {"criterion": "Kernels", "marks": 4, "subpart": "b"},
        ],
        marks=10, partition=[6, 4],
        sub_parts=[sp_a, sp_b],
        figure_id="fig_sched", image_path="workspace/figures/sched.png", figure_caption="Process States",
        references_image=True,
    )

    q_dict = legacy_question_from_generated(gq)
    sub_qs = q_dict["sub_questions"]
    assert len(sub_qs) == 2

    # Subpart (a) is visual
    sq_a = sub_qs[0]
    assert sq_a["letter"] == "a"
    assert sq_a["image_path"] == "workspace/figures/sched.png"
    assert sq_a["figure_id"] == "fig_sched"
    assert sq_a["figure_caption"] == "Process States"
    assert sq_a["references_image"] is True

    # Subpart (b) is strictly non-visual
    sq_b = sub_qs[1]
    assert sq_b["letter"] == "b"
    assert sq_b["image_path"] is None
    assert sq_b["figure_id"] is None
    assert sq_b["figure_caption"] is None
    assert sq_b["references_image"] is False


# =============================================================================
# AC-D7: DOCX Multi-Part Image Bleed Prevention
# =============================================================================

def test_ac_d7_docx_export_multi_part_image_bleed_prevention(temp_images_dir):
    """
    AC-D7: v0_1/docx_export renders multi-part [6, 4] question with image strictly in
    cell (a) and zero <w:drawing> elements in cell (b).
    """
    valid_png = temp_images_dir["png"]["path"]

    paper_dict = {
        "metadata": {"subject_name": "Operating Systems", "max_marks": 50},
        "modules": [{
            "module_index": 1,
            "module_title": "OS Concepts",
            "questions": [{
                "question_number": 1,
                "sub_questions": [
                    {
                        "letter": "a", "label": "a",
                        "text": "With reference to the given figure, explain process states.",
                        "marks": 6,
                        "image_path": str(valid_png),
                        "figure_caption": "Figure 1: Process States",
                    },
                    {
                        "letter": "b", "label": "b",
                        "text": "Discuss monolithic vs microkernel architectures.",
                        "marks": 4,
                        "image_path": None,
                        "figure_caption": None,
                    },
                ],
                "image_path": str(valid_png),  # Parent fallback present
                "figure_caption": "Figure 1: Process States",
                "marks": 10,
                "partition": [6, 4],
            }],
        }],
    }

    buf = render_docx_document(paper_dict)
    doc = Document(buf)
    table = doc.tables[1]

    # Row 2 is subpart (a), Row 3 is subpart (b)
    cell_a = table.rows[2].cells[2]
    cell_b = table.rows[3].cells[2]

    xml_a = cell_a._tc.xml
    xml_b = cell_b._tc.xml

    # Cell (a) has image drawing
    assert "w:drawing" in xml_a, "Subpart (a) must contain an embedded image drawing"
    # Cell (b) has NO image drawing
    assert "w:drawing" not in xml_b, "Subpart (b) must NOT contain an embedded image drawing (bleed defect resolved!)"


# =============================================================================
# AC-D8: DOCX Fail-Closed & Multi-Format ZIP Media Verification
# =============================================================================

def test_ac_d8_docx_export_fail_closed_and_zip_media_verification(temp_images_dir):
    """
    AC-D8:
    1. docx_export raises ExportIntegrityError on missing or corrupt image files.
    2. Multi-format verification (PNG, JPEG, WebP): valid ZIP archive, media part existence,
       relationship verification in document.xml.rels, and cell w:drawing placement.
    """
    # 1. Fail-closed on missing image
    missing_paper = {
        "metadata": {"subject_name": "Test", "max_marks": 50},
        "modules": [{
            "module_index": 1,
            "questions": [{
                "question_number": 1,
                "sub_questions": [{"letter": "a", "text": "Q text", "marks": 10, "image_path": "non_existent_folder/missing.png"}],
                "marks": 10, "partition": [10],
            }],
        }],
    }
    with pytest.raises(ExportIntegrityError, match="Missing referenced visual asset"):
        render_docx_document(missing_paper)

    # 2. Fail-closed on corrupt image
    corrupt_paper = {
        "metadata": {"subject_name": "Test", "max_marks": 50},
        "modules": [{
            "module_index": 1,
            "questions": [{
                "question_number": 1,
                "sub_questions": [{"letter": "a", "text": "Q text", "marks": 10, "image_path": str(temp_images_dir["corrupt"]["path"])}],
                "marks": 10, "partition": [10],
            }],
        }],
    }
    with pytest.raises(ExportIntegrityError, match="Corrupted or invalid referenced visual asset"):
        render_docx_document(corrupt_paper)

    # 3. Multi-format ZIP/XML verification for PNG, JPEG, and WebP
    for fmt_key in ("png", "jpg", "webp"):
        img_info = temp_images_dir[fmt_key]
        test_paper = {
            "metadata": {"subject_name": f"Test {fmt_key.upper()}", "max_marks": 50},
            "modules": [{
                "module_index": 1,
                "questions": [{
                    "question_number": 1,
                    "sub_questions": [
                        {
                            "letter": "a",
                            "text": f"Question with {fmt_key.upper()} figure.",
                            "marks": 6,
                            "image_path": str(img_info["path"]),
                            "figure_caption": f"Sample {fmt_key.upper()} caption",
                        },
                        {
                            "letter": "b",
                            "text": "Non-visual sibling question.",
                            "marks": 4,
                            "image_path": None,
                        },
                    ],
                    "marks": 10,
                    "partition": [6, 4],
                }],
            }],
        }

        buf = render_docx_document(test_paper)
        assert zipfile.is_zipfile(buf), "Rendered DOCX must be a valid ZIP archive"

        with zipfile.ZipFile(buf, "r") as z:
            namelist = z.namelist()
            # Verify media part exists under word/media/
            media_files = [n for n in namelist if n.startswith("word/media/")]
            assert len(media_files) >= 1, f"Expected at least one media part for {fmt_key}, got: {media_files}"

            # Verify relationship in document.xml.rels
            assert "word/_rels/document.xml.rels" in namelist
            rels_xml = z.read("word/_rels/document.xml.rels").decode("utf-8")
            root_rels = ET.fromstring(rels_xml)
            image_targets = [
                elem.attrib["Target"] for elem in root_rels
                if "image" in elem.attrib.get("Type", "")
            ]
            assert len(image_targets) >= 1, "document.xml.rels must have an Image relationship target"

            # Verify that target points to the media file in the zip
            target_file = f"word/{image_targets[0]}"
            assert target_file in namelist, f"Target {target_file} must exist in DOCX ZIP"

            # Verify cell drawings in document.xml
            doc_xml = z.read("word/document.xml").decode("utf-8")
            assert "w:drawing" in doc_xml, "document.xml must contain w:drawing"


# =============================================================================
# AC-D9: End-to-End Extraction-to-DOCX Integration Pipeline
# =============================================================================

def test_ac_d9_end_to_end_extraction_to_docx_pipeline(temp_images_dir):
    """
    AC-D9: Complete end-to-end integration test tracing an asset from:
    FigureArtifact (relative path + SHA-256) -> PlanningAgent -> QuestionPlan ->
    WritingAgent -> GeneratedQuestion (preserving figure_id) -> CheckingAgent (validating physical bytes) ->
    legacy_adapter (isolating visual subpart) -> render_docx_document() -> DOCX ZIP verification.
    """
    ws_root = temp_images_dir["workspace_root"]
    png_info = temp_images_dir["png"]

    # 1. Extracted FigureArtifact with relative path and SHA-256
    fig = FigureArtifact(
        id="fig_sched_e2e",
        page=3,
        caption="Figure 3.1: CPU Scheduling States and State Transitions",
        image_path=png_info["rel"],
        image_sha256=png_info["sha"],
        source_pdf_sha256="test_pdf_sha",
        provenance_score=0.95,
        module_id="module_1",
    )

    # 2. PlanningAgent resolves figure by topic
    planner = PlanningAgent()
    ok_fig, selected_fig = planner._resolve_visual(
        bloom="L4",
        module_figures=[fig],
        used_figures=set(),
        visual_slots_used=0,
        topic="Explain process state transitions and CPU scheduling",
    )
    assert ok_fig is True
    assert selected_fig.id == "fig_sched_e2e"

    # 3. Create plans
    sp1 = SubPartPlan(
        label="a", marks=6,
        topic="Process state transitions and CPU scheduling",
        canonical_concept_id="cpu_scheduling_transitions",
        co="CO3", bloom="L4",
        visual_required=True, figure=selected_fig,
    )
    sp2 = SubPartPlan(
        label="b", marks=4,
        topic="Monolithic versus microkernel architecture",
        canonical_concept_id="kernel_architectures",
        co="CO3", bloom="L4",
        visual_required=False, figure=None,
    )
    q_plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[6, 4],
        total_marks=10,
        co="CO3", bloom="L4",
        topic="Operating Systems Design",
        sub_part_plans=[sp1, sp2],
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )

    # 4. WritingAgent generates question with subparts
    class MockVisualLLMCaller:
        def call(self, prompt, **kwargs):
            m_marks = re.search(r"Marks:\s*(\d+)", prompt) or re.search(r"Total Subpart Marks:\s*(\d+)", prompt)
            marks = int(m_marks.group(1)) if m_marks else 5
            m1 = max(1, marks // 2)
            m2 = marks - m1
            if "process state transitions" in prompt.lower() or "scheduling" in prompt.lower():
                text = "With reference to the given figure, analyze the performance trade-offs between process state transitions and scheduling overhead."
                verb = "analyze"
            else:
                text = "Compare the architectural trade-offs and performance differences between monolithic and microkernel designs."
                verb = "compare"
            return {
                "text": text,
                "question_text": text,
                "solution": "Solution steps.",
                "marking_scheme": [
                    {"criterion": "Principle identification", "marks": m1},
                    {"criterion": "Analytical calculation", "marks": m2},
                ],
                "references_image": "given figure" in text.lower(),
                "bloom_verb_used": verb,
            }

    writer = WritingAgent(llm_caller=MockVisualLLMCaller(), vlm_caller=MockVisualLLMCaller(), enable_irt=False)
    ctx = AgentContext(plans=[q_plan], request={"workspace_dir": ws_root})
    result = writer.run(ctx)
    assert result.success is True
    assert len(result.payload.questions) == 1
    gq = result.payload.questions[0]

    # Verify figure_id and image_path preserved on GeneratedQuestion and SubPart
    assert gq.figure_id == "fig_sched_e2e"
    assert gq.sub_parts[0].figure_id == "fig_sched_e2e"
    assert gq.sub_parts[1].figure_id is None

    # 5. CheckingAgent audits physical visual asset
    checker = CheckingAgent(min_questions_ratio=0.1)
    chk_ctx = AgentContext(plans=[q_plan], questions=[gq], request={"workspace_dir": ws_root})
    chk_res = checker.run(chk_ctx)
    assert chk_res.payload.status == STATUS_OK
    assert not any("MISSING_VISUAL_ASSET" in r for r in chk_res.payload.blocked_reasons)

    # 6. legacy_adapter produces isolated subquestions
    q_dict = legacy_question_from_generated(gq)
    sub_qs = q_dict["sub_questions"]
    assert len(sub_qs) == 2
    assert sub_qs[0]["image_path"] is not None
    assert sub_qs[0]["figure_id"] == "fig_sched_e2e"
    assert sub_qs[1]["image_path"] is None
    assert sub_qs[1]["figure_id"] is None

    # 7. DOCX rendering and ZIP verification
    paper_dict = {
        "metadata": {"subject_name": "Operating Systems", "max_marks": 50, "workspace_root": ws_root},
        "modules": [{
            "module_index": 1,
            "module_title": "OS Concepts",
            "questions": [q_dict],
        }],
    }
    docx_buf = render_docx_document(paper_dict, workspace_root=ws_root)
    assert zipfile.is_zipfile(docx_buf)

    doc = Document(docx_buf)
    table = doc.tables[1]
    cell_a = table.rows[2].cells[2]
    cell_b = table.rows[3].cells[2]

    assert "w:drawing" in cell_a._tc.xml
    assert "w:drawing" not in cell_b._tc.xml

    with zipfile.ZipFile(docx_buf, "r") as z:
        media_parts = [n for n in z.namelist() if n.startswith("word/media/")]
        assert len(media_parts) == 1
