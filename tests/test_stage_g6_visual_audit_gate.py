"""
Stage G6.1 Acceptance Suite: Visual Integrity & Answer-Leakage Hardening.

Covers:
- All 12+ critical visual correctness & answer-leakage scenarios
- False-positive control (given input value equals numerical answer, e.g. R1 = 15 ohm)
- Equivalent numeric/unit formats & SymPy symbolic equivalence
- OCR unavailable & low-confidence OCR -> INDETERMINATE (never PASS)
- Stage A pre-selection candidate filtering & alternative candidate selection
- Stage B mandatory CheckingAgent enforcement (no audit bypass)
- Subpart (b) visual assignment & VisualTaskContract.prohibited_features compatibility
- Presentation-only caption normalization
- Row-level DOCX placement verification (wrong row, swapped, missing, unexpected, caption mismatch)
- Real Flask test client verification of POST /api/export/docx
"""

import hashlib
import io
from pathlib import Path
from PIL import Image, ImageDraw

import pytest

from core.contracts.document_artifact import DocumentArtifact, FigureArtifact, TextBlock
from core.contracts.paper_spec import PaperSpec
from core.extraction.module_segmenter import ModuleSegment, SegmentationStrategy
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_agent import CheckingAgent
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.local_evaluators import (
    VisualContentExtractor,
    audit_visual_asset,
    verify_exported_docx_visuals,
)
from core.generation.agents.planning_agent import PlanningAgent
from core.generation.agents.question_plan import (
    ExpectedResponseContract,
    QuestionPlan,
    SubPartPlan,
    VisualAssetAudit,
    VisualTaskContract,
)
from tests.test_stage_g1d_integration import MockKnowledgeGraph
from v0_1.docx_export import render_docx_document


def _make_png(path: Path, label: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 180), color=(248, 250, 252))
    draw = ImageDraw.Draw(img)
    draw.rectangle([4, 4, 395, 175], outline=(20, 50, 100), width=2)
    draw.text((16, 20), label, fill=(10, 20, 30))
    img.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_circuit_fig(tmp_path: Path, **overrides) -> dict:
    p = tmp_path / "circuit_orig.png"
    sha = _make_png(p, "R1 = 10 ohm, R2 = 5 ohm, Series Circuit")
    base = {
        "id": "fig_circuit_1",
        "source_document_id": "ee_circuits.pdf",
        "source_revision": "rev_abc_123",
        "image_sha256": sha,
        "image_path": str(p),
        "module_id": "module_1",
        "page": 4,
        "caption": "Series circuit topology with resistor values R1 and R2",
        "detected_text": ["Series Circuit Topology", "R1 = 10 ohm", "R2 = 5 ohm", "Connection: Series"],
        "detected_values": ["10 ohm", "5 ohm"],
        "detected_equations": [],
        "detected_annotations": [],
    }
    base.update(overrides)
    return base


# =============================================================================
# 1. Contract & Serialization Compatibility
# =============================================================================


def test_visual_task_contract_prohibited_features_and_compat():
    """VisualTaskContract preserves 4-arg positional compat and supports prohibited_features."""
    vt_legacy = VisualTaskContract("fig_1", True, "calculate", ["circuit_topology"])
    assert vt_legacy.prohibited_features == ()
    assert "prohibited_features" not in vt_legacy.to_dict()
    assert VisualTaskContract.from_dict(vt_legacy.to_dict()) == vt_legacy

    vt_full = VisualTaskContract(
        "fig_1", True, "calculate",
        ("circuit_topology", "resistor_values"),
        ("final_equivalent_resistance", "worked_calculation"),
    )
    assert vt_full.prohibited_features == ("final_equivalent_resistance", "worked_calculation")
    restored = VisualTaskContract.from_dict(vt_full.to_dict())
    assert restored == vt_full

    with pytest.raises(TypeError):
        VisualTaskContract("fig_1", True, "calculate", ["a"], prohibited_features="not_a_seq")  # type: ignore


def test_visual_asset_audit_contract_strict_validation():
    """VisualAssetAudit validates field types, leakage states, and decision enum."""
    audit = VisualAssetAudit(
        figure_id="fig_1",
        source_document_id="doc.pdf",
        source_revision="rev1",
        image_sha256="abc123",
        module_id="module_1",
        page_number=2,
        required_features_verified=True,
        question_alignment_verified=True,
        answer_leakage="NONE",
        decision="PASS",
    )
    assert VisualAssetAudit.from_dict(audit.to_dict()) == audit

    with pytest.raises(ValueError):
        VisualAssetAudit(
            figure_id="fig_1", source_document_id="doc.pdf", source_revision="rev1",
            image_sha256="abc123", module_id="module_1", page_number=2,
            required_features_verified=True, question_alignment_verified=True,
            answer_leakage="INVALID", decision="PASS",
        )
    with pytest.raises(TypeError):
        VisualAssetAudit(
            figure_id="fig_1", source_document_id="doc.pdf", source_revision="rev1",
            image_sha256="abc123", module_id="module_1", page_number=True,  # bool rejected
            required_features_verified=True, question_alignment_verified=True,
        )


# =============================================================================
# 2. Core Visual Correctness, Leakage & False-Positive Control Tests
# =============================================================================


def test_01_correct_circuit_with_original_values_passes(tmp_path):
    """1. Correct circuit with original values -> PASS."""
    fig = _base_circuit_fig(tmp_path)
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values", "series"])
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance of the given circuit.",
        expected_answer="15 ohm",
        visual_contract=vt,
        registered_document_id="ee_circuits.pdf",
        registered_revision="rev_abc_123",
        expected_module_id="module_1",
        module_page_range=(1, 10),
        vlm_verifier=lambda **kw: {
            "required_features_verified": True,
            "question_alignment_verified": True,
            "topic_module_match": True,
            "completed_solution_detected": False,
            "answer_leakage": "NONE",
        },
    )
    assert audit.decision == "PASS"
    assert audit.answer_leakage == "NONE"


def test_01b_false_positive_control_input_equals_answer_passes(tmp_path):
    """False-positive control: Given input R1 = 15 ohm equals expected answer 15 ohm -> PASS (not a leak)."""
    fig = _base_circuit_fig(
        tmp_path,
        detected_text=["Series Circuit Topology", "R1 = 15 ohm", "R2 = 0 ohm"],
        detected_values=["15 ohm", "0 ohm"],
        detected_equations=["R1 = 15 ohm", "R2 = 0 ohm"],
    )
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance of the given circuit.",
        expected_answer="15 ohm",
        visual_contract=vt,
    )
    assert audit.decision == "PASS"
    assert audit.answer_leakage == "NONE"


def test_02_circuit_from_different_syllabus_topic_fails(tmp_path):
    """2. Circuit from a different syllabus topic -> FAIL."""
    fig = _base_circuit_fig(tmp_path, caption="Op-Amp Inverting Amplifier")
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance of the given circuit.",
        visual_contract=vt,
        vlm_verifier=lambda **kw: {
            "required_features_verified": True,
            "question_alignment_verified": False,
            "topic_module_match": False,
            "answer_leakage": "NONE",
        },
    )
    assert audit.decision == "FAIL"


def test_03_correct_topic_but_wrong_circuit_configuration_fails(tmp_path):
    """3. Correct topic but wrong circuit configuration -> FAIL."""
    fig = _base_circuit_fig(tmp_path, caption="Parallel Resistor Network")
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance of the series circuit.",
        visual_contract=vt,
        vlm_verifier=lambda **kw: {
            "required_features_verified": True,
            "question_alignment_verified": False,
            "topic_module_match": True,
            "answer_leakage": "NONE",
        },
    )
    assert audit.decision == "FAIL"


def test_04_numeric_unit_and_symbolic_equivalence_leakage_fails(tmp_path):
    """4. Equivalent numeric/unit formats ('15.0 ohm', '0.015 kohm') and SymPy symbolic equivalence ('3*5') -> FAIL."""
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])

    # Numeric/unit format equivalence
    fig_unit = _base_circuit_fig(tmp_path, detected_equations=["Equivalent resistance = 0.015 kohm"])
    assert audit_visual_asset(fig_unit, "Calculate equivalent resistance.", expected_answer="15 ohm", visual_contract=vt).decision == "FAIL"

    # Symbolic equivalence via SymPy (2*(x+2) vs 2*x + 4)
    fig_sym = _base_circuit_fig(tmp_path, detected_equations=["Result = 2*(x + 2)"])
    assert audit_visual_asset(fig_sym, "Simplify the impedance expression.", expected_answer="2*x + 4", visual_contract=vt).decision == "FAIL"


def test_05_worked_calculation_and_mcq_highlight_and_prohibited_feature_fails(tmp_path):
    """5. Worked calculation, highlighted MCQ option, and prohibited_features -> FAIL."""
    vt = VisualTaskContract(
        "fig_circuit_1", True, "calculate",
        ["circuit topology", "resistor values"],
        prohibited_features=("worked_calculation", "answer_annotation"),
    )

    fig_worked = _base_circuit_fig(tmp_path, detected_equations=["R = R1 + R2 = 15"])
    assert audit_visual_asset(fig_worked, "Calculate equivalent resistance.", visual_contract=vt).decision == "FAIL"

    fig_mcq = _base_circuit_fig(tmp_path, detected_annotations=["[x] Option B"])
    assert audit_visual_asset(fig_mcq, "Identify the correct topology.", visual_contract=vt).decision == "FAIL"


def test_06_graph_with_required_input_data_passes(tmp_path):
    """6. Graph with its required input data -> PASS."""
    p = tmp_path / "graph_input.png"
    sha = _make_png(p, "Velocity vs Time Graph (t=0..10s)")
    fig = {
        "id": "fig_graph_1",
        "source_document_id": "physics.pdf",
        "source_revision": "rev1",
        "image_sha256": sha,
        "image_path": str(p),
        "module_id": "module_2",
        "page": 12,
        "caption": "Velocity-Time graph with axis labels and input data",
        "detected_text": ["Velocity (m/s)", "Time (s)", "v(0)=0", "v(10)=20"],
        "detected_values": ["0", "10", "20"],
    }
    vt = VisualTaskContract("fig_graph_1", True, "calculate", ["velocity", "time"])
    audit = audit_visual_asset(fig, "Calculate the acceleration from the given graph.", expected_answer="2 m/s^2", visual_contract=vt)
    assert audit.decision == "PASS"


def test_07_graph_with_requested_output_pre_annotated_fails_unless_read_task(tmp_path):
    """7. Graph with requested output pre-annotated -> FAIL unless reading output is the task."""
    p = tmp_path / "graph_annot.png"
    sha = _make_png(p, "Velocity-Time Graph, Acceleration = 2 m/s^2")
    fig = {
        "id": "fig_graph_2",
        "source_document_id": "physics.pdf",
        "source_revision": "rev1",
        "image_sha256": sha,
        "image_path": str(p),
        "module_id": "module_2",
        "page": 12,
        "caption": "Velocity-Time graph",
        "detected_text": ["Velocity (m/s)", "Time (s)", "Pre-annotated output: 2 m/s^2"],
    }
    vt_calc = VisualTaskContract("fig_graph_2", True, "calculate", ["velocity", "time"])
    assert audit_visual_asset(fig, "Calculate acceleration.", expected_answer="2 m/s^2", visual_contract=vt_calc).decision == "FAIL"

    vt_read = VisualTaskContract("fig_graph_2", True, "read_output", ["velocity", "time"])
    assert audit_visual_asset(fig, "Read the annotated acceleration.", expected_answer="2 m/s^2", visual_contract=vt_read).decision == "PASS"


def test_08_corrupted_image_hash_mismatch_wrong_revision_wrong_module_fail(tmp_path):
    """8. Image corruption, hash mismatch, wrong revision, and wrong module provenance -> FAIL."""
    # Corrupted / undecodable image bytes
    corrupt_p = tmp_path / "corrupt.png"
    corrupt_p.write_bytes(b"NOT_A_VALID_PNG_STREAM")
    corrupt_sha = hashlib.sha256(corrupt_p.read_bytes()).hexdigest()
    fig_corrupt = _base_circuit_fig(tmp_path, image_path=str(corrupt_p), image_sha256=corrupt_sha)
    assert audit_visual_asset(fig_corrupt, "Calculate resistance.").decision == "FAIL"

    # Hash mismatch
    fig_bad_sha = _base_circuit_fig(tmp_path, image_sha256="deadbeef" * 8)
    assert audit_visual_asset(fig_bad_sha, "Calculate resistance.").decision == "FAIL"

    # Wrong revision & wrong module
    fig_rev = _base_circuit_fig(tmp_path, source_revision="rev_old")
    assert audit_visual_asset(fig_rev, "Calculate resistance.", registered_revision="rev_abc_123").decision == "FAIL"
    fig_mod = _base_circuit_fig(tmp_path, module_id="module_4")
    assert audit_visual_asset(fig_mod, "Calculate resistance.", expected_module_id="module_1").decision == "FAIL"


def test_09_ocr_unavailable_low_confidence_and_unreadable_labels_indeterminate(tmp_path):
    """9. OCR unavailable, low-confidence OCR, and unreadable labels -> INDETERMINATE (never PASS)."""
    p = tmp_path / "raw_circuit.png"
    sha = _make_png(p, "Raw Circuit")
    raw_fig = {
        "id": "fig_raw",
        "source_document_id": "ee.pdf",
        "source_revision": "rev1",
        "image_sha256": sha,
        "image_path": str(p),
        "module_id": "module_1",
        "page": 2,
        "caption": "Circuit",
    }

    # OCR unavailable
    audit_unavail = audit_visual_asset(
        raw_fig,
        "Calculate resistance.",
        ocr_extractor=lambda _p: {"available": False, "error": "ocr_backend_unavailable", "confidence": 0.0},
    )
    assert audit_unavail.decision == "INDETERMINATE"

    # Low-confidence OCR
    audit_low = audit_visual_asset(
        raw_fig,
        "Calculate resistance.",
        ocr_extractor=lambda _p: {"available": True, "text_regions": ["R1=10"], "confidence": 0.25},
    )
    assert audit_low.decision == "INDETERMINATE"

    # Unreadable labels
    fig_unreadable = _base_circuit_fig(tmp_path, detected_text=["Series Circuit", "R1 = ???"])
    assert audit_visual_asset(fig_unreadable, "Calculate resistance.").decision == "INDETERMINATE"


# =============================================================================
# 3. Stage A Selection, Subpart (b) Eligibility, Caption Norm & Stage B Gate
# =============================================================================


def test_stage_a_skips_leaky_candidate_and_selects_clean_alternative(tmp_path):
    """Stage A screening rejects leaky first candidate and selects clean second candidate."""
    p_bad = tmp_path / "bad_leak.png"
    sha_bad = _make_png(p_bad, "Series Circuit Ans: 15")
    p_good = tmp_path / "good_clean.png"
    sha_good = _make_png(p_good, "Series Circuit R1=10 R2=5")

    fig_bad = {
        "id": "fig_a_leaky",
        "page": 2,
        "caption": "Series Circuit Topology with Resistor Values",
        "image_path": str(p_bad),
        "image_sha256": sha_bad,
        "provenance_score": 0.99,  # higher overlap/score, but leaks answer!
        "module_id": "module_1",
        "detected_text": ["Series Circuit Topology", "R1=10", "Ans: 15"],
    }
    fig_good = {
        "id": "fig_b_clean",
        "page": 3,
        "caption": "Series Circuit Topology",
        "image_path": str(p_good),
        "image_sha256": sha_good,
        "provenance_score": 0.90,
        "module_id": "module_1",
        "detected_text": ["circuit_topology", "resistor_values", "connection_types", "R1=10", "R2=5"],
    }

    planner = PlanningAgent()
    ok, selected = planner._resolve_visual(
        bloom="L3",
        module_figures=[fig_bad, fig_good],
        used_figures=set(),
        visual_slots_used=0,
        topic="Series Circuit Topology",
        expected_module_id="module_1",
        strict_stage_a=True,
    )
    assert ok is True
    assert selected["id"] == "fig_b_clean"


def test_subpart_b_visual_assignment_and_caption_normalization(tmp_path):
    """Subpart (b) can receive a visual when (a) is text-only; display caption strips 'Figure X.Y:'."""
    p_fig = tmp_path / "bplus.png"
    sha = _make_png(p_fig, "B+ Tree Root Nodes and Leaves")
    fig = FigureArtifact(
        id="fig_bplus",
        page=3,
        caption="Figure 5.1: B+ Tree Index Structure",
        image_path=str(p_fig),
        image_sha256=sha,
        source_pdf_sha256="rev_sub_b",
        provenance_score=0.95,
        module_id="module_1",
    )
    assert PlanningAgent.normalize_display_caption(fig.caption) == "B+ Tree Index Structure"

    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10)
    blocks = [
        TextBlock(id="b1", text="SQL normalization removes transitive dependencies.", page=2, block_role="BODY"),
        TextBlock(id="b2", text="B+ Tree balanced index maintains linked leaf nodes.", page=4, block_role="BODY"),
    ]
    artifact = DocumentArtifact(source_pdf_sha256="rev_sub_b", source_pdf_path="dbms.pdf", text_blocks=blocks, figures=[fig])
    seg = ModuleSegment(module_id=1, title="Module 1", page_start=1, page_end=5, confidence=0.95, strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING)
    concepts = {"module_1": ["SQL Normalization", "B+ Tree Index Structure"]}
    ctx = AgentContext(
        paper_spec=spec,
        artifact=artifact,
        knowledge_graph=MockKnowledgeGraph(concepts_by_module=concepts),
        request={"scheme_id": "VTU_2022_NEP", "course_type": "AEC_GENERAL_THEORY", "assessment_type": "CIE_IAT", "blueprint_id": "VTU_AEC_ANNEXURE_II"},
    )
    ctx.module_segments = [seg]
    res = PlanningAgent().run(ctx)
    assert res.success is True
    sps = res.payload[0].sub_part_plans
    assert sps[0].visual_required is False
    assert sps[1].visual_required is True
    assert sps[1].figure.id == "fig_bplus"


def test_checking_agent_mandatory_stage_b_no_audit_bypass(tmp_path):
    """Stage B CheckingAgent never passes a visual subpart without a valid completed audit."""
    from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry

    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH", course_type="SYNTHETIC_TEST", assessment_type="CIE_IAT",
        blueprint_id="BP_10M", total_marks_per_question=10, permitted_partitions=((10,),),
        max_subparts=1, reference_source="Test Stage B",
    ))

    sp = SubPart(
        subpart_index=0, label="a", text="With reference to the given diagram, explain the topology.",
        marks=10, bloom="L2", co="CO1",
        figure_id="fig_missing_audit",
        image_path=str(tmp_path / "nonexistent.png"),
        visual_task=VisualTaskContract("fig_missing_audit", True, "explain", ["topology"]),
    )
    q = GeneratedQuestion(
        question_id="Q1", slot_id="module_1_Q1",
        question_text="(a) With reference to the given diagram, explain the topology.",
        marks=10, bloom="L2", co="CO1", module_id="module_1",
        solution="Detailed topology explanation.",
        marking_scheme=[{"criterion": "Topology", "marks": 10}],
        sub_parts=[sp],
        figure_id="fig_missing_audit",
        image_path=str(tmp_path / "nonexistent.png"),
    )
    plan = QuestionPlan(
        slot_id="module_1_Q1", module_id="module_1", module_idx=1, slot_idx=0, global_q_idx=1,
        partition=[10], total_marks=10, co="CO1", bloom="L2", topic="Topology",
        sub_part_plans=[SubPartPlan(
            label="a", marks=10, topic="Topology", bloom="L2", co="CO1",
            visual_required=True,
            figure={"id": "fig_missing_audit", "image_path": str(tmp_path / "nonexistent.png")},
        )],
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
        status="PLANNED",
    )
    checker = CheckingAgent()
    ctx = AgentContext(paper_spec=PaperSpec("IAT1", 1, 1, 10), plans=[plan], questions=[q])
    c_res = checker.run(ctx)
    # Must NOT be PASS when image file is missing / unverified
    assert c_res.payload.status == "BLOCKED"
    assert any("MISSING_VISUAL_ASSET" in str(r) for r in c_res.payload.blocked_reasons)


# =============================================================================
# 4. Row-Level DOCX Placement Verification & /api/export/docx Route Test
# =============================================================================


def test_docx_row_level_placement_swap_missing_unexpected_and_caption(tmp_path):
    """DOCX verifier checks actual row placement, swapped images, missing, unexpected, and captions."""
    p1 = tmp_path / "fig_q1a.png"
    sha1 = _make_png(p1, "Circuit Q1a")
    p2 = tmp_path / "fig_wrong.png"
    sha2 = _make_png(p2, "Wrong Circuit")

    def _render_paper(img_q1a, sha_q1a, cap_q1a, img_q1b=None, sha_q1b=None, name="paper.docx"):
        out = tmp_path / name
        sub_qs = [
            {
                "label": "(a)", "text": "With reference to the given diagram, calculate resistance.",
                "marks": 5, "co": "CO1", "bloom": "L3",
                "image_path": img_q1a, "image_sha256": sha_q1a, "figure_caption": cap_q1a,
            },
            {
                "label": "(b)", "text": "Explain Ohm's law.",
                "marks": 5, "co": "CO1", "bloom": "L2",
                **({"image_path": img_q1b, "image_sha256": sha_q1b, "figure_caption": "Unexpected"} if img_q1b else {}),
            },
        ]
        render_docx_document({
            "assessmentType": "IA", "exam_type": "IAT1", "subject": "Circuits", "subject_code": "BEE101",
            "max_marks": 10, "duration": "30 Minutes",
            "questions": [{"question_number": 1, "module_id": "module_1", "marks": 10, "co": "CO1", "bloom": "L3", "sub_questions": sub_qs}],
        }, output_path=str(out))
        return out

    manifest = [{
        "question_id": "Q1", "subpart_id": "Q1_a", "figure_id": "fig_q1a",
        "image_sha256": sha1, "expected_caption": "Series Circuit", "approved": True, "decision": "PASS",
    }]

    # 1. Valid placement & caption -> PASS
    good_doc = _render_paper(str(p1), sha1, "Series Circuit", name="good.docx")
    ok, errs = verify_exported_docx_visuals(str(good_doc), manifest)
    assert ok is True and errs == []

    # 2. Swapped image in Q1_a -> FAIL
    swapped_doc = _render_paper(str(p2), sha2, "Series Circuit", name="swapped.docx")
    ok_sw, errs_sw = verify_exported_docx_visuals(str(swapped_doc), manifest)
    assert ok_sw is False and any("Swapped or wrong image" in e for e in errs_sw)

    # 3. Missing image in Q1_a (embedded in Q1_b instead -> wrong question/subpart!) -> FAIL
    misplaced_doc = _render_paper(None, None, "", img_q1b=str(p1), sha_q1b=sha1, name="misplaced.docx")
    ok_mp, errs_mp = verify_exported_docx_visuals(str(misplaced_doc), manifest)
    assert ok_mp is False and any("Missing approved image in row Q1_a" in e for e in errs_mp)

    # 4. Unexpected extra image in Q1_b -> FAIL
    extra_doc = _render_paper(str(p1), sha1, "Series Circuit", img_q1b=str(p2), sha_q1b=sha2, name="extra.docx")
    ok_ex, errs_ex = verify_exported_docx_visuals(str(extra_doc), manifest)
    assert ok_ex is False and any("Unexpected/unapproved image embedded in row Q1_b" in e for e in errs_ex)

    # 5. Caption mismatch -> FAIL
    cap_doc = _render_paper(str(p1), sha1, "Completely Wrong Caption", name="badcap.docx")
    ok_cp, errs_cp = verify_exported_docx_visuals(str(cap_doc), manifest)
    assert ok_cp is False and any("Caption mismatch" in e for e in errs_cp)


def test_api_export_docx_route_enforces_visual_manifest(tmp_path, monkeypatch):
    """Exercise actual POST /api/export/docx route: approves clean visual, blocks leaky/unverified visual."""
    from types import SimpleNamespace
    import aion_api
    from core.validation.export_gate import ExportGate
    from core.validation.paper_export_registry import PaperExportRegistry, canonicalize_paper_payload

    p_clean = tmp_path / "api_clean.png"
    sha_clean = _make_png(p_clean, "R1=10 ohm, R2=5 ohm, Series")
    p_leaky = tmp_path / "api_leaky.png"
    sha_leaky = _make_png(p_leaky, "Series Circuit Ans: 15 ohm")

    active_payload = {}
    monkeypatch.setattr(
        PaperExportRegistry,
        "get",
        staticmethod(lambda pid: (SimpleNamespace(
            export_eligible=True,
            blocked_reasons=[],
            canonical_hash=canonicalize_paper_payload(active_payload["body"]),
        ), "OK")),
    )
    monkeypatch.setattr(ExportGate, "evaluate_dict", staticmethod(lambda _b: SimpleNamespace(passed=True, message="OK", failures=[])))

    client = aion_api.app.test_client()

    # 1. Valid visual paper with clean OCR payload -> HTTP 200 & valid DOCX bytes
    q_clean = {
        "question_number": 1, "module_id": "module_1", "marks": 10, "co": "CO1", "bloom": "L3",
        "sub_questions": [{
            "label": "(a)",
            "text": "With reference to the given diagram, calculate the equivalent resistance.",
            "solution": "R = 15 ohm",
            "marks": 10, "co": "CO1", "bloom": "L3",
            "figure_id": "fig_api_1",
            "image_path": str(p_clean),
            "image_sha256": sha_clean,
            "figure_caption": "Series Circuit",
            "ocr_payload": {"available": True, "text_regions": ["R1 = 10 ohm", "R2 = 5 ohm"], "confidence": 0.95},
        }],
    }
    valid_payload = {
        "paper_id": "paper_vis_ok_001",
        "assessmentType": "IA", "exam_type": "IAT1", "subject": "Circuits", "subject_code": "BEE101",
        "max_marks": 10, "duration": "30 Minutes",
        "modules": [{"module_id": "module_1", "title": "Module 1: DC Circuits", "questions": [q_clean]}],
        "questions": [q_clean],
    }
    active_payload["body"] = valid_payload
    resp_ok = client.post("/api/export/docx", json=valid_payload)
    assert resp_ok.status_code == 200, f"Unexpected {resp_ok.status_code}: {resp_ok.get_json()}"
    ok_bytes, errs_bytes = verify_exported_docx_visuals(
        io.BytesIO(resp_ok.data),
        [{
            "question_id": "Q1", "subpart_id": "Q1_a", "figure_id": "fig_api_1",
            "image_sha256": sha_clean, "expected_caption": "Series Circuit", "approved": True, "decision": "PASS",
        }],
    )
    assert ok_bytes is True and errs_bytes == []

    # 2. Leaky visual (even if client tries to send 'approved: True') -> HTTP 422 blocked!
    q_leaky = {
        "question_number": 1, "module_id": "module_1", "marks": 10, "co": "CO1", "bloom": "L3",
        "sub_questions": [{
            "label": "(a)",
            "text": "With reference to the given diagram, calculate the equivalent resistance.",
            "solution": "15 ohm",
            "marks": 10, "co": "CO1", "bloom": "L3",
            "figure_id": "fig_api_bad",
            "image_path": str(p_leaky),
            "image_sha256": sha_leaky,
            "figure_caption": "Series Circuit",
            "approved": True,  # Untrusted client flag must be ignored!
            "ocr_payload": {"available": True, "text_regions": ["Series Circuit", "Ans: 15 ohm"], "confidence": 0.95},
        }],
    }
    leaky_payload = {
        "paper_id": "paper_vis_bad_001",
        "assessmentType": "IA", "exam_type": "IAT1", "subject": "Circuits", "subject_code": "BEE101",
        "max_marks": 10, "duration": "30 Minutes",
        "modules": [{"module_id": "module_1", "title": "Module 1: DC Circuits", "questions": [q_leaky]}],
        "questions": [q_leaky],
    }
    active_payload["body"] = leaky_payload
    resp_bad = client.post("/api/export/docx", json=leaky_payload)
    assert resp_bad.status_code == 422
    assert "visual manifest verification failed" in resp_bad.get_json()["error"].lower()
