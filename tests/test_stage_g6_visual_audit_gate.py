"""
Stage G6.1 Acceptance Suite: Visual Asset Correctness & Answer-Leakage Gate.

Covers all 12 critical scenarios, multi-distractor image selection,
independent two-mechanism verification (OCR + VLM verifier),
VisualAssetAudit contract validation, and pre-generation + post-DOCX-export double audit.
"""

import hashlib
from pathlib import Path
from PIL import Image, ImageDraw

import pytest

from core.contracts.document_artifact import FigureArtifact
from core.generation.agents.question_plan import VisualAssetAudit, VisualTaskContract, ExpectedResponseContract
from core.generation.agents.local_evaluators import audit_visual_asset, verify_exported_docx_visuals
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
    restored = VisualAssetAudit.from_dict(audit.to_dict())
    assert restored == audit

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
    assert audit.required_features_verified is True


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
    assert audit.question_alignment_verified is False


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


def test_04_circuit_with_final_answer_printed_fails(tmp_path):
    """4. Circuit with final answer printed -> FAIL."""
    fig = _base_circuit_fig(
        tmp_path,
        detected_text=["R1 = 10 ohm", "R2 = 5 ohm", "R_eq = 15 ohm"],
        detected_equations=["R_eq = 15"],
    )
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(fig, "Calculate the equivalent resistance.", expected_answer="15 ohm", visual_contract=vt)
    assert audit.decision == "FAIL"
    assert audit.answer_leakage == "CONFIRMED"


def test_05_circuit_with_worked_solution_beside_it_fails(tmp_path):
    """5. Circuit with worked solution beside it -> FAIL."""
    fig = _base_circuit_fig(
        tmp_path,
        detected_equations=["R = R1 + R2 = 15"],
        detected_annotations=["Worked solution"],
    )
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(fig, "Calculate the equivalent resistance.", visual_contract=vt)
    assert audit.decision == "FAIL"
    assert audit.answer_leakage == "CONFIRMED"


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
    assert audit.answer_leakage == "NONE"


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
    audit_calc = audit_visual_asset(fig, "Calculate the acceleration from the given graph.", expected_answer="2 m/s^2", visual_contract=vt_calc)
    assert audit_calc.decision == "FAIL"
    assert audit_calc.answer_leakage == "CONFIRMED"

    vt_read = VisualTaskContract("fig_graph_2", True, "read_output", ["velocity", "time"])
    audit_read = audit_visual_asset(fig, "Read the annotated acceleration from the given graph.", expected_answer="2 m/s^2", visual_contract=vt_read)
    assert audit_read.decision == "PASS"


def test_08_image_from_different_document_revision_fails(tmp_path):
    """8. Image from a different document revision -> FAIL."""
    fig = _base_circuit_fig(tmp_path, source_revision="rev_old_000")
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance.",
        registered_revision="rev_abc_123",
    )
    assert audit.decision == "FAIL"


def test_09_figure_with_unreadable_numerical_labels_indeterminate(tmp_path):
    """9. Figure with unreadable numerical labels -> INDETERMINATE (never promoted to PASS on valid hash)."""
    fig = _base_circuit_fig(tmp_path, detected_text=["Series Circuit Topology", "R1 = ???", "R2 = ???"])
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology"])
    audit = audit_visual_asset(fig, "Calculate the equivalent resistance.", visual_contract=vt)
    assert audit.decision == "INDETERMINATE"


def test_10_correct_image_with_missing_required_feature_fails(tmp_path):
    """10. Correct image with missing required feature -> FAIL."""
    fig = _base_circuit_fig(tmp_path, detected_text=["Series Circuit Topology"], detected_values=[])
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "inductance"])
    audit = audit_visual_asset(fig, "Calculate the impedance.", visual_contract=vt)
    assert audit.decision == "FAIL"
    assert audit.required_features_verified is False


def test_11_correct_image_with_ambiguous_handwritten_annotation_indeterminate(tmp_path):
    """11. Correct image with ambiguous handwritten annotation -> INDETERMINATE."""
    fig = _base_circuit_fig(tmp_path, detected_annotations=["ambiguous handwritten annotation near R2"])
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(fig, "Calculate the equivalent resistance.", visual_contract=vt)
    assert audit.decision == "INDETERMINATE"
    assert audit.answer_leakage == "POSSIBLE"


def test_12_correct_image_and_caption_but_wrong_module_provenance_fails(tmp_path):
    """12. Correct image and caption but wrong module provenance -> FAIL."""
    fig = _base_circuit_fig(tmp_path, module_id="module_4")
    audit = audit_visual_asset(
        fig,
        "Calculate the equivalent resistance.",
        expected_module_id="module_1",
    )
    assert audit.decision == "FAIL"


def test_plausible_incorrect_images_distractor_rejection(tmp_path):
    """Test the same question against several plausible but incorrect images from the same subject."""
    vt = VisualTaskContract("fig_target", True, "calculate", ["circuit topology", "resistor values"])
    question = "Calculate the equivalent resistance of the series circuit."

    valid = _base_circuit_fig(tmp_path, id="fig_target")
    wrong_topic = _base_circuit_fig(tmp_path, id="fig_d1", caption="Kirchhoff Voltage Law Mesh")
    leaked_ans = _base_circuit_fig(tmp_path, id="fig_d2", detected_equations=["Req = 15"])
    wrong_mod = _base_circuit_fig(tmp_path, id="fig_d3", module_id="module_5")

    assert audit_visual_asset(valid, question, visual_contract=vt, expected_module_id="module_1").decision == "PASS"
    assert audit_visual_asset(
        wrong_topic, question, visual_contract=vt, expected_module_id="module_1",
        vlm_verifier=lambda **kw: {"question_alignment_verified": False, "topic_module_match": False}
    ).decision == "FAIL"
    assert audit_visual_asset(leaked_ans, question, visual_contract=vt, expected_module_id="module_1").decision == "FAIL"
    assert audit_visual_asset(wrong_mod, question, visual_contract=vt, expected_module_id="module_1").decision == "FAIL"


def test_post_export_docx_manifest_verification(tmp_path):
    """Section 7: Double audit (pre-generation PASS + post-DOCX-export embedded image verification)."""
    p_good = tmp_path / "approved.png"
    sha_good = _make_png(p_good, "Approved Circuit Diagram")
    p_swapped = tmp_path / "swapped.png"
    sha_swapped = _make_png(p_swapped, "Wrong Swapped Diagram")

    fig = _base_circuit_fig(tmp_path, image_path=str(p_good), image_sha256=sha_good)
    vt = VisualTaskContract("fig_circuit_1", True, "calculate", ["circuit topology", "resistor values"])
    audit = audit_visual_asset(fig, "Calculate the equivalent resistance.", visual_contract=vt)
    assert audit.decision == "PASS"

    def _export(img_path: str, img_sha: str, out_name: str) -> Path:
        out_path = tmp_path / out_name
        paper_data = {
            "assessmentType": "IA", "exam_type": "IAT1", "subject": "Circuits", "subject_code": "BEE101",
            "max_marks": 10, "duration": "30 Minutes",
            "questions": [{
                "question_number": 1, "module_id": "module_1", "marks": 10, "co": "CO1", "bloom": "L3",
                "sub_questions": [{
                    "label": "(a)", "text": "Refer to the given diagram and calculate resistance.",
                    "marks": 10, "co": "CO1", "bloom": "L3",
                    "image_path": img_path, "image_sha256": img_sha, "figure_caption": "Circuit",
                }],
            }],
        }
        render_docx_document(paper_data, output_path=str(out_path))
        return out_path

    good_docx = _export(str(p_good), sha_good, "good.docx")
    ok, errs = verify_exported_docx_visuals(str(good_docx), [audit])
    assert ok is True and errs == []

    # If exporter attaches a swapped image, post-export audit MUST catch it
    swapped_docx = _export(str(p_swapped), sha_swapped, "swapped.docx")
    ok_bad, errs_bad = verify_exported_docx_visuals(str(swapped_docx), [audit])
    assert ok_bad is False
    assert any("missing from exported DOCX" in e for e in errs_bad)
