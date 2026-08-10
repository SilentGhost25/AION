"""
Adversarial Integration Test Suite for AION VRE v1.0 & Unified Pipeline.
Validates real-world failure modes, departmental questions (DSA, AVL, ECE),
corrupted extraction handling, and FinalPaper asset serialization.
"""

import pytest
from PIL import Image
from v0_1.contracts import ContractViolation
from v0_1.unified_pipeline import run_unified
from v0_1.vre import (
    CircuitSolver, FSC, FigureExtractionResult, FigureInput, GraphSolver,
    PaperVisualValidator, QPVDE, RenderMode, SemanticQuestionValidator,
    TreeSolver, VKOC, VKOValidator, VQGBuilder, VQGR, VREDecisionState,
    VREEngine, VRERequest, VisualCritic
)


@pytest.fixture
def dsa_graph_image(tmp_path) -> str:
    img_path = tmp_path / "dsa_weighted_graph.png"
    img = Image.new("RGB", (450, 300), color=(255, 255, 255))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def ece_circuit_image(tmp_path) -> str:
    img_path = tmp_path / "ece_resistor_circuit.png"
    img = Image.new("RGB", (450, 200), color=(255, 255, 255))
    img.save(img_path)
    return str(img_path)


@pytest.fixture
def avl_tree_image(tmp_path) -> str:
    img_path = tmp_path / "avl_tree_diagram.png"
    img = Image.new("RGB", (400, 250), color=(255, 255, 255))
    img.save(img_path)
    return str(img_path)


# ── Test 1: DSA Weighted Graph (Dijkstra) ─────────────────────────────────────

def test_adversarial_dsa_dijkstra(dsa_graph_image):
    request = VRERequest(
        request_id="adv_dsa_01",
        subject="Data Structures",
        department="CSE",
        module="module_3",
        topic="dijkstra_algorithm",
        bloom_level="L3",
        marks=7,
        figure_candidates=[FigureInput(image_path=dsa_graph_image, page_number=1, confidence=0.90)],
    )

    output = VREEngine.execute(request)

    assert output.success is True
    assert output.decision_state == VREDecisionState.IMAGE_NEEDED_AND_VALID
    assert output.reference_solution is not None
    assert output.reference_solution["operation"] == "DIJKSTRA"
    assert output.reference_solution["unique_solution"] is True

    # Assert question contains action verbs and no descriptive phrases
    text = output.text.lower()
    assert any(v in text for v in ["apply", "determine", "calculate", "find"])
    assert not any(p in text for p in ["describe the figure", "explain the diagram", "what is shown"])
    assert output.figure_svg is not None
    assert output.question_plan_hash != ""


# ── Test 2: AI / DSA AVL Tree Insertion & Rotation ───────────────────────────

def test_adversarial_avl_tree_insert(avl_tree_image):
    request = VRERequest(
        request_id="adv_avl_02",
        subject="Data Structures",
        department="CSE",
        module="module_2",
        topic="avl_rotations",
        bloom_level="L3",
        marks=8,
        figure_candidates=[FigureInput(image_path=avl_tree_image, page_number=1, confidence=0.92)],
    )

    output = VREEngine.execute(request)

    assert output.success is True
    assert output.decision_state == VREDecisionState.IMAGE_NEEDED_AND_VALID
    assert "avl" in output.text.lower() or "tree" in output.text.lower()
    # Must NOT be a descriptive question
    assert not any(p in output.text.lower() for p in ["describe the tree", "explain the picture"])
    assert output.figure_svg is not None


# ── Test 3: ECE Circuit Analysis (Equivalent Resistance) ──────────────────────

def test_adversarial_ece_circuit_analysis(ece_circuit_image):
    request = VRERequest(
        request_id="adv_ece_03",
        subject="Network Analysis",
        department="ECE",
        module="module_1",
        topic="kvl_equivalent_resistance",
        bloom_level="L3",
        marks=6,
        figure_candidates=[FigureInput(image_path=ece_circuit_image, page_number=1, confidence=0.88)],
    )

    output = VREEngine.execute(request)

    assert output.success is True
    assert output.decision_state == VREDecisionState.IMAGE_NEEDED_AND_VALID
    assert output.reference_solution["operation"] == "EQUIVALENT_RESISTANCE"
    assert output.reference_solution["r_equivalent"] > 0
    assert "resistance" in output.text.lower() or "circuit" in output.text.lower()
    assert output.figure_svg is not None


# ── Test 4: Corrupted & Low Word Count Input Guard ────────────────────────────

def test_adversarial_corrupted_input_rejection(tmp_path):
    corrupted_file = tmp_path / "corrupted_document.txt"
    corrupted_file.write_text("Short noise text.", encoding="utf-8")

    with pytest.raises((ContractViolation, RuntimeError)):
        run_unified(
            file_path=str(corrupted_file),
            exam_type="IA",
            difficulty="Mixed",
            subject="Data Structures",
            max_questions=4,
        )


# ── Test 5: Full FinalPaper JSON & Asset Serialization Integrity ─────────────

def test_adversarial_full_paper_export_integrity(tmp_path):
    paper_file = tmp_path / "valid_academic_paper.txt"
    paper_file.write_text(
        "Data Structures and Algorithms course content. "
        "Dijkstra's algorithm is used for finding single-source shortest paths in weighted graphs with non-negative weights. "
        "AVL trees are self-balancing binary search trees maintaining height balance factor between -1 and +1. "
        "Resistive circuit networks obey Kirchhoff's Voltage Law (KVL) to calculate node voltages and equivalent resistance. "
        "Simply supported structural beams undergo shear force and bending moment reactions under concentrated point loads.",
        encoding="utf-8",
    )

    final_paper = run_unified(
        file_path=str(paper_file),
        exam_type="IA",
        difficulty="Mixed",
        subject="Data Structures",
        max_questions=4,
    )

    assert final_paper.exportable is True
    assert final_paper.qa_score >= 40

    paper_dict = {
        "doc_id": final_paper.doc_id,
        "modules": final_paper.modules,
        "exam_type": final_paper.exam_type,
        "subject": final_paper.subject,
    }

    valid, errors = PaperVisualValidator.validate_paper(paper_dict)
    assert valid is True
    assert len(errors) == 0
