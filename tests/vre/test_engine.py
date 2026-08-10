"""
Integration tests for VREEngine orchestrator & fail-closed directives.
"""

import pytest
from pathlib import Path
from PIL import Image
from v0_1.vre.engine import VREEngine
from v0_1.vre.contracts import VRERequest, FigureInput, VREDecisionState


@pytest.fixture
def dummy_image_file(tmp_path) -> str:
    img_path = tmp_path / "test_graph.png"
    img = Image.new("RGB", (400, 300), color=(255, 255, 255))
    img.save(img_path)
    return str(img_path)


def test_vre_engine_clean_graph_execution(dummy_image_file):
    request = VRERequest(
        request_id="req_101",
        subject="Data Structures",
        department="CSE",
        module="Module 3",
        topic="dijkstra_algorithm",
        bloom_level="L3",
        marks=7,
        figure_candidates=[
            FigureInput(image_path=dummy_image_file, page_number=1, confidence=0.90)
        ],
    )

    output = VREEngine.execute(request)
    assert output.success is True
    assert output.decision_state == VREDecisionState.IMAGE_NEEDED_AND_VALID
    assert "Dijkstra" in output.text or "weighted graph" in output.text
    assert output.figure_svg is not None
    assert output.provenance is not None
    assert output.reference_solution is not None


def test_vre_engine_text_only_fallback_for_conceptual(dummy_image_file):
    request = VRERequest(
        request_id="req_102",
        subject="Data Structures",
        department="CSE",
        module="Module 1",
        topic="definition_questions",
        bloom_level="L1",
        marks=4,
        figure_candidates=[
            FigureInput(image_path=dummy_image_file, page_number=1, confidence=0.90)
        ],
    )

    output = VREEngine.execute(request)
    assert output.success is True
    assert output.decision_state == VREDecisionState.IMAGE_NOT_NEEDED
    assert output.figure_svg is None
    assert "not_needed" in output.reason.lower() or "strategy_text_only" in output.reason.lower()
