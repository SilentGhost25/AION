"""
ECE Departmental Regression Test
"""

from v0_1.vre import (
    CircuitSolver, FigureClassification, FigureExtractionResult, OperationChain,
    OperationStep, VKOC, VKOValidator
)


def test_ece_resistive_circuit_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="ECE", figure_class="CIRCUIT_RESISTIVE", operations=["EQUIVALENT_RESISTANCE"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, errors = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="EQUIVALENT_RESISTANCE", input_type="circuit", output_type="resistance")])
    sol = CircuitSolver.solve(vko, chain=chain)
    assert sol["operation"] == "EQUIVALENT_RESISTANCE"
    assert sol["r_equivalent"] > 0
    assert sol["total_current"] > 0
