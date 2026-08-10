"""
EEE Electrical Engineering Regression Test
"""

from v0_1.vre import (
    CircuitSolver, FigureClassification, FigureExtractionResult, OperationChain,
    OperationStep, VKOC, VKOValidator
)


def test_eee_kvl_circuit_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="EEE", figure_class="CIRCUIT_RESISTIVE", operations=["KVL"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, _ = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="KVL", input_type="circuit", output_type="kvl_eq")])
    sol = CircuitSolver.solve(vko, chain=chain)
    assert sol["operation"] in ("KVL", "EQUIVALENT_RESISTANCE")
    assert sol["v_source"] > 0
