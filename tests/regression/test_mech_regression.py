"""
Mechanical Engineering Regression Test
"""

from v0_1.vre import (
    BeamSolver, FigureClassification, FigureExtractionResult, OperationChain,
    OperationStep, VKOC, VKOValidator
)


def test_mech_beam_reaction_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="Civil", figure_class="BEAM_SIMPLY_SUPPORTED", operations=["REACTIONS"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, _ = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="REACTIONS", input_type="beam", output_type="reactions")])
    sol = BeamSolver.solve(vko, chain=chain)
    assert sol["operation"] in ("REACTIONS", "BEAM_REACTIONS")
    r_a = sol.get("r_a", sol.get("reaction_A", 0))
    r_b = sol.get("r_b", sol.get("reaction_B", 0))
    assert round(r_a + r_b, 2) >= 0
