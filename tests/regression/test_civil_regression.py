"""
Civil Engineering Regression Test
"""

from v0_1.vre import (
    BeamSolver, FigureClassification, FigureExtractionResult, OperationChain,
    OperationStep, VKOC, VKOValidator
)


def test_civil_sfd_bmd_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="Civil", figure_class="BEAM_SIMPLY_SUPPORTED", operations=["SFD_BMD"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, _ = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="SFD_BMD", input_type="beam", output_type="sfd_bmd")])
    sol = BeamSolver.solve(vko, chain=chain)
    assert sol["operation"] in ("SFD_BMD", "BEAM_SFD_BMD", "REACTIONS")
