"""
AI / Machine Learning Departmental Regression Test
"""

from v0_1.vre import (
    FigureClassification, FigureExtractionResult, OperationChain, OperationStep,
    TreeSolver, VKOC, VKOValidator
)


def test_ai_decision_tree_entropy_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="AI", figure_class="BINARY_SEARCH_TREE", operations=["HEIGHT"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, _ = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="HEIGHT", input_type="tree", output_type="height")])
    sol = TreeSolver.solve(vko, chain=chain)
    assert sol["operation"] in ("TREE_HEIGHT", "AVL_INSERT_ROTATE")
