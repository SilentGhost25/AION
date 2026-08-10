"""
DSA Departmental Regression Test
"""

from v0_1.vre import (
    FigureClassification, FigureExtractionResult, GraphSolver, OperationChain,
    OperationStep, TreeSolver, VKO, VKOValidator, VKOC
)


def make_vko(domain: str, figure_class: str, ops: list[str]) -> VKO:
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain=domain, figure_class=figure_class, operations=ops, confidence=0.9)
    return VKOC.build(ext, cls)


def test_dsa_dijkstra_solver_invariant():
    vko = make_vko("DSA", "WEIGHTED_GRAPH", ["DIJKSTRA"])
    valid, errors = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="DIJKSTRA", input_type="graph", output_type="path")])
    solution = GraphSolver.solve(vko, chain=chain)
    assert solution["operation"] == "DIJKSTRA"
    assert solution["total_cost"] > 0
    assert solution["unique_solution"] is True


def test_dsa_avl_tree_solver_invariant():
    vko = make_vko("DSA", "AVL_TREE", ["INSERT_ROTATE"])
    valid, errors = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c2", bloom_level="L3", steps=[OperationStep(step_number=1, operation="INSERT_ROTATE", input_type="tree", output_type="tree")])
    solution = TreeSolver.solve(vko, chain=chain)
    assert solution["operation"] == "AVL_INSERT_ROTATE"
    assert solution["required_rotation"] in ("LL", "RR", "LR", "RL")
