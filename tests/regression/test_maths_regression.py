"""
Engineering Mathematics Regression Test
"""

from v0_1.vre import (
    FigureClassification, FigureExtractionResult, GraphSolver, OperationChain,
    OperationStep, VKOC, VKOValidator
)


def test_maths_graph_mst_invariant():
    ext = FigureExtractionResult(status="SUCCESS", image_path="fake.png", page_number=1, bbox=(0, 0, 100, 100), confidence=0.9, extraction_method="OCR")
    cls = FigureClassification(domain="DSA", figure_class="WEIGHTED_GRAPH", operations=["PRIM"], confidence=0.9)
    vko = VKOC.build(ext, cls)

    valid, _ = VKOValidator.validate(vko)
    assert valid is True

    chain = OperationChain(chain_id="c1", bloom_level="L3", steps=[OperationStep(step_number=1, operation="PRIM", input_type="graph", output_type="mst")])
    sol = GraphSolver.solve(vko, chain=chain)
    assert sol["operation"] in ("MST", "PRIM", "DIJKSTRA")
    assert sol.get("total_cost", 0) > 0 or sol.get("total_mst_weight", 0) > 0 or "mst_edges" in sol
