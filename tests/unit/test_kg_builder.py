"""Unit tests for KnowledgeGraph, ModuleGraph, and KnowledgeGraphBuilder."""

from dataclasses import dataclass

import pytest

from core.knowledge import (
    KnowledgeGraph,
    KnowledgeGraphBuilder,
    ModuleGraph,
    Triple,
)


# -----------------------------------------------------------------------------
# Test fixtures
# -----------------------------------------------------------------------------


@dataclass
class FakeBlock:
    id: str
    text: str
    page: int
    block_role: str = "BODY"


def make_triple(s: str, p: str, o: str, page: int = 1, confidence: float = 0.8) -> Triple:
    return Triple(
        subject=s,
        predicate=p,
        object=o,
        source_block_id=f"block_{page}",
        source_page=page,
        confidence=confidence,
    )


# -----------------------------------------------------------------------------
# ModuleGraph — add_triple, PageRank, neighborhoods
# -----------------------------------------------------------------------------


def test_module_graph_initial_state():
    g = ModuleGraph("module_1")
    assert g.triple_count() == 0
    assert g.node_count() == 0
    assert g.top_concepts(10) == []


def test_add_triple_populates_adjacency():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("Docker", "is_a", "platform"))
    assert g.triple_count() == 1
    # 2 nodes, both reachable
    assert g.node_count() == 2


def test_pagerank_weights_sum_to_one():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    g.add_triple(make_triple("B", "is_a", "C"))
    g.add_triple(make_triple("C", "is_a", "D"))
    result = g.compute_pagerank()
    total = sum(result.weights.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    assert result.converged is True


def test_pagerank_empty_graph():
    g = ModuleGraph("module_1")
    result = g.compute_pagerank()
    assert result.weights == {}
    assert result.converged is True
    assert result.iterations == 0


def test_pagerank_centrality_ranks_high_degree_nodes_first():
    g = ModuleGraph("module_1")
    # Star topology centered on "Central"
    g.add_triple(make_triple("Central", "is_a", "A"))
    g.add_triple(make_triple("Central", "is_a", "B"))
    g.add_triple(make_triple("Central", "is_a", "C"))
    g.add_triple(make_triple("Central", "is_a", "D"))
    g.compute_pagerank()
    top = g.top_concepts(1)
    assert top[0][0] == "Central"


def test_neighborhood_depth_1():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    g.add_triple(make_triple("B", "is_a", "C"))
    neighborhood = g.get_neighborhood("A", max_depth=1)
    assert len(neighborhood) == 1
    assert neighborhood[0].subject == "A"


def test_neighborhood_depth_2():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    g.add_triple(make_triple("B", "is_a", "C"))
    g.add_triple(make_triple("C", "is_a", "D"))
    neighborhood = g.get_neighborhood("A", max_depth=2)
    predicates = {t.subject for t in neighborhood}
    assert "A" in predicates
    assert "B" in predicates
    # Depth 2 from A reaches the A-B and B-C triples
    assert len(neighborhood) == 2


def test_neighborhood_depth_zero_returns_empty():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    assert g.get_neighborhood("A", max_depth=0) == []


def test_neighborhood_unknown_concept():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    assert g.get_neighborhood("Z", max_depth=2) == []


def test_neighborhood_confidence_filter():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B", confidence=0.4))
    g.add_triple(make_triple("A", "is_a", "C", confidence=0.9))
    neighborhood = g.get_neighborhood("A", max_depth=1, min_confidence=0.5)
    assert len(neighborhood) == 1
    assert neighborhood[0].object == "C"


def test_pagerank_invalidated_on_add():
    g = ModuleGraph("module_1")
    g.add_triple(make_triple("A", "is_a", "B"))
    g.compute_pagerank()
    w1 = g.concept_weight("A")
    # Add a new triple
    g.add_triple(make_triple("A", "is_a", "C"))
    w2 = g.concept_weight("A")
    # Weights should have changed (A now has degree 2)
    assert w1 != w2


def test_invalid_module_id_rejected():
    with pytest.raises(ValueError):
        ModuleGraph("")


# -----------------------------------------------------------------------------
# KnowledgeGraph — container and isolation
# -----------------------------------------------------------------------------


def test_kg_add_module_is_idempotent():
    kg = KnowledgeGraph()
    m1 = kg.add_module("module_1")
    m2 = kg.add_module("module_1")
    assert m1 is m2


def test_kg_get_unknown_module_raises():
    kg = KnowledgeGraph()
    with pytest.raises(KeyError):
        kg.get_module("nonexistent")


def test_kg_has_module():
    kg = KnowledgeGraph()
    kg.add_module("module_1")
    assert kg.has_module("module_1") is True
    assert kg.has_module("module_2") is False


def test_kg_module_ids():
    kg = KnowledgeGraph()
    kg.add_module("module_1")
    kg.add_module("module_2")
    assert sorted(kg.module_ids()) == ["module_1", "module_2"]


# -----------------------------------------------------------------------------
# Isolation invariant (the key test)
# -----------------------------------------------------------------------------


def test_module_isolation_no_cross_module_leakage():
    """A concept in module_1 must not appear in module_2's graph."""
    kg = KnowledgeGraph()
    m1 = kg.add_module("module_1")
    m2 = kg.add_module("module_2")

    m1.add_triple(make_triple("Docker", "is_a", "platform"))
    m2.add_triple(make_triple("TCP", "is_a", "protocol"))

    # Docker neighborhood exists in module_1 only
    assert len(kg.get_concept_neighborhood("module_1", "Docker")) == 1
    # Docker neighborhood is empty in module_2
    assert len(kg.get_concept_neighborhood("module_2", "Docker")) == 0

    # TCP neighborhood is empty in module_1
    assert len(kg.get_concept_neighborhood("module_1", "TCP")) == 0
    # TCP neighborhood exists in module_2 only
    assert len(kg.get_concept_neighborhood("module_2", "TCP")) == 1


def test_kg_total_counts():
    kg = KnowledgeGraph()
    m1 = kg.add_module("module_1")
    m2 = kg.add_module("module_2")
    m1.add_triple(make_triple("A", "is_a", "B"))
    m1.add_triple(make_triple("C", "is_a", "D"))
    m2.add_triple(make_triple("E", "is_a", "F"))
    assert kg.total_triples() == 3


# -----------------------------------------------------------------------------
# KnowledgeGraphBuilder — end-to-end construction
# -----------------------------------------------------------------------------


def test_builder_build_module_from_blocks():
    builder = KnowledgeGraphBuilder()
    blocks = [
        FakeBlock(id="b1", text="Docker is a containerization platform.", page=1),
        FakeBlock(id="b2", text="Kubernetes requires a container runtime.", page=1),
        FakeBlock(id="b3", text="Encryption enables secure communication.", page=1),
    ]
    graph = builder.build_module("module_1", blocks)
    assert graph.triple_count() >= 3
    assert graph.node_count() >= 3


def test_builder_skips_non_body_blocks():
    builder = KnowledgeGraphBuilder()
    blocks = [
        FakeBlock(id="b1", text="Docker is a containerization platform.", page=1, block_role="BODY"),
        FakeBlock(id="b2", text="Reference: Packt Publishing 2020", page=1, block_role="BIBLIOGRAPHY"),
        FakeBlock(id="b3", text="DSATM-ISE Header", page=1, block_role="METADATA"),
    ]
    graph = builder.build_module("module_1", blocks)
    # Only the BODY block should contribute triples
    for triple in graph.all_triples():
        assert triple.source_block_id == "b1"


def test_builder_build_all_with_page_ranges():
    @dataclass
    class FakeArtifact:
        text_blocks: list

    artifact = FakeArtifact(text_blocks=[
        FakeBlock(id="b1", text="Docker is a containerization platform.", page=1),
        FakeBlock(id="b2", text="TCP is a transport protocol.", page=6),
        FakeBlock(id="b3", text="Encryption enables secure communication.", page=7),
    ])

    builder = KnowledgeGraphBuilder()
    kg = builder.build_all(
        artifact=artifact,
        module_page_ranges={
            "module_1": (1, 5),
            "module_2": (6, 10),
        },
    )

    assert kg.has_module("module_1")
    assert kg.has_module("module_2")
    assert kg.get_module("module_1").triple_count() >= 1
    assert kg.get_module("module_2").triple_count() >= 2


def test_builder_skips_blocks_outside_ranges():
    @dataclass
    class FakeArtifact:
        text_blocks: list

    artifact = FakeArtifact(text_blocks=[
        FakeBlock(id="b1", text="Docker is a containerization platform.", page=100),
    ])

    builder = KnowledgeGraphBuilder()
    kg = builder.build_all(
        artifact=artifact,
        module_page_ranges={"module_1": (1, 5)},
    )
    assert kg.get_module("module_1").triple_count() == 0


def test_builder_pagerank_runs_at_build_time():
    builder = KnowledgeGraphBuilder()
    blocks = [
        FakeBlock(id="b1", text="Docker is a containerization platform.", page=1),
        FakeBlock(id="b2", text="Docker requires a runtime.", page=1),
    ]
    graph = builder.build_module("module_1", blocks)
    # Weights should be available immediately without an explicit call
    top = graph.top_concepts(1)
    assert len(top) == 1
    # "Docker" appears in both triples → highest centrality
    assert top[0][0] == "Docker"


def test_builder_realistic_academic_text():
    builder = KnowledgeGraphBuilder()
    blocks = [
        FakeBlock(
            id="b1",
            text=(
                "Virtualization is a fundamental technology. "
                "Virtualization creates multiple simulated environments. "
                "A hypervisor consists of a software layer. "
                "Cloud storage is used for remote data persistence. "
                "Encryption enables secure communication."
            ),
            page=1,
        ),
    ]
    graph = builder.build_module("module_1", blocks)
    assert graph.triple_count() >= 3
    concepts = {t.subject for t in graph.all_triples()} | {t.object for t in graph.all_triples()}
    # Sanity: "Virtualization" should appear
    assert any("irtualization" in c for c in concepts)
