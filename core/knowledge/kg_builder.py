"""
Knowledge graph construction with module isolation and PageRank weighting.

The core invariant: **triples cannot cross module boundaries.** Each
module has its own graph. Adding a triple requires specifying the module
it belongs to. Querying for a concept requires specifying the module.
There is no cross-module API.

This invariant protects VTU's strict per-module examination scope: a
question about Module 2 must never be generated from a Module 3 concept.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .triple_extractor import RawTriple, RuleBasedTripleExtractor


# -----------------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Triple:
    """A triple with full provenance."""
    subject: str
    predicate: str
    object: str
    source_block_id: str
    source_page: int
    confidence: float


@dataclass
class PageRankResult:
    weights: Dict[str, float]
    iterations: int
    converged: bool


# -----------------------------------------------------------------------------
# Module graph
# -----------------------------------------------------------------------------


class ModuleGraph:
    """
    A single module's knowledge graph.

    The graph is undirected for PageRank purposes (each triple adds both
    directions to the adjacency map). This is a design choice: for
    concept centrality, directionality is less important than connectivity.
    """

    def __init__(self, module_id: str) -> None:
        if not module_id:
            raise ValueError("module_id must be non-empty")
        self.module_id = module_id
        self._triples: List[Triple] = []
        self._adjacency: Dict[str, List[str]] = defaultdict(list)
        self._weights: Dict[str, float] = {}
        self._pagerank_computed = False

    # ------------------------------------------------------------- Mutation

    def add_triple(self, triple: Triple) -> None:
        self._triples.append(triple)
        self._adjacency[triple.subject].append(triple.object)
        self._adjacency[triple.object].append(triple.subject)
        # Invalidate cached weights
        self._pagerank_computed = False

    # ------------------------------------------------------------- PageRank

    def compute_pagerank(
        self,
        damping: float = 0.85,
        max_iterations: int = 50,
        tolerance: float = 1e-6,
    ) -> PageRankResult:
        """
        Compute PageRank weights over the module's concept graph.

        Uses an undirected interpretation (each triple is a symmetric edge).
        Converges when the L1 change between iterations is < tolerance,
        or when max_iterations is reached.
        """
        nodes = list(self._adjacency.keys())
        n = len(nodes)

        if n == 0:
            self._weights = {}
            self._pagerank_computed = True
            return PageRankResult(weights={}, iterations=0, converged=True)

        rank: Dict[str, float] = {node: 1.0 / n for node in nodes}

        converged = False
        iterations_run = 0
        for iteration in range(max_iterations):
            iterations_run = iteration + 1
            new_rank: Dict[str, float] = {}
            for node in nodes:
                incoming = 0.0
                for other in nodes:
                    neighbors = self._adjacency.get(other, [])
                    if node in neighbors:
                        degree = len(neighbors)
                        if degree > 0:
                            incoming += rank[other] / degree
                new_rank[node] = (1.0 - damping) / n + damping * incoming

            delta = sum(abs(new_rank[n] - rank[n]) for n in nodes)
            rank = new_rank
            if delta < tolerance:
                converged = True
                break

        self._weights = rank
        self._pagerank_computed = True
        return PageRankResult(
            weights=dict(rank),
            iterations=iterations_run,
            converged=converged,
        )

    def _ensure_pagerank(self) -> None:
        if not self._pagerank_computed:
            self.compute_pagerank()

    # ------------------------------------------------------------- Queries

    def get_neighborhood(
        self,
        concept: str,
        max_depth: int = 2,
        min_confidence: float = 0.0,
    ) -> List[Triple]:
        """
        Return triples within `max_depth` hops of `concept`.

        depth=0 → no triples (the concept itself)
        depth=1 → triples where concept is subject or object
        depth=2 → triples one hop beyond those
        """
        if max_depth < 1:
            return []

        concept_lc = concept.lower()
        visited_nodes = {concept_lc}
        frontier = {concept_lc}
        result: List[Triple] = []
        seen_triples: set[int] = set()

        for _ in range(max_depth):
            next_frontier: set[str] = set()
            for triple_idx, triple in enumerate(self._triples):
                if triple_idx in seen_triples:
                    continue
                s = triple.subject.lower()
                o = triple.object.lower()
                if s in frontier or o in frontier:
                    if triple.confidence < min_confidence:
                        continue
                    result.append(triple)
                    seen_triples.add(triple_idx)
                    if s not in visited_nodes:
                        next_frontier.add(s)
                    if o not in visited_nodes:
                        next_frontier.add(o)

            visited_nodes |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        return result

    def concept_weight(self, concept: str) -> float:
        """Return the PageRank weight for a concept, or 0.0 if unknown."""
        self._ensure_pagerank()
        return self._weights.get(concept, 0.0)

    def top_concepts(self, n: int = 10) -> List[Tuple[str, float]]:
        self._ensure_pagerank()
        sorted_nodes = sorted(self._weights.items(), key=lambda kv: kv[1], reverse=True)
        return sorted_nodes[:n]

    def triple_count(self) -> int:
        return len(self._triples)

    def node_count(self) -> int:
        return len(self._adjacency)

    def all_triples(self) -> List[Triple]:
        return list(self._triples)


# -----------------------------------------------------------------------------
# Knowledge graph container
# -----------------------------------------------------------------------------


class KnowledgeGraph:
    """
    Container for multiple module graphs.

    There is deliberately no method to add a cross-module triple.
    Each module has its own graph. Query APIs require an explicit
    module_id, so cross-module lookups are impossible by construction.
    """

    def __init__(self) -> None:
        self._modules: Dict[str, ModuleGraph] = {}

    # ------------------------------------------------------------- Modules

    def add_module(self, module_id: str) -> ModuleGraph:
        if module_id in self._modules:
            return self._modules[module_id]
        graph = ModuleGraph(module_id)
        self._modules[module_id] = graph
        return graph

    def set_module(self, module_id: str, graph: ModuleGraph) -> None:
        """Register or replace a module graph."""
        if not module_id:
            raise ValueError("module_id must be non-empty")
        self._modules[module_id] = graph

    def get_module(self, module_id: str) -> ModuleGraph:
        if module_id not in self._modules:
            raise KeyError(f"Unknown module: {module_id}")
        return self._modules[module_id]

    def has_module(self, module_id: str) -> bool:
        return module_id in self._modules

    def module_ids(self) -> List[str]:
        return list(self._modules.keys())

    def total_triples(self) -> int:
        return sum(m.triple_count() for m in self._modules.values())

    def total_nodes(self) -> int:
        return sum(m.node_count() for m in self._modules.values())

    # ------------------------------------------------------------- Convenience

    def get_concept_neighborhood(
        self,
        module_id: str,
        concept: str,
        max_depth: int = 2,
        min_confidence: float = 0.0,
    ) -> List[Triple]:
        """Convenience wrapper — module_id is required, no cross-module leakage."""
        return self.get_module(module_id).get_neighborhood(
            concept, max_depth=max_depth, min_confidence=min_confidence
        )


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------


class KnowledgeGraphBuilder:
    """
    Orchestrates triple extraction and graph construction.

    Parameters
    ----------
    extractor : TripleExtractor | None
        Defaults to RuleBasedTripleExtractor with default config.
    """

    def __init__(self, extractor=None) -> None:
        self._extractor = extractor or RuleBasedTripleExtractor()

    def build_module(
        self,
        module_id: str,
        blocks: Iterable,
        skip_roles: Tuple[str, ...] = ("CAPTION", "METADATA", "ADMIN", "BIBLIOGRAPHY", "TOC", "EXERCISE"),
    ) -> ModuleGraph:
        """
        Build a single module's graph from a list of text blocks.

        Blocks with block_role in `skip_roles` are ignored. Default skips
        everything except BODY and HEADING.
        """
        graph = ModuleGraph(module_id)

        for block in blocks:
            role = getattr(block, "block_role", "BODY")
            if role in skip_roles:
                continue

            raw_triples = self._extractor.extract(block.text)
            for raw in raw_triples:
                graph.add_triple(
                    Triple(
                        subject=raw.subject,
                        predicate=raw.predicate,
                        object=raw.object,
                        source_block_id=getattr(block, "id", ""),
                        source_page=getattr(block, "page", 0),
                        confidence=raw.confidence,
                    )
                )

        graph.compute_pagerank()
        return graph

    def build_all(
        self,
        artifact,
        module_page_ranges: Dict[str, Tuple[int, int]],
        skip_roles: Tuple[str, ...] = ("CAPTION", "METADATA", "ADMIN", "BIBLIOGRAPHY", "TOC", "EXERCISE"),
    ) -> KnowledgeGraph:
        """
        Build a KnowledgeGraph with one graph per module.

        Parameters
        ----------
        artifact : DocumentArtifact
            Must expose `.text_blocks`.
        module_page_ranges : Dict[str, Tuple[int, int]]
            Maps module_id → (start_page_inclusive, end_page_inclusive).
            Blocks are assigned to modules by their `.page` attribute.

        Raises
        ------
        ValueError
            If any block's page falls outside the provided ranges.
        """
        kg = KnowledgeGraph()
        blocks_by_module: Dict[str, List] = defaultdict(list)

        ranges_sorted = sorted(
            module_page_ranges.items(),
            key=lambda kv: kv[1][0],
        )

        for block in artifact.text_blocks:
            page = getattr(block, "page", 0)
            matched = None
            for module_id, (start, end) in ranges_sorted:
                if start <= page <= end:
                    matched = module_id
                    break
            if matched is None:
                # Block outside any declared module — skip rather than
                # silently attach to the nearest one.
                continue
            blocks_by_module[matched].append(block)

        for module_id in module_page_ranges.keys():
            module_graph = self.build_module(
                module_id=module_id,
                blocks=blocks_by_module.get(module_id, []),
                skip_roles=skip_roles,
            )
            kg.set_module(module_id, module_graph)

        return kg
