"""
AION Knowledge Graph Layer.

Extracts Subject-Predicate-Object triples from academic text blocks,
builds one graph per module, and computes PageRank weights for
concept centrality. Enforces module isolation as an API invariant.

Public surface:
    - Triple              : one SPO triple with provenance
    - RawTriple           : triple before provenance attachment
    - ModuleGraph         : per-module graph with PageRank weights
    - KnowledgeGraph      : container for multiple module graphs
    - KnowledgeGraphBuilder: orchestrates extraction + graph construction
    - RuleBasedTripleExtractor: deterministic SPO extraction from text
"""

from .kg_builder import (
    Triple,
    RawTriple,
    ModuleGraph,
    KnowledgeGraph,
    KnowledgeGraphBuilder,
)
from .triple_extractor import RuleBasedTripleExtractor

__all__ = [
    "Triple",
    "RawTriple",
    "ModuleGraph",
    "KnowledgeGraph",
    "KnowledgeGraphBuilder",
    "RuleBasedTripleExtractor",
]
