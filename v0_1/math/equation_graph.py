"""
AION Equation Graph Builder
============================
Builds a mathematical knowledge graph where nodes are
MathObjects and edges are semantic relationships.

Relationships:
  USES_VARIABLE    — equation uses this variable
  DERIVED_FROM     — this formula is derived from another
  APPLIED_IN       — this concept applies in a context
  PREREQUISITE     — must understand this before this
  EQUIVALENT_TO    — mathematically equivalent forms
  SOLVES           — this method solves this type of problem
  PARAMETERIZES    — this template can generate this instance
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict
from .math_object import MathObject


@dataclass
class MathRelation:
    relation_type: str
    source_id:     str
    target_id:     str
    description:   str = ""
    weight:        float = 1.0


@dataclass
class EquationGraph:
    """
    Subject-scoped mathematical knowledge graph.
    Nodes: MathObject instances
    Edges: MathRelation instances
    """
    subject:   str
    nodes:     Dict[str, MathObject]   = field(default_factory=dict)
    edges:     List[MathRelation]      = field(default_factory=list)
    topics:    Dict[str, List[str]]    = field(default_factory=dict)

    def add(self, obj: MathObject):
        self.nodes[obj.math_id] = obj
        topic = obj.topic or obj.math_type.value
        self.topics.setdefault(topic, []).append(obj.math_id)

    def relate(
        self,
        source_id:     str,
        relation_type: str,
        target_id:     str,
        description:   str = "",
    ):
        self.edges.append(MathRelation(
            relation_type = relation_type,
            source_id     = source_id,
            target_id     = target_id,
            description   = description,
        ))

    def get_by_type(self, math_type: str) -> List[MathObject]:
        return [
            obj for obj in self.nodes.values()
            if obj.math_type.value == math_type
        ]

    def get_by_topic(self, topic: str) -> List[MathObject]:
        ids = self.topics.get(topic, [])
        return [self.nodes[i] for i in ids if i in self.nodes]

    def get_prerequisites(self, math_id: str) -> List[MathObject]:
        prereq_ids = [
            e.target_id for e in self.edges
            if e.source_id == math_id and e.relation_type == "PREREQUISITE"
        ]
        return [self.nodes[i] for i in prereq_ids if i in self.nodes]

    def get_applications(self, math_id: str) -> List[str]:
        return [
            e.description for e in self.edges
            if e.source_id == math_id and e.relation_type == "APPLIED_IN"
        ]

    def summary(self) -> dict:
        return {
            "subject":       self.subject,
            "total_formulas": len(self.nodes),
            "total_relations": len(self.edges),
            "topics":        list(self.topics.keys()),
            "by_type":       {
                t: len(v) for t, v in
                {obj.math_type.value: [] for obj in self.nodes.values()}.items()
            },
        }


class EquationGraphBuilder:
    """
    Builds an EquationGraph from a list of MathObjects.
    Infers relationships between formulas automatically.
    """

    # Known inter-formula relationships
    KNOWN_RELATIONS = [
        # (keyword_in_a, keyword_in_b, relation_type, description)
        ("integral", "derivative", "EQUIVALENT_TO",
         "Fundamental Theorem of Calculus"),
        ("laplace",  "fourier",    "DERIVED_FROM",
         "Fourier transform is Laplace on imaginary axis"),
        ("laplace",  "transfer_function", "APPLIED_IN",
         "Laplace transforms used in control systems"),
        ("eigenvalue", "matrix",   "USES_VARIABLE",
         "Eigenvalue defined for square matrices"),
        ("fourier",  "signal",     "APPLIED_IN",
         "Fourier analysis of signals"),
    ]

    def build(
        self,
        objects: List[MathObject],
        subject: str = "",
    ) -> EquationGraph:
        graph = EquationGraph(subject=subject)

        # Add all objects as nodes
        for obj in objects:
            graph.add(obj)

        # Infer relationships
        obj_list = list(objects)
        for i, a in enumerate(obj_list):
            for j, b in enumerate(obj_list):
                if i == j:
                    continue
                self._infer_relations(graph, a, b)

        # Add shared variable relationships
        self._add_variable_relations(graph, obj_list)

        return graph

    def _infer_relations(
        self,
        graph: EquationGraph,
        a: MathObject,
        b: MathObject,
    ):
        a_type = a.math_type.value
        b_type = b.math_type.value

        for kw_a, kw_b, rel, desc in self.KNOWN_RELATIONS:
            if kw_a in a_type and kw_b in b_type:
                graph.relate(a.math_id, rel, b.math_id, desc)
                break

        # Prerequisite: named formulas often require simpler ones
        if a.is_template and not b.is_template:
            if any(v.symbol in b.canonical for v in a.variables):
                graph.relate(
                    b.math_id, "PREREQUISITE", a.math_id,
                    f"Understanding {b.description} requires {a.named_as or a.description}"
                )

    def _add_variable_relations(
        self,
        graph: EquationGraph,
        objects: List[MathObject],
    ):
        var_to_formulas: Dict[str, List[str]] = {}
        for obj in objects:
            for var in obj.variables:
                var_to_formulas.setdefault(var.symbol, []).append(obj.math_id)

        for var, formula_ids in var_to_formulas.items():
            if len(formula_ids) > 1:
                for fid in formula_ids:
                    for other_fid in formula_ids:
                        if fid != other_fid:
                            graph.relate(
                                fid, "USES_VARIABLE", other_fid,
                                f"Both use variable '{var}'"
                            )
