"""
AION Mathematical Knowledge Engine
Expression Tree
===============
Internal representation of mathematical expressions as trees.
This is how SymPy, Mathematica, and Maple represent mathematics internally.

The model NEVER sees raw symbols.
It reasons over the tree structure.

Node types:
    NUMBER      — literal numeric value
    VARIABLE    — symbolic variable (x, t, n, ...)
    CONSTANT    — named constant (π, e, g, c, ...)
    OPERATOR    — binary/unary operation (+, -, *, /, ^)
    FUNCTION    — named function (sin, cos, exp, ln, ...)
    INTEGRAL    — definite or indefinite integral
    DERIVATIVE  — nth order derivative
    LIMIT       — limit expression
    SUMMATION   — finite or infinite sum
    MATRIX      — matrix expression
    VECTOR      — vector expression
    EQUATION    — lhs = rhs
    INEQUALITY  — lhs </>/<=/>=/>= rhs
"""

from __future__ import annotations
import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Union, Optional, Any, List, Set, Dict


# ── Node Types ────────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    NUMBER      = "number"
    VARIABLE    = "variable"
    CONSTANT    = "constant"
    OPERATOR    = "operator"
    FUNCTION    = "function"
    INTEGRAL    = "integral"
    DERIVATIVE  = "derivative"
    LIMIT       = "limit"
    SUMMATION   = "summation"
    PRODUCT_OP  = "product_op"
    MATRIX      = "matrix"
    VECTOR      = "vector"
    EQUATION    = "equation"
    INEQUALITY  = "inequality"
    FRACTION    = "fraction"
    SQRT        = "sqrt"
    TRANSFORM   = "transform"
    EXPRESSION  = "expression"


# ── Named Constants ───────────────────────────────────────────────────────────

NAMED_CONSTANTS: Dict[str, tuple] = {
    "pi":     ("π",  3.14159265358979),
    "e":      ("e",  2.71828182845905),
    "i":      ("i",  None),             # imaginary unit
    "inf":    ("∞",  float('inf')),
    "g":      ("g",  9.80665),          # gravitational acceleration
    "c":      ("c",  3e8),              # speed of light
    "h":      ("h",  6.626e-34),        # Planck constant
    "k":      ("k",  1.380e-23),        # Boltzmann constant
}

# ── Operator Metadata ─────────────────────────────────────────────────────────

OPERATOR_META: Dict[str, dict] = {
    "+":   {"arity": 2, "precedence": 1, "name": "addition",       "latex": "+"},
    "-":   {"arity": 2, "precedence": 1, "name": "subtraction",    "latex": "-"},
    "*":   {"arity": 2, "precedence": 2, "name": "multiplication",  "latex": "\\cdot"},
    "/":   {"arity": 2, "precedence": 2, "name": "division",        "latex": "\\frac"},
    "^":   {"arity": 2, "precedence": 3, "name": "exponentiation",  "latex": "^"},
    "neg": {"arity": 1, "precedence": 3, "name": "negation",        "latex": "-"},
    "abs": {"arity": 1, "precedence": 4, "name": "absolute value",  "latex": "|"},
}

FUNCTION_META: Dict[str, dict] = {
    "sin":   {"latex": "\\sin",   "inverse": "asin"},
    "cos":   {"latex": "\\cos",   "inverse": "acos"},
    "tan":   {"latex": "\\tan",   "inverse": "atan"},
    "exp":   {"latex": "e^",      "inverse": "ln"},
    "ln":    {"latex": "\\ln",    "inverse": "exp"},
    "log":   {"latex": "\\log",   "inverse": None},
    "sqrt":  {"latex": "\\sqrt",  "inverse": "square"},
    "abs":   {"latex": "|",       "inverse": None},
    "sgn":   {"latex": "\\text{sgn}", "inverse": None},
}


# ── Core Node ─────────────────────────────────────────────────────────────────

@dataclass
class ExprNode:
    """
    A single node in the expression tree.
    Every mathematical expression is a tree of ExprNodes.
    """
    node_type:  NodeType
    value:      Any                         = None    # number, variable name, operator
    children:   List["ExprNode"]            = field(default_factory=list)
    metadata:   Dict                        = field(default_factory=dict)

    # For INTEGRAL, DERIVATIVE, LIMIT, SUMMATION
    variable:   Optional[str]               = None    # integration/diff variable
    lower:      Optional["ExprNode"]        = None    # lower bound
    upper:      Optional["ExprNode"]        = None    # upper bound
    order:      int                         = 1       # derivative order

    # For EQUATION / INEQUALITY
    lhs:        Optional["ExprNode"]        = None
    rhs:        Optional["ExprNode"]        = None
    relation:   str                         = "="     # =, <, >, <=, >=, !=

    def __repr__(self) -> str:
        if self.node_type == NodeType.NUMBER:
            return str(self.value)
        if self.node_type == NodeType.VARIABLE:
            return str(self.value)
        if self.node_type == NodeType.CONSTANT:
            return NAMED_CONSTANTS.get(self.value, (self.value,))[0]
        if self.node_type == NodeType.OPERATOR:
            if len(self.children) == 2:
                return f"({self.children[0]} {self.value} {self.children[1]})"
            return f"{self.value}({self.children[0]})"
        if self.node_type == NodeType.FUNCTION:
            args = ", ".join(repr(c) for c in self.children)
            return f"{self.value}({args})"
        if self.node_type == NodeType.EQUATION:
            return f"{self.lhs} {self.relation} {self.rhs}"
        if self.node_type == NodeType.INTEGRAL:
            bounds = f"_{{{self.lower}}}^{{{self.upper}}}" if self.lower else ""
            expr   = self.children[0] if self.children else "f"
            return f"∫{bounds} {expr} d{self.variable}"
        if self.node_type == NodeType.DERIVATIVE:
            expr = self.children[0] if self.children else "f"
            return f"d^{self.order}/d{self.variable}^{self.order} ({expr})"
        if self.node_type == NodeType.LIMIT:
            expr = self.children[0] if self.children else "f"
            return f"lim({self.variable}→{self.upper}) {expr}"
        if self.node_type == NodeType.SUMMATION:
            expr = self.children[0] if self.children else "a_n"
            return f"∑({self.variable}={self.lower} to {self.upper}) {expr}"
        return f"<{self.node_type.value}:{self.value}>"

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        d: dict = {"type": self.node_type.value}
        if self.value is not None:
            d["value"] = self.value
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        if self.variable:
            d["variable"] = self.variable
        if self.lower is not None:
            d["lower"] = self.lower.to_dict() if isinstance(self.lower, ExprNode) else self.lower
        if self.upper is not None:
            d["upper"] = self.upper.to_dict() if isinstance(self.upper, ExprNode) else self.upper
        if self.order != 1:
            d["order"] = self.order
        if self.lhs is not None:
            d["lhs"] = self.lhs.to_dict()
        if self.rhs is not None:
            d["rhs"] = self.rhs.to_dict()
        if self.relation != "=":
            d["relation"] = self.relation
        if self.metadata:
            d["metadata"] = self.metadata
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def variables(self) -> Set[str]:
        """Collect all variable names in this subtree."""
        result = set()
        if self.node_type == NodeType.VARIABLE:
            result.add(str(self.value))
        if self.variable:
            result.add(self.variable)
        for child in self.children:
            result |= child.variables()
        if self.lhs:
            result |= self.lhs.variables()
        if self.rhs:
            result |= self.rhs.variables()
        if isinstance(self.lower, ExprNode):
            result |= self.lower.variables()
        if isinstance(self.upper, ExprNode):
            result |= self.upper.variables()
        return result

    def depth(self) -> int:
        """Maximum depth of the expression tree."""
        if not self.children:
            return 1
        return 1 + max(c.depth() for c in self.children)

    def complexity(self) -> int:
        """Count of all nodes in the tree — proxy for expression complexity."""
        count = 1
        for child in self.children:
            count += child.complexity()
        if self.lhs:
            count += self.lhs.complexity()
        if self.rhs:
            count += self.rhs.complexity()
        return count


# ── Node Factory Functions ────────────────────────────────────────────────────

def num(value: Union[int, float]) -> ExprNode:
    return ExprNode(NodeType.NUMBER, value=value)

def var(name: str) -> ExprNode:
    return ExprNode(NodeType.VARIABLE, value=name)

def const(name: str) -> ExprNode:
    return ExprNode(NodeType.CONSTANT, value=name)

def op(operator: str, *children: ExprNode) -> ExprNode:
    return ExprNode(NodeType.OPERATOR, value=operator, children=list(children))

def fn(name: str, *args: ExprNode) -> ExprNode:
    return ExprNode(NodeType.FUNCTION, value=name, children=list(args))

def integral(
    expr:     ExprNode,
    variable: str,
    lower:    Optional[ExprNode] = None,
    upper:    Optional[ExprNode] = None,
) -> ExprNode:
    return ExprNode(
        NodeType.INTEGRAL,
        children = [expr],
        variable = variable,
        lower    = lower,
        upper    = upper,
    )

def derivative(
    expr:     ExprNode,
    variable: str,
    order:    int = 1,
) -> ExprNode:
    return ExprNode(
        NodeType.DERIVATIVE,
        children = [expr],
        variable = variable,
        order    = order,
    )

def limit(
    expr:     ExprNode,
    variable: str,
    approach: ExprNode,
) -> ExprNode:
    return ExprNode(
        NodeType.LIMIT,
        children = [expr],
        variable = variable,
        upper    = approach,
    )

def summation(
    expr:     ExprNode,
    index:    str,
    lower:    ExprNode,
    upper:    ExprNode,
) -> ExprNode:
    return ExprNode(
        NodeType.SUMMATION,
        children = [expr],
        variable = index,
        lower    = lower,
        upper    = upper,
    )

def equation(lhs: ExprNode, rhs: ExprNode, relation: str = "=") -> ExprNode:
    return ExprNode(
        NodeType.EQUATION,
        lhs      = lhs,
        rhs      = rhs,
        relation = relation,
    )

def fraction(numerator: ExprNode, denominator: ExprNode) -> ExprNode:
    return ExprNode(
        NodeType.FRACTION,
        children = [numerator, denominator],
    )

def sqrt(expr: ExprNode) -> ExprNode:
    return ExprNode(NodeType.SQRT, children=[expr])
