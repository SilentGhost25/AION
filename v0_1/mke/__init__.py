"""
AION Mathematical Knowledge Engine (MKE)
=========================================
Pratt-parsed expression trees, Equation Knowledge Objects (EKO),
automatic solvers, problem generators, and multi-format renderers.
"""

from .expression_tree import (
    ExprNode, NodeType,
    num, var, const, op, fn,
    integral, derivative, limit, summation, equation, fraction, sqrt,
    NAMED_CONSTANTS, OPERATOR_META, FUNCTION_META,
)
from .equation_knowledge_object import (
    EquationKnowledgeObject, VariableSpec,
    QuestionBlueprint, DerivationStep,
)
from .expression_parser import ExpressionParser, PrattParser, tokenize, Token
from .renderer import TreeRenderer
from .mathematical_knowledge_engine import MathematicalKnowledgeEngine

__all__ = [
    "ExprNode",
    "NodeType",
    "num", "var", "const", "op", "fn",
    "integral", "derivative", "limit", "summation", "equation", "fraction", "sqrt",
    "NAMED_CONSTANTS", "OPERATOR_META", "FUNCTION_META",
    "EquationKnowledgeObject",
    "VariableSpec",
    "QuestionBlueprint",
    "DerivationStep",
    "ExpressionParser",
    "PrattParser",
    "tokenize",
    "Token",
    "TreeRenderer",
    "MathematicalKnowledgeEngine",
]
