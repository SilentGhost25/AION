"""
AION Mathematical Understanding Engine (MUE)
=============================================
Provides canonical math tokenization, MathObject internal representation,
deterministic parsing, equation graph building, and parameterized numerical problem generation.
"""

from .tokens import MathType, token_to_unicode, token_to_latex, UNICODE_MAP, LATEX_MAP
from .math_object import MathObject, Variable
from .math_parser import MathParser
from .equation_graph import EquationGraph, EquationGraphBuilder, MathRelation
from .param_question_generator import ParameterizedQuestionGenerator, NumericalProblem

__all__ = [
    "MathType",
    "token_to_unicode",
    "token_to_latex",
    "UNICODE_MAP",
    "LATEX_MAP",
    "MathObject",
    "Variable",
    "MathParser",
    "EquationGraph",
    "EquationGraphBuilder",
    "MathRelation",
    "ParameterizedQuestionGenerator",
    "NumericalProblem",
]
