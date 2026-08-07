"""
AION Mathematical Token System
===============================
Canonical internal representation for all mathematical expressions.
The model NEVER reasons over Unicode directly.
Unicode appears only at the rendering stage.

Token format: <TOKEN_NAME param=value ...>
"""

from dataclasses import dataclass, field
from typing import Optional, Union, Dict
from enum import Enum


# ── Token Types ───────────────────────────────────────────────────────────────

class MathType(str, Enum):
    # Calculus
    INTEGRAL     = "integral"
    DERIVATIVE   = "derivative"
    LIMIT        = "limit"
    SUMMATION    = "summation"
    PRODUCT      = "product"

    # Algebra
    EQUATION     = "equation"
    INEQUALITY   = "inequality"
    POLYNOMIAL   = "polynomial"
    MATRIX       = "matrix"
    DETERMINANT  = "determinant"
    VECTOR       = "vector"

    # Functions
    FUNCTION     = "function"
    TRIG         = "trig"
    LOG          = "log"
    EXPONENTIAL  = "exponential"

    # Transforms
    LAPLACE      = "laplace"
    FOURIER      = "fourier"
    Z_TRANSFORM  = "z_transform"

    # Special
    FORMULA      = "formula"      # named formula (e.g. Ohm's law)
    EXPRESSION   = "expression"   # generic algebraic expression
    CONSTANT     = "constant"     # π, e, g, c, etc.
    FRACTION     = "fraction"
    SQRT         = "sqrt"
    PROBABILITY  = "probability"
    STATISTIC    = "statistic"

    # Engineering
    TRANSFER_FN  = "transfer_function"
    CIRCUIT_EQ   = "circuit_equation"
    SIGNAL       = "signal"


# ── Unicode Render Dictionary ─────────────────────────────────────────────────
# ONLY used at final rendering stage. Never used for reasoning.

UNICODE_MAP: Dict[str, str] = {
    # Greek letters
    "<ALPHA>":    "α",   "<BETA>":     "β",   "<GAMMA>":    "γ",
    "<DELTA>":    "δ",   "<EPSILON>":  "ε",   "<ZETA>":     "ζ",
    "<ETA>":      "η",   "<THETA>":    "θ",   "<IOTA>":     "ι",
    "<KAPPA>":    "κ",   "<LAMBDA>":   "λ",   "<MU>":       "μ",
    "<NU>":       "ν",   "<XI>":       "ξ",   "<PI>":       "π",
    "<RHO>":      "ρ",   "<SIGMA>":    "σ",   "<TAU>":      "τ",
    "<UPSILON>":  "υ",   "<PHI>":      "φ",   "<CHI>":      "χ",
    "<PSI>":      "ψ",   "<OMEGA>":    "ω",

    # Capitals
    "<GAMMA_U>":  "Γ",   "<DELTA_U>":  "Δ",   "<THETA_U>":  "Θ",
    "<LAMBDA_U>": "Λ",   "<XI_U>":     "Ξ",   "<PI_U>":     "Π",
    "<SIGMA_U>":  "Σ",   "<UPSILON_U>":"Υ",   "<PHI_U>":    "Φ",
    "<PSI_U>":    "Ψ",   "<OMEGA_U>":  "Ω",

    # Operators
    "<INT>":       "∫",   "<IINT>":    "∬",   "<IIINT>":   "∭",
    "<OINT>":      "∮",   "<SUM>":     "∑",   "<PROD>":    "∏",
    "<SQRT>":      "√",   "<CBRT>":    "∛",   "<INF>":     "∞",
    "<PARTIAL>":   "∂",   "<NABLA>":   "∇",   "<DEL>":     "∇",
    "<PM>":        "±",   "<MP>":      "∓",   "<TIMES>":   "×",
    "<DIV_SYM>":   "÷",   "<DOT>":     "·",   "<CDOT>":    "⋅",

    # Relations
    "<LEQ>":       "≤",   "<GEQ>":     "≥",   "<NEQ>":     "≠",
    "<APPROX>":    "≈",   "<EQUIV>":   "≡",   "<PROP>":    "∝",
    "<SUBSET>":    "⊂",   "<SUPSET>":  "⊃",   "<IN>":      "∈",
    "<NOT_IN>":    "∉",   "<FORALL>":  "∀",   "<EXISTS>":  "∃",

    # Arrows
    "<RARR>":      "→",   "<LARR>":    "←",   "<DARR>":    "↓",
    "<UARR>":      "↑",   "<LRARR>":   "↔",   "<IMPLIES>": "⇒",
    "<IFF>":       "⟺",

    # Superscripts
    "<SUP0>": "⁰", "<SUP1>": "¹", "<SUP2>": "²", "<SUP3>": "³",
    "<SUP4>": "⁴", "<SUP5>": "⁵", "<SUP6>": "⁶", "<SUP7>": "⁷",
    "<SUP8>": "⁸", "<SUP9>": "⁹", "<SUPN>": "ⁿ",

    # Subscripts
    "<SUB0>": "₀", "<SUB1>": "₁", "<SUB2>": "₂", "<SUB3>": "₃",
    "<SUB4>": "₄", "<SUB5>": "₅", "<SUB6>": "₆", "<SUB7>": "₇",
    "<SUB8>": "₈", "<SUB9>": "₉",

    # Constants
    "<EULER>": "e",  "<HBAR>": "ℏ",  "<PLANCK>": "h",
    "<BOLTZ>": "k",  "<AVOG>": "Nₐ",

    # Brackets
    "<LFLOOR>": "⌊", "<RFLOOR>": "⌋",
    "<LCEIL>":  "⌈", "<RCEIL>":  "⌉",
    "<LANGLE>": "⟨", "<RANGLE>": "⟩",
}

# LaTeX render dictionary — for PDF/HTML MathJax output
LATEX_MAP: Dict[str, str] = {
    "<ALPHA>":   r"\alpha",    "<BETA>":    r"\beta",
    "<GAMMA>":   r"\gamma",    "<DELTA>":   r"\delta",
    "<PI>":      r"\pi",       "<SIGMA>":   r"\sigma",
    "<OMEGA>":   r"\omega",    "<THETA>":   r"\theta",
    "<LAMBDA>":  r"\lambda",   "<MU>":      r"\mu",
    "<INT>":     r"\int",      "<SUM>":     r"\sum",
    "<PROD>":    r"\prod",     "<PARTIAL>": r"\partial",
    "<NABLA>":   r"\nabla",    "<INF>":     r"\infty",
    "<SQRT>":    r"\sqrt",     "<LEQ>":     r"\leq",
    "<GEQ>":     r"\geq",      "<NEQ>":     r"\neq",
    "<APPROX>":  r"\approx",   "<EQUIV>":   r"\equiv",
    "<TIMES>":   r"\times",    "<FORALL>":  r"\forall",
    "<EXISTS>":  r"\exists",   "<IN>":      r"\in",
    "<IMPLIES>": r"\Rightarrow", "<IFF>":   r"\Leftrightarrow",
}


def token_to_unicode(token: str) -> str:
    """Convert a canonical token to its Unicode character."""
    return UNICODE_MAP.get(token, token)


def token_to_latex(token: str) -> str:
    """Convert a canonical token to its LaTeX command."""
    return LATEX_MAP.get(token, token)
