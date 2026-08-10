"""
AION Math Integrity Architecture — Math Symbol Registry
=========================================================
Controlled translation table between Unicode <-> LaTeX <-> MathML.
"""

from __future__ import annotations

from typing import Dict, Optional
from .contracts import MathSymbol


MATH_SYMBOL_REGISTRY: Dict[str, MathSymbol] = {
    # ── GREEK LOWERCASE ───────────────────────────────────────────────────────
    "alpha":   MathSymbol("alpha",   "α", "\\alpha",   "&alpha;",  "&alpha;",  "a",   "greek", ["mathematics"]),
    "beta":    MathSymbol("beta",    "β", "\\beta",    "&beta;",   "&beta;",   "b",   "greek", ["mathematics"]),
    "gamma":   MathSymbol("gamma",   "γ", "\\gamma",   "&gamma;",  "&gamma;",  "g",   "greek", ["mathematics", "physics"]),
    "delta":   MathSymbol("delta",   "δ", "\\delta",   "&delta;",  "&delta;",  "d",   "greek", ["mathematics", "physics"]),
    "epsilon": MathSymbol("epsilon", "ε", "\\epsilon", "&epsilon;","&epsilon;","e",   "greek", ["mathematics"]),
    "zeta":    MathSymbol("zeta",    "ζ", "\\zeta",    "&zeta;",   "&zeta;",   "z",   "greek", ["mathematics"]),
    "eta":     MathSymbol("eta",     "η", "\\eta",     "&eta;",    "&eta;",    "n",   "greek", ["mathematics", "thermodynamics"]),
    "theta":   MathSymbol("theta",   "θ", "\\theta",   "&theta;",  "&theta;",  "th",  "greek", ["mathematics", "electrical"]),
    "lambda":  MathSymbol("lambda",  "λ", "\\lambda",  "&lambda;", "&lambda;", "lam", "greek", ["mathematics", "physics"]),
    "mu":      MathSymbol("mu",      "μ", "\\mu",      "&mu;",     "&mu;",     "u",   "greek", ["mathematics", "electrical"]),
    "nu":      MathSymbol("nu",      "ν", "\\nu",      "&nu;",     "&nu;",     "v",   "greek", ["mathematics", "physics"]),
    "pi":      MathSymbol("pi",      "π", "\\pi",      "&pi;",     "&pi;",     "pi",  "greek", ["mathematics"]),
    "rho":     MathSymbol("rho",     "ρ", "\\rho",     "&rho;",    "&rho;",    "r",   "greek", ["mathematics", "electrical"]),
    "sigma":   MathSymbol("sigma",   "σ", "\\sigma",   "&sigma;",  "&sigma;",  "s",   "greek", ["mathematics", "statistics"]),
    "tau":     MathSymbol("tau",     "τ", "\\tau",     "&tau;",    "&tau;",    "t",   "greek", ["mathematics", "electrical"]),
    "phi":     MathSymbol("phi",     "φ", "\\phi",     "&phi;",    "&phi;",    "ph",  "greek", ["mathematics", "electrical"]),
    "psi":     MathSymbol("psi",     "ψ", "\\psi",     "&psi;",    "&psi;",    "ps",  "greek", ["mathematics", "physics"]),
    "omega":   MathSymbol("omega",   "ω", "\\omega",   "&omega;",  "&omega;",  "w",   "greek", ["mathematics", "electrical", "mechanics"]),

    # ── GREEK UPPERCASE ───────────────────────────────────────────────────────
    "Gamma":   MathSymbol("Gamma",   "Γ", "\\Gamma",   "&Gamma;",  "&Gamma;",  "G",   "greek", ["mathematics"]),
    "Delta":   MathSymbol("Delta",   "Δ", "\\Delta",   "&Delta;",  "&Delta;",  "D",   "greek", ["mathematics", "physics"]),
    "Theta":   MathSymbol("Theta",   "Θ", "\\Theta",   "&Theta;",  "&Theta;",  "TH",  "greek", ["mathematics"]),
    "Lambda":  MathSymbol("Lambda",  "Λ", "\\Lambda",  "&Lambda;", "&Lambda;", "LAM", "greek", ["mathematics"]),
    "Pi":      MathSymbol("Pi",      "Π", "\\Pi",      "&Pi;",     "&Pi;",     "PI",  "greek", ["mathematics"]),
    "Sigma":   MathSymbol("Sigma",   "Σ", "\\Sigma",   "&Sigma;",  "&Sigma;",  "S",   "greek", ["mathematics", "statistics"]),
    "Phi":     MathSymbol("Phi",     "Φ", "\\Phi",     "&Phi;",    "&Phi;",    "PH",  "greek", ["mathematics", "electrical"]),
    "Omega":   MathSymbol("Omega",   "Ω", "\\Omega",   "&Omega;",  "&Omega;",  "Ohm", "greek", ["mathematics", "electrical"]),

    # ── OPERATORS ─────────────────────────────────────────────────────────────
    "times":      MathSymbol("times",      "×", "\\times",      "&times;",  "&times;",  "*",   "operator", ["mathematics"]),
    "div":        MathSymbol("div",        "÷", "\\div",        "&divide;", "&divide;", "/",   "operator", ["mathematics"]),
    "plus_minus": MathSymbol("plus_minus", "±", "\\pm",         "&plusmn;", "&plusmn;", "+-",  "operator", ["mathematics"]),
    "cdot":       MathSymbol("cdot",       "·", "\\cdot",       "&middot;", "&middot;", ".",   "operator", ["mathematics"]),
    "partial":    MathSymbol("partial",    "∂", "\\partial",    "&part;",   "&part;",   "d",   "operator", ["mathematics", "physics"]),
    "nabla":      MathSymbol("nabla",      "∇", "\\nabla",      "&nabla;",  "&nabla;",  "del", "operator", ["mathematics", "physics"]),
    "sqrt":       MathSymbol("sqrt",       "√", "\\sqrt{}",     "&radic;",  "&radic;",  "sqrt","operator", ["mathematics"]),
    "inf":        MathSymbol("inf",        "∞", "\\infty",      "&infin;",  "&infin;",  "inf", "operator", ["mathematics"]),
    "integral":   MathSymbol("integral",   "∫", "\\int",        "&int;",    "&int;",    "int", "operator", ["mathematics"]),
    "sum":        MathSymbol("sum",        "∑", "\\sum",        "&sum;",    "&sum;",    "sum", "operator", ["mathematics"]),
    "product":    MathSymbol("product",    "∏", "\\prod",       "&prod;",   "&prod;",   "prod","operator", ["mathematics"]),

    # ── RELATIONS ─────────────────────────────────────────────────────────────
    "leq":          MathSymbol("leq",          "≤", "\\leq",    "&le;",     "&le;",     "<=",  "relation", ["mathematics"]),
    "geq":          MathSymbol("geq",          "≥", "\\geq",    "&ge;",     "&ge;",     ">=",  "relation", ["mathematics"]),
    "neq":          MathSymbol("neq",          "≠", "\\neq",    "&ne;",     "&ne;",     "!=",  "relation", ["mathematics"]),
    "approx":       MathSymbol("approx",       "≈", "\\approx", "&asymp;",  "&asymp;",  "~=",  "relation", ["mathematics"]),
    "equiv":        MathSymbol("equiv",        "≡", "\\equiv",  "&equiv;",  "&equiv;",  "===", "relation", ["mathematics"]),
    "proportional": MathSymbol("proportional", "∝", "\\propto", "&prop;",   "&prop;",   "~",   "relation", ["mathematics", "physics"]),
    "in_set":       MathSymbol("in_set",       "∈", "\\in",     "&isin;",   "&isin;",   "in",  "relation", ["mathematics"]),

    # ── DOMAIN-SPECIFIC ───────────────────────────────────────────────────────
    "ohm":    MathSymbol("ohm",    "Ω", "\\Omega",   "&#937;", "&#937;", "Ohm", "electrical", ["electrical", "electronics"]),
    "angle":  MathSymbol("angle",  "∠", "\\angle",   "&ang;",  "&ang;",  "<|",  "geometry",   ["mathematics", "geometry", "electrical"]),
    "degree": MathSymbol("degree", "°", "^{\\circ}", "&deg;",  "&deg;",  "deg", "geometry",   ["mathematics", "geometry"]),
    "micro":  MathSymbol("micro",  "μ", "\\mu",      "&micro;","&micro;","u",   "prefix",     ["electrical", "physics"]),

    # ── ARROWS ────────────────────────────────────────────────────────────────
    "rightarrow":     MathSymbol("rightarrow",     "→", "\\rightarrow",     "&rarr;", "&rarr;", "->",  "arrow", ["mathematics"]),
    "Rightarrow":     MathSymbol("Rightarrow",     "⇒", "\\Rightarrow",     "&rArr;", "&rArr;", "=>",  "arrow", ["mathematics", "logic"]),
    "leftarrow":      MathSymbol("leftarrow",      "←", "\\leftarrow",      "&larr;", "&larr;", "<-",  "arrow", ["mathematics"]),
    "leftrightarrow": MathSymbol("leftrightarrow", "↔", "\\leftrightarrow", "&harr;", "&harr;", "<->", "arrow", ["mathematics"]),
}


def unicode_to_latex(char: str) -> Optional[str]:
    """Translate a single Unicode character to its LaTeX command."""
    for symbol in MATH_SYMBOL_REGISTRY.values():
        if symbol.unicode == char:
            return symbol.latex
    return None


def latex_to_unicode(latex_cmd: str) -> Optional[str]:
    """Translate a LaTeX command to its Unicode character."""
    key = latex_cmd.lstrip("\\").rstrip("{}")
    if key in MATH_SYMBOL_REGISTRY:
        return MATH_SYMBOL_REGISTRY[key].unicode
    return None
