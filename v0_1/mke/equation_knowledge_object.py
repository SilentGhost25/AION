"""
AION Equation Knowledge Object (EKO)
=====================================
The complete semantic wrapper around a mathematical equation.
Stores not just the expression tree but everything AION needs
to understand, teach, question, and solve the equation.

This is the atom of the Mathematical Knowledge Engine.
"""

import uuid
import json
import random
from dataclasses import dataclass, field
from typing import Optional, Any, Callable, List, Tuple, Dict
from .expression_tree import ExprNode, NodeType


@dataclass
class VariableSpec:
    """Specification for a variable in an equation."""
    symbol:      str
    name:        str           = ""
    unit:        str           = ""
    domain:      str           = "real"          # real, positive, integer, complex
    constraint:  str           = ""              # "non-zero", "> 0", etc.
    typical_range: Tuple[float, float] = (1.0, 100.0)
    description: str           = ""

    def generate_value(self) -> float:
        """Generate a random valid value for this variable."""
        lo, hi = self.typical_range
        if self.domain == "integer":
            return float(random.randint(int(lo), int(hi)))
        if self.domain == "positive":
            val = random.uniform(max(0.1, lo), hi)
            return round(val, 2)
        return round(random.uniform(lo, hi), 2)


@dataclass
class QuestionBlueprint:
    """A template for generating questions from this equation."""
    bloom_level:  int
    bloom_name:   str
    verb:         str
    template:     str          # question text with {variable} placeholders
    requires:     List[str]    = field(default_factory=list)   # prerequisite concepts
    marks_range:  Tuple[int, int] = (5, 20)

    def render(self, values: dict, equation_str: str) -> str:
        """Fill in the template with actual values."""
        try:
            ctx = {**values, "equation": equation_str}
            return self.template.format(**ctx)
        except KeyError:
            return self.template


@dataclass
class DerivationStep:
    """A single step in the derivation of an equation."""
    step_number: int
    description: str
    expression:  Optional[ExprNode] = None
    justification: str = ""         # theorem, rule, or axiom used


@dataclass
class EquationKnowledgeObject:
    """
    The complete knowledge representation of a mathematical equation.

    This is the fundamental unit of the Mathematical Knowledge Engine.
    Every equation in AION is stored as an EKO — never as a string.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    eko_id:       str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name:         str = ""               # "Quadratic Formula", "Ohm's Law"
    aliases:      List[str] = field(default_factory=list)

    # ── Mathematical Core ─────────────────────────────────────────────────────
    tree:         Optional[ExprNode] = None    # the expression tree
    sympy_expr:   Any = None                   # sympy representation for solving
    canonical:    str = ""                     # canonical string form

    # ── Variables ─────────────────────────────────────────────────────────────
    variables:    List[VariableSpec] = field(default_factory=list)
    constants:    List[VariableSpec] = field(default_factory=list)
    unknowns:     List[str] = field(default_factory=list)   # what is being solved for

    # ── Context ───────────────────────────────────────────────────────────────
    subject:      str = ""
    topic:        str = ""
    subtopic:     str = ""
    domains:      List[str] = field(default_factory=list)   # where it applies
    prerequisites: List[str] = field(default_factory=list) # concept names

    # ── Question Generation ───────────────────────────────────────────────────
    blueprints:   List[QuestionBlueprint] = field(default_factory=list)
    param_generator: Optional[Callable] = None    # function that generates values
    auto_solver:  Optional[Callable] = None       # function that solves for unknowns

    # ── Knowledge ─────────────────────────────────────────────────────────────
    derivation:   List[DerivationStep] = field(default_factory=list)
    special_cases: List[dict] = field(default_factory=list)
    limitations:  List[str] = field(default_factory=list)
    applications: List[str] = field(default_factory=list)
    related_ecos: List[str] = field(default_factory=list)   # other EKO ids

    # ── Quality ───────────────────────────────────────────────────────────────
    difficulty:   str = "medium"     # easy / medium / hard
    confidence:   float = 1.0
    verified:     bool = False
    source:       str = ""

    # ── Rendering Cache ───────────────────────────────────────────────────────
    _latex_cache: str = field(default="", repr=False)
    _unicode_cache: str = field(default="", repr=False)

    def generate_values(self) -> dict:
        """
        Generate a fresh set of valid numerical values for all variables.
        This is the parameterized generation — numbers come from here, never from LLM.
        """
        values = {}
        for spec in self.variables:
            values[spec.symbol] = spec.generate_value()
        for spec in self.constants:
            values[spec.symbol] = spec.typical_range[0]   # use defined constant value
        return values

    def generate_problem(
        self,
        bloom:   int = 3,
        marks:   int = 10,
        unknown: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Generate a complete numerical problem from this equation.

        Returns:
            {
                "question": str,
                "given": dict,
                "find": str,
                "answer": value,
                "steps": list[str],
                "marks": int,
                "bloom": int
            }
        """
        values  = self.generate_values()
        find_var = unknown or (self.unknowns[0] if self.unknowns else None)

        if not find_var:
            return None

        # Solve for the unknown
        answer, steps = self._solve(values, find_var)

        # Select appropriate blueprint for bloom level
        blueprint = self._select_blueprint(bloom)

        # Format given values for question text
        given_text = self._format_given(values, find_var)

        if blueprint:
            question = blueprint.render(
                {**values, "find": find_var, "name": self.name},
                self.canonical,
            )
        else:
            question = self._default_question(given_text, find_var, marks)

        return {
            "question": question,
            "given":    {k: v for k, v in values.items() if k != find_var},
            "find":     find_var,
            "answer":   answer,
            "steps":    steps,
            "marks":    marks,
            "bloom":    bloom,
            "eko_id":   self.eko_id,
            "equation": self.name or self.canonical,
        }

    def explain(self) -> str:
        """
        Generate a structured explanation of this equation.
        Used when the LLM needs to write about this concept.
        """
        parts = [f"EQUATION: {self.name or self.canonical}"]

        if self.topic:
            parts.append(f"TOPIC: {self.topic}")

        if self.variables:
            parts.append("VARIABLES:")
            for v in self.variables:
                unit_str = f" [{v.unit}]" if v.unit else ""
                parts.append(
                    f"  {v.symbol}{unit_str} — {v.name or v.description}"
                    + (f" ({v.constraint})" if v.constraint else "")
                )

        if self.applications:
            parts.append(f"APPLICATIONS: {', '.join(self.applications)}")

        if self.limitations:
            parts.append(f"LIMITATIONS: {', '.join(self.limitations)}")

        if self.special_cases:
            parts.append("SPECIAL CASES:")
            for sc in self.special_cases:
                parts.append(f"  When {sc.get('condition', '?')}: {sc.get('result', '?')}")

        if self.derivation:
            parts.append("DERIVATION OUTLINE:")
            for step in self.derivation[:3]:
                parts.append(f"  {step.step_number}. {step.description}")

        return "\n".join(parts)

    def to_dict(self) -> dict:
        return {
            "eko_id":       self.eko_id,
            "name":         self.name,
            "canonical":    self.canonical,
            "subject":      self.subject,
            "topic":        self.topic,
            "difficulty":   self.difficulty,
            "variables":    [
                {"symbol": v.symbol, "name": v.name, "unit": v.unit,
                 "domain": v.domain, "constraint": v.constraint}
                for v in self.variables
            ],
            "applications": self.applications,
            "limitations":  self.limitations,
            "unknowns":     self.unknowns,
            "domains":      self.domains,
            "tree":         self.tree.to_dict() if self.tree else None,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _solve(
        self, values: dict, unknown: str
    ) -> Tuple[Any, List[str]]:
        """Solve the equation for the unknown given all other values."""
        if self.auto_solver:
            try:
                return self.auto_solver(values, unknown)
            except Exception:
                pass

        # Try sympy
        if self.sympy_expr is not None:
            try:
                import sympy as sp
                sym   = sp.Symbol(unknown)
                known = {k: v for k, v in values.items() if k != unknown}
                expr  = self.sympy_expr.subs(known)
                sol   = sp.solve(expr, sym)
                if sol:
                    val = float(sol[0]) if getattr(sol[0], 'is_real', True) else str(sol[0])
                    return val, [f"Solved: {unknown} = {val}"]
            except Exception:
                pass

        return "See solution", [f"Apply {self.name or self.canonical}",
                                "Substitute given values", f"Solve for {unknown}"]

    def _select_blueprint(self, bloom: int) -> Optional[QuestionBlueprint]:
        matches = [b for b in self.blueprints if b.bloom_level == bloom]
        if matches:
            return random.choice(matches)
        if self.blueprints:
            return min(self.blueprints, key=lambda b: abs(b.bloom_level - bloom))
        return None

    def _format_given(self, values: dict, unknown: str) -> str:
        parts = []
        for spec in self.variables:
            if spec.symbol == unknown:
                continue
            val  = values.get(spec.symbol, "?")
            unit = f" {spec.unit}" if spec.unit else ""
            name = spec.name or spec.symbol
            parts.append(f"{name} ({spec.symbol}) = {val}{unit}")
        return ", ".join(parts)

    def _default_question(
        self, given_text: str, unknown: str, marks: int
    ) -> str:
        spec = next(
            (v for v in self.variables if v.symbol == unknown), None
        )
        find_name = spec.name if spec else unknown
        split_a   = marks * 6 // 10
        split_b   = marks - split_a

        return (
            f"Using {self.name or 'the given equation'}, given: {given_text}.\n\n"
            f"({split_a} marks) Calculate {find_name} ({unknown}), "
            f"showing all steps with appropriate units.\n"
            f"({split_b} marks) Explain the physical significance of "
            f"{find_name} in the context of {self.topic or self.subject}."
        )
