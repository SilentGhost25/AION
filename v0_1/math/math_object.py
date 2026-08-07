"""
AION Math Object
================
Structured internal representation of a mathematical expression.
Replaces raw Unicode/LaTeX strings in the Knowledge Graph.

Every formula, equation, or expression in AION is stored as a MathObject.
The LLM reasons over the structured fields, not the raw symbols.
"""

import uuid
import json
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
from .tokens import MathType


@dataclass
class Variable:
    """A mathematical variable with optional constraints."""
    symbol:      str
    domain:      str   = "real"           # real, integer, positive, complex
    constraint:  str   = ""               # e.g. "non-zero", "> 0", "integer"
    description: str   = ""               # human-readable meaning


@dataclass
class MathObject:
    """
    Canonical structured representation of a mathematical expression.

    This is the single internal format for ALL mathematics in AION.
    Never store raw Unicode or LaTeX in the Knowledge Graph —
    always convert to MathObject first.
    """

    # Identity
    math_id:     str      = field(default_factory=lambda: str(uuid.uuid4())[:8])
    math_type:   MathType = MathType.EXPRESSION

    # Core content (canonical token form, not Unicode)
    canonical:   str      = ""    # e.g. "<INT lower=0 upper=1 var=x> x^2 </INT>"
    latex:       str      = ""    # LaTeX form for rendering
    description: str      = ""    # human-readable description

    # Structure
    variables:   List[Variable]    = field(default_factory=list)
    parameters:  Dict[str, Any]    = field(default_factory=dict)
    subparts:    List["MathObject"] = field(default_factory=list)

    # Subject context
    subject:     str      = ""    # e.g. "Engineering Mathematics"
    topic:       str      = ""    # e.g. "Integral Calculus"
    named_as:    str      = ""    # e.g. "Euler's Formula", "Ohm's Law"

    # Solution metadata
    solution:    str      = ""    # symbolic or numeric solution
    solution_steps: List[str] = field(default_factory=list)
    difficulty:  str      = "medium"  # easy / medium / hard

    # Parameterization
    is_template:     bool = False   # True if variables can be substituted
    param_constraints: Dict = field(default_factory=dict)

    # Source
    source_page:  int     = 0
    confidence:   float   = 0.9
    raw_text:     str     = ""    # original text before parsing

    def to_dict(self) -> dict:
        return {
            "math_id":     self.math_id,
            "math_type":   self.math_type.value,
            "canonical":   self.canonical,
            "latex":       self.latex,
            "description": self.description,
            "variables":   [
                {"symbol": v.symbol, "domain": v.domain,
                 "constraint": v.constraint, "description": v.description}
                for v in self.variables
            ],
            "parameters":  self.parameters,
            "subject":     self.subject,
            "topic":       self.topic,
            "named_as":    self.named_as,
            "solution":    self.solution,
            "is_template": self.is_template,
            "difficulty":  self.difficulty,
            "confidence":  self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_prompt_context(self) -> str:
        """
        Convert to a string suitable for injecting into an LLM prompt.
        The model reasons over this structured text, not raw symbols.
        """
        parts = [f"MATH TYPE: {self.math_type.value.upper()}"]

        if self.named_as:
            parts.append(f"FORMULA NAME: {self.named_as}")

        if self.description:
            parts.append(f"DESCRIPTION: {self.description}")

        if self.topic:
            parts.append(f"TOPIC: {self.topic}")

        if self.variables:
            var_desc = ", ".join(
                f"{v.symbol} ({v.description or v.domain})"
                for v in self.variables
            )
            parts.append(f"VARIABLES: {var_desc}")

        if self.parameters:
            params = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
            parts.append(f"PARAMETERS: {params}")

        if self.solution:
            parts.append(f"KNOWN SOLUTION: {self.solution}")

        if self.solution_steps:
            parts.append("SOLUTION APPROACH:")
            for i, step in enumerate(self.solution_steps, 1):
                parts.append(f"  {i}. {step}")

        if self.canonical:
            parts.append(f"EXPRESSION: {self.canonical}")

        return "\n".join(parts)
