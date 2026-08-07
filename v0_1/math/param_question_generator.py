"""
AION Parameterized Mathematical Question Generator
===================================================
Generates fresh numerical problems from MathObject templates.
The LLM writes the question narrative.
This engine generates the numbers and verifies the solution.

Key principle: Numbers come from THIS engine. Never from the LLM.
"""

import random
import math
import sympy as sp
from dataclasses import dataclass, field
from typing import Optional, Any, List, Dict
from .math_object import MathObject, Variable
from .tokens import MathType


@dataclass
class NumericalProblem:
    """A fully parameterized mathematical problem ready for examination."""
    math_type:      str
    question_text:  str           # narrative question
    given_values:   Dict          # parameter → value with units
    expected_answer: Any          # computed answer
    solution_steps: List[str]     # step-by-step working
    marks:          int
    bloom_level:    int
    units:          str = ""
    latex_question: str = ""


class ParameterizedQuestionGenerator:
    """
    Generates fresh numerical problems from MathObject templates.
    Works for all VTU engineering subjects.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)

    def generate(
        self,
        math_obj:   MathObject,
        marks:      int = 10,
        bloom:      int = 3,
        subject:    str = "",
    ) -> Optional[NumericalProblem]:
        """
        Generate a parameterized problem from a MathObject template.
        Returns None if the type is not supported.
        """
        generators = {
            MathType.INTEGRAL:     self._gen_integral,
            MathType.DERIVATIVE:   self._gen_derivative,
            MathType.SUMMATION:    self._gen_summation,
            MathType.LIMIT:        self._gen_limit,
            MathType.EQUATION:     self._gen_equation,
            MathType.MATRIX:       self._gen_matrix,
            MathType.FORMULA:      self._gen_named_formula,
            MathType.LAPLACE:      self._gen_laplace,
            MathType.Z_TRANSFORM:  self._gen_z_transform,
            MathType.FOURIER:      self._gen_fourier,
        }

        gen_fn = generators.get(math_obj.math_type)
        if gen_fn:
            return gen_fn(math_obj, marks, bloom)

        # Fallback for named formulas
        if math_obj.named_as:
            return self._gen_named_formula(math_obj, marks, bloom)

        return None

    # ── Integral ──────────────────────────────────────────────────────────────

    def _gen_integral(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        x     = sp.Symbol('x')
        power = random.randint(1, 4)
        coeff = random.randint(1, 5)
        lower = random.randint(0, 3)
        upper = lower + random.randint(1, 5)

        integrand  = coeff * x**power
        result     = sp.integrate(integrand, (x, lower, upper))
        result_val = float(result)

        expr_str   = f"{coeff}x^{power}" if coeff > 1 else f"x^{power}"

        split_a    = marks * 6 // 10
        split_b    = marks - split_a

        steps = [
            f"Apply the power rule: ∫x^n dx = x^(n+1)/(n+1)",
            f"∫{coeff}x^{power} dx = {coeff}x^{power+1}/{power+1}",
            f"Evaluate from {lower} to {upper}:",
            f"= [{coeff}/{power+1} × ({upper})^{power+1}] - [{coeff}/{power+1} × ({lower})^{power+1}]",
            f"= {result_val:.4f}",
        ]

        return NumericalProblem(
            math_type   = "integral",
            question_text = (
                f"Evaluate the definite integral ∫_{lower}^{upper} {expr_str} dx.\n\n"
                f"({split_a} marks) Show all steps of integration using the power rule.\n"
                f"({split_b} marks) Verify your result by differentiating the antiderivative."
            ),
            given_values  = {
                "integrand": f"{coeff}x^{power}",
                "lower_limit": lower,
                "upper_limit": upper,
            },
            expected_answer = round(result_val, 4),
            solution_steps  = steps,
            marks           = marks,
            bloom_level     = bloom,
            latex_question  = (
                f"\\int_{{{lower}}}^{{{upper}}} {coeff}x^{{{power}}} \\, dx"
            ),
        )

    # ── Derivative ────────────────────────────────────────────────────────────

    def _gen_derivative(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        x      = sp.Symbol('x')
        power  = random.randint(2, 5)
        coeff  = random.randint(2, 8)
        offset = random.randint(1, 6)
        sign   = random.choice([1, -1])

        func   = coeff * x**power + sign * offset * x
        deriv  = sp.diff(func, x)
        func_str  = f"{coeff}x^{power} {'+' if sign>0 else '-'} {offset}x"
        deriv_str = str(deriv).replace("**", "^").replace("*", "")

        split_a = marks * 6 // 10
        split_b = marks - split_a

        steps = [
            f"Apply the power rule: d/dx(x^n) = nx^(n-1)",
            f"d/dx({coeff}x^{power}) = {coeff*power}x^{power-1}",
            f"d/dx({sign*offset}x) = {sign*offset}",
            f"Therefore: d/dx({func_str}) = {deriv_str}",
        ]

        return NumericalProblem(
            math_type     = "derivative",
            question_text = (
                f"Find the derivative of f(x) = {func_str}.\n\n"
                f"({split_a} marks) Differentiate using the power rule, "
                f"showing each step clearly.\n"
                f"({split_b} marks) Find the value of f'(x) at x = 2 and "
                f"interpret its geometric meaning."
            ),
            given_values    = {"function": func_str},
            expected_answer = deriv_str,
            solution_steps  = steps,
            marks           = marks,
            bloom_level     = bloom,
            latex_question  = f"f(x) = {coeff}x^{{{power}}} + {sign*offset}x",
        )

    # ── Summation ─────────────────────────────────────────────────────────────

    def _gen_summation(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        i      = sp.Symbol('i')
        upper  = random.randint(4, 10)
        coeff  = random.randint(1, 4)

        series_type = random.choice(["arithmetic", "geometric", "squares"])

        if series_type == "arithmetic":
            expr    = coeff * i
            result  = sum(coeff * k for k in range(1, upper + 1))
            formula = f"∑(i=1 to {upper}) {coeff}i"
        elif series_type == "geometric":
            r      = random.randint(2, 4)
            expr   = r**i
            result = sum(r**k for k in range(1, upper + 1))
            formula = f"∑(i=1 to {upper}) {r}^i"
        else:
            expr   = i**2
            result = sum(k**2 for k in range(1, upper + 1))
            formula = f"∑(i=1 to {upper}) i²"

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalProblem(
            math_type     = "summation",
            question_text = (
                f"Evaluate the summation: {formula}.\n\n"
                f"({split_a} marks) Compute the sum term by term and verify "
                f"using the standard formula.\n"
                f"({split_b} marks) Prove the standard formula used above "
                f"by mathematical induction."
            ),
            given_values    = {"upper_limit": upper, "type": series_type},
            expected_answer = result,
            solution_steps  = [
                f"Sum terms from 1 to {upper}",
                f"Sum = {result}",
            ],
            marks         = marks,
            bloom_level   = bloom,
        )

    # ── Limit ─────────────────────────────────────────────────────────────────

    def _gen_limit(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        x       = sp.Symbol('x')
        choices = [
            (x**2 - 4) / (x - 2),       # L'Hopital type, limit=4
            sp.sin(x) / x,               # classic, limit=1
            (sp.exp(x) - 1) / x,         # limit=1
            (x**3 - 1) / (x - 1),        # limit=3
        ]

        func, approach, result_val = choices[0], 2, 4.0
        try:
            chosen = random.choice([(choices[0], 2, 4.0), (choices[3], 1, 3.0)])
            func, approach, result_val = chosen
        except Exception:
            pass

        func_str = str(func).replace("**", "^").replace("*", "")
        split_a  = marks * 6 // 10
        split_b  = marks - split_a

        return NumericalProblem(
            math_type     = "limit",
            question_text = (
                f"Evaluate the limit lim_(x->{approach}) ({func_str}).\n\n"
                f"({split_a} marks) Evaluate using algebraic simplification or L'Hopital's rule.\n"
                f"({split_b} marks) Explain why direct substitution fails at x = {approach}."
            ),
            given_values    = {"expression": func_str, "approach": approach},
            expected_answer = result_val,
            solution_steps  = [
                f"Direct substitution gives 0/0 indeterminate form.",
                f"Factor or differentiate numerator and denominator.",
                f"Limit value = {result_val}",
            ],
            marks         = marks,
            bloom_level   = bloom,
        )

    # ── Equation ──────────────────────────────────────────────────────────────

    def _gen_equation(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        a = random.randint(1, 5)
        b = random.randint(-10, 10)
        c = random.randint(-10, 10)

        # ax^2 + bx + c = 0
        x = sp.Symbol('x')
        eq = a * x**2 + b * x + c
        solutions = sp.solve(eq, x)
        sol_strs  = [str(s) for s in solutions]

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalProblem(
            math_type     = "equation",
            question_text = (
                f"Solve the quadratic equation {a}x² + ({b})x + ({c}) = 0.\n\n"
                f"({split_a} marks) Use the quadratic formula to find all roots.\n"
                f"({split_b} marks) Determine the nature of the roots using the discriminant."
            ),
            given_values    = {"a": a, "b": b, "c": c},
            expected_answer = ", ".join(sol_strs),
            solution_steps  = [
                f"Discriminant Δ = b² - 4ac = ({b})² - 4({a})({c}) = {b**2 - 4*a*c}",
                f"Apply x = (-b ± √Δ) / 2a",
                f"Roots: {', '.join(sol_strs)}",
            ],
            marks         = marks,
            bloom_level   = bloom,
        )

    # ── Matrix ────────────────────────────────────────────────────────────────

    def _gen_matrix(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        a = random.randint(1, 5)
        b = random.randint(0, 4)
        c = random.randint(0, 4)
        d = random.randint(1, 5)

        det = a * d - b * c

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalProblem(
            math_type     = "matrix",
            question_text = (
                f"Given the matrix A = [[{a}, {b}], [{c}, {d}]]:\n\n"
                f"({split_a} marks) Find the determinant det(A).\n"
                f"({split_b} marks) Compute the inverse matrix A⁻¹ if det(A) ≠ 0."
            ),
            given_values    = {"matrix": [[a, b], [c, d]]},
            expected_answer = {"det": det, "invertible": det != 0},
            solution_steps  = [
                f"det(A) = ad - bc = ({a})({d}) - ({b})({c}) = {det}",
                f"A⁻¹ = (1/{det}) * [[{d}, -{b}], [-{c}, {a}]]" if det != 0 else "Matrix is singular (not invertible)."
            ],
            marks         = marks,
            bloom_level   = bloom,
        )

    # ── Named Formula ─────────────────────────────────────────────────────────

    def _gen_named_formula(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        name = obj.named_as or "Formula Problem"

        if "ohm" in name.lower() or "voltage" in name.lower():
            r = random.randint(10, 100)
            i = random.randint(1, 10)
            v = i * r
            return NumericalProblem(
                math_type     = "formula",
                question_text = (
                    f"A circuit resistor of R = {r} Ω has a current of I = {i} A flowing through it.\n\n"
                    f"({marks//2} marks) State Ohm's Law and calculate the voltage drop V across the resistor.\n"
                    f"({marks - marks//2} marks) Calculate the power dissipated in the resistor."
                ),
                given_values    = {"R": f"{r} Ω", "I": f"{i} A"},
                expected_answer = f"{v} V",
                solution_steps  = [f"V = I × R = {i} × {r} = {v} V", f"P = I² × R = {i**2 * r} W"],
                marks           = marks,
                bloom_level     = bloom,
                units           = "V",
            )

        # Default generic formula question
        return NumericalProblem(
            math_type     = "formula",
            question_text = f"State and explain {name}. Derive its expression from first principles.",
            given_values    = {},
            expected_answer = f"Theoretical derivation of {name}",
            solution_steps  = [f"Define terms for {name}", f"State governing equation", f"Derive step-by-step"],
            marks           = marks,
            bloom_level     = bloom,
        )

    # ── Laplace ───────────────────────────────────────────────────────────────

    def _gen_laplace(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        a = random.randint(1, 5)
        return NumericalProblem(
            math_type     = "laplace",
            question_text = (
                f"Find the Laplace transform of f(t) = e^{{-{a}t}} \\sin({a}t).\n\n"
                f"({marks//2} marks) Apply the frequency shifting theorem.\n"
                f"({marks - marks//2} marks) Verify using direct integration."
            ),
            given_values    = {"a": a},
            expected_answer = f"{a} / ((s + {a})^2 + {a}^2)",
            solution_steps  = [
                f"L{{sin({a}t)}} = {a}/(s² + {a}²)",
                f"Apply e^(-{a}t) shift s -> s + {a}",
                f"F(s) = {a} / ((s + {a})² + {a}²)",
            ],
            marks           = marks,
            bloom_level     = bloom,
        )

    # ── Z-Transform ───────────────────────────────────────────────────────────

    def _gen_z_transform(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        a = random.randint(2, 5)
        return NumericalProblem(
            math_type     = "z_transform",
            question_text = (
                f"Find the Z-transform of x[n] = ({a})^n u[n].\n\n"
                f"({marks//2} marks) Derive the expression using the summation definition.\n"
                f"({marks - marks//2} marks) Specify the Region of Convergence (ROC)."
            ),
            given_values    = {"a": a},
            expected_answer = f"z / (z - {a}), ROC: |z| > {a}",
            solution_steps  = [
                f"X(z) = ∑ (a)^n z^(-n) from n=0 to ∞",
                f"= ∑ (a z⁻¹)^n = 1 / (1 - a z⁻¹)",
                f"= z / (z - {a}) for |z| > {a}",
            ],
            marks           = marks,
            bloom_level     = bloom,
        )

    # ── Fourier ───────────────────────────────────────────────────────────────

    def _gen_fourier(
        self, obj: MathObject, marks: int, bloom: int
    ) -> NumericalProblem:
        a = random.randint(1, 4)
        return NumericalProblem(
            math_type     = "fourier",
            question_text = (
                f"Find the Fourier transform of the rectangular pulse signal x(t) = 1 for |t| ≤ {a}, and 0 otherwise.\n\n"
                f"({marks//2} marks) Compute the integral definition.\n"
                f"({marks - marks//2} marks) Express the answer in terms of the sinc function."
            ),
            given_values    = {"a": a},
            expected_answer = f"2 * sin(w * {a}) / w = {2*a} * sinc(w * {a} / pi)",
            solution_steps  = [
                f"X(ω) = ∫_{-{a}}^{{{a}}} 1 · e^(-jωt) dt",
                f"= [e^(-jωt) / (-jω)] from -{a} to {a}",
                f"= 2 sin({a}ω) / ω",
            ],
            marks           = marks,
            bloom_level     = bloom,
        )
