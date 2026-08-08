"""
AION Mathematical Knowledge Engine (MKE)
=========================================
Central registry of all Equation Knowledge Objects.
Provides search, generation, explanation, and rendering.

This is the mathematical equivalent of the Academic Knowledge Graph.
"""

import json
import sympy as sp
from pathlib import Path
from typing import Optional, List, Dict
from .expression_tree import (
    ExprNode, NodeType,
    num, var, const, op, fn,
    integral, derivative, limit, summation, equation, fraction, sqrt,
)
from .equation_knowledge_object import (
    EquationKnowledgeObject, VariableSpec,
    QuestionBlueprint, DerivationStep
)
from .expression_parser import ExpressionParser
from .renderer import TreeRenderer


class MathematicalKnowledgeEngine:
    """
    Central registry and query engine for all mathematical equations.

    Usage:
        mke = MathematicalKnowledgeEngine()
        eko = mke.get("Ohm's Law")
        problem = mke.generate_problem(eko, marks=10, bloom=3)
        print(problem["question"])
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self.registry:  Dict[str, EquationKnowledgeObject] = {}
        self.by_name:   Dict[str, str] = {}      # name → eko_id
        self.by_topic:  Dict[str, List[str]] = {}
        self.by_subject: Dict[str, List[str]] = {}
        self.parser     = ExpressionParser()
        self.renderer   = TreeRenderer()
        self.persist    = persist_path

        # Load built-in equations
        self._load_builtin_equations()

        # Load persisted equations if available
        if persist_path and persist_path.exists():
            self._load_from_disk(persist_path)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, eko: EquationKnowledgeObject):
        """Add an EKO to the registry."""
        self.registry[eko.eko_id] = eko

        # Index by name and aliases
        for name in [eko.name] + eko.aliases:
            if name:
                self.by_name[name.lower()] = eko.eko_id

        # Index by topic and subject
        if eko.topic:
            self.by_topic.setdefault(eko.topic, []).append(eko.eko_id)
        if eko.subject:
            self.by_subject.setdefault(eko.subject, []).append(eko.eko_id)

    # ── Query ─────────────────────────────────────────────────────────────────

    def get(self, name: str) -> Optional[EquationKnowledgeObject]:
        """Look up an EKO by name."""
        eko_id = self.by_name.get(name.lower())
        return self.registry.get(eko_id) if eko_id else None

    def get_by_topic(self, topic: str) -> List[EquationKnowledgeObject]:
        ids = self.by_topic.get(topic, [])
        return [self.registry[i] for i in ids if i in self.registry]

    def get_by_subject(self, subject: str) -> List[EquationKnowledgeObject]:
        ids = self.by_subject.get(subject, [])
        return [self.registry[i] for i in ids if i in self.registry]

    def search(self, query: str) -> List[EquationKnowledgeObject]:
        """Full-text search across all EKOs."""
        q = query.lower()
        results = []
        for eko in self.registry.values():
            text = " ".join([
                eko.name, eko.topic, eko.subject, eko.canonical,
                " ".join(eko.applications), " ".join(eko.domains),
            ]).lower()
            if q in text:
                results.append(eko)
        return results

    def detect_in_text(self, text: str) -> List[EquationKnowledgeObject]:
        """Find known equations mentioned in a block of text."""
        text_lower = text.lower()
        found = []
        for name, eko_id in self.by_name.items():
            if name in text_lower:
                eko = self.registry.get(eko_id)
                if eko and eko not in found:
                    found.append(eko)

        # Also parse any raw expressions
        regions = self.parser.detect_math_regions(text)
        for expr_str, _, _ in regions:
            tree = self.parser.parse(expr_str)
            if tree:
                eko = self._tree_to_eko(tree, expr_str)
                found.append(eko)

        return found

    # ── Generation ────────────────────────────────────────────────────────────

    def generate_problem(
        self,
        eko:    EquationKnowledgeObject,
        marks:  int = 10,
        bloom:  int = 3,
        unknown: Optional[str] = None,
    ) -> Optional[dict]:
        """Generate a fresh numerical problem from an EKO."""
        return eko.generate_problem(marks=marks, bloom=bloom, unknown=unknown)

    def generate_conceptual_question(
        self,
        eko:   EquationKnowledgeObject,
        bloom: int = 2,
        marks: int = 10,
    ) -> str:
        """Generate a conceptual (non-numerical) question from an EKO."""
        blueprints = {
            1: f"State {eko.name} and define all variables with their units.",
            2: f"Explain {eko.name} ({eko.canonical}) and describe its significance in {eko.topic or eko.subject}.",
            3: f"Illustrate {eko.name} with a suitable example, showing all steps of calculation.",
            4: f"Analyze the limitations of {eko.name} and compare it with at least one alternative approach.",
            5: f"Evaluate the applicability of {eko.name} in {', '.join(eko.applications[:2]) or eko.topic}. Justify with examples.",
            6: f"Design a system that applies {eko.name} to solve a real-world problem in {eko.subject}. Derive all necessary parameters.",
        }

        split_a = marks * 6 // 10
        split_b = marks - split_a
        base    = blueprints.get(bloom, blueprints[2])

        if eko.derivation:
            derive_q = f"Derive {eko.name} from first principles."
            return (
                f"({split_a} marks) {base}\n"
                f"({split_b} marks) {derive_q}"
            )

        return (
            f"({split_a} marks) {base}\n"
            f"({split_b} marks) Explain how {eko.name} applies in {eko.domains[0] if eko.domains else eko.topic}."
        )

    # ── Rendering ─────────────────────────────────────────────────────────────

    def render_unicode(self, eko: EquationKnowledgeObject) -> str:
        if eko.tree:
            return self.renderer.to_unicode(eko.tree)
        return eko.canonical

    def render_latex(self, eko: EquationKnowledgeObject) -> str:
        if eko.tree:
            return self.renderer.to_latex(eko.tree)
        return eko.canonical

    # ── Built-in Equations ────────────────────────────────────────────────────

    def _load_builtin_equations(self):
        """Load the built-in equation knowledge base."""
        equations_to_register = [
            self._make_ohms_law(),
            self._make_power_law(),
            self._make_kirchhoff_voltage(),
            self._make_carnot(),
            self._make_first_law(),
            self._make_quadratic_formula(),
            self._make_power_rule(),
            self._make_integration_power_rule(),
            self._make_eulers_formula(),
            self._make_laplace_definition(),
            self._make_fourier_definition(),
            self._make_z_transform(),
            self._make_fspl(),
            self._make_eirp(),
        ]
        for eko in equations_to_register:
            self.register(eko)

    def _make_ohms_law(self) -> EquationKnowledgeObject:
        tree = equation(var("V"), op("*", var("I"), var("R")))
        V_sym, I_sym, R_sym = sp.symbols('V I R', positive=True)
        return EquationKnowledgeObject(
            name="Ohm's Law",
            aliases=["V=IR", "Ohm law"],
            tree=tree,
            sympy_expr=sp.Eq(V_sym, I_sym * R_sym),
            canonical="V = I * R",
            variables=[
                VariableSpec("V", "Voltage", "V", "positive", "> 0", (1.0, 240.0), "Electrical potential difference"),
                VariableSpec("I", "Current", "A", "positive", "> 0", (0.1, 20.0), "Electric current"),
                VariableSpec("R", "Resistance", "Ω", "positive", "> 0", (1.0, 1000.0), "Electrical resistance"),
            ],
            unknowns=["V", "I", "R"],
            subject="Basic Electrical Engineering",
            topic="DC Circuits",
            applications=["Circuit analysis", "Power distribution", "Electronic design"],
            domains=["Electrical Engineering", "Physics"],
            blueprints=[
                QuestionBlueprint(2, "Understand", "Explain", "State Ohm's Law ({equation}) and define all variables."),
                QuestionBlueprint(3, "Apply", "Calculate", "A circuit resistor R = {R} Ω carries current I = {I} A. Calculate the voltage V across it."),
            ]
        )

    def _make_power_law(self) -> EquationKnowledgeObject:
        tree = equation(var("P"), op("*", var("V"), var("I")))
        P_sym, V_sym, I_sym = sp.symbols('P V I', positive=True)
        return EquationKnowledgeObject(
            name="Electrical Power Law",
            aliases=["P=VI", "Electric Power"],
            tree=tree,
            sympy_expr=sp.Eq(P_sym, V_sym * I_sym),
            canonical="P = V * I",
            variables=[
                VariableSpec("P", "Power", "W", "positive", "> 0", (10.0, 5000.0), "Electrical power"),
                VariableSpec("V", "Voltage", "V", "positive", "> 0", (10.0, 240.0), "Voltage"),
                VariableSpec("I", "Current", "A", "positive", "> 0", (0.5, 50.0), "Current"),
            ],
            unknowns=["P", "V", "I"],
            subject="Basic Electrical Engineering",
            topic="DC Circuits",
            applications=["Energy auditing", "Circuit rating", "Power electronics"],
        )

    def _make_kirchhoff_voltage(self) -> EquationKnowledgeObject:
        tree = equation(summation(var("V_k"), "k", num(1), var("N")), num(0))
        return EquationKnowledgeObject(
            name="Kirchhoff's Voltage Law",
            aliases=["KVL", "Kirchhoff Voltage Law"],
            tree=tree,
            canonical="∑_{k=1}^{N} V_k = 0",
            subject="Basic Electrical Engineering",
            topic="Circuit Theorems",
            applications=["Loop analysis", "Mesh analysis"],
        )

    def _make_carnot(self) -> EquationKnowledgeObject:
        tree = equation(var("eta"), op("-", num(1), fraction(var("T_L"), var("T_H"))))
        return EquationKnowledgeObject(
            name="Carnot Efficiency Formula",
            aliases=["Carnot Efficiency", "eta=1-TL/TH"],
            tree=tree,
            canonical="η = 1 - (T_L / T_H)",
            variables=[
                VariableSpec("eta", "Efficiency", "", "positive", "0 < eta < 1", (0.2, 0.8), "Thermal efficiency"),
                VariableSpec("T_L", "Sink Temperature", "K", "positive", "> 0", (273.0, 350.0), "Cold reservoir temperature"),
                VariableSpec("T_H", "Source Temperature", "K", "positive", "> T_L", (400.0, 1200.0), "Hot reservoir temperature"),
            ],
            unknowns=["eta", "T_L", "T_H"],
            subject="Thermodynamics",
            topic="Second Law of Thermodynamics",
            applications=["Heat engines", "Refrigeration cycles"],
        )

    def _make_first_law(self) -> EquationKnowledgeObject:
        tree = equation(var("dQ"), op("+", var("dU"), var("dW")))
        return EquationKnowledgeObject(
            name="First Law of Thermodynamics",
            aliases=["First Law", "dQ = dU + dW"],
            tree=tree,
            canonical="dQ = dU + dW",
            subject="Thermodynamics",
            topic="Energy Conservation",
        )

    def _make_quadratic_formula(self) -> EquationKnowledgeObject:
        x_sym, a_sym, b_sym, c_sym = sp.symbols('x a b c')
        tree = equation(var("x"), fraction(op("+", op("neg", var("b")), sqrt(op("-", op("^", var("b"), num(2)), op("*", num(4), op("*", var("a"), var("c")))))), op("*", num(2), var("a"))))
        return EquationKnowledgeObject(
            name="Quadratic Formula",
            aliases=["Quadratic equation", "roots formula"],
            tree=tree,
            sympy_expr=sp.Eq(a_sym * x_sym**2 + b_sym * x_sym + c_sym, 0),
            canonical="x = (-b ± √(b² - 4ac)) / (2a)",
            variables=[
                VariableSpec("a", "Coefficient a", "", "real", "≠ 0", (1.0, 10.0), "Leading coefficient"),
                VariableSpec("b", "Coefficient b", "", "real", "", (-10.0, 10.0), "Linear coefficient"),
                VariableSpec("c", "Constant c", "", "real", "", (-20.0, 20.0), "Constant term"),
            ],
            unknowns=["x"],
            subject="Engineering Mathematics",
            topic="Algebra",
        )

    def _make_power_rule(self) -> EquationKnowledgeObject:
        tree = equation(derivative(op("^", var("x"), var("n")), "x"), op("*", var("n"), op("^", var("x"), op("-", var("n"), num(1)))))
        return EquationKnowledgeObject(
            name="Power Rule for Differentiation",
            aliases=["d/dx x^n", "Power Rule"],
            tree=tree,
            canonical="d/dx(x^n) = n * x^(n-1)",
            subject="Engineering Mathematics",
            topic="Differential Calculus",
        )

    def _make_integration_power_rule(self) -> EquationKnowledgeObject:
        tree = equation(integral(op("^", var("x"), var("n")), "x"), fraction(op("^", var("x"), op("+", var("n"), num(1))), op("+", var("n"), num(1))))
        return EquationKnowledgeObject(
            name="Power Rule for Integration",
            aliases=["int x^n dx", "Integration Power Rule"],
            tree=tree,
            canonical="∫ x^n dx = x^(n+1) / (n+1)",
            subject="Engineering Mathematics",
            topic="Integral Calculus",
        )

    def _make_eulers_formula(self) -> EquationKnowledgeObject:
        tree = equation(op("^", const("e"), op("*", const("i"), var("theta"))), op("+", fn("cos", var("theta")), op("*", const("i"), fn("sin", var("theta")))))
        return EquationKnowledgeObject(
            name="Euler's Formula",
            aliases=["e^(i theta)", "Euler identity"],
            tree=tree,
            canonical="e^(i·θ) = cos(θ) + i·sin(θ)",
            subject="Engineering Mathematics",
            topic="Complex Analysis",
        )

    def _make_laplace_definition(self) -> EquationKnowledgeObject:
        tree = equation(fn("F", var("s")), integral(op("*", fn("f", var("t")), op("^", const("e"), op("*", op("neg", var("s")), var("t")))), "t", num(0), const("inf")))
        return EquationKnowledgeObject(
            name="Laplace Transform Definition",
            aliases=["Laplace Transform", "F(s)"],
            tree=tree,
            canonical="F(s) = ∫_0^∞ f(t)·e^(-st) dt",
            subject="Signals and Systems",
            topic="Laplace Transforms",
        )

    def _make_fourier_definition(self) -> EquationKnowledgeObject:
        tree = equation(fn("F", var("omega")), integral(op("*", fn("f", var("t")), op("^", const("e"), op("neg", op("*", const("i"), op("*", var("omega"), var("t")))))), "t", op("neg", const("inf")), const("inf")))
        return EquationKnowledgeObject(
            name="Fourier Transform Definition",
            aliases=["Fourier Transform", "F(omega)"],
            tree=tree,
            canonical="F(ω) = ∫_{-∞}^{∞} f(t)·e^(-jωt) dt",
            subject="Signals and Systems",
            topic="Fourier Analysis",
        )

    def _make_z_transform(self) -> EquationKnowledgeObject:
        tree = equation(fn("X", var("z")), summation(op("*", fn("x", var("n")), op("^", var("z"), op("neg", var("n")))), "n", num(0), const("inf")))
        return EquationKnowledgeObject(
            name="Z-Transform Definition",
            aliases=["Z Transform", "X(z)"],
            tree=tree,
            canonical="X(z) = ∑_{n=0}^{∞} x[n]·z^(-n)",
            subject="Signals and Systems",
            topic="Z Transforms",
        )

    def _make_fspl(self) -> EquationKnowledgeObject:
        tree = equation(var("FSPL"), op("^", fraction(op("*", num(4), op("*", const("pi"), op("*", var("d"), var("f")))), const("c")), num(2)))
        return EquationKnowledgeObject(
            name="Free Space Path Loss",
            aliases=["FSPL", "Free Space Path Loss formula"],
            tree=tree,
            canonical="FSPL = (4πdf / c)²",
            subject="Satellite Communication",
            topic="Link Budget Analysis",
        )

    def _make_eirp(self) -> EquationKnowledgeObject:
        tree = equation(var("EIRP"), op("*", var("P_t"), var("G_t")))
        return EquationKnowledgeObject(
            name="Equivalent Isotropic Radiated Power",
            aliases=["EIRP", "P_t G_t"],
            tree=tree,
            canonical="EIRP = P_t * G_t",
            subject="Satellite Communication",
            topic="RF Transmitters",
        )

    def _tree_to_eko(self, tree: ExprNode, text: str) -> EquationKnowledgeObject:
        return EquationKnowledgeObject(
            name=f"Parsed Expression: {text[:20]}",
            tree=tree,
            canonical=text,
            subject="Mathematics",
        )

    def _load_from_disk(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for item in data:
                eko = EquationKnowledgeObject(
                    eko_id=item.get("eko_id"),
                    name=item.get("name", ""),
                    canonical=item.get("canonical", ""),
                    subject=item.get("subject", ""),
                    topic=item.get("topic", ""),
                )
                self.register(eko)
        except Exception as e:
            print(f"[MKE] Warning loading persisted knowledge: {e}")
