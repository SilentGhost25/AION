"""
AION Numerical Engine
=====================
Detects whether a topic has numerical potential
and generates fresh parameter sets deterministically.

The LLM never generates numbers.
This engine generates the numbers and inserts them into the prompt.
The LLM only writes the question narrative around the numbers.

Supported domains:
  - Data Structures (sorting, searching, complexity)
  - Network Theory (circuits, Ohm's law, power)
  - Signals & Systems (Fourier, Z-transform)
  - Satellite Communication (link budget, FSPL, EIRP)
  - Thermodynamics (Carnot, efficiency)
  - Mathematics (integration, matrices, eigenvalues)
  - Digital Electronics (Boolean, gates)
  - Fluid Mechanics (Bernoulli, flow rate)
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any


@dataclass
class NumericalTemplate:
    """A template for a numerical question with fresh parameter values."""
    domain:       str
    topic:        str
    template:     str          # Question text with {param} placeholders
    params:       dict         # Generated parameter values
    solution_hint: str         # What the answer involves (not the answer itself)
    marks_hint:   str          # Suggested marks allocation
    bloom_level:  str          # Always L3 or L4 for numerical


class NumericalEngine:
    """
    Detects numerical topics and generates fresh parameter sets.
    Called by the Planner when a module has numerical content.
    """

    # Keywords that indicate numerical question potential
    NUMERICAL_INDICATORS = {
        "data_structures": [
            "sort", "sorting", "search", "complexity", "O(n)", "O(log n)",
            "array", "heap", "quicksort", "mergesort", "binary search",
            "time complexity", "space complexity", "Big O"
        ],
        "network_theory": [
            "ohm", "resistance", "current", "voltage", "power", "impedance",
            "thevenin", "norton", "kirchhoff", "KVL", "KCL", "circuit",
            "capacitor", "inductor", "frequency", "resonance"
        ],
        "signals_systems": [
            "fourier", "laplace", "z-transform", "convolution", "sampling",
            "frequency", "transfer function", "impulse", "step response",
            "bandwidth", "nyquist", "filter"
        ],
        "satellite_comm": [
            "EIRP", "link budget", "path loss", "FSPL", "free space",
            "gain", "noise", "SNR", "carrier", "decibel", "dB", "GHz",
            "transponder", "TDMA", "FDMA", "bandwidth"
        ],
        "thermodynamics": [
            "carnot", "efficiency", "entropy", "enthalpy", "heat",
            "temperature", "pressure", "work", "cycle", "isothermal",
            "adiabatic", "compressor", "turbine"
        ],
        "mathematics": [
            "integral", "derivative", "matrix", "eigenvalue", "determinant",
            "differential equation", "Laplace", "Fourier", "series",
            "convergence", "transform"
        ],
        "digital_electronics": [
            "logic gate", "boolean", "karnaugh", "K-map", "flip-flop",
            "counter", "binary", "hexadecimal", "truth table", "register"
        ],
        "fluid_mechanics": [
            "bernoulli", "reynolds", "flow rate", "viscosity", "pressure",
            "velocity", "head loss", "pipe", "continuity equation"
        ],
    }

class UniversalFormulaSolver:
    """
    Tier 1: Universal SymPy Formula Parser & Solver.
    Parses equations with real symbol binding, dimensional unit awareness,
    parameter range bounding, and step-by-step Scheme of Valuation generation.
    """

    COMMON_UNITS = {
        "v": "V", "voltage": "V", "i": "A", "current": "A", "r": "Ω", "resistance": "Ω",
        "p": "W", "power": "W", "c": "F", "capacitance": "F", "l": "H", "inductance": "H",
        "f": "Hz", "frequency": "Hz", "t": "K", "temperature": "K", "m": "kg", "mass": "kg",
        "v1": "m/s", "v2": "m/s", "velocity": "m/s", "p1": "kPa", "p2": "kPa", "pressure": "kPa",
        "eirp": "dBW", "fspl": "dB", "snr": "dB", "ber": "", "efficiency": "%"
    }

    def extract_equations(self, text: str) -> List[str]:
        """Extract candidate equation strings from raw text or LaTeX."""
        equations = []
        # LaTeX dollar formulas
        for m in re.finditer(r'\$([A-Za-z0-9_+\-*/\^=()\s]{4,60})\$', text):
            eq = m.group(1).strip()
            if "=" in eq and len(eq) >= 5:
                equations.append(eq)
        # Inline plain equations: variable = expression
        for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/\^().\s]{3,50})', text):
            eq = m.group(1).strip()
            if "=" in eq and not re.search(r'\b(?:def|class|if|for|while)\b', eq):
                equations.append(eq)
        return equations

    def solve_equation(
        self,
        equation_str: str,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Symbolically solve the equation for the primary target variable,
        sample independent parameters within realistic ranges, and compute exact ground truth.
        """
        import sympy as sp
        if seed is not None:
            random.seed(seed)

        try:
            # Clean LaTeX / syntax markers
            clean_eq = re.sub(r'\\(?:text|mathrm|mathbf)\{([^}]+)\}', r'\1', equation_str)
            clean_eq = clean_eq.replace('^', '**').replace('×', '*').strip('$ \n\r')
            if "=" not in clean_eq:
                return None

            lhs_str, rhs_str = clean_eq.split("=", 1)
            lhs_str = lhs_str.strip()
            rhs_str = rhs_str.strip()

            # Isolate tokens and create symbol namespace with real=True to avoid 'I' as imaginary unit
            tokens = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', clean_eq))
            if not tokens:
                return None

            local_dict = {t: sp.Symbol(t, real=True, positive=True) for t in tokens}
            lhs_sym = sp.sympify(lhs_str, locals=local_dict)
            rhs_sym = sp.sympify(rhs_str, locals=local_dict)
            eq = sp.Eq(lhs_sym, rhs_sym)

            free_syms = list(eq.free_symbols)
            if len(free_syms) < 2:
                return None

            # Target variable is LHS if simple symbol, else first symbol
            target_var = lhs_sym if isinstance(lhs_sym, sp.Symbol) else free_syms[0]
            independent_syms = [s for s in free_syms if s != target_var]

            sols = sp.solve(eq, target_var)
            if not sols:
                return None
            sol_expr = sols[0]

            # Sample values for independent parameters
            sampled_values = {}
            for s in independent_syms:
                s_name = str(s).lower()
                if parameter_ranges and s_name in parameter_ranges:
                    low, high = parameter_ranges[s_name]
                    val = round(random.uniform(low, high), 2)
                else:
                    val = round(random.uniform(2.0, 25.0), 2)
                sampled_values[s] = val

            ans_raw = sol_expr.subs(sampled_values).evalf()
            ans_float = round(float(ans_raw), 4)

            # Map units
            target_unit = self.COMMON_UNITS.get(str(target_var).lower(), "")
            param_str_list = []
            for s, v in sampled_values.items():
                unit = self.COMMON_UNITS.get(str(s).lower(), "")
                param_str_list.append(f"{s} = {v}{' ' + unit if unit else ''}")

            return {
                "equation": clean_eq,
                "target": str(target_var),
                "target_unit": target_unit,
                "parameters": {str(k): v for k, v in sampled_values.items()},
                "param_display": ", ".join(param_str_list),
                "solution_expression": str(sol_expr),
                "answer": ans_float,
            }
        except Exception as e:
            return None

    def generate_template(
        self,
        topic: str,
        marks: int,
        context_text: str = "",
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: Optional[int] = None
    ) -> Optional[NumericalTemplate]:
        equations = self.extract_equations(context_text) if context_text else []
        solved = None
        for eq_str in equations:
            res = self.solve_equation(eq_str, parameter_ranges=parameter_ranges, seed=seed)
            if res is not None:
                solved = res
                break

        if not solved:
            # Fallback baseline formula if text lacks explicit closed form
            solved = self.solve_equation("P = V * I", seed=seed)
            if not solved:
                return None

        split_a = max(2, marks * 3 // 10)
        split_b = max(2, marks * 4 // 10)
        split_c = marks - split_a - split_b

        target = solved["target"]
        target_unit = f" (in {solved['target_unit']})" if solved["target_unit"] else ""
        template_text = (
            f"In a technical analysis of {topic}, governing conditions follow the relationship: "
            f"${solved['equation']}$.\n"
            f"Given the system operating parameters: {solved['param_display']}:\n\n"
            f"({split_a} marks) State the governing formula and identify all independent variables.\n"
            f"({split_b} marks) Substitute the given operational values into the model with proper unit alignment.\n"
            f"({split_c} marks) Calculate the resulting value of {target}{target_unit} and state the final result with appropriate units."
        )

        solution_hint = (
            f"Formula: {solved['equation']} -> {target} = {solved['solution_expression']}. "
            f"Substitution: {solved['param_display']}. "
            f"Final Answer: {solved['target']} = {solved['answer']} {solved['target_unit']}."
        )

        return NumericalTemplate(
            domain="universal_formula",
            topic=topic,
            template=template_text,
            params={"parameters": solved["parameters"], "answer": solved["answer"], "unit": solved["target_unit"]},
            solution_hint=solution_hint,
            marks_hint=f"{split_a}+{split_b}+{split_c}",
            bloom_level="L3",
        )


class AlgorithmicStateTracer:
    """Tier 2: Pure Python Algorithmic Simulation Traces for discrete computational structures."""

    def generate_template(self, topic: str, marks: int, seed: Optional[int] = None) -> NumericalTemplate:
        if seed is not None:
            random.seed(seed)

        arr = random.sample(range(12, 88), 7)
        split_a = marks * 6 // 10
        split_b = marks - split_a

        template_text = (
            f"Consider the following discrete numerical sequence: {arr}.\n\n"
            f"({split_a} marks) Trace the step-by-step state transformation under {topic}, "
            f"showing the sequence configuration after each iteration/pass.\n"
            f"({split_b} marks) Calculate the total number of element comparisons and swap/update "
            f"operations executed to reach completion."
        )

        return NumericalTemplate(
            domain="algorithmic_trace",
            topic=topic,
            template=template_text,
            params={"sequence": arr},
            solution_hint=f"Step-by-step iteration table for {arr} with exact comparison count",
            marks_hint=f"{split_a}+{split_b}",
            bloom_level="L3",
        )


class DualVLLMVerifier:
    """
    Tier 3: Dual-vLLM Double-Blind Cross-Verification.
    Verifies that synthesizer solution and independent auditor solution agree within ±3% tolerance.
    """

    @classmethod
    def verify(
        cls,
        synthesizer_answer: float,
        auditor_answer: float,
        tolerance_pct: float = 3.0
    ) -> bool:
        if abs(synthesizer_answer) < 1e-6 and abs(auditor_answer) < 1e-6:
            return True
        denom = max(abs(synthesizer_answer), abs(auditor_answer), 1e-6)
        rel_diff = abs(synthesizer_answer - auditor_answer) / denom * 100.0
        return rel_diff <= tolerance_pct


class NumericalEngine:
    """
    Detects numerical topics and generates fresh parameter sets.
    Implements a 3-tier universal synthesis cascade:
      Tier 1: Universal SymPy Formula Solver (closed-form equations)
      Tier 2: Algorithmic State Tracer (discrete computational sequences)
      Tier 3: Dual-vLLM Cross-Verification (double-blind validation)
      Safety: Clean automatic downgrade to qualitative archetype when no formulas exist.
    """

    def __init__(self):
        self.formula_solver = UniversalFormulaSolver()
        self.state_tracer = AlgorithmicStateTracer()
        self.verifier = DualVLLMVerifier()

    # Keywords that indicate numerical question potential
    NUMERICAL_INDICATORS = {
        "data_structures": [
            "sort", "sorting", "search", "complexity", "O(n)", "O(log n)",
            "array", "heap", "quicksort", "mergesort", "binary search",
            "time complexity", "space complexity", "Big O"
        ],
        "network_theory": [
            "ohm", "resistance", "current", "voltage", "power", "impedance",
            "thevenin", "norton", "kirchhoff", "KVL", "KCL", "circuit",
            "capacitor", "inductor", "frequency", "resonance"
        ],
        "signals_systems": [
            "fourier", "laplace", "z-transform", "convolution", "sampling",
            "frequency", "transfer function", "impulse", "step response",
            "bandwidth", "nyquist", "filter"
        ],
        "satellite_comm": [
            "EIRP", "link budget", "path loss", "FSPL", "free space",
            "gain", "noise", "SNR", "carrier", "decibel", "dB", "GHz",
            "transponder", "TDMA", "FDMA", "bandwidth"
        ],
        "thermodynamics": [
            "carnot", "efficiency", "entropy", "enthalpy", "heat",
            "temperature", "pressure", "work", "cycle", "isothermal",
            "adiabatic", "compressor", "turbine"
        ],
        "mathematics": [
            "integral", "derivative", "matrix", "eigenvalue", "determinant",
            "differential equation", "Laplace", "Fourier", "series",
            "convergence", "transform"
        ],
        "digital_electronics": [
            "logic gate", "boolean", "karnaugh", "K-map", "flip-flop",
            "counter", "binary", "hexadecimal", "truth table", "register"
        ],
        "fluid_mechanics": [
            "bernoulli", "reynolds", "flow rate", "viscosity", "pressure",
            "velocity", "head loss", "pipe", "continuity equation"
        ],
    }

    def detect_domain(self, text: str) -> Optional[str]:
        """Detect if text has numerical potential and return domain name."""
        text_lower = text.lower()
        scores = {}
        for domain, keywords in self.NUMERICAL_INDICATORS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[domain] = score

        if scores:
            return max(scores, key=scores.get)

        # Subject-agnostic formula detection: check for equations or calculation intent
        has_equation = bool(re.search(r'\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[A-Za-z0-9_+\-*/\^().\s]{3,50}', text))
        has_calc_word = any(w in text_lower for w in ("calculate", "compute", "determine the value", "find the value", "evaluate"))
        has_digits = bool(re.search(r'\b\d+(?:\.\d+)?\s*(?:kg|m|s|v|a|w|hz|k|pa|%|db|ohm|μf)\b', text_lower))

        if has_equation or (has_calc_word and has_digits):
            return "universal_formula"

        if has_calc_word and any(w in text_lower for w in ("sequence", "array", "steps", "trace", "iteration")):
            return "algorithmic_trace"

        # Pure qualitative text: cleanly downgrade by returning None
        return None

    def is_numerical(self, chunks: list[dict], threshold: int = 2) -> bool:
        """Return True if chunks have enough numerical indicators."""
        combined = " ".join(c.get("text", "") for c in chunks)
        domain = self.detect_domain(combined)
        if not domain:
            return False

        if domain in self.NUMERICAL_INDICATORS:
            text_lower = combined.lower()
            count = sum(1 for kw in self.NUMERICAL_INDICATORS[domain] if kw.lower() in text_lower)
            return count >= threshold

        return True

    def generate(
        self,
        domain: str,
        topic: str,
        marks: int,
        seed: Optional[int] = None,
        context_text: str = "",
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Optional[NumericalTemplate]:
        """
        Generate a NumericalTemplate with fresh parameter values across the 3-tier cascade.
        """
        if seed is not None:
            random.seed(seed)

        if domain == "universal_formula":
            return self.formula_solver.generate_template(
                topic, marks, context_text=context_text, parameter_ranges=parameter_ranges, seed=seed
            )
        elif domain == "algorithmic_trace":
            return self.state_tracer.generate_template(topic, marks, seed=seed)

        generators = {
            "data_structures": self._gen_data_structures,
            "network_theory":  self._gen_network_theory,
            "satellite_comm":  self._gen_satellite_comm,
            "thermodynamics":  self._gen_thermodynamics,
            "mathematics":     self._gen_mathematics,
            "signals_systems": self._gen_signals_systems,
            "digital_electronics": self._gen_digital_electronics,
            "fluid_mechanics": self._gen_fluid_mechanics,
        }

        gen_fn = generators.get(domain)
        if not gen_fn:
            return self.formula_solver.generate_template(
                topic, marks, context_text=context_text, parameter_ranges=parameter_ranges, seed=seed
            )

        return gen_fn(topic, marks)

    def generate_from_chunks(
        self,
        chunks: list[dict],
        marks: int,
        seed: Optional[int] = None,
        parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Optional[NumericalTemplate]:
        """Auto-detect domain from chunks and generate template with full context."""
        combined = " ".join(c.get("text", "") for c in chunks)
        domain = self.detect_domain(combined)
        if not domain:
            return None

        topic = self._extract_topic(combined, domain)
        return self.generate(
            domain, topic, marks, seed=seed, context_text=combined, parameter_ranges=parameter_ranges
        )

    # -- Domain-specific generators --------------------------------------------

    def _gen_data_structures(self, topic: str, marks: int) -> NumericalTemplate:
        n      = random.randint(6, 10)
        values = random.sample(range(10, 99), n)

        algo_choices = [
            {
                "name": "Quick Sort",
                "instruction": f"Sort the following array using Quick Sort. Show the array after each partition step.",
                "hint": "pivot selection, partitioning, recursive calls",
            },
            {
                "name": "Merge Sort",
                "instruction": f"Sort the following array using Merge Sort. Show the divide and merge steps clearly.",
                "hint": "divide phase showing splits, merge phase showing sorted merges",
            },
            {
                "name": "Insertion Sort",
                "instruction": f"Sort the following array using Insertion Sort. Show the array state after each insertion.",
                "hint": "state of array after each insertion step",
            },
            {
                "name": "Selection Sort",
                "instruction": f"Sort the following array using Selection Sort. Identify the minimum at each pass.",
                "hint": "minimum element selected and swapped at each pass",
            },
            {
                "name": "Heap Sort",
                "instruction": f"Build a max-heap from the following array, then sort it. Show the heap at each step.",
                "hint": "heapify process, extract-max operations",
            },
        ]

        algo = random.choice(algo_choices)
        arr  = str(values).replace("[", "").replace("]", "")

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "data_structures",
            topic     = algo["name"],
            template  = (
                f"{algo['instruction']}\n"
                f"Array: [{arr}]\n\n"
                f"({split_a} marks) Show all steps of {algo['name']}.\n"
                f"({split_b} marks) State the time complexity of {algo['name']} "
                f"in best, average, and worst cases."
            ),
            params    = {"array": values, "n": n, "algorithm": algo["name"]},
            solution_hint = algo["hint"],
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _gen_network_theory(self, topic: str, marks: int) -> NumericalTemplate:
        config = random.randint(1, 3)

        if config == 1:
            # Series-parallel circuit
            r1 = random.choice([10, 15, 20, 22, 33, 47, 68, 100])
            r2 = random.choice([10, 15, 20, 22, 33, 47, 68, 100])
            r3 = random.choice([10, 15, 20, 22, 33, 47, 68, 100])
            vs = random.choice([5, 9, 10, 12, 15, 24])

            split_a = marks * 6 // 10
            split_b = marks - split_a

            return NumericalTemplate(
                domain    = "network_theory",
                topic     = "Series-Parallel Circuit",
                template  = (
                    f"In the circuit below, R1 = {r1}Ω, R2 = {r2}Ω, and R3 = {r3}Ω "
                    f"are connected such that R2 and R3 are in parallel, "
                    f"and this combination is in series with R1. "
                    f"The supply voltage Vs = {vs}V.\n\n"
                    f"({split_a} marks) Find the total resistance and total current drawn from the supply.\n"
                    f"({split_b} marks) Calculate the voltage across R2 and the power dissipated in R3."
                ),
                params    = {"R1": r1, "R2": r2, "R3": r3, "Vs": vs},
                solution_hint = "Req = R1 + (R2||R3), I_total = Vs/Req, V_R2 = I_total * (R2||R3)",
                marks_hint    = f"{split_a}+{split_b}",
                bloom_level   = "L3",
            )

        elif config == 2:
            # Thevenin equivalent
            r1 = random.choice([10, 20, 30, 40, 50])
            r2 = random.choice([10, 20, 30, 40, 50])
            vs = random.choice([10, 20, 30, 40, 50])

            split_a = marks * 6 // 10
            split_b = marks - split_a

            return NumericalTemplate(
                domain    = "network_theory",
                topic     = "Thevenin Equivalent",
                template  = (
                    f"A circuit has a voltage source Vs = {vs}V with R1 = {r1}Ω in series "
                    f"and R2 = {r2}Ω connected across the output terminals AB.\n\n"
                    f"({split_a} marks) Find the Thevenin equivalent voltage (Vth) "
                    f"and Thevenin equivalent resistance (Rth) at terminals AB.\n"
                    f"({split_b} marks) If a load resistance RL = {r1}Ω is connected "
                    f"across AB, find the load current and power delivered to RL."
                ),
                params    = {"R1": r1, "R2": r2, "Vs": vs},
                solution_hint = "Vth = Vs * R2/(R1+R2), Rth = R1||R2",
                marks_hint    = f"{split_a}+{split_b}",
                bloom_level   = "L4",
            )

        else:
            # Power calculation
            v = random.choice([110, 120, 220, 230])
            r = random.choice([40, 50, 60, 75, 100])

            split_a = marks * 6 // 10
            split_b = marks - split_a

            return NumericalTemplate(
                domain    = "network_theory",
                topic     = "Power in AC/DC Circuit",
                template  = (
                    f"A resistive load of {r}Ω is connected to a {v}V supply.\n\n"
                    f"({split_a} marks) Calculate the current through the load, "
                    f"the power dissipated, and the energy consumed in 2 hours.\n"
                    f"({split_b} marks) If the supply voltage drops by 10%, "
                    f"calculate the percentage change in power dissipated."
                ),
                params    = {"V": v, "R": r},
                solution_hint = "I = V/R, P = V²/R = I²R, E = P×t",
                marks_hint    = f"{split_a}+{split_b}",
                bloom_level   = "L3",
            )

    def _gen_satellite_comm(self, topic: str, marks: int) -> NumericalTemplate:
        # Link budget calculation
        pt_dbw  = random.randint(-10, 20)         # transmit power dBW
        gt_dbi  = random.randint(20, 50)          # transmit antenna gain dBi
        freq    = random.choice([4, 6, 11, 14])   # GHz
        dist    = random.choice([35786])           # km (GEO)
        gr_dbi  = random.randint(25, 45)          # receive antenna gain dBi
        ts_k    = random.choice([100, 150, 200, 290])  # system noise temp K

        fspl = round(20 * (freq ** 0.5) + 92.4 + 20, 1)  # simplified
        eirp = pt_dbw + gt_dbi

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "satellite_comm",
            topic     = "Link Budget",
            template  = (
                f"A GEO satellite communication link has the following parameters:\n"
                f"Transmit power: {pt_dbw} dBW\n"
                f"Transmit antenna gain: {gt_dbi} dBi\n"
                f"Operating frequency: {freq} GHz\n"
                f"Slant range: {dist} km\n"
                f"Receive antenna gain: {gr_dbi} dBi\n"
                f"System noise temperature: {ts_k} K\n\n"
                f"({split_a} marks) Calculate the EIRP, Free Space Path Loss (FSPL), "
                f"and the received carrier power at the earth station.\n"
                f"({split_b} marks) Calculate the G/T of the receiving system "
                f"and the carrier-to-noise density ratio C/N₀."
            ),
            params    = {
                "Pt_dBW": pt_dbw, "Gt_dBi": gt_dbi,
                "freq_GHz": freq, "dist_km": dist,
                "Gr_dBi": gr_dbi, "Ts_K": ts_k,
            },
            solution_hint = "EIRP=Pt+Gt, FSPL=20log(d)+20log(f)+92.4, C=EIRP-FSPL+Gr",
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _gen_thermodynamics(self, topic: str, marks: int) -> NumericalTemplate:
        t_hot  = random.randint(500, 1000)   # K
        t_cold = random.randint(300, 400)    # K
        q_in   = random.randint(500, 2000)   # kJ

        eta    = round((1 - t_cold / t_hot) * 100, 1)
        w_net  = round(q_in * (1 - t_cold / t_hot), 1)

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "thermodynamics",
            topic     = "Carnot Cycle",
            template  = (
                f"A Carnot heat engine operates between a source at {t_hot} K "
                f"and a sink at {t_cold} K. The heat supplied per cycle is {q_in} kJ.\n\n"
                f"({split_a} marks) Calculate the thermal efficiency of the Carnot engine "
                f"and the net work output per cycle.\n"
                f"({split_b} marks) Calculate the heat rejected to the sink "
                f"and the coefficient of performance if the same cycle operates as a refrigerator."
            ),
            params    = {"T_H": t_hot, "T_L": t_cold, "Q_in": q_in},
            solution_hint = f"η = 1 - T_L/T_H = {eta}%, W_net = η × Q_in = {w_net} kJ",
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _gen_mathematics(self, topic: str, marks: int) -> NumericalTemplate:
        config = random.randint(1, 2)

        if config == 1:
            # Matrix eigenvalues
            a = random.randint(1, 5)
            b = random.randint(1, 4)
            c = random.randint(1, 4)
            d = random.randint(1, 5)

            split_a = marks * 6 // 10
            split_b = marks - split_a

            return NumericalTemplate(
                domain    = "mathematics",
                topic     = "Eigenvalues and Eigenvectors",
                template  = (
                    f"Given the matrix A = [[{a}, {b}], [{c}, {d}]]:\n\n"
                    f"({split_a} marks) Find the eigenvalues of matrix A "
                    f"using the characteristic equation.\n"
                    f"({split_b} marks) Find the eigenvectors corresponding "
                    f"to each eigenvalue and verify using the Cayley-Hamilton theorem."
                ),
                params    = {"a": a, "b": b, "c": c, "d": d},
                solution_hint = f"char eq: λ² - ({a+d})λ + ({a*d-b*c}) = 0",
                marks_hint    = f"{split_a}+{split_b}",
                bloom_level   = "L3",
            )
        else:
            # Numerical integration
            a_val = random.randint(0, 2)
            b_val = a_val + random.randint(2, 4)
            n     = random.choice([4, 6, 8])

            split_a = marks * 6 // 10
            split_b = marks - split_a

            return NumericalTemplate(
                domain    = "mathematics",
                topic     = "Numerical Integration",
                template  = (
                    f"Evaluate the integral ∫from {a_val} to {b_val} of (1 + x²) dx "
                    f"using:\n\n"
                    f"({split_a} marks) Simpson's 1/3 rule with n = {n} subintervals. "
                    f"Show the tabulated values and the final result.\n"
                    f"({split_b} marks) Trapezoidal rule with the same n = {n} subintervals. "
                    f"Compare with the exact value and calculate the percentage error."
                ),
                params    = {"a": a_val, "b": b_val, "n": n},
                solution_hint = "h = (b-a)/n, evaluate f at each point, apply formula",
                marks_hint    = f"{split_a}+{split_b}",
                bloom_level   = "L3",
            )

    def _gen_signals_systems(self, topic: str, marks: int) -> NumericalTemplate:
        # Z-transform
        a = random.choice([0.5, 0.25, 0.75, 2, 3])

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "signals_systems",
            topic     = "Z-Transform",
            template  = (
                f"For the discrete-time signal x[n] = ({a})^n · u[n]:\n\n"
                f"({split_a} marks) Find the Z-transform X(z) and state the "
                f"Region of Convergence (ROC).\n"
                f"({split_b} marks) Find the inverse Z-transform of "
                f"X(z) = z / (z - {a}) and determine if the system is stable."
            ),
            params    = {"a": a},
            solution_hint = f"X(z) = z/(z-{a}), ROC: |z| > {a}",
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _gen_digital_electronics(self, topic: str, marks: int) -> NumericalTemplate:
        # K-map simplification
        minterms = sorted(random.sample(range(16), random.randint(4, 8)))
        mt_str   = ", ".join(str(m) for m in minterms)

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "digital_electronics",
            topic     = "K-Map Simplification",
            template  = (
                f"A Boolean function F(A,B,C,D) is defined by the minterms: "
                f"Σm({mt_str}).\n\n"
                f"({split_a} marks) Draw the 4-variable Karnaugh map, "
                f"group the minterms, and obtain the minimized Sum of Products (SOP) expression.\n"
                f"({split_b} marks) Implement the minimized expression using "
                f"only NAND gates and draw the logic circuit."
            ),
            params    = {"minterms": minterms},
            solution_hint = "Group adjacent 1s in powers of 2 (1,2,4,8), read off simplified terms",
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _gen_fluid_mechanics(self, topic: str, marks: int) -> NumericalTemplate:
        v1   = random.randint(2, 8)       # m/s
        d1   = random.choice([0.1, 0.15, 0.2])   # m diameter
        d2   = round(d1 / 2, 3)
        p1   = random.randint(100, 300)   # kPa

        split_a = marks * 6 // 10
        split_b = marks - split_a

        return NumericalTemplate(
            domain    = "fluid_mechanics",
            topic     = "Bernoulli and Continuity",
            template  = (
                f"Water flows through a horizontal pipe that narrows from "
                f"diameter D1 = {d1}m to D2 = {d2}m. "
                f"At section 1, velocity V1 = {v1} m/s and pressure P1 = {p1} kPa. "
                f"Assume ideal flow (density = 1000 kg/m³).\n\n"
                f"({split_a} marks) Using the continuity equation, find the velocity V2 "
                f"at section 2. Then apply Bernoulli's equation to find the pressure P2.\n"
                f"({split_b} marks) Calculate the volume flow rate Q and the "
                f"force exerted by the fluid on the pipe contraction."
            ),
            params    = {"V1": v1, "D1": d1, "D2": d2, "P1": p1},
            solution_hint = "A1V1=A2V2 -> V2=V1(D1/D2)², P1+½ρV1²=P2+½ρV2²",
            marks_hint    = f"{split_a}+{split_b}",
            bloom_level   = "L3",
        )

    def _extract_topic(self, text: str, domain: str) -> str:
        """Extract the most relevant topic from text for a domain."""
        keywords = self.NUMERICAL_INDICATORS.get(domain, [])
        text_lower = text.lower()
        for kw in keywords:
            if kw.lower() in text_lower:
                return kw.title()
        return domain.replace("_", " ").title()
