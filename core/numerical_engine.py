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
from typing import Optional


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

    def detect_domain(self, text: str) -> Optional[str]:
        """Detect if text has numerical potential and return domain name."""
        text_lower = text.lower()
        scores     = {}
        for domain, keywords in self.NUMERICAL_INDICATORS.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[domain] = score

        if not scores:
            return None

        return max(scores, key=scores.get)

    def is_numerical(self, chunks: list[dict], threshold: int = 2) -> bool:
        """Return True if chunks have enough numerical indicators."""
        combined = " ".join(c.get("text", "") for c in chunks)
        domain   = self.detect_domain(combined)
        if not domain:
            return False

        text_lower = combined.lower()
        count = sum(
            1 for kw in self.NUMERICAL_INDICATORS[domain]
            if kw.lower() in text_lower
        )
        return count >= threshold

    def generate(
        self,
        domain:   str,
        topic:    str,
        marks:    int,
        seed:     Optional[int] = None,
    ) -> Optional[NumericalTemplate]:
        """
        Generate a NumericalTemplate with fresh parameter values.
        Returns None if domain not supported.
        """
        if seed is not None:
            random.seed(seed)

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
            return None

        return gen_fn(topic, marks)

    def generate_from_chunks(
        self,
        chunks: list[dict],
        marks:  int,
        seed:   Optional[int] = None,
    ) -> Optional[NumericalTemplate]:
        """Auto-detect domain from chunks and generate template."""
        combined = " ".join(c.get("text", "") for c in chunks)
        domain   = self.detect_domain(combined)
        if not domain:
            return None

        topic = self._extract_topic(combined, domain)
        return self.generate(domain, topic, marks, seed)

    # ── Domain-specific generators ────────────────────────────────────────────

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
            solution_hint = "A1V1=A2V2 → V2=V1(D1/D2)², P1+½ρV1²=P2+½ρV2²",
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
