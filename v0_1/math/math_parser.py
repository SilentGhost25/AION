"""
AION Math Parser
================
Converts raw text containing mathematical expressions
into structured MathObject instances.

Algorithm:
  1. Detect math regions using pattern matching
  2. Classify the type of expression
  3. Extract variables, parameters, bounds
  4. Build canonical token representation
  5. Compute LaTeX equivalent
  6. Return MathObject

No LLM calls in this module — pure deterministic parsing.
"""

import re
from typing import Optional, List, Tuple
from .math_object import MathObject, Variable
from .tokens import MathType, UNICODE_MAP


# ── Detection Patterns ────────────────────────────────────────────────────────

MATH_REGIONS = [
    # LaTeX inline and display
    (r'\$\$(.+?)\$\$',           "latex_display"),
    (r'\$(.+?)\$',               "latex_inline"),
    (r'\\begin\{equation\}(.+?)\\end\{equation\}', "latex_equation"),

    # Unicode operators
    (r'[∫∬∭∮∑∏√∛∂∇±×÷·⋅≤≥≠≈≡∝∀∃∈∉→←↔⇒⟺αβγδεζηθικλμνξπρστυφχψωΓΔΘΛΞΠΣΥΦΨΩ]+'
     r'[^.!?\n]{3,80}',          "unicode_expr"),

    # Common formula patterns
    (r'[A-Za-z]\s*=\s*[^=\n]{3,60}', "assignment"),
    (r'[A-Za-z]\([A-Za-z,\s]+\)\s*=', "function_def"),
    (r'(?:lim|Lim)\s*[_\[]',    "limit"),
    (r'(?:∫|\\int)\s*',         "integral"),
    (r'(?:∑|\\sum)\s*',         "summation"),
    (r'd/d[a-zA-Z]|∂/∂',       "derivative"),
    (r'[A-Z]\(s\)\s*=|[A-Z]\(j[ω\w]\)\s*=', "transfer_fn"),
    (r'\b[A-Za-z]\^[{2-9n}]',   "power"),
]

# Named formula database
NAMED_FORMULAS = {
    "ohm":          ("V = IR",           MathType.FORMULA, "Ohm's Law",
                     [Variable("V","real","","Voltage"),
                      Variable("I","real","","Current"),
                      Variable("R","positive","","Resistance")]),
    "power":        ("P = VI",           MathType.FORMULA, "Electrical Power",
                     [Variable("P","real","","Power"),
                      Variable("V","real","","Voltage"),
                      Variable("I","real","","Current")]),
    "carnot":       ("η = 1 - T_L/T_H", MathType.FORMULA, "Carnot Efficiency",
                     [Variable("η","real","0<η<1","Efficiency"),
                      Variable("T_L","positive","","Cold temperature"),
                      Variable("T_H","positive","T_H>T_L","Hot temperature")]),
    "quadratic":    ("x = (-b ± √(b²-4ac)) / 2a", MathType.FORMULA,
                     "Quadratic Formula",
                     [Variable("a","real","non-zero","Leading coefficient"),
                      Variable("b","real","","Linear coefficient"),
                      Variable("c","real","","Constant term")]),
    "euler":        ("e^(iθ) = cos(θ) + i·sin(θ)", MathType.FORMULA,
                     "Euler's Formula",
                     [Variable("θ","real","","Angle in radians"),
                      Variable("e","constant","","Euler's number"),
                      Variable("i","complex","i²=-1","Imaginary unit")]),
    "fspl":         ("FSPL = (4πd/λ)²", MathType.FORMULA,
                     "Free Space Path Loss",
                     [Variable("d","positive","","Distance"),
                      Variable("λ","positive","","Wavelength"),
                      Variable("π","constant","","Pi")]),
    "eirp":         ("EIRP = P_t × G_t", MathType.FORMULA,
                     "Equivalent Isotropic Radiated Power",
                     [Variable("P_t","positive","","Transmit power"),
                      Variable("G_t","positive","","Transmit antenna gain")]),
    "bernoulli":    ("P + ½ρv² + ρgh = constant", MathType.FORMULA,
                     "Bernoulli's Equation",
                     [Variable("P","real","","Pressure"),
                      Variable("ρ","positive","","Fluid density"),
                      Variable("v","real","","Velocity"),
                      Variable("g","constant","9.81","Gravity"),
                      Variable("h","real","","Height")]),
    "kepler3":      ("T² ∝ a³", MathType.FORMULA,
                     "Kepler's Third Law",
                     [Variable("T","positive","","Orbital period"),
                      Variable("a","positive","","Semi-major axis")]),
    "fourier":      ("F(ω) = ∫_{-∞}^{∞} f(t)e^{-jωt} dt", MathType.FORMULA,
                     "Fourier Transform",
                     [Variable("f","function","","Time domain signal"),
                      Variable("ω","real","","Angular frequency"),
                      Variable("t","real","","Time variable")]),
    "laplace":      ("F(s) = ∫_0^∞ f(t)e^{-st} dt", MathType.FORMULA,
                     "Laplace Transform",
                     [Variable("f","function","","Time domain function"),
                      Variable("s","complex","","Complex frequency")]),
    "eigenvalue":   ("Av = λv", MathType.FORMULA,
                     "Eigenvalue Equation",
                     [Variable("A","matrix","square","Matrix"),
                      Variable("v","vector","non-zero","Eigenvector"),
                      Variable("λ","real","","Eigenvalue")]),
}


class MathParser:
    """
    Parses mathematical expressions from raw academic text.
    Returns structured MathObject instances.
    """

    def parse_text(self, text: str, subject: str = "") -> List[MathObject]:
        """
        Extract all mathematical expressions from a block of text.
        Returns list of MathObject, ordered by appearance.
        """
        results = []
        seen    = set()

        # Check for named formulas first
        text_lower = text.lower()
        for keyword, (expr, mtype, name, variables) in NAMED_FORMULAS.items():
            if keyword in text_lower and name not in seen:
                obj = MathObject(
                    math_type   = mtype,
                    canonical   = self._to_canonical(expr),
                    latex       = self._to_latex(expr),
                    description = name,
                    named_as    = name,
                    variables   = variables,
                    subject     = subject,
                    is_template = True,
                    confidence  = 0.95,
                    raw_text    = expr,
                )
                results.append(obj)
                seen.add(name)

        # Detect math regions
        for pattern, region_type in MATH_REGIONS:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                raw = match.group().strip()
                if len(raw) < 3 or raw in seen:
                    continue
                seen.add(raw)

                obj = self._parse_expression(raw, region_type, subject)
                if obj:
                    results.append(obj)

        return results

    def parse_expression(
        self,
        expr:    str,
        subject: str = "",
    ) -> MathObject:
        """Parse a single mathematical expression string."""
        region_type = self._detect_region_type(expr)
        return self._parse_expression(expr, region_type, subject) or MathObject(
            math_type   = MathType.EXPRESSION,
            canonical   = self._to_canonical(expr),
            latex       = self._to_latex(expr),
            description = "Mathematical expression",
            raw_text    = expr,
            subject     = subject,
        )

    # ── Core parsing logic ────────────────────────────────────────────────────

    def _parse_expression(
        self,
        raw:         str,
        region_type: str,
        subject:     str,
    ) -> Optional[MathObject]:

        raw = raw.strip()

        # Integral
        if re.search(r'∫|\\int', raw, re.IGNORECASE):
            return self._parse_integral(raw, subject)

        # Summation
        if re.search(r'∑|\\sum', raw, re.IGNORECASE):
            return self._parse_summation(raw, subject)

        # Limit
        if re.search(r'\blim\b', raw, re.IGNORECASE):
            return self._parse_limit(raw, subject)

        # Derivative
        if re.search(r'd/d[a-zA-Z]|∂/∂|dy/dx', raw, re.IGNORECASE):
            return self._parse_derivative(raw, subject)

        # Laplace / Z / Fourier Transform
        if re.search(r'[FGH]\(s\)\s*=|laplace', raw, re.IGNORECASE):
            return self._parse_transform(raw, MathType.LAPLACE, subject)
        if re.search(r'[FGH]\(z\)\s*=|z-transform', raw, re.IGNORECASE):
            return self._parse_transform(raw, MathType.Z_TRANSFORM, subject)
        if re.search(r'[FGH]\(j?ω\)\s*=|fourier', raw, re.IGNORECASE):
            return self._parse_transform(raw, MathType.FOURIER, subject)

        # Matrix
        if re.search(r'\[.*\]\[.*\]|\bmatrix\b', raw, re.IGNORECASE):
            return self._parse_matrix(raw, subject)

        # Equation (has = sign)
        if "=" in raw:
            return self._parse_equation(raw, subject)

        # Generic expression
        return MathObject(
            math_type   = MathType.EXPRESSION,
            canonical   = self._to_canonical(raw),
            latex       = self._to_latex(raw),
            description = "Mathematical expression",
            variables   = self._extract_variables(raw),
            subject     = subject,
            confidence  = 0.7,
            raw_text    = raw,
        )

    def _parse_integral(self, raw: str, subject: str) -> MathObject:
        lower  = self._extract_bound(raw, "lower")
        upper  = self._extract_bound(raw, "upper")
        var    = self._extract_integration_variable(raw)
        integrand = self._extract_integrand(raw)

        canonical = f"<INT lower={lower} upper={upper} var={var}>{integrand}</INT>"
        latex = f"\\int_{{{lower}}}^{{{upper}}} {integrand} \\, d{var}"

        return MathObject(
            math_type   = MathType.INTEGRAL,
            canonical   = canonical,
            latex       = latex,
            description = f"Integral of {integrand} with respect to {var}",
            parameters  = {"lower": lower, "upper": upper, "variable": var,
                           "integrand": integrand},
            variables   = [Variable(var, "real", "", "Integration variable")],
            subject     = subject,
            is_template = True,
            confidence  = 0.88,
            raw_text    = raw,
        )

    def _parse_summation(self, raw: str, subject: str) -> MathObject:
        index  = self._extract_sum_index(raw)
        start  = self._extract_bound(raw, "lower")
        end    = self._extract_bound(raw, "upper")
        expr   = self._extract_sum_expression(raw)

        canonical = f"<SUM index={index} start={start} end={end}>{expr}</SUM>"
        latex     = f"\\sum_{{{index}={start}}}^{{{end}}} {expr}"

        return MathObject(
            math_type   = MathType.SUMMATION,
            canonical   = canonical,
            latex       = latex,
            description = f"Summation of {expr} from {index}={start} to {end}",
            parameters  = {"index": index, "start": start, "end": end,
                           "expression": expr},
            variables   = [Variable(index, "integer", f"{start}≤{index}≤{end}",
                                    "Summation index")],
            subject     = subject,
            is_template = True,
            confidence  = 0.88,
            raw_text    = raw,
        )

    def _parse_limit(self, raw: str, subject: str) -> MathObject:
        var      = self._extract_limit_variable(raw)
        approach = self._extract_limit_approach(raw)
        expr     = self._extract_limit_expression(raw)

        canonical = f"<LIMIT var={var} approach={approach}>{expr}</LIMIT>"
        latex     = f"\\lim_{{{var} \\to {approach}}} {expr}"

        return MathObject(
            math_type   = MathType.LIMIT,
            canonical   = canonical,
            latex       = latex,
            description = f"Limit of {expr} as {var} approaches {approach}",
            parameters  = {"variable": var, "approach": approach,
                           "expression": expr},
            variables   = [Variable(var, "real", "", "Limit variable")],
            subject     = subject,
            confidence  = 0.85,
            raw_text    = raw,
        )

    def _parse_derivative(self, raw: str, subject: str) -> MathObject:
        order  = 1 if "d²" not in raw and "d^2" not in raw else 2
        var    = self._extract_diff_variable(raw)
        func   = self._extract_diff_function(raw)

        canonical = f"<DERIV order={order} var={var}>{func}</DERIV>"
        if order == 1:
            latex = f"\\frac{{d{func}}}{{d{var}}}"
        else:
            latex = f"\\frac{{d^{order}{func}}}{{d{var}^{order}}}"

        return MathObject(
            math_type   = MathType.DERIVATIVE,
            canonical   = canonical,
            latex       = latex,
            description = f"{'Second' if order==2 else 'First'} derivative of {func} with respect to {var}",
            parameters  = {"order": order, "variable": var, "function": func},
            variables   = [Variable(var, "real", "", "Differentiation variable")],
            subject     = subject,
            confidence  = 0.85,
            raw_text    = raw,
        )

    def _parse_transform(
        self, raw: str, transform_type: MathType, subject: str
    ) -> MathObject:
        domain_map = {
            MathType.LAPLACE:     ("s", "t", "Laplace Transform"),
            MathType.Z_TRANSFORM: ("z", "n", "Z-Transform"),
            MathType.FOURIER:     ("ω", "t", "Fourier Transform"),
        }
        freq_var, time_var, name = domain_map[transform_type]

        canonical = f"<TRANSFORM type={transform_type.value} freq={freq_var} time={time_var}>{raw}</TRANSFORM>"
        latex     = raw  # preserve original LaTeX if present

        return MathObject(
            math_type   = transform_type,
            canonical   = canonical,
            latex       = latex,
            description = f"{name} expression",
            named_as    = name,
            parameters  = {"frequency_variable": freq_var, "time_variable": time_var},
            variables   = [
                Variable(freq_var, "complex", "", f"Frequency domain variable ({name})"),
                Variable(time_var, "real",    "", "Time domain variable"),
            ],
            subject     = subject,
            confidence  = 0.9,
            raw_text    = raw,
        )

    def _parse_matrix(self, raw: str, subject: str) -> MathObject:
        dims = self._extract_matrix_dimensions(raw)
        return MathObject(
            math_type   = MathType.MATRIX,
            canonical   = f"<MATRIX rows={dims[0]} cols={dims[1]}>{raw}</MATRIX>",
            latex       = self._to_latex(raw),
            description = f"{dims[0]}×{dims[1]} matrix",
            parameters  = {"rows": dims[0], "cols": dims[1]},
            subject     = subject,
            confidence  = 0.8,
            raw_text    = raw,
        )

    def _parse_equation(self, raw: str, subject: str) -> MathObject:
        parts     = raw.split("=", 1)
        lhs       = parts[0].strip()
        rhs       = parts[1].strip() if len(parts) > 1 else ""
        variables = self._extract_variables(raw)

        return MathObject(
            math_type   = MathType.EQUATION,
            canonical   = f"<EQ lhs={lhs} rhs={rhs}>{raw}</EQ>",
            latex       = self._to_latex(raw),
            description = f"Equation: {raw}",
            parameters  = {"lhs": lhs, "rhs": rhs},
            variables   = variables,
            subject     = subject,
            is_template = len(variables) > 1,
            confidence  = 0.85,
            raw_text    = raw,
        )

    # ── Extraction helpers ────────────────────────────────────────────────────

    def _extract_bound(self, text: str, bound: str) -> str:
        if bound == "lower":
            m = re.search(r'[_\[{]([^}\]]+)[}\]]', text)
            return m.group(1).strip() if m else "0"
        else:
            m = re.search(r'\^[{(]([^})]+)[})]', text)
            return m.group(1).strip() if m else "∞"

    def _extract_integration_variable(self, text: str) -> str:
        m = re.search(r'd([a-zA-Z])\b', text)
        return m.group(1) if m else "x"

    def _extract_integrand(self, text: str) -> str:
        text = re.sub(r'[∫\\int].*?(?=\s)', '', text, count=1)
        text = re.sub(r'd[a-zA-Z]\s*$', '', text.strip())
        return text.strip() or "f(x)"

    def _extract_sum_index(self, text: str) -> str:
        m = re.search(r'[_\[{]([a-zA-Z])\s*=', text)
        return m.group(1) if m else "i"

    def _extract_sum_expression(self, text: str) -> str:
        parts = re.split(r'[∑\\sum]', text, maxsplit=1)
        return parts[-1].strip() if len(parts) > 1 else "a_i"

    def _extract_limit_variable(self, text: str) -> str:
        m = re.search(r'(?:lim|Lim)\s*[_\[{]?\s*([a-zA-Z])\s*(?:→|->|\\to)', text)
        return m.group(1) if m else "x"

    def _extract_limit_approach(self, text: str) -> str:
        m = re.search(r'(?:→|->|\\to)\s*([^\s}\]]+)', text)
        return m.group(1) if m else "∞"

    def _extract_limit_expression(self, text: str) -> str:
        m = re.search(r'\]\s*(.+)$', text)
        return m.group(1).strip() if m else "f(x)"

    def _extract_diff_variable(self, text: str) -> str:
        m = re.search(r'd/d([a-zA-Z])|∂/∂([a-zA-Z])', text)
        if m:
            return m.group(1) or m.group(2)
        m = re.search(r'dy/d([a-zA-Z])', text)
        return m.group(1) if m else "x"

    def _extract_diff_function(self, text: str) -> str:
        m = re.search(r'd([A-Za-z])/d', text)
        return m.group(1) if m else "y"

    def _extract_matrix_dimensions(self, text: str) -> Tuple[int, int]:
        rows = text.count("\\\\") + 1 if "\\\\" in text else 2
        cols = text.count("&") + 1   if "&"    in text else 2
        return rows, cols

    def _extract_variables(self, text: str) -> List[Variable]:
        symbols = re.findall(r'\b([A-Za-z_][A-Za-z_0-9]*)\b', text)
        skip    = {"sin","cos","tan","log","ln","exp","lim","sum","int",
                   "if","else","for","in","and","or","not","is"}
        seen    = set()
        result  = []
        for sym in symbols:
            if sym.lower() not in skip and sym not in seen and len(sym) <= 3:
                seen.add(sym)
                result.append(Variable(sym, "real", "", ""))
        return result

    def _detect_region_type(self, text: str) -> str:
        if text.startswith("$"):   return "latex"
        if "∫" in text:            return "integral"
        if "∑" in text:            return "summation"
        if "lim" in text.lower():  return "limit"
        if "=" in text:            return "assignment"
        return "expression"

    def _to_canonical(self, text: str) -> str:
        """Convert raw expression to canonical token form."""
        rev_map = {v: k for k, v in UNICODE_MAP.items()}
        result  = text
        for unicode_ch, token in rev_map.items():
            result = result.replace(unicode_ch, token)
        return result

    def _to_latex(self, text: str) -> str:
        """Best-effort conversion of expression to LaTeX."""
        result = text
        latex_subs = {
            "α": r"\alpha", "β": r"\beta",  "γ": r"\gamma",
            "δ": r"\delta", "π": r"\pi",    "σ": r"\sigma",
            "ω": r"\omega", "∞": r"\infty", "∫": r"\int",
            "∑": r"\sum",   "∏": r"\prod",  "√": r"\sqrt",
            "≤": r"\leq",   "≥": r"\geq",   "≠": r"\neq",
            "≈": r"\approx","±": r"\pm",    "×": r"\times",
            "∂": r"\partial","∇": r"\nabla",
        }
        for ch, latex in latex_subs.items():
            result = result.replace(ch, latex)
        return result
