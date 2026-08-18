"""
AION Expression Parser
======================
Converts raw text or LaTeX into an ExpressionTree.

Algorithm:
    1. Detect mathematical regions in text
    2. Tokenize the expression
    3. Apply Pratt parser (top-down operator precedence)
    4. Build ExprNode tree bottom-up
    5. Identify equation type and wrap in EKO

References:
    - Pratt, V.R. (1973) "Top Down Operator Precedence"
    - Same algorithm used in Python's ast module
"""

import re
from typing import Optional, List, Tuple
from .expression_tree import (
    ExprNode, NodeType,
    num, var, const, op, fn,
    integral, derivative, limit, summation, equation, fraction, sqrt,
    NAMED_CONSTANTS, OPERATOR_META, FUNCTION_META,
)


# -- Tokenizer -----------------------------------------------------------------

class Token:
    def __init__(self, kind: str, value: str):
        self.kind  = kind    # NUM, VAR, OP, FUNC, LPAREN, RPAREN, etc.
        self.value = value

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


def tokenize(expr: str) -> List[Token]:
    """
    Convert expression string to token list.
    Handles Unicode math symbols, LaTeX commands, and ASCII.
    """
    tokens = []
    i      = 0
    expr   = _normalize(expr)

    while i < len(expr):
        ch = expr[i]

        # Whitespace
        if ch.isspace():
            i += 1
            continue

        # Numbers
        if ch.isdigit() or (ch == '.' and i+1 < len(expr) and expr[i+1].isdigit()):
            j = i
            while j < len(expr) and (expr[j].isdigit() or expr[j] == '.'):
                j += 1
            tokens.append(Token("NUM", expr[i:j]))
            i = j
            continue

        # Identifiers and functions
        if ch.isalpha() or ch == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            word = expr[i:j]
            if word in FUNCTION_META or word in ("int", "sum", "lim", "prod"):
                tokens.append(Token("FUNC", word))
            elif word in NAMED_CONSTANTS:
                tokens.append(Token("CONST", word))
            else:
                tokens.append(Token("VAR", word))
            i = j
            continue

        # Operators and brackets
        if ch in "+-*/^":
            tokens.append(Token("OP", ch))
            i += 1
        elif ch == '(':
            tokens.append(Token("LPAREN", ch))
            i += 1
        elif ch == ')':
            tokens.append(Token("RPAREN", ch))
            i += 1
        elif ch == ',':
            tokens.append(Token("COMMA", ch))
            i += 1
        elif ch == '=':
            tokens.append(Token("EQ", ch))
            i += 1
        elif ch in '<>':
            rel = ch
            if i+1 < len(expr) and expr[i+1] == '=':
                rel += '='
                i += 1
            tokens.append(Token("REL", rel))
            i += 1
        elif ch == '_':
            tokens.append(Token("SUB", ch))
            i += 1
        else:
            i += 1  # skip unknown

    return tokens


def _normalize(expr: str) -> str:
    """Normalize Unicode math symbols to ASCII equivalents for tokenization."""
    replacements = {
        '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5',
        '₀': '_0', '₁': '_1', '₂': '_2', '₃': '_3',
        '×': '*', '÷': '/', '·': '*',
        '√': 'sqrt', '∛': 'cbrt',
        '∫': 'int', '∑': 'sum', '∏': 'prod',
        '∂': 'd', '∇': 'nabla',
        '∞': 'inf', 'π': 'pi', 'α': 'alpha', 'β': 'beta',
        'γ': 'gamma', 'δ': 'delta', 'θ': 'theta', 'λ': 'lambda',
        'μ': 'mu', 'σ': 'sigma', 'ω': 'omega', 'φ': 'phi',
        '->': '->', '←': '<-', '↔': '<->',
        '≤': '<=', '≥': '>=', '≠': '!=', '≈': '~=',
        '±': 'pm',
    }
    for unicode_ch, ascii_eq in replacements.items():
        expr = expr.replace(unicode_ch, ascii_eq)

    # Remove LaTeX commands
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}',
                  r'(\1)/(\2)', expr)
    expr = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', expr)
    expr = re.sub(r'\\left|\\right', '', expr)
    expr = re.sub(r'\\[a-zA-Z]+', lambda m: m.group()[1:], expr)
    expr = re.sub(r'\{([^}]*)\}', r'(\1)', expr)

    return expr


# -- Pratt Parser --------------------------------------------------------------

class PrattParser:
    """
    Top-down operator precedence parser.
    Converts token list into ExprNode tree.

    Precedence levels:
        0 — lowest (equation)
        1 — addition, subtraction
        2 — multiplication, division
        3 — exponentiation
        4 — unary operators
        5 — function calls, subscripts
    """

    PREC = {
        "EQ":  0, "REL": 0,
        "+":   1, "-":   1,
        "*":   2, "/":   2,
        "^":   3,
    }

    def __init__(self, tokens: List[Token]):
        self.tokens  = tokens
        self.pos     = 0

    def peek(self) -> Optional[Token]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, kind: str = None) -> Optional[Token]:
        tok = self.peek()
        if tok is None:
            return None
        if kind and tok.kind != kind:
            return None
        self.pos += 1
        return tok

    def parse(self) -> Optional[ExprNode]:
        """Parse the full expression."""
        if not self.tokens:
            return None
        node = self._parse_expr(0)

        # Check for equation
        if self.peek() and self.peek().kind in ("EQ", "REL"):
            rel_tok = self.consume()
            rhs     = self._parse_expr(0)
            return ExprNode(
                NodeType.EQUATION,
                lhs      = node,
                rhs      = rhs,
                relation = rel_tok.value,
            )
        return node

    def _parse_expr(self, min_prec: int) -> Optional[ExprNode]:
        left = self._parse_unary()
        if left is None:
            return None

        while True:
            tok = self.peek()
            if tok is None:
                break
            if tok.kind == "OP":
                prec = self.PREC.get(tok.value, -1)
                if prec <= min_prec:
                    break
                self.consume()
                right = self._parse_expr(prec)
                if right is None:
                    break
                if tok.value == "/":
                    left = fraction(left, right)
                else:
                    left = op(tok.value, left, right)
            else:
                break

        return left

    def _parse_unary(self) -> Optional[ExprNode]:
        tok = self.peek()
        if tok and tok.kind == "OP" and tok.value == "-":
            self.consume()
            operand = self._parse_unary()
            return op("neg", operand) if operand else None
        return self._parse_primary()

    def _parse_primary(self) -> Optional[ExprNode]:
        tok = self.peek()
        if tok is None:
            return None

        # Number
        if tok.kind == "NUM":
            self.consume()
            val = float(tok.value)
            return num(int(val) if val == int(val) else val)

        # Constant
        if tok.kind == "CONST":
            self.consume()
            return const(tok.value)

        # Function call
        if tok.kind == "FUNC":
            return self._parse_function()

        # Variable
        if tok.kind == "VAR":
            self.consume()
            node = var(tok.value)
            # Check for subscript
            if self.peek() and self.peek().kind == "SUB":
                self.consume()
                sub = self._parse_primary()
                node.metadata["subscript"] = sub
            return node

        # Parenthesized expression
        if tok.kind == "LPAREN":
            self.consume()
            inner = self._parse_expr(0)
            self.consume("RPAREN")
            return inner

        return None

    def _parse_function(self) -> Optional[ExprNode]:
        tok = self.consume()
        name = tok.value

        # Special forms
        if name == "int":
            return self._parse_integral_form()
        if name == "sum":
            return self._parse_sum_form()
        if name == "lim":
            return self._parse_limit_form()
        if name == "sqrt":
            self.consume("LPAREN")
            arg = self._parse_expr(0)
            self.consume("RPAREN")
            return sqrt(arg) if arg else ExprNode(NodeType.SQRT)

        # Regular function f(args)
        args = []
        if self.peek() and self.peek().kind == "LPAREN":
            self.consume()
            while self.peek() and self.peek().kind != "RPAREN":
                arg = self._parse_expr(0)
                if arg:
                    args.append(arg)
                if self.peek() and self.peek().kind == "COMMA":
                    self.consume()
            self.consume("RPAREN")

        return fn(name, *args)

    def _parse_integral_form(self) -> ExprNode:
        """Parse: int_{lower}^{upper} expr dvariable"""
        lower_node = None
        upper_node = None

        tok = self.peek()
        if tok and tok.kind == "SUB":
            self.consume()
            lower_node = self._parse_primary()
        if self.peek() and self.peek().kind == "OP" and self.peek().value == "^":
            self.consume()
            upper_node = self._parse_primary()

        expr_node = self._parse_expr(0)
        variable  = "x"

        tok = self.peek()
        if tok and tok.kind == "VAR" and tok.value.startswith("d"):
            variable = tok.value[1:] or "x"
            self.consume()

        return integral(expr_node or num(0), variable, lower_node, upper_node)

    def _parse_sum_form(self) -> ExprNode:
        """Parse: sum_{index=lower}^{upper} expr"""
        index    = "i"
        lower_n  = num(1)
        upper_n  = const("inf")

        if self.peek() and self.peek().kind == "SUB":
            self.consume()
            # Try to parse index=lower
            if self.peek() and self.peek().kind == "VAR":
                index = self.consume().value
            if self.peek() and self.peek().kind == "EQ":
                self.consume()
                lower_n = self._parse_primary() or num(1)

        if self.peek() and self.peek().kind == "OP" and self.peek().value == "^":
            self.consume()
            upper_n = self._parse_primary() or const("inf")

        expr_node = self._parse_expr(0)
        return summation(expr_node or var("a_n"), index, lower_n, upper_n)

    def _parse_limit_form(self) -> ExprNode:
        """Parse: lim_{var->approach} expr"""
        variable = "x"
        approach = const("inf")

        if self.peek() and self.peek().kind == "SUB":
            self.consume()
            if self.peek() and self.peek().kind == "VAR":
                variable = self.consume().value
            # Skip -> or ->
            if self.peek() and self.peek().kind == "OP":
                self.consume()
            approach = self._parse_primary() or const("inf")

        expr_node = self._parse_expr(0)
        return limit(expr_node or var("f"), variable, approach)


# -- Main Parser Interface -----------------------------------------------------

class ExpressionParser:
    """Main interface for parsing mathematical expressions."""

    def parse(self, text: str) -> Optional[ExprNode]:
        """
        Parse a mathematical expression string into an ExprNode tree.
        Handles plain text, LaTeX, and Unicode.
        """
        text = text.strip()
        if not text:
            return None
        try:
            tokens = tokenize(text)
            parser = PrattParser(tokens)
            return parser.parse()
        except Exception:
            return None

    def parse_equation(self, text: str) -> Optional[ExprNode]:
        """Parse an equation (must contain = sign)."""
        if "=" not in text:
            return None
        return self.parse(text)

    def detect_math_regions(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Find all mathematical sub-expressions in a block of text.
        Returns list of (expression_string, start, end).
        """
        regions = []
        patterns = [
            r'\$\$(.+?)\$\$',
            r'\$(.+?)\$',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'[A-Za-z]\s*=\s*[A-Za-z0-9\+\-\*/\^\(\)√∫∑\s\.]{3,60}',
            r'[∫∑∏√∂]\s*[^\n\.]{3,80}',
            r'\b(?:lim|int|sum)\b[^\n]{3,80}',
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
                expr = (match.group(1) if match.lastindex else match.group()).strip()
                if len(expr) >= 3:
                    regions.append((expr, match.start(), match.end()))
        return regions
