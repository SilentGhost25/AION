"""
AION Tree Renderer
==================
Converts an ExprNode tree into LaTeX, Unicode, or Plain Text strings.
"""

from .expression_tree import (
    ExprNode, NodeType,
    NAMED_CONSTANTS, OPERATOR_META, FUNCTION_META
)


class TreeRenderer:
    """
    Renders ExprNode trees into LaTeX, Unicode, or ASCII text.
    Separates mathematical structure from visual rendering.
    """

    def to_unicode(self, node: ExprNode) -> str:
        """Render ExprNode as clean Unicode string."""
        if node is None:
            return ""

        if node.node_type == NodeType.NUMBER:
            return str(node.value)
        if node.node_type == NodeType.VARIABLE:
            sub = node.metadata.get("subscript")
            sub_str = f"_{self.to_unicode(sub)}" if sub else ""
            return f"{node.value}{sub_str}"
        if node.node_type == NodeType.CONSTANT:
            return NAMED_CONSTANTS.get(node.value, (node.value,))[0]

        if node.node_type == NodeType.OPERATOR:
            if len(node.children) == 2:
                op_sym = " × " if node.value == "*" else f" {node.value} "
                return f"({self.to_unicode(node.children[0])}{op_sym}{self.to_unicode(node.children[1])})"
            if node.value == "neg":
                return f"-{self.to_unicode(node.children[0])}"
            return f"{node.value}({self.to_unicode(node.children[0])})"

        if node.node_type == NodeType.FRACTION:
            num_str = self.to_unicode(node.children[0])
            den_str = self.to_unicode(node.children[1])
            return f"({num_str})/({den_str})"

        if node.node_type == NodeType.SQRT:
            return f"√({self.to_unicode(node.children[0])})"

        if node.node_type == NodeType.FUNCTION:
            args = ", ".join(self.to_unicode(c) for c in node.children)
            return f"{node.value}({args})"

        if node.node_type == NodeType.EQUATION:
            return f"{self.to_unicode(node.lhs)} {node.relation} {self.to_unicode(node.rhs)}"

        if node.node_type == NodeType.INTEGRAL:
            lower = self.to_unicode(node.lower) if node.lower else ""
            upper = self.to_unicode(node.upper) if node.upper else ""
            bounds = f"_{{{lower}}}^{{{upper}}}" if lower or upper else ""
            body = self.to_unicode(node.children[0]) if node.children else ""
            return f"∫{bounds} {body} d{node.variable or 'x'}"

        if node.node_type == NodeType.DERIVATIVE:
            body = self.to_unicode(node.children[0]) if node.children else ""
            return f"d^{node.order}/d{node.variable}^{node.order} ({body})"

        if node.node_type == NodeType.LIMIT:
            app = self.to_unicode(node.upper) if node.upper else "∞"
            body = self.to_unicode(node.children[0]) if node.children else ""
            return f"lim({node.variable}->{app}) {body}"

        if node.node_type == NodeType.SUMMATION:
            low = self.to_unicode(node.lower) if node.lower else "1"
            up = self.to_unicode(node.upper) if node.upper else "∞"
            body = self.to_unicode(node.children[0]) if node.children else ""
            return f"∑({node.variable}={low} to {up}) {body}"

        return repr(node)

    def to_latex(self, node: ExprNode) -> str:
        """Render ExprNode as LaTeX string for MathJax / PDF rendering."""
        if node is None:
            return ""

        if node.node_type == NodeType.NUMBER:
            return str(node.value)
        if node.node_type == NodeType.VARIABLE:
            sub = node.metadata.get("subscript")
            sub_str = f"_{{{self.to_latex(sub)}}}" if sub else ""
            return f"{node.value}{sub_str}"
        if node.node_type == NodeType.CONSTANT:
            const_latex = {
                "pi": r"\pi", "e": "e", "i": "i", "inf": r"\infty",
                "g": "g", "c": "c", "h": "h", "k": "k"
            }
            return const_latex.get(node.value, node.value)

        if node.node_type == NodeType.OPERATOR:
            if len(node.children) == 2:
                if node.value == "*":
                    return f"{self.to_latex(node.children[0])} \\cdot {self.to_latex(node.children[1])}"
                if node.value == "^":
                    return f"{self.to_latex(node.children[0])}^{{{self.to_latex(node.children[1])}}}"
                return f"{self.to_latex(node.children[0])} {node.value} {self.to_latex(node.children[1])}"
            if node.value == "neg":
                return f"-{self.to_latex(node.children[0])}"
            return f"{node.value}({self.to_latex(node.children[0])})"

        if node.node_type == NodeType.FRACTION:
            return f"\\frac{{{self.to_latex(node.children[0])}}}{{{self.to_latex(node.children[1])}}}"

        if node.node_type == NodeType.SQRT:
            return f"\\sqrt{{{self.to_latex(node.children[0])}}}"

        if node.node_type == NodeType.FUNCTION:
            fn_meta = FUNCTION_META.get(node.value, {})
            fn_latex = fn_meta.get("latex", f"\\text{{{node.value}}}")
            args = ", ".join(self.to_latex(c) for c in node.children)
            return f"{fn_latex}({args})"

        if node.node_type == NodeType.EQUATION:
            rel = r"\leq" if node.relation == "<=" else (r"\geq" if node.relation == ">=" else node.relation)
            return f"{self.to_latex(node.lhs)} {rel} {self.to_latex(node.rhs)}"

        if node.node_type == NodeType.INTEGRAL:
            low = f"_{{{self.to_latex(node.lower)}}}" if node.lower else ""
            up = f"^{{{self.to_latex(node.upper)}}}" if node.upper else ""
            body = self.to_latex(node.children[0]) if node.children else ""
            return f"\\int{low}{up} {body} \\, d{node.variable or 'x'}"

        if node.node_type == NodeType.DERIVATIVE:
            body = self.to_latex(node.children[0]) if node.children else ""
            order_str = f"^{{{node.order}}}" if node.order > 1 else ""
            return f"\\frac{{d{order_str}}}{{d{node.variable}{order_str}}} \\left({body}\\right)"

        if node.node_type == NodeType.LIMIT:
            app = self.to_latex(node.upper) if node.upper else r"\infty"
            body = self.to_latex(node.children[0]) if node.children else ""
            return f"\\lim_{{{node.variable} \\to {app}}} {body}"

        if node.node_type == NodeType.SUMMATION:
            low = self.to_latex(node.lower) if node.lower else "1"
            up = self.to_latex(node.upper) if node.upper else r"\infty"
            body = self.to_latex(node.children[0]) if node.children else ""
            return f"\\sum_{{{node.variable}={low}}}^{{{up}}} {body}"

        return str(node)
