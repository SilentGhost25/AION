"""
AION VRE Visual Question Graph Renderer (VQGR)
==============================================
Declarative Figure Synthesis (DFS) rendering pure-Python SVG representations
for updated VKOs (Algorithm 8).
"""

from __future__ import annotations

from .contracts import RenderMode, VKO


class VQGR:
    """Visual Question Graph Renderer (Algorithm 8)."""

    @classmethod
    def render(cls, vko: VKO, render_mode: RenderMode = RenderMode.SVG) -> str:
        if render_mode == RenderMode.ORIGINAL_SOURCE and vko.source_image:
            return f'<img src="{vko.source_image}" alt="Original Figure"/>'
        elif render_mode == RenderMode.ANNOTATED_SOURCE and vko.source_image:
            return f'<div class="annotated-figure"><img src="{vko.source_image}"/><svg class="overlay"></svg></div>'

        # Default SVG synthesis
        fig_cls = vko.figure_class
        if "GRAPH" in fig_cls:
            return cls._render_graph_svg(vko)
        elif "TREE" in fig_cls:
            return cls._render_tree_svg(vko)
        elif "CIRCUIT" in fig_cls:
            return cls._render_circuit_svg(vko)
        elif fig_cls == "BEAM":
            return cls._render_beam_svg(vko)
        return cls._render_generic_svg(vko)

    @staticmethod
    def _render_graph_svg(vko: VKO) -> str:
        svg_parts = [
            '<svg width="500" height="300" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff; font-family:sans-serif;">',
            '  <defs>',
            '    <marker id="arrow" viewBox="0 0 10 10" refX="22" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">',
            '      <path d="M 0 0 L 10 5 L 0 10 z" fill="#333"/>',
            '    </marker>',
            '  </defs>',
        ]

        # Node positions map
        node_pos = {}
        for n in vko.topology.nodes:
            node_pos[n.id] = n.position if n.position != (0, 0) else (100, 100)

        # Render edges
        for edge in vko.topology.edges:
            x1, y1 = node_pos.get(edge.from_node, (50, 50))
            x2, y2 = node_pos.get(edge.to_node, (200, 200))
            w = vko.quantities.edge_weights.get(edge.id, 1.0)
            marker = ' marker-end="url(#arrow)"' if edge.directed else ''

            svg_parts.append(f'  <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#333" stroke-width="2"{marker}/>')

            # Weight label midpoint
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2 - 5
            svg_parts.append(f'  <text x="{mx}" y="{my}" font-size="14" fill="#c00" font-weight="bold">{w:g}</text>')

        # Render nodes
        for n in vko.topology.nodes:
            x, y = node_pos.get(n.id, (100, 100))
            fill = "#e0f2fe" if n.is_source else "#fef3c7" if n.is_sink else "#ffffff"
            label = vko.labels.node_labels.get(n.id, n.id)

            svg_parts.append(f'  <circle cx="{x}" cy="{y}" r="18" fill="{fill}" stroke="#0284c7" stroke-width="2"/>')
            svg_parts.append(f'  <text x="{x}" y="{y+5}" font-size="14" font-weight="bold" text-anchor="middle" fill="#0f172a">{label}</text>')

        svg_parts.append('</svg>')
        return "\n".join(svg_parts)

    @staticmethod
    def _render_tree_svg(vko: VKO) -> str:
        return (
            '<svg width="400" height="250" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;">'
            '<circle cx="200" cy="50" r="18" fill="#e0f2fe" stroke="#0284c7" stroke-width="2"/>'
            '<text x="200" y="55" font-size="14" text-anchor="middle">10</text>'
            '<line x1="200" y1="50" x2="100" y2="120" stroke="#333" stroke-width="2"/>'
            '<circle cx="100" cy="120" r="18" fill="#ffffff" stroke="#0284c7" stroke-width="2"/>'
            '<text x="100" y="125" font-size="14" text-anchor="middle">5</text>'
            '<line x1="200" y1="50" x2="300" y2="120" stroke="#333" stroke-width="2"/>'
            '<circle cx="300" cy="120" r="18" fill="#ffffff" stroke="#0284c7" stroke-width="2"/>'
            '<text x="300" y="125" font-size="14" text-anchor="middle">15</text>'
            '</svg>'
        )

    @staticmethod
    def _render_circuit_svg(vko: VKO) -> str:
        comp_vals = vko.quantities.component_values
        v1 = comp_vals.get("V1", (12.0, "V"))[0]
        r1 = comp_vals.get("R1", (10.0, "Ω"))[0]
        r2 = comp_vals.get("R2", (20.0, "Ω"))[0]
        r3 = comp_vals.get("R3", (30.0, "Ω"))[0]

        return (
            f'<svg width="500" height="220" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;">'
            f'<rect x="50" y="40" width="400" height="130" fill="none" stroke="#333" stroke-width="2"/>'
            f'<text x="20" y="105" font-size="14" font-weight="bold" fill="#0284c7">V1={v1:g}V</text>'
            f'<text x="120" y="30" font-size="14" font-weight="bold" fill="#c00">R1={r1:g}Ω</text>'
            f'<text x="250" y="30" font-size="14" font-weight="bold" fill="#c00">R2={r2:g}Ω</text>'
            f'<text x="370" y="30" font-size="14" font-weight="bold" fill="#c00">R3={r3:g}Ω</text>'
            f'</svg>'
        )

    @staticmethod
    def _render_beam_svg(vko: VKO) -> str:
        span = vko.quantities.span_length or 6.0
        load_val = vko.quantities.component_values.get("load_P1", (20.0, "kN"))[0]

        return (
            f'<svg width="500" height="180" xmlns="http://www.w3.org/2000/svg" style="background:#ffffff;">'
            f'<rect x="50" y="90" width="400" height="15" fill="#94a3b8" stroke="#333"/>'
            f'<text x="250" y="130" font-size="14" text-anchor="middle">Span = {span:g} m</text>'
            f'<line x1="250" y1="30" x2="250" y2="85" stroke="#dc2626" stroke-width="3"/>'
            f'<text x="250" y="25" font-size="14" font-weight="bold" text-anchor="middle" fill="#dc2626">P1 = {load_val:g} kN</text>'
            f'</svg>'
        )

    @staticmethod
    def _render_generic_svg(vko: VKO) -> str:
        return (
            '<svg width="300" height="150" xmlns="http://www.w3.org/2000/svg">'
            '<rect width="300" height="150" fill="#f8fafc"/>'
            '<text x="150" y="80" text-anchor="middle" font-size="14">Generic Figure</text>'
            '</svg>'
        )
