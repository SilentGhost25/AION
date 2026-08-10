"""
AION VRE Render QA Validator
============================
Verifies that rendered SVG string matches the VKO parameters exactly.
"""

from __future__ import annotations

from typing import List, Tuple
from .contracts import VKO


class RenderValidator:
    """Render QA Validator enforcing SVG vs VKO fidelity."""

    @classmethod
    def validate(cls, svg_content: str, vko: VKO) -> Tuple[bool, List[str]]:
        errors = []
        if not svg_content or not svg_content.startswith("<svg"):
            return (False, ["INVALID_SVG_MARKUP"])

        # Validate presence of VKO node labels or component values
        if "GRAPH" in vko.figure_class:
            for node_id, label in vko.labels.node_labels.items():
                if label not in svg_content:
                    errors.append(f"MISSING_NODE_LABEL_IN_SVG:{label}")

            for edge_id, weight in vko.quantities.edge_weights.items():
                w_str = f"{weight:g}"
                if w_str not in svg_content:
                    errors.append(f"MISSING_EDGE_WEIGHT_IN_SVG:{w_str}")

        elif "CIRCUIT" in vko.figure_class:
            for comp_id, (val, unit) in vko.quantities.component_values.items():
                val_str = f"{val:g}"
                if val_str not in svg_content:
                    errors.append(f"MISSING_COMPONENT_VALUE_IN_SVG:{comp_id}={val_str}")

        elif vko.figure_class == "BEAM":
            if vko.quantities.span_length is not None:
                span_str = f"{vko.quantities.span_length:g}"
                if span_str not in svg_content:
                    errors.append(f"MISSING_SPAN_LENGTH_IN_SVG:{span_str}")

        return (len(errors) == 0, errors)
