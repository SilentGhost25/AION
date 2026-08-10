"""
AION VRE Grounded Generator (GG)
================================
Formulates QuestionPlan with sha256 hash and solver-authoritative prompt rules.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Dict
from .contracts import OperationChain, QuestionPlan, VKO
from .taxonomy import GROUNDING_VOCABULARY


class GG:
    """Grounded Generator (Algorithm 6)."""

    @classmethod
    def generate_question_plan(
        cls,
        vko: VKO,
        chain: OperationChain,
        bloom_level: str,
        marks: int,
        reference_solution: Any,
    ) -> QuestionPlan:
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        op = chain.steps[0].operation if chain.steps else "ANALYZE"

        src = "A"
        dst = "D"
        if vko.topology.nodes:
            src = vko.topology.nodes[0].id
            dst = vko.topology.nodes[-1].id

        anchors = cls._extract_anchors(vko, op, reference_solution)

        # Compute sha256 QuestionPlan hash
        raw_payload = {
            "op": op,
            "vko_id": vko.id,
            "bloom": bloom_level,
            "marks": marks,
            "src": src,
            "dst": dst,
            "solution": str(reference_solution),
        }
        plan_hash = hashlib.sha256(json.dumps(raw_payload, sort_keys=True).encode()).hexdigest()[:16]

        return QuestionPlan(
            plan_id=plan_id,
            operation=op,
            vko_id=vko.id,
            bloom_level=bloom_level,
            marks=marks,
            source_element=src,
            destination_element=dst,
            expected_output=chain.expected_output_type,
            question_plan_hash=plan_hash,
            anchors=anchors,
            reference_solution=reference_solution,
        )

    @classmethod
    def render_question_text(cls, plan: QuestionPlan, vko: VKO) -> str:
        """
        Renders question text using solver-authoritative reference solutions.
        The solver determines mathematical/academic validity; LLM formulates language.
        """
        op = plan.operation

        if op == "DIJKSTRA":
            src = plan.anchors.get("source", plan.source_element)
            dst = plan.anchors.get("destination", plan.destination_element)
            text = (
                f"Using the given weighted graph, apply Dijkstra's algorithm to determine "
                f"the shortest path and total minimum cost from vertex {src} to vertex {dst}."
            )

        elif op in ("EQUIVALENT_RESISTANCE", "KVL"):
            v_val = plan.anchors.get("voltage_source", "12V")
            text = (
                f"With reference to the given circuit diagram with DC supply of {v_val}, "
                f"calculate the total equivalent resistance and total source current."
            )

        elif op in ("REACTIONS", "SFD", "BMD"):
            span = plan.anchors.get("span_length", "6m")
            text = (
                f"Using the simply supported beam of span {span} shown in the figure, "
                f"determine the support reaction forces at supports A and B."
            )

        elif op in ("INSERT_ROTATE", "INSERT"):
            text = (
                f"Construct the balanced AVL tree by inserting the sequence of keys "
                f"into the given initial tree structure, showing all necessary rotations."
            )

        else:
            text = f"With reference to the given figure, analyze and evaluate the operational parameters."

        vocab = GROUNDING_VOCABULARY.get(vko.figure_class, {})
        for forbidden in vocab.get("FORBIDDEN", []):
            if forbidden.lower() in text.lower():
                text = text.replace(forbidden, "figure")

        return text

    @staticmethod
    def format_solver_authoritative_prompt(plan: QuestionPlan, vko: VKO) -> str:
        """Constructs solver-authoritative prompt instructions for Qwen/LLM."""
        return (
            f"OPERATION:\n{plan.operation}\n\n"
            f"GRAPH / TOPOLOGY:\n{vko.figure_class}\n\n"
            f"SOURCE:\n{plan.source_element}\n\n"
            f"DESTINATION:\n{plan.destination_element}\n\n"
            f"REFERENCE SOLUTION:\n{plan.reference_solution}\n\n"
            f"TASK:\n"
            f"Generate a VTU-style {plan.bloom_level} question using this figure.\n"
            f"Do not modify figure values.\n"
            f"Do not invent another answer.\n"
            f"Do not ask the student to describe the figure."
        )

    @staticmethod
    def _extract_anchors(vko: VKO, operation: str, solution: Any) -> Dict[str, Any]:
        anchors = {}
        if operation == "DIJKSTRA" and isinstance(solution, dict):
            anchors["source"] = solution.get("source", "A")
            anchors["destination"] = solution.get("destination", "D")
            anchors["shortest_path"] = solution.get("shortest_path", [])
            anchors["total_cost"] = solution.get("total_cost", 0.0)

        elif operation in ("EQUIVALENT_RESISTANCE", "KVL") and isinstance(solution, dict):
            anchors["voltage_source"] = f"{solution.get('v_source', 12)}V"
            anchors["r_equivalent"] = f"{solution.get('r_equivalent', 60)}Ω"

        elif operation in ("REACTIONS", "SFD", "BMD") and isinstance(solution, dict):
            anchors["span_length"] = f"{solution.get('span_length', 6)}m"
            anchors["reaction_A"] = f"{solution.get('reaction_A', 10)}kN"

        return anchors
