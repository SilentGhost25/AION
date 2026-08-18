"""
AION Master Production Specification — Paper Structure Planner
===============================================================
Deterministic engine that constructs an immutable PaperStructurePlan before any Qwen calls begin.
Executes Phase 1 (MDE mark distribution), Phase 2 (JBMCS Bloom resolution), Phase 3 (OR parity question numbering),
and Phase 4 (Plan finalization).
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timezone
from typing import List, Tuple

from core.contracts.generation_request import GenerationRequest
from core.contracts.paper_structure import (
    ORPairDescriptor,
    PaperStructurePlan,
    SlotDescriptor,
)
from v0_1.structural_v2.contracts import BloomLevel
from v0_1.structural_v2.jbmcs import JointBloomMarkConstraintSolver
from v0_1.structural_v2.mde import MarkDistributionEngine


class PaperPlannerError(Exception):
    """Raised when paper structure planning fails."""
    pass


VTU_SLOT_CONSTRAINTS = {
    4:  {"max_co": 2, "allowed_blooms": ["L1", "L2", "L3"]},
    5:  {"max_co": 3, "allowed_blooms": ["L2", "L3"]},
    6:  {"max_co": 3, "allowed_blooms": ["L2", "L3", "L4"]},
    8:  {"max_co": 4, "allowed_blooms": ["L3", "L4", "L5"]},
    10: {"max_co": 5, "allowed_blooms": ["L3", "L4", "L5"]},
}


def _align_to_vtu(marks: int, co_val: str, bloom_val: str) -> Tuple[str, str]:
    if marks in VTU_SLOT_CONSTRAINTS:
        rules = VTU_SLOT_CONSTRAINTS[marks]
        try:
            co_num = int(co_val.replace("CO", ""))
            if co_num > rules["max_co"]:
                co_val = f"CO{rules['max_co']}"
        except ValueError:
            pass
        if bloom_val not in rules["allowed_blooms"]:
            bloom_val = rules["allowed_blooms"][-1]
    return co_val, bloom_val


class PaperStructurePlanner:
    """Deterministic Paper Structure Planner."""

    @classmethod
    def build(cls, request: GenerationRequest) -> PaperStructurePlan:
        # Validate request first
        if not request.validated:
            request.validate()

        marks_per_module = request.total_marks // len(request.modules)
        if marks_per_module < request.subquestion_count:
            raise PaperPlannerError(
                f"marks_per_module ({marks_per_module}) < subquestion_count ({request.subquestion_count})"
            )

        # Phase 1: Mark Distribution
        mark_dist = MarkDistributionEngine.compute(
            total=marks_per_module,
            n=request.subquestion_count,
            policy=request.distribution_policy,
            custom=request.custom_distribution,
        )
        D: Tuple[int, ...] = tuple(mark_dist)

        # Phase 2: Bloom Profile Resolution
        rng = random.Random(request.seed or 42)
        bloom_enums = []
        for b_str in request.bloom_levels:
            if hasattr(BloomLevel, b_str):
                bloom_enums.append(getattr(BloomLevel, b_str))
        if not bloom_enums:
            bloom_enums = [BloomLevel.L2, BloomLevel.L3]

        bloom_profile_enums = JointBloomMarkConstraintSolver.solve(
            D,
            bloom_enums,
            request.subquestion_count,
            rng,
        )
        bloom_profile: Tuple[str, ...] = tuple(b.name if hasattr(b, 'name') else str(b) for b in bloom_profile_enums)

        # Phase 3: Question Numbering & OR Pair Construction
        sub_labels = ["a", "b", "c", "d"][: request.subquestion_count]
        or_pairs: List[ORPairDescriptor] = []
        question_counter = 1
        n_mods = len(request.modules)

        for mod_pos, mod in enumerate(request.modules):
            alt_a_no = question_counter
            alt_b_no = question_counter + 1
            question_counter += 2

            if request.co_mapping and mod in request.co_mapping:
                co_val = request.co_mapping[mod]
            elif n_mods <= 3:
                co_val = f"CO{mod_pos + 1}"
            elif n_mods <= 5:
                if mod_pos < 2:   co_val = "CO1"
                elif mod_pos < 4: co_val = "CO2"
                else:             co_val = "CO3"
            else:
                co_idx = min(5, (mod_pos * 5 // n_mods) + 1)
                co_val = f"CO{co_idx}"

            slots_a: List[SlotDescriptor] = []
            slots_b: List[SlotDescriptor] = []

            for i, (marks, bloom) in enumerate(zip(D, bloom_profile)):
                q_type = request.question_types[i % len(request.question_types)]
                aligned_co, aligned_bloom = _align_to_vtu(marks, co_val, bloom)

                slot_a = SlotDescriptor(
                    slot_id=f"Q{alt_a_no}{sub_labels[i]}",
                    question_no=alt_a_no,
                    sub_label=sub_labels[i],
                    module_id=mod,
                    marks=marks,          # LOCKED
                    co=aligned_co,        # LOCKED
                    bloom=aligned_bloom,  # LOCKED
                    question_type=q_type, # LOCKED
                )

                slot_b = SlotDescriptor(
                    slot_id=f"Q{alt_b_no}{sub_labels[i]}",
                    question_no=alt_b_no,
                    sub_label=sub_labels[i],
                    module_id=mod,
                    marks=marks,          # LOCKED — SAME as slot_a
                    co=aligned_co,        # LOCKED — SAME as slot_a
                    bloom=aligned_bloom,  # LOCKED — SAME as slot_a
                    question_type=q_type, # LOCKED
                )

                slots_a.append(slot_a)
                slots_b.append(slot_b)

            or_pair = ORPairDescriptor(
                module_id=mod,
                alt_a_question_no=alt_a_no,
                alt_b_question_no=alt_b_no,
                total_marks=marks_per_module,
                subquestion_count=request.subquestion_count,
                mark_distribution=D,
                slots_a=tuple(slots_a),
                slots_b=tuple(slots_b),
            )
            or_pairs.append(or_pair)

        # Phase 4: Finalize Plan
        plan = PaperStructurePlan(
            plan_id=str(uuid.uuid4()),
            request_id=request.request_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            total_marks=request.total_marks,
            module_count=len(request.modules),
            marks_per_module=marks_per_module,
            subquestion_count=request.subquestion_count,
            distribution_policy=request.distribution_policy,
            mark_distribution=D,
            or_pairs=tuple(or_pairs),
            total_questions=len(request.modules) * 2,
            total_attemptable=request.total_marks,
        )

        return plan
