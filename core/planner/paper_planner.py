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

        for mod in request.modules:
            alt_a_no = question_counter
            alt_b_no = question_counter + 1
            question_counter += 2

            co_val = request.co_mapping.get(mod, f"CO{mod}")

            slots_a: List[SlotDescriptor] = []
            slots_b: List[SlotDescriptor] = []

            for i, (marks, bloom) in enumerate(zip(D, bloom_profile)):
                q_type = request.question_types[i % len(request.question_types)]

                slot_a = SlotDescriptor(
                    slot_id=f"Q{alt_a_no}{sub_labels[i]}",
                    question_no=alt_a_no,
                    sub_label=sub_labels[i],
                    module_id=mod,
                    marks=marks,          # LOCKED
                    co=co_val,            # LOCKED
                    bloom=bloom,          # LOCKED
                    question_type=q_type, # LOCKED
                )

                slot_b = SlotDescriptor(
                    slot_id=f"Q{alt_b_no}{sub_labels[i]}",
                    question_no=alt_b_no,
                    sub_label=sub_labels[i],
                    module_id=mod,
                    marks=marks,          # LOCKED — SAME as slot_a
                    co=co_val,            # LOCKED — SAME as slot_a
                    bloom=bloom,          # LOCKED — SAME as slot_a
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
