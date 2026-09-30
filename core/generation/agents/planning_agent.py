"""
Planning Agent — slot allocation, KG grounding, visual flag.

Inputs (read from AgentContext):
    - paper_spec        : PaperSpec
    - artifact          : DocumentArtifact
    - knowledge_graph   : KnowledgeGraph
    - marks_split       : List[List[int]]

Output (AgentResult.payload):
    - List[QuestionPlan]

Determinism:
    - No LLM calls.
    - No randomness.
    - Same inputs → same plans.

Module isolation:
    - All evidence for a slot is drawn from that slot's module only.
    - Figure binding never crosses module boundaries.
    - KG neighborhoods are read from the module's own graph.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional

from .base import Agent, AgentContext, AgentResult
from .question_plan import QuestionPlan


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------


# Marks-to-Bloom mapping per VTU IAT convention.
# Keys are the max mark in the partition (e.g. [6, 4] uses 6).
BLOOM_BY_MARKS: Dict[int, str] = {
    2: "L1",
    3: "L2",
    4: "L2",
    5: "L3",
    6: "L3",
    7: "L3",
    8: "L4",
    9: "L4",
    10: "L4",
    12: "L5",
    15: "L5",
    20: "L5",
}

# Bloom levels that benefit from visual grounding.
VISUAL_BLOOM_LEVELS = frozenset({"L4", "L5"})

# Roles that are eligible as evidence for generation.
EVIDENCE_ROLES = frozenset({"BODY", "HEADING"})


class PlanningAgent(Agent):
    name = "planning"

    def __init__(
        self,
        max_visual_slots_per_module: int = 1,
        min_figure_quality: float = 0.60,
        max_evidence_blocks_per_slot: int = 5,
    ) -> None:
        if max_visual_slots_per_module < 0:
            raise ValueError("max_visual_slots_per_module must be >= 0")
        if not (0.0 <= min_figure_quality <= 1.0):
            raise ValueError("min_figure_quality must be in [0, 1]")
        if max_evidence_blocks_per_slot < 1:
            raise ValueError("max_evidence_blocks_per_slot must be >= 1")

        self.max_visual_slots_per_module = max_visual_slots_per_module
        self.min_figure_quality = min_figure_quality
        self.max_evidence_blocks_per_slot = max_evidence_blocks_per_slot

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> AgentResult:
        try:
            plans = self._plan(context)
        except Exception as e:
            self.log("failed", error=str(e))
            return AgentResult(
                success=False,
                failure_code="PLANNING_FAILED",
                failure_detail=str(e),
            )

        self.log(
            "done",
            plans=len(plans),
            visual=sum(1 for p in plans if p.visual_required),
        )
        return AgentResult(
            success=True,
            payload=plans,
            telemetry={
                "plans_count": len(plans),
                "visual_count": sum(1 for p in plans if p.visual_required),
                "modules_with_plans": len({p.module_id for p in plans}),
            },
        )

    # -------------------------------------------------------------- Planning

    def _plan(self, context: AgentContext) -> List[QuestionPlan]:
        spec = context.paper_spec
        artifact = context.artifact
        kg = context.knowledge_graph
        marks_split = context.marks_split

        if spec is None:
            raise ValueError("AgentContext.paper_spec is required")
        if artifact is None:
            raise ValueError("AgentContext.artifact is required")
        if not marks_split:
            raise ValueError("AgentContext.marks_split is required")

        # Validate marks_split length
        expected = spec.total_questions
        if len(marks_split) != expected:
            raise ValueError(
                f"marks_split has {len(marks_split)} partitions; "
                f"PaperSpec requires {expected}"
            )

        # 1. Allocate partitions to (module_idx, within_module_idx) slots.
        assignments = spec.allocate_partitions(marks_split)

        # 2. Compute CO map for the paper.
        co_map = self._build_co_map(spec)

        # 3. Group figures by module.
        figures_by_module = self._group_figures_by_module(artifact, spec.module_count)

        # 4. Track used topics and figures across the whole plan.
        used_topics: Dict[str, set] = defaultdict(set)
        used_figures: set = set()

        # 5. Build plans slot by slot.
        plans: List[QuestionPlan] = []
        global_q_idx = 1

        for module_idx in range(1, spec.module_count + 1):
            module_id = f"module_{module_idx}"
            co = co_map.get(str(module_idx), f"CO{module_idx}")

            module_figures = figures_by_module.get(module_idx, [])
            visual_slots_used = 0

            for within_idx in range(spec.questions_per_module):
                assignment_key = (module_idx - 1, within_idx)
                partition = assignments.get(assignment_key)
                if not partition:
                    continue

                total_marks = sum(partition)
                bloom = self._assign_bloom(partition)

                topic = self._pick_topic(
                    kg=kg,
                    module_id=module_id,
                    module_idx=module_idx,
                    exclude=used_topics[module_id],
                )
                if topic:
                    used_topics[module_id].add(topic.lower())
                else:
                    topic = f"Module {module_idx} — Topic {within_idx + 1}"
                    used_topics[module_id].add(topic.lower())

                neighborhood = self._get_neighborhood(kg, module_id, topic)
                evidence_blocks = self._pick_evidence_blocks(
                    artifact=artifact,
                    module_idx=module_idx,
                    module_count=spec.module_count,
                    slot_idx=within_idx,
                )

                visual_required, figure = self._resolve_visual(
                    bloom=bloom,
                    module_figures=module_figures,
                    used_figures=used_figures,
                    visual_slots_used=visual_slots_used,
                )
                if figure is not None:
                    used_figures.add(figure.id)
                    visual_slots_used += 1

                slot_id = f"{module_id}_Q{global_q_idx}"

                plans.append(
                    QuestionPlan(
                        slot_id=slot_id,
                        module_id=module_id,
                        module_idx=module_idx,
                        slot_idx=within_idx,
                        global_q_idx=global_q_idx,
                        partition=list(partition),
                        total_marks=total_marks,
                        co=co,
                        bloom=bloom,
                        topic=topic or f"Module {module_idx}",
                        concept_neighborhood=neighborhood,
                        evidence_blocks=evidence_blocks,
                        visual_required=visual_required,
                        figure=figure,
                        telemetry={
                            "topic_source": "kg" if neighborhood else "fallback",
                            "figures_available": len(module_figures),
                        },
                    )
                )
                global_q_idx += 1

        return plans

    # -------------------------------------------------------------- Helpers

    def _build_co_map(self, spec) -> Dict[str, str]:
        """Prefer spec.custom_co_map; else compute module→CO mapping."""
        if getattr(spec, "custom_co_map", None):
            return dict(spec.custom_co_map)
        # Try importing the existing helper.
        try:
            from core.generation.difficulty_policy import build_co_map
            return build_co_map(spec.module_count, spec.co_count)
        except Exception:
            # Fallback: module M → CO(min(M, co_count))
            return {
                str(m): f"CO{min(m, spec.co_count)}"
                for m in range(1, spec.module_count + 1)
            }

    def _assign_bloom(self, partition: List[int]) -> str:
        """Bloom level from the max mark in the partition."""
        if not partition:
            return "L2"
        max_mark = max(partition)
        # Nearest lower key if not exact
        keys = sorted(BLOOM_BY_MARKS.keys())
        for k in reversed(keys):
            if max_mark >= k:
                return BLOOM_BY_MARKS[k]
        return "L1"

    def _group_figures_by_module(
        self,
        artifact,
        module_count: int,
    ) -> Dict[int, List[Any]]:
        """
        Return a dict {module_idx: [FigureArtifact]}.

        Figures are grouped by their `module_id` attribute if present,
        else by dividing pages into equal ranges.
        """
        result: Dict[int, List[Any]] = {i: [] for i in range(1, module_count + 1)}
        figures = getattr(artifact, "figures", []) or []

        for fig in figures:
            module_id = getattr(fig, "module_id", None)
            if module_id and module_id.startswith("module_"):
                try:
                    idx = int(module_id.split("_")[1])
                    if 1 <= idx <= module_count:
                        result[idx].append(fig)
                        continue
                except (ValueError, IndexError):
                    pass

            # Fallback: assign by page-range heuristic
            page = getattr(fig, "page", 0)
            total_pages = self._max_page(artifact)
            idx = self._page_to_module_idx(page, total_pages, module_count)
            if idx is not None:
                result[idx].append(fig)

        return result

    def _max_page(self, artifact) -> int:
        blocks = getattr(artifact, "text_blocks", []) or []
        return max((getattr(b, "page", 0) for b in blocks), default=0)

    def _page_to_module_idx(
        self, page: int, total_pages: int, module_count: int
    ) -> Optional[int]:
        if total_pages <= 0 or page <= 0:
            return None
        # Simple equal-partition assumption. The Evidence Planner will
        # refine this when module boundaries are known.
        pages_per_module = max(1, total_pages // module_count)
        idx = (page - 1) // pages_per_module + 1
        return min(idx, module_count)

    def _pick_topic(
        self,
        kg,
        module_id: str,
        module_idx: int,
        exclude: set,
    ) -> str:
        """Pick the highest-ranked unused concept from the module's KG."""
        if kg is None or not hasattr(kg, "has_module"):
            return ""
        try:
            if not kg.has_module(module_id):
                return ""
            graph = kg.get_module(module_id)
            top = graph.top_concepts(100)
            for concept, _weight in top:
                if concept.lower() not in exclude:
                    return concept
        except Exception:
            return ""
        return ""

    def _get_neighborhood(self, kg, module_id: str, topic: str) -> List[Any]:
        if not topic or kg is None:
            return []
        try:
            if not kg.has_module(module_id):
                return []
            return kg.get_concept_neighborhood(module_id, topic, max_depth=2)
        except Exception:
            return []

    def _pick_evidence_blocks(
        self,
        artifact,
        module_idx: int,
        module_count: int,
        slot_idx: int = 0,
    ) -> List[Any]:
        """
        Return up to N evidence blocks from this module, windowed by slot_idx
        so sibling slots receive distinct/offset evidence contexts.
        """
        blocks = getattr(artifact, "text_blocks", []) or []
        max_page = self._max_page(artifact)
        if max_page <= 0:
            return []

        pages_per_module = max(1, max_page // module_count)
        start_page = (module_idx - 1) * pages_per_module + 1
        end_page = module_idx * pages_per_module
        if module_idx == module_count:
            end_page = max_page

        module_blocks = [
            b
            for b in blocks
            if start_page <= getattr(b, "page", 0) <= end_page
            and getattr(b, "block_role", "BODY") in EVIDENCE_ROLES
        ]

        # Filter BODY blocks (fall back to module_blocks if none marked BODY)
        body = [b for b in module_blocks if getattr(b, "block_role", "") == "BODY"]
        candidates = body if body else module_blocks
        if not candidates:
            return []

        # Window/offset by slot_idx to vary context across sibling questions in same module
        limit = self.max_evidence_blocks_per_slot
        step = max(1, limit // 2)
        start_offset = (slot_idx * step) % len(candidates)
        
        # Wrap-around slicing if start_offset + limit exceeds length
        chosen = candidates[start_offset: start_offset + limit]
        if len(chosen) < limit and len(candidates) > len(chosen):
            chosen += candidates[: limit - len(chosen)]
        return chosen

    def _resolve_visual(
        self,
        bloom: str,
        module_figures: List[Any],
        used_figures: set,
        visual_slots_used: int,
    ):
        """
        Return (visual_required, figure).
        Both False/None if the slot should not carry a figure.
        """
        # Gate 1: Bloom level
        if bloom not in VISUAL_BLOOM_LEVELS:
            return False, None

        # Gate 2: module quota
        if visual_slots_used >= self.max_visual_slots_per_module:
            return False, None

        # Gate 3: find an unused, high-quality figure
        for fig in module_figures:
            fig_id = getattr(fig, "id", None)
            if not fig_id or fig_id in used_figures:
                continue
            score = getattr(fig, "provenance_score", 1.0)
            if score < self.min_figure_quality:
                continue
            return True, fig

        return False, None
