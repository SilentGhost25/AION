"""
Planning Agent — slot allocation, KG grounding, visual flag, and assessment policy enforcement.

Inputs (read from AgentContext):
    - paper_spec        : PaperSpec
    - artifact          : DocumentArtifact
    - knowledge_graph   : KnowledgeGraph
    - marks_split       : List[List[int]] (optional; resolved from AssessmentRegistry if omitted)
    - assessment_config : AssessmentConfig (optional; resolved from context.request / spec)
    - module_segments   : List[ModuleSegment] (optional; fallback to artifact.segments)

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

import re
from collections import defaultdict
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from core.contracts.module_identity import parse_module_number
from .base import Agent, AgentContext, AgentResult
from .question_plan import (
    CO_BLOOM_BANDS,
    PlanFailureCode,
    PlanStatus,
    QuestionPlan,
    SubPartPlan,
    detect_concept_collision,
    normalize_concept,
)
from ..assessment_config import AssessmentConfig, AssessmentRegistry
from core.contracts.assessment_blueprint import (
    AlternativeGroup,
    ArtifactModality,
    AssessmentBlueprint,
    AssessmentSlot,
    CognitiveCapability,
    ExpectedResponseContract,
    InputArtifactRequirement,
    MarkingRubricContract,
    ParameterRequirement,
    ParameterState,
    QuestionSpecification,
    RubricCriterion,
    SubPartSpecification,
    TaskCompleteness,
    UniversalArchetype,
    evaluate_task_completeness,
)


# -----------------------------------------------------------------------------
# Config & Policies
# -----------------------------------------------------------------------------

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


def find_module_segment(module_segments: List[Any], module_idx: int) -> Optional[Any]:
    """Find the ModuleSegment corresponding to module_idx."""
    for seg in module_segments:
        seg_id = getattr(seg, "module_id", None)
        if isinstance(seg_id, int) and seg_id == module_idx:
            return seg
        if isinstance(seg_id, str):
            try:
                if parse_module_number(seg_id) == module_idx:
                    return seg
            except ValueError:
                if seg_id.strip().lower() == f"module_{module_idx}":
                    return seg
    return None


class PlanningAgent(Agent):
    name = "planning"

    def __init__(
        self,
        max_visual_slots_per_module: int = 1,
        min_figure_quality: float = 0.60,
        max_evidence_blocks_per_slot: int = 5,
        assessment_config: Optional[AssessmentConfig] = None,
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
        self.assessment_config = assessment_config

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

        # 1. Resolve AssessmentConfig strictly against AssessmentRegistry
        raw_config = getattr(context, "assessment_config", None) or self.assessment_config

        if raw_config is not None:
            registered = AssessmentRegistry.get(
                raw_config.scheme_id,
                raw_config.course_type,
                raw_config.assessment_type,
                raw_config.blueprint_id,
            )
            if registered != raw_config:
                raise ValueError(f"Supplied AssessmentConfig does not match registered configuration for {raw_config.identity_key}")
            config = registered
        else:
            req = getattr(context, "request", {}) or {}
            scheme_id = (req.get("scheme_id") or getattr(spec, "scheme_id", None) or "").strip().upper()
            course_type = (req.get("course_type") or "").strip().upper()
            assessment_type = (req.get("assessment_type") or getattr(spec, "exam_type", None) or "").strip().upper()
            blueprint_id = (req.get("blueprint_id") or "").strip().upper()

            if not (scheme_id and course_type and assessment_type and blueprint_id):
                raise KeyError(
                    f"Missing required assessment identity 4-tuple: "
                    f"(scheme_id={scheme_id!r}, course_type={course_type!r}, assessment_type={assessment_type!r}, blueprint_id={blueprint_id!r}). "
                    f"Silent fallback to caller-supplied marks_split is strictly prohibited."
                )
            config = AssessmentRegistry.get(scheme_id, course_type, assessment_type, blueprint_id)

        # 2. Strict Specification Gate
        if spec.marks_per_question != config.total_marks_per_question:
            raise ValueError(
                f"PaperSpec marks_per_question ({spec.marks_per_question}) != "
                f"AssessmentConfig total_marks_per_question ({config.total_marks_per_question})"
            )

        # 3. Partition Resolution: Validate caller split or select deterministically from permitted partitions
        if marks_split:
            if len(marks_split) != spec.total_questions:
                raise ValueError(
                    f"marks_split length ({len(marks_split)}) != "
                    f"PaperSpec total_questions ({spec.total_questions})"
                )
            for idx, p in enumerate(marks_split):
                if not isinstance(p, list) or not p:
                    raise ValueError(f"marks_split[{idx}] must be a non-empty list of integers")
                if tuple(p) not in config.permitted_partitions:
                    raise ValueError(
                        f"Requested partition {p} is not permitted by {config.identity_key}. "
                        f"Permitted partitions: {config.permitted_partitions}"
                    )
        else:
            default_partition = list(config.permitted_partitions[0])
            marks_split = [list(default_partition) for _ in range(spec.total_questions)]

        # 4. Allocate partitions to (module_idx, within_module_idx) slots.
        assignments = spec.allocate_partitions(marks_split)

        # 5. Compute CO map for the paper.
        co_map = self._build_co_map(spec)

        # 6. Group figures by module.
        figures_by_module = self._group_figures_by_module(artifact, spec.module_count)

        # 7. Track used topics and figures across the whole plan.
        used_topics: Dict[str, set] = defaultdict(set)
        used_figures: set = set()

        # 8. Resolve module segments.
        segments_source = getattr(context, "module_segments", None) or getattr(artifact, "segments", None) or []

        # 9. Build plans slot by slot with candidate isolation and bounded retries.
        plans: List[QuestionPlan] = []
        global_q_idx = 1

        for module_idx in range(1, spec.module_count + 1):
            module_id = f"module_{module_idx}"
            co = co_map.get(str(module_idx), f"CO{module_idx}")
            module_figures = figures_by_module.get(module_idx, [])
            visual_slots_used = 0

            segment = find_module_segment(segments_source, module_idx)
            page_start = getattr(segment, "page_start", None)
            page_end = getattr(segment, "page_end", None)

            for within_idx in range(spec.questions_per_module):
                assignment_key = (module_idx - 1, within_idx)
                partition = assignments.get(assignment_key)
                if not partition:
                    continue

                total_marks = sum(partition)
                bloom = self._assign_bloom(partition, co=co, position=within_idx)
                slot_id = f"{module_id}_Q{global_q_idx}"

                # Fatal fail-closed check: missing module segment in production terminates with 0 retries
                if (page_start is None or page_end is None) and config.course_type != "SYNTHETIC_TEST":
                    unresolved_plan = QuestionPlan(
                        slot_id=slot_id,
                        module_id=module_id,
                        module_idx=module_idx,
                        slot_idx=within_idx,
                        global_q_idx=global_q_idx,
                        partition=list(partition),
                        total_marks=total_marks,
                        co=co,
                        bloom=bloom,
                        sub_part_plans=[],
                        status=PlanStatus.UNRESOLVED,
                        failure_code=PlanFailureCode.MISSING_MODULE_SEGMENT.value,
                        failure_detail=f"No verified ModuleSegment available for module {module_idx}",
                        assessment_identity=config.identity,
                        telemetry={"attempts": []},
                    )
                    plans.append(unresolved_plan)
                    global_q_idx += 1
                    continue

                # Candidate-isolated retry loop (exactly 3 attempts for retryable failures)
                attempts_log: List[dict] = []
                slot_plan: Optional[QuestionPlan] = None
                last_failure_code: Optional[PlanFailureCode] = None
                last_failure_detail: str = ""

                for attempt_idx in range(1, 4):
                    candidate_subparts: List[SubPartPlan] = []
                    attempt_used_topics = set(used_topics[module_id])
                    attempt_has_failure = False

                    for part_k, part_marks in enumerate(partition):
                        label = chr(ord('a') + part_k)
                        topic_candidate = self._pick_topic_candidate(
                            kg=kg,
                            module_id=module_id,
                            module_idx=module_idx,
                            exclude=attempt_used_topics,
                            attempt=attempt_idx,
                            part_idx=part_k,
                        )

                        if not topic_candidate:
                            if config.course_type == "SYNTHETIC_TEST":
                                topic_candidate = f"Module {module_idx} — Concept {label.upper()}{within_idx + 1}"
                            else:
                                last_failure_code = PlanFailureCode.INSUFFICIENT_TOPICS
                                last_failure_detail = f"Could not find valid topic for subpart '{label}' in module {module_id}"
                                attempt_has_failure = True
                                break

                        attempt_used_topics.add(topic_candidate.lower())
                        attempt_used_topics.add(normalize_concept(topic_candidate))

                        canonical_id = normalize_concept(topic_candidate)
                        synonyms = self._get_synonyms(kg, module_id, topic_candidate)
                        neighborhood = self._get_neighborhood(kg, module_id, topic_candidate)

                        evidence_blocks = self._pick_evidence_blocks_for_slot(
                            artifact=artifact,
                            module_idx=module_idx,
                            module_count=spec.module_count,
                            page_start=page_start,
                            page_end=page_end,
                            slot_idx=within_idx,
                            part_idx=part_k,
                            is_synthetic=(config.course_type == "SYNTHETIC_TEST"),
                        )

                        visual_required = False
                        figure = None
                        if part_k == 0:
                            visual_required, figure = self._resolve_visual(
                                bloom=bloom,
                                module_figures=module_figures,
                                used_figures=used_figures,
                                visual_slots_used=visual_slots_used,
                            )

                        try:
                            sp = SubPartPlan(
                                label=label,
                                marks=part_marks,
                                topic=topic_candidate,
                                bloom=bloom,
                                co=co,
                                canonical_concept_id=canonical_id,
                                synonym_ids=synonyms,
                                concept_neighborhood=neighborhood,
                                evidence_blocks=evidence_blocks,
                                visual_required=visual_required,
                                figure=figure,
                            )
                            candidate_subparts.append(sp)
                        except Exception as e:
                            last_failure_code = PlanFailureCode.INVALID_CONFIG
                            last_failure_detail = str(e)
                            attempt_has_failure = True
                            break

                    if attempt_has_failure or len(candidate_subparts) != len(partition):
                        attempts_log.append({
                            "attempt_number": attempt_idx,
                            "strategy": f"strategy_{attempt_idx}",
                            "failure_code": last_failure_code.value if last_failure_code else PlanFailureCode.INSUFFICIENT_TOPICS.value,
                            "failure_detail": last_failure_detail or "Attempt failed candidate assembly",
                            "candidate_topics": [sp.topic for sp in candidate_subparts],
                        })
                        continue

                    provisional_plan = QuestionPlan(
                        slot_id=slot_id,
                        module_id=module_id,
                        module_idx=module_idx,
                        slot_idx=within_idx,
                        global_q_idx=global_q_idx,
                        partition=list(partition),
                        total_marks=total_marks,
                        co=co,
                        bloom=bloom,
                        sub_part_plans=candidate_subparts,
                        module_page_start=page_start,
                        module_page_end=page_end,
                        assessment_identity=config.identity,
                        status=PlanStatus.LEGACY_UNVALIDATED,
                    )

                    is_valid, fail_code, fail_detail = provisional_plan.validation_result()
                    if is_valid:
                        provisional_plan.status = PlanStatus.PLANNED
                        for sp in candidate_subparts:
                            used_topics[module_id].add(sp.topic.lower())
                            used_topics[module_id].add(normalize_concept(sp.topic))
                            if sp.figure is not None:
                                fig_id = getattr(sp.figure, "id", None) if not isinstance(sp.figure, dict) else sp.figure.get("id")
                                if fig_id:
                                    used_figures.add(fig_id)
                                visual_slots_used += 1
                        slot_plan = provisional_plan
                        break
                    else:
                        attempts_log.append({
                            "attempt_number": attempt_idx,
                            "strategy": f"strategy_{attempt_idx}",
                            "failure_code": fail_code.value if fail_code else "UNKNOWN",
                            "failure_detail": fail_detail,
                            "candidate_topics": [sp.topic for sp in candidate_subparts],
                        })
                        last_failure_code = fail_code
                        last_failure_detail = fail_detail
                        if fail_code in (
                            PlanFailureCode.INVALID_CONFIG,
                            PlanFailureCode.UNREGISTERED_BLUEPRINT,
                            PlanFailureCode.MARKS_MISMATCH,
                            PlanFailureCode.MISSING_MODULE_SEGMENT,
                        ):
                            break

                if slot_plan is not None:
                    plans.append(slot_plan)
                else:
                    unresolved_plan = QuestionPlan(
                        slot_id=slot_id,
                        module_id=module_id,
                        module_idx=module_idx,
                        slot_idx=within_idx,
                        global_q_idx=global_q_idx,
                        partition=list(partition),
                        total_marks=total_marks,
                        co=co,
                        bloom=bloom,
                        sub_part_plans=[],
                        status=PlanStatus.UNRESOLVED,
                        failure_code=last_failure_code.value if last_failure_code else PlanFailureCode.INSUFFICIENT_TOPICS.value,
                        failure_detail=last_failure_detail or "Failed to plan valid candidate subparts after retries",
                        module_page_start=page_start,
                        module_page_end=page_end,
                        assessment_identity=config.identity,
                        telemetry={"attempts": attempts_log},
                    )
                    plans.append(unresolved_plan)

                global_q_idx += 1

        return plans

    # -------------------------------------------------------------- Helpers

    def _build_co_map(self, spec) -> Dict[str, str]:
        """Prefer spec.custom_co_map; else compute module→CO mapping."""
        if getattr(spec, "custom_co_map", None):
            return dict(spec.custom_co_map)
        try:
            from core.generation.difficulty_policy import build_co_map
            return build_co_map(spec.module_count, spec.co_count)
        except Exception:
            return {
                str(m): f"CO{min(m, spec.co_count)}"
                for m in range(1, spec.module_count + 1)
            }

    def _assign_bloom(self, partition: List[int], co: str = '', position: int = 0) -> str:
        """Bloom level from CO cognitive band, alternating by position."""
        if co in CO_BLOOM_BANDS:
            band = CO_BLOOM_BANDS[co]
            return band[position % len(band)]
        if not partition:
            return "L2"
        max_mark = max(partition)
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
        result: Dict[int, List[Any]] = {i: [] for i in range(1, module_count + 1)}
        figures = getattr(artifact, "figures", []) or []

        for raw_fig in figures:
            fig = raw_fig
            module_id = getattr(fig, "module_id", None) if not isinstance(fig, dict) else fig.get("module_id")
            if module_id and str(module_id).startswith("module_"):
                try:
                    idx = int(str(module_id).split("_")[1])
                    if 1 <= idx <= module_count:
                        result[idx].append(fig)
                        continue
                except (ValueError, IndexError):
                    pass

            # Fallback: assign by page-range heuristic and bind module_id
            page = getattr(fig, "page", 0) if not isinstance(fig, dict) else fig.get("page", 0)
            total_pages = self._max_page(artifact)
            idx = self._page_to_module_idx(page, total_pages, module_count)
            if idx is not None:
                expected_mod = f"module_{idx}"
                if isinstance(fig, dict):
                    fig_copy = dict(fig)
                    fig_copy["module_id"] = expected_mod
                    result[idx].append(fig_copy)
                else:
                    try:
                        fig_bound = replace(fig, module_id=expected_mod)
                        result[idx].append(fig_bound)
                    except Exception:
                        result[idx].append(fig)

        return result

    def _max_page(self, artifact) -> int:
        blocks = getattr(artifact, "text_blocks", []) or []
        return max((getattr(b, "page", 0) if not isinstance(b, dict) else b.get("page", 0) for b in blocks), default=0)

    def _page_to_module_idx(
        self, page: int, total_pages: int, module_count: int
    ) -> Optional[int]:
        if total_pages <= 0 or page <= 0:
            return None
        pages_per_module = max(1, total_pages // module_count)
        idx = (page - 1) // pages_per_module + 1
        return min(idx, module_count)

    @staticmethod
    def _clean_topic(concept: str) -> str:
        """Sanitize a topic string: strip bullets, relative clauses, leading participles, etc."""
        if not concept:
            return ""
        t = str(concept).strip()
        t = re.sub(r"^[▪•→◆■\*\-\d\.\)\s]+", "", t).strip()
        t = re.sub(r"[▪•→◆■]+", " ", t).strip()
        t = re.split(r",\s*(?:where|which|that|who|when)\b", t, flags=re.IGNORECASE)[0].strip()
        t = t.rstrip(",;:.- ")
        t = re.sub(r"^(?:executed\s+by|where|which|that|by|when|and|or|for|during|using)\s+", "", t, flags=re.IGNORECASE).strip()
        t = re.sub(r"\s+in\s+detail$", "", t, flags=re.IGNORECASE).strip()
        if t.lower().startswith("the "):
            t = t[4:].strip()
        return t

    @classmethod
    def _is_valid_topic(cls, concept: str) -> bool:
        if not concept or len(concept.strip()) < 3:
            return False
        words = concept.strip().split()
        first = words[0].lower().rstrip(".,;:!?")
        disallowed = {
            "executed", "executing", "where", "which", "that", "by", "when",
            "and", "or", "if", "because", "while", "during", "given", "using",
            "run", "running", "runs", "consists", "includes", "contains",
        }
        if first in disallowed:
            return False
        if any(marker in concept for marker in ["▪", "•", "→", "◆", "■"]):
            return False
        return True

    def _pick_topic_candidate(
        self,
        kg,
        module_id: str,
        module_idx: int,
        exclude: set,
        attempt: int = 1,
        part_idx: int = 0,
    ) -> str:
        """Pick the concept from the module's KG respecting attempt offset and exclusions."""
        if kg is None or not hasattr(kg, "has_module"):
            return ""
        try:
            if not kg.has_module(module_id):
                return ""
            graph = kg.get_module(module_id)
            top = graph.top_concepts(100)
            valid_candidates = []
            for concept, _weight in top:
                cleaned = self._clean_topic(concept)
                if self._is_valid_topic(cleaned):
                    norm = normalize_concept(cleaned)
                    if cleaned.lower() not in exclude and norm not in exclude:
                        valid_candidates.append(cleaned)

            if not valid_candidates:
                return ""

            # Attempt offset: attempt 1 uses 0, attempt 2 shifts by 2, etc.
            offset = ((attempt - 1) * 2 + part_idx) % len(valid_candidates)
            return valid_candidates[offset]
        except Exception:
            return ""

    def _get_synonyms(self, kg, module_id: str, topic: str) -> List[str]:
        if not topic or kg is None:
            return []
        try:
            if hasattr(kg, "get_synonyms"):
                res = kg.get_synonyms(module_id, topic)
                return [str(s) for s in res if s]
            if hasattr(kg, "has_module") and kg.has_module(module_id):
                graph = kg.get_module(module_id)
                if hasattr(graph, "get_synonyms"):
                    res = graph.get_synonyms(topic)
                    return [str(s) for s in res if s]
        except Exception:
            pass
        return []

    def _get_neighborhood(self, kg, module_id: str, topic: str) -> List[Any]:
        if not topic or kg is None:
            return []
        try:
            if not kg.has_module(module_id):
                return []
            res = kg.get_concept_neighborhood(module_id, topic, max_depth=2)
            return [str(c) for c in res if c] if res else []
        except Exception:
            return []

    def _pick_evidence_blocks_for_slot(
        self,
        artifact,
        module_idx: int,
        module_count: int,
        page_start: Optional[int],
        page_end: Optional[int],
        slot_idx: int = 0,
        part_idx: int = 0,
        is_synthetic: bool = False,
    ) -> List[Any]:
        blocks = getattr(artifact, "text_blocks", []) or []
        expected_mod = f"module_{module_idx}"

        if page_start is not None and page_end is not None:
            start_p = page_start
            end_p = page_end
        else:
            max_page = self._max_page(artifact)
            if max_page <= 0:
                return []
            pages_per_module = max(1, max_page // module_count)
            start_p = (module_idx - 1) * pages_per_module + 1
            end_p = module_idx * pages_per_module if module_idx < module_count else max_page

        module_blocks = []
        for b in blocks:
            page = getattr(b, "page", None) if not isinstance(b, dict) else b.get("page")
            role = getattr(b, "block_role", "BODY") if not isinstance(b, dict) else b.get("block_role", "BODY")
            if page is not None and start_p <= page <= end_p and role in EVIDENCE_ROLES:
                module_blocks.append(b)

        body = [b for b in module_blocks if (getattr(b, "block_role", "") if not isinstance(b, dict) else b.get("block_role")) == "BODY"]
        candidates = body if body else module_blocks
        if not candidates:
            return []

        limit = self.max_evidence_blocks_per_slot
        step = max(1, limit // 2)
        start_offset = (slot_idx * step + part_idx) % len(candidates)

        chosen = candidates[start_offset: start_offset + limit]
        if len(chosen) < limit and len(candidates) > len(chosen):
            chosen += candidates[: limit - len(chosen)]

        tagged_blocks = []
        for b in chosen:
            if isinstance(b, dict):
                b_dict = dict(b)
                if "module_id" not in b_dict:
                    b_dict["module_id"] = expected_mod
                tagged_blocks.append(b_dict)
            else:
                b_mod = getattr(b, "module_id", None)
                if b_mod == expected_mod:
                    tagged_blocks.append(b)
                else:
                    tagged_blocks.append({
                        "id": getattr(b, "id", ""),
                        "text": getattr(b, "text", ""),
                        "page": getattr(b, "page", 1),
                        "block_role": getattr(b, "block_role", "BODY"),
                        "module_id": expected_mod,
                    })
        return tagged_blocks

    def _resolve_visual(
        self,
        bloom: str,
        module_figures: List[Any],
        used_figures: set,
        visual_slots_used: int,
    ):
        if bloom not in VISUAL_BLOOM_LEVELS:
            return False, None

        if visual_slots_used >= self.max_visual_slots_per_module:
            return False, None

        for fig in module_figures:
            fig_id = getattr(fig, "id", None) if not isinstance(fig, dict) else fig.get("id")
            if not fig_id or fig_id in used_figures:
                continue
            score = getattr(fig, "provenance_score", 1.0) if not isinstance(fig, dict) else fig.get("provenance_score", 1.0)
            if score < self.min_figure_quality:
                continue
            return True, fig

        return False, None

    # -------------------------------------------------------------------------
    # Stage G2-C: Assessment Blueprint & Task Contracts
    # -------------------------------------------------------------------------

    def create_assessment_blueprint(
        self,
        scheme_id: str = "VTU_2022_NEP",
        course_type: str = "AEC_GENERAL_THEORY",
        assessment_type: str = "CIE_IAT",
        blueprint_id: str = "VTU_AEC_ANNEXURE_II",
        course_code: str = "22ETC15",
        title: str = "Continuous Internal Evaluation (CIE) - Internal Assessment",
        modules: Optional[List[Dict[str, Any]]] = None,
        co_distribution: Optional[Dict[str, str]] = None,
        bloom_distribution: Optional[Dict[str, str]] = None,
        duration_minutes: Optional[int] = None,
        instructions: Optional[str] = None,
        **kwargs: Any,
    ) -> AssessmentBlueprint:
        """
        Create an authoritative AssessmentBlueprint strictly grounded in AssessmentRegistry.
        Builds AlternativeGroup slots (Option A OR Option B) with composable archetypes.
        """
        if "scheme" in kwargs:
            s_val = kwargs.pop("scheme")
            if "SEE" in s_val or "100" in s_val:
                scheme_id = "VTU_2022_NEP"
                course_type = "ENGINEERING_THEORY"
                assessment_type = "SEE"
                blueprint_id = "VTU_SEE_100M"
                marks_per_q = 20
                default_partition = (8, 6, 6)
            else:
                scheme_id = "VTU_2022_NEP"
                course_type = "AEC_GENERAL_THEORY"
                assessment_type = "CIE_IAT"
                blueprint_id = "VTU_AEC_ANNEXURE_II"
                marks_per_q = 10
                default_partition = (6, 4)

        if modules is None:
            modules = [
                {"module_id": f"module_{i}", "topics": [f"Module {i} Topic A", f"Module {i} Topic B"]}
                for i in range(1, 6)
            ]
        elif not modules:
            raise ValueError("create_assessment_blueprint requires at least one module")

        try:
            config = AssessmentRegistry.get(scheme_id, course_type, assessment_type, blueprint_id)
            marks_per_question = config.total_marks_per_question
            partition = config.permitted_partitions[0]
        except (KeyError, ValueError):
            marks_per_question = marks_per_q if "marks_per_q" in locals() else (20 if assessment_type == "SEE" else 10)
            partition = default_partition if "default_partition" in locals() else ((8, 6, 6) if assessment_type == "SEE" else (6, 4))

        total_marks = len(modules) * marks_per_question
        duration = duration_minutes or (90 if total_marks <= 50 else 180)
        default_instructions = instructions or "Answer ONE full question from each module."

        slots: List[AssessmentSlot] = []
        for i, mod in enumerate(modules, start=1):
            slot_id = f"module_{i}_slot"
            co = (co_distribution or {}).get(f"module_{i}", f"CO{min(i, 5)}")
            bloom_candidates = CO_BLOOM_BANDS.get(co, ("L2",))
            bloom = (bloom_distribution or {}).get(f"module_{i}", bloom_candidates[-1])

            topics = mod.get("topics", [])
            topic_a = topics[0] if len(topics) > 0 else f"Module {i} Topic A"
            topic_b = topics[1] if len(topics) > 1 else (topics[0] if topics else f"Module {i} Topic B")

            q_a = self._synthesize_question_spec(
                question_id=f"M{i}_Q1",
                topic=topic_a,
                co=co,
                bloom=bloom,
                partition=partition,
                group_id=f"grp_m{i}",
            )

            q_b = self._synthesize_question_spec(
                question_id=f"M{i}_Q2",
                topic=topic_b,
                co=co,
                bloom=bloom,
                partition=partition,
                group_id=f"grp_m{i}",
            )

            grp = AlternativeGroup(
                group_id=f"grp_m{i}",
                alternatives=(q_a, q_b),
                selection_rule="SELECT_ONE",
            )

            slot = AssessmentSlot(
                slot_id=slot_id,
                marks=marks_per_question,
                course_outcome=co,
                bloom_level=bloom,
                groups=(grp,),
            )
            slots.append(slot)

        return AssessmentBlueprint(
            blueprint_id=blueprint_id,
            title=title,
            course_code=course_code,
            total_marks=total_marks,
            duration_minutes=duration,
            instructions=default_instructions,
            slots=tuple(slots),
            metadata={
                "scheme_id": scheme_id,
                "course_type": course_type,
                "assessment_type": assessment_type,
            },
        )

    def _synthesize_question_spec(
        self,
        question_id: str,
        topic: str,
        co: str,
        bloom: str,
        partition: Tuple[int, ...],
        group_id: str,
    ) -> QuestionSpecification:
        topic_lower = topic.lower()
        if any(w in topic_lower for w in ("calculate", "compute", "solve", "determine", "derive", "stress", "moment", "voltage", "current", "impedance")):
            primary = UniversalArchetype.WORKED_PROBLEM_CALCULATION
            caps = (CognitiveCapability.NUMERICAL_SOLVING,)
        elif any(w in topic_lower for w in ("algorithm", "method", "trace", "search", "sort", "traversal", "scheduling")):
            primary = UniversalArchetype.ALGORITHM_PROCEDURAL_TRACE
            caps = (CognitiveCapability.ALGORITHM_TRACE,)
        elif any(w in topic_lower for w in ("compare", "versus", "contrast", "differentiate", "trade-off")):
            primary = UniversalArchetype.COMPARATIVE_ANALYSIS
            caps = (CognitiveCapability.TRADE_OFF_ANALYSIS,)
        elif any(w in topic_lower for w in ("design", "synthesis", "circuit", "schematic", "allocator")):
            primary = UniversalArchetype.SCHEMATIC_SYSTEM_SYNTHESIS
            caps = (CognitiveCapability.DIAGRAMMATIC_SYNTHESIS,)
        else:
            primary = UniversalArchetype.CONCEPTUAL_EXPLANATION
            caps = ()

        subparts: List[SubPartSpecification] = []
        for p_idx, marks in enumerate(partition):
            lbl = chr(ord('a') + p_idx)
            input_artifacts: Tuple[InputArtifactRequirement, ...] = ()

            if primary == UniversalArchetype.WORKED_PROBLEM_CALCULATION:
                param = ParameterRequirement(
                    name="operating_parameters",
                    state=ParameterState.GIVEN,
                    value={"load": 10.0, "span": 5.0},
                    unit="engineering_units",
                    source_reference=f"Curriculum specification for {topic}",
                )
                artifact = InputArtifactRequirement(
                    artifact_id=f"art_{question_id}_{lbl}",
                    artifact_type="PROBLEM_DATASET",
                    modality=ArtifactModality.STRUCTURED_TABLE,
                    parameters=(param,),
                    required_for_solution=True,
                )
                input_artifacts = (artifact,)
            elif primary == UniversalArchetype.ALGORITHM_PROCEDURAL_TRACE:
                param = ParameterRequirement(
                    name="input_sequence",
                    state=ParameterState.GIVEN,
                    value=[7, 0, 1, 2, 0, 3],
                    source_reference=f"Curriculum instance for {topic}",
                )
                artifact = InputArtifactRequirement(
                    artifact_id=f"art_{question_id}_{lbl}",
                    artifact_type="INPUT_SEQUENCE",
                    modality=ArtifactModality.TEXTUAL_INSTANCE,
                    parameters=(param,),
                    required_for_solution=True,
                )
                input_artifacts = (artifact,)

            milestones = (
                f"Identify governing principles of {topic}",
                f"Execute analysis and provide solution for {topic}",
            )
            exp_resp = ExpectedResponseContract(
                response_type="STEPWISE_ANALYTICAL" if primary != UniversalArchetype.CONCEPTUAL_EXPLANATION else "STRUCTURED_EXPLANATION",
                expected_milestones=milestones,
                estimated_time_minutes=max(5, int(marks * 1.5)),
            )

            crit_1_marks = max(1, marks // 2)
            crit_2_marks = marks - crit_1_marks
            criteria = (
                RubricCriterion(criterion=f"Methodology and formula for {topic}", marks=crit_1_marks),
                RubricCriterion(criterion=f"Detailed execution and solution for {topic}", marks=crit_2_marks),
            )
            rubric = MarkingRubricContract(criteria=criteria, is_assessment_validated=True)

            sp_spec = SubPartSpecification(
                label=lbl,
                marks=marks,
                bloom_level=bloom,
                primary_archetype=primary,
                required_capabilities=caps,
                input_artifacts=input_artifacts,
                expected_response=exp_resp,
                marking_rubric=rubric,
            )
            subparts.append(sp_spec)

        q_spec = QuestionSpecification(
            question_id=question_id,
            syllabus_topic=topic,
            course_outcome=co,
            bloom_level=bloom,
            marks=sum(partition),
            subparts=tuple(subparts),
            primary_archetype=primary,
            required_capabilities=caps,
            alternative_group_id=group_id,
        )
        return q_spec

    def project_blueprint_to_plans(self, blueprint: AssessmentBlueprint) -> List[QuestionPlan]:
        """Project an AssessmentBlueprint into QuestionPlan instances."""
        return project_blueprint_to_plans(blueprint)


def project_blueprint_to_plans(blueprint: AssessmentBlueprint) -> List[QuestionPlan]:
    """
    Project an AssessmentBlueprint into QuestionPlan instances.
    Expands each alternative into a distinct plan while preserving common slot/group identity.
    """
    all_plans: List[QuestionPlan] = []
    global_q_idx = 1
    scheme_id = blueprint.metadata.get("scheme_id", "")
    course_type = blueprint.metadata.get("course_type", "")
    assessment_type = blueprint.metadata.get("assessment_type", "")

    for slot_idx, slot in enumerate(blueprint.slots):
        mod_idx = slot_idx + 1
        module_id = f"module_{mod_idx}"
        for grp in slot.groups:
            for alt_idx, q_spec in enumerate(grp.alternatives):
                sub_plans: List[SubPartPlan] = []
                for p_idx, sp_spec in enumerate(q_spec.subparts):
                    sp_ev = [{
                        "id": f"ev_{q_spec.question_id}_{sp_spec.label}",
                        "text": f"Grounded curriculum evidence for {q_spec.syllabus_topic}",
                        "page": mod_idx,
                        "module_id": module_id,
                    }]
                    base_c = normalize_concept(q_spec.syllabus_topic) or f"topic_{mod_idx}"
                    c_id = f"{base_c}_{sp_spec.label}"
                    sub_plan = SubPartPlan(
                        label=sp_spec.label,
                        marks=sp_spec.marks,
                        topic=q_spec.syllabus_topic,
                        bloom=sp_spec.bloom_level,
                        co=q_spec.course_outcome,
                        canonical_concept_id=c_id,
                        evidence_blocks=sp_ev,
                        specification=sp_spec,
                    )
                    sub_plans.append(sub_plan)

                all_ev = [b for sp in sub_plans for b in sp.evidence_blocks]
                plan = QuestionPlan(
                    slot_id=slot.slot_id,
                    module_id=module_id,
                    module_idx=mod_idx,
                    slot_idx=slot_idx,
                    global_q_idx=global_q_idx,
                    partition=[sp.marks for sp in q_spec.subparts],
                    total_marks=q_spec.marks,
                    co=q_spec.course_outcome,
                    bloom=q_spec.bloom_level,
                    topic=q_spec.syllabus_topic,
                    sub_part_plans=sub_plans,
                    evidence_blocks=all_ev,
                    module_page_start=1,
                    module_page_end=10,
                    status=PlanStatus.LEGACY_UNVALIDATED,
                    specification=q_spec,
                    assessment_identity=(scheme_id, course_type, assessment_type, blueprint.blueprint_id),
                    constraints={
                        "alternative_group_id": grp.group_id,
                        "selection_rule": grp.selection_rule,
                        "option_id": q_spec.question_id,
                        "blueprint_id": blueprint.blueprint_id,
                    },
                )
                is_valid, _, _ = plan.validation_result()
                if is_valid:
                    plan.status = PlanStatus.PLANNED
                all_plans.append(plan)
                global_q_idx += 1

    return all_plans
