"""
Unit tests for the Planning Agent.

Uses a minimal stub PaperSpec, DocumentArtifact, and KnowledgeGraph so
each behavior can be tested in isolation.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pytest

from core.generation.agents import (
    AgentContext,
    PlanningAgent,
    QuestionPlan,
)


# -----------------------------------------------------------------------------
# Stubs
# -----------------------------------------------------------------------------


@dataclass
class StubBlock:
    id: str
    text: str
    page: int
    block_role: str = "BODY"


@dataclass
class StubFigure:
    id: str
    page: int
    module_id: str = ""
    provenance_score: float = 1.0
    caption: str = ""


@dataclass
class StubArtifact:
    text_blocks: List[StubBlock] = field(default_factory=list)
    figures: List[StubFigure] = field(default_factory=list)


class StubModuleGraph:
    def __init__(self, triples: List[Tuple[str, str, str]]):
        self._triples = triples

    def top_concepts(self, n: int):
        seen = {}
        for s, p, o in self._triples:
            seen[s] = seen.get(s, 0) + 1
            seen[o] = seen.get(o, 0) + 1
        return sorted(seen.items(), key=lambda kv: -kv[1])[:n]

    def get_neighborhood(self, concept: str, max_depth: int = 2):
        return [t for t in self._triples if concept in (t[0], t[2])]


class StubKG:
    def __init__(self, modules: Dict[str, StubModuleGraph]):
        self._modules = modules

    def has_module(self, module_id: str) -> bool:
        return module_id in self._modules

    def get_module(self, module_id: str) -> StubModuleGraph:
        return self._modules[module_id]

    def get_concept_neighborhood(self, module_id: str, concept: str, max_depth: int = 2):
        return self._modules[module_id].get_neighborhood(concept, max_depth)


class StubPaperSpec:
    def __init__(
        self,
        module_count: int = 5,
        questions_per_module: int = 2,
        co_count: int = 5,
        custom_co_map: Dict[str, str] = None,
    ):
        self.module_count = module_count
        self.questions_per_module = questions_per_module
        self.co_count = co_count
        self.custom_co_map = custom_co_map

    @property
    def total_questions(self) -> int:
        return self.module_count * self.questions_per_module

    def allocate_partitions(self, marks_split):
        assignments = {}
        cumulative = 0
        for m in range(self.module_count):
            for q in range(self.questions_per_module):
                assignments[(m, q)] = marks_split[cumulative]
                cumulative += 1
        return assignments


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def basic_artifact():
    blocks = []
    for page in range(1, 51):
        blocks.append(StubBlock(
            id=f"b{page}",
            text=f"Body content on page {page}.",
            page=page,
            block_role="BODY",
        ))
    return StubArtifact(text_blocks=blocks, figures=[])


@pytest.fixture
def artifact_with_figures():
    blocks = [
        StubBlock(id=f"b{p}", text=f"Body {p}.", page=p) for p in range(1, 51)
    ]
    figures = [
        StubFigure(id="fig_m1", page=5, module_id="module_1", provenance_score=0.9),
        StubFigure(id="fig_m3", page=25, module_id="module_3", provenance_score=0.8),
        StubFigure(id="fig_m5", page=45, module_id="module_5", provenance_score=0.95),
    ]
    return StubArtifact(text_blocks=blocks, figures=figures)


@pytest.fixture
def basic_kg():
    return StubKG({
        "module_1": StubModuleGraph([
            ("Docker", "is_a", "platform"),
            ("Docker", "requires", "runtime"),
            ("runtime", "is_a", "software"),
        ]),
        "module_2": StubModuleGraph([
            ("Virtualization", "is_a", "technology"),
            ("Virtualization", "enables", "isolation"),
        ]),
        "module_3": StubModuleGraph([
            ("TCP", "is_a", "protocol"),
            ("TCP", "requires", "handshake"),
        ]),
        "module_4": StubModuleGraph([
            ("Load balancer", "is_a", "component"),
        ]),
        "module_5": StubModuleGraph([
            ("QoS", "is_a", "policy"),
        ]),
    })


@pytest.fixture
def iat1_split():
    # 10 slots: [10], [10], [6,4]×4, [10]×4
    return [[10], [10], [6, 4], [6, 4], [6, 4], [6, 4], [10], [10], [10], [10]]


# -----------------------------------------------------------------------------
# Basic behaviour
# -----------------------------------------------------------------------------


def test_planning_produces_expected_count(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    result = PlanningAgent().run(ctx)
    assert result.success is True
    assert len(result.payload) == 10


def test_slot_ids_are_sequential(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    for i, plan in enumerate(plans, start=1):
        assert plan.global_q_idx == i


def test_marks_match_partition(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    assert plans[0].total_marks == 10
    assert plans[2].total_marks == 10
    assert plans[2].partition == [6, 4]
    assert plans[2].is_multi_part is True
    assert plans[0].is_multi_part is False


def test_co_mapping_is_per_module(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    # Module 1 slots → CO1, Module 2 → CO2, etc.
    assert plans[0].co == "CO1"
    assert plans[1].co == "CO1"
    assert plans[2].co == "CO2"
    assert plans[-1].co == "CO5"


def test_custom_co_map_overrides_default(
    basic_artifact, basic_kg, iat1_split
):
    spec = StubPaperSpec(custom_co_map={"1": "CO1", "2": "CO1", "3": "CO2",
                                        "4": "CO3", "5": "CO4"})
    ctx = AgentContext(
        paper_spec=spec,
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    assert plans[2].co == "CO1"  # Module 2 → CO1 per custom map


# -----------------------------------------------------------------------------
# Bloom assignment
# -----------------------------------------------------------------------------


def test_bloom_l2_for_4_marks(basic_artifact, basic_kg):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=1),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=[[4]],
    )
    plan = PlanningAgent().run(ctx).payload[0]
    assert plan.bloom == "L2"


def test_bloom_l3_for_6_marks(basic_artifact, basic_kg):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=1),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=[[6]],
    )
    plan = PlanningAgent().run(ctx).payload[0]
    assert plan.bloom == "L3"


def test_bloom_l4_for_10_marks(basic_artifact, basic_kg):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=1),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=[[10]],
    )
    plan = PlanningAgent().run(ctx).payload[0]
    assert plan.bloom == "L4"


# -----------------------------------------------------------------------------
# Topic grounding
# -----------------------------------------------------------------------------


def test_topics_drawn_from_kg(basic_artifact, basic_kg, iat1_split):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    # Module 1's top concept is "Docker" — first slot should use it
    assert plans[0].topic == "Docker"
    # Module 1's second slot should use a different concept
    assert plans[1].topic != "Docker"


def test_concept_neighborhood_attached(basic_artifact, basic_kg, iat1_split):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    assert len(plans[0].concept_neighborhood) > 0


def test_missing_kg_falls_back(basic_artifact, iat1_split):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=None,
        marks_split=iat1_split,
    )
    result = PlanningAgent().run(ctx)
    assert result.success is True
    # Fallback topic format
    assert "Module 1" in result.payload[0].topic


# -----------------------------------------------------------------------------
# Visual flag
# -----------------------------------------------------------------------------


def test_visual_flag_not_set_for_l3(
    artifact_with_figures, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=artifact_with_figures,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    # Slot 4 is 6 marks (L3) in module 3 (which has a figure) → no visual because L3
    assert plans[4].visual_required is False
    # Slot 6 is 10 marks (L4) in module 4, but module 4 has no figures → no visual
    assert plans[6].visual_required is False


def test_visual_flag_set_for_l4_with_figure(
    artifact_with_figures, basic_kg
):
    # Single-module spec where module 1 has a figure
    # Reassign the figure to module 1
    for fig in artifact_with_figures.figures:
        fig.module_id = "module_1"
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=1),
        artifact=artifact_with_figures,
        knowledge_graph=basic_kg,
        marks_split=[[10]],
    )
    plan = PlanningAgent().run(ctx).payload[0]
    assert plan.visual_required is True
    assert plan.figure is not None


def test_visual_quota_per_module(
    artifact_with_figures, basic_kg
):
    # Give module 1 two figures; two L4 slots; quota is 1
    for fig in artifact_with_figures.figures:
        fig.module_id = "module_1"
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=2),
        artifact=artifact_with_figures,
        knowledge_graph=basic_kg,
        marks_split=[[10], [10]],
    )
    plans = PlanningAgent().run(ctx).payload
    assert sum(1 for p in plans if p.visual_required) == 1


def test_visual_not_required_when_no_figures(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,  # no figures
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    assert all(not p.visual_required for p in plans)


def test_visual_quality_gate(artifact_with_figures, basic_kg):
    for fig in artifact_with_figures.figures:
        fig.module_id = "module_1"
        fig.provenance_score = 0.3  # below default 0.60
    ctx = AgentContext(
        paper_spec=StubPaperSpec(module_count=1, questions_per_module=1),
        artifact=artifact_with_figures,
        knowledge_graph=basic_kg,
        marks_split=[[10]],
    )
    plan = PlanningAgent().run(ctx).payload[0]
    assert plan.visual_required is False


# -----------------------------------------------------------------------------
# Evidence
# -----------------------------------------------------------------------------


def test_evidence_blocks_from_correct_module(
    basic_artifact, basic_kg, iat1_split
):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = PlanningAgent().run(ctx).payload
    # Module 1 slot 0: pages 1-10
    m1_pages = [b.page for b in plans[0].evidence_blocks]
    assert all(1 <= p <= 10 for p in m1_pages)
    # Module 3 slot: pages 21-30
    m3_slot = next(p for p in plans if p.module_idx == 3)
    m3_pages = [b.page for b in m3_slot.evidence_blocks]
    assert all(21 <= p <= 30 for p in m3_pages)


def test_evidence_respects_max(basic_artifact, basic_kg, iat1_split):
    agent = PlanningAgent(max_evidence_blocks_per_slot=3)
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    plans = agent.run(ctx).payload
    for plan in plans:
        assert len(plan.evidence_blocks) <= 3


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def test_mismatched_marks_split_length_raises(basic_artifact, basic_kg):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=[[10], [10]],  # only 2, need 10
    )
    result = PlanningAgent().run(ctx)
    assert result.success is False
    assert result.failure_code == "PLANNING_FAILED"


def test_missing_paper_spec_raises(basic_artifact, basic_kg, iat1_split):
    ctx = AgentContext(
        paper_spec=None,
        artifact=basic_artifact,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    result = PlanningAgent().run(ctx)
    assert result.success is False


def test_missing_artifact_raises(basic_kg, iat1_split):
    ctx = AgentContext(
        paper_spec=StubPaperSpec(),
        artifact=None,
        knowledge_graph=basic_kg,
        marks_split=iat1_split,
    )
    result = PlanningAgent().run(ctx)
    assert result.success is False


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_inputs_produce_same_plans(
    basic_artifact, basic_kg, iat1_split
):
    def run_once():
        ctx = AgentContext(
            paper_spec=StubPaperSpec(),
            artifact=basic_artifact,
            knowledge_graph=basic_kg,
            marks_split=iat1_split,
        )
        return PlanningAgent().run(ctx).payload

    p1 = run_once()
    p2 = run_once()
    assert [p.slot_id for p in p1] == [p.slot_id for p in p2]
    assert [p.topic for p in p1] == [p.topic for p in p2]
    assert [p.bloom for p in p1] == [p.bloom for p in p2]
    assert [p.visual_required for p in p1] == [p.visual_required for p in p2]


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PlanningAgent(max_visual_slots_per_module=-1)
    with pytest.raises(ValueError):
        PlanningAgent(min_figure_quality=1.5)
    with pytest.raises(ValueError):
        PlanningAgent(max_evidence_blocks_per_slot=0)
