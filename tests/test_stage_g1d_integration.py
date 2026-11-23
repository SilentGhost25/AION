"""
Integration and Acceptance Test Suite for Stage G1-D & G1-D.1:
End-to-End Integration and Acceptance across all six pipeline stages.

Covers the component and full-pipeline integration scenarios:
1. Single-part [10] question
2. Two-part [6,4] question with different topics
3. Two-part [5,5] question with independent solutions and marking schemes
4. Registered multi-part blueprint resolution
5. Insufficient evidence or topic coverage — fail closed
6. One failed subpart after retries — no partial question or marks redistribution
7. Visual question using an object-based figure
8. Visual question using a dictionary-based figure
9. Successful generation through actual DOCX export
10. Failed generation propagated through the real orchestrator and CheckingAgent
11. Stage G1-D.1: Genuine five-agent pipeline success (Planning -> Writing -> Evaluation -> Refinement -> Checking -> DOCX)
12. Stage G1-D.1: Genuine pipeline failure propagation from real PlanningAgent to CheckingAgent blocking
13. Stage G1-D.1: Refinement preservation: subpart labels, marks, partition, and total marks immutable across repair
"""

import io
import os
import pytest
from typing import Any, Dict, List, Optional

import docx
from core.contracts.document_artifact import DocumentArtifact, TextBlock
from core.contracts.paper_spec import PaperSpec
from core.extraction.module_segmenter import ModuleSegment, SegmentationStrategy
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.base import AgentContext
from core.generation.agents.agent_orchestrator import AgentOrchestrator
from core.generation.agents.checking_agent import CheckingAgent
from core.generation.agents.checking_contracts import (
    BLOCK_REASON_UNRESOLVED_SLOTS,
    PaperBlockedError,
    STATUS_BLOCKED,
    STATUS_DEGRADED,
    STATUS_PASS,
)
from core.generation.agents.evaluation_contracts import AuditReport, SlotVerdict, BLOOM_MISMATCH
from core.generation.agents.evaluation_agent import EvaluationAgent
from core.generation.agents.generated_question import (
    GeneratedQuestion,
    SlotFailure,
    SubPart,
    WritingAgentOutput,
)
from core.generation.agents.legacy_adapter import to_legacy_shape
from core.generation.agents.planning_agent import (
    PlanFailureCode,
    PlanStatus,
    PlanningAgent,
    QuestionPlan,
    SubPartPlan,
)
from core.generation.agents.refinement_agent import RefinementAgent
from core.generation.agents.writing_agent import (
    GenerationFailed,
    WritingAgent,
)
from v0_1.docx_export import render_docx_document


# =============================================================================
# Test Mocks & Fixtures
# =============================================================================


class MockFigureObject:
    def __init__(self, id: str = "fig1", image_path: str = "/images/fig1.png", caption: str = "Architecture Diagram", page: int = 1):
        self.id = id
        self.image_path = image_path
        self.caption = caption
        self.page = page


class MockLLMCaller:
    def __init__(self, responses: Optional[List[Any]] = None, side_effect: Optional[Any] = None):
        self.responses = list(responses) if responses is not None else []
        self.side_effect = side_effect
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt: str, schema=None, image_path=None, seed=None):
        self.calls.append({
            "prompt": prompt,
            "schema": schema,
            "image_path": image_path,
            "seed": seed,
        })
        if self.side_effect:
            return self.side_effect(prompt, schema=schema, image_path=image_path, seed=seed)
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, dict):
                import json
                return json.dumps(resp)
            return resp
        # Default structured subpart response
        import json
        return json.dumps({
            "text": "Explain the fundamental principles of operating systems.",
            "solution": "An operating system manages hardware resources and provides common services.",
            "marking_scheme": [{"criterion": "Principles explained", "marks": 5}],
            "bloom_verb_used": "explain",
        })


class MockVLMCaller:
    def __init__(self, responses: Optional[List[Any]] = None):
        self.responses = list(responses) if responses is not None else []
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt: str, image_path: str, schema=None, seed=None):
        self.calls.append({
            "prompt": prompt,
            "image_path": image_path,
            "schema": schema,
            "seed": seed,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            if isinstance(resp, dict):
                import json
                return json.dumps(resp)
            return resp
        import json
        return json.dumps({
            "text": "With the aid of the given diagram, explain the process transition model.",
            "solution": "The process state transitions include ready, running, and waiting states.",
            "marking_scheme": [{"criterion": "State transitions explained", "marks": 10}],
            "references_image": True,
            "bloom_verb_used": "explain",
        })


class MockKnowledgeGraph:
    """Mock KnowledgeGraph providing module concepts and topology for PlanningAgent."""
    def __init__(self, concepts_by_module: Optional[Dict[str, List[str]]] = None):
        self._concepts = concepts_by_module or {
            "module_1": ["Process Scheduling", "CPU Dispatching", "Context Switching", "Virtual Memory Paging"]
        }

    def has_module(self, module_id: str) -> bool:
        return module_id in self._concepts

    def get_module(self, module_id: str):
        class _MG:
            def __init__(self, concepts):
                self._concepts = concepts
            def top_concepts(self, n: int = 100):
                return [(c, 1.0) for c in self._concepts[:n]]
            def get_synonyms(self, topic):
                return [f"{topic} Paradigm"]
            def get_neighborhood(self, topic, max_depth=2, min_confidence=0.0):
                return []
        return _MG(self._concepts.get(module_id, []))

    def get_synonyms(self, module_id: str, topic: str):
        return [f"{topic} Paradigm"]

    def get_concept_neighborhood(self, module_id: str, topic: str, max_depth=2):
        return [f"{topic} Neighbor"]


@pytest.fixture(autouse=True)
def setup_test_registry():
    """Register synthetic and verified assessment configs for integration tests."""
    AssessmentRegistry._reset_for_tests()
    # 1. Register production baseline
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (4, 6), (3, 7), (10,)),
        max_subparts=2,
        reference_source="VTU 2022 Scheme AEC Regulations (2024), Annexure-II, Clause 4(a)",
    ))
    # 2. Register synthetic test configs
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="CIE_IAT",
        blueprint_id="BP_10M",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4), (10,)),
        max_subparts=2,
        reference_source="Test Specs 10M",
    ))
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="SEE_EXAM",
        blueprint_id="BP_20M",
        total_marks_per_question=20,
        permitted_partitions=((8, 6, 6), (10, 10), (20,)),
        max_subparts=3,
        reference_source="Test Specs 20M",
    ))
    yield
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (4, 6), (3, 7), (10,)),
        max_subparts=2,
        reference_source="VTU 2022 Scheme AEC Regulations (2024), Annexure-II, Clause 4(a)",
    ))
    AssessmentRegistry.freeze()


def make_test_context(
    plans: Optional[List[QuestionPlan]] = None,
    module_count: int = 1,
    questions_per_module: int = 1,
    marks_per_question: int = 10,
) -> AgentContext:
    spec = PaperSpec(
        exam_type="IAT1",
        module_count=module_count,
        questions_per_module=questions_per_module,
        marks_per_question=marks_per_question,
    )
    ctx = AgentContext(paper_spec=spec)
    if plans:
        ctx.plans = plans
    return ctx


def make_planned_question(
    partition: List[int],
    slot_id: str = "module_1_Q1",
    module_id: str = "module_1",
    global_q_idx: int = 1,
    topics: Optional[List[str]] = None,
    visual_required: bool = False,
    figure: Optional[Any] = None,
    assessment_identity: Optional[tuple] = None,
    module_page_start: int = 1,
    module_page_end: int = 10,
    evidence_blocks: Optional[List[Dict[str, Any]]] = None,
) -> QuestionPlan:
    """Helper to construct a fully validated QuestionPlan."""
    total_marks = sum(partition)
    topics = topics or [f"Topic {chr(ord('a') + i)}" for i in range(len(partition))]
    identity = assessment_identity or ("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M" if total_marks == 10 else "BP_20M")
    is_synth = (identity[1] == "SYNTHETIC_TEST")

    sub_part_plans = []
    for i, marks in enumerate(partition):
        label = chr(ord('a') + i)
        part_fig = figure if (i == 0 and visual_required) else None
        if part_fig is not None and not is_synth:
            if isinstance(part_fig, dict):
                part_fig = dict(part_fig)
                part_fig.setdefault("module_id", module_id)
            elif hasattr(part_fig, "__dict__") and not getattr(part_fig, "module_id", None):
                part_fig.module_id = module_id

        part_evidence = evidence_blocks if evidence_blocks is not None else [
            {"id": f"blk_{label}_{i}", "text": f"Authoritative evidence text for {topics[i]}", "page": 1, "module_id": module_id}
        ]

        sub_part_plans.append(
            SubPartPlan(
                label=label,
                marks=marks,
                topic=topics[i],
                bloom="L2",
                co="CO1",
                canonical_concept_id=f"concept_{label}_{i}",
                evidence_blocks=part_evidence,
                visual_required=visual_required if i == 0 else False,
                figure=part_fig,
            )
        )

    return QuestionPlan(
        slot_id=slot_id,
        module_id=module_id,
        module_idx=1,
        slot_idx=0,
        global_q_idx=global_q_idx,
        total_marks=total_marks,
        partition=list(partition),
        bloom="L2",
        co="CO1",
        topic=topics[0],
        status=PlanStatus.PLANNED,
        sub_part_plans=sub_part_plans,
        visual_required=visual_required,
        figure=figure,
        module_page_start=module_page_start,
        module_page_end=module_page_end,
        assessment_identity=identity,
    )


# =============================================================================
# The Ten Required Integration Scenarios
# =============================================================================


def test_scenario_01_single_part_10m_question_end_to_end():
    """Scenario 1: Single-part [10] question from planning through DOCX export."""
    plan = make_planned_question([10], slot_id="module_1_Q1", topics=["Virtual Memory Paging"])
    ctx = make_test_context(plans=[plan])

    resp = {
        "text": "Explain the concept of virtual memory paging and address translation.",
        "solution": "Virtual memory paging divides physical memory into frames and logical memory into pages.",
        "marking_scheme": [{"criterion": "Paging concept and translation", "marks": 10}],
        "bloom_verb_used": "explain",
    }
    llm = MockLLMCaller(responses=[resp])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    evaluator = EvaluationAgent()
    checker = CheckingAgent()

    # Run writing through checking
    w_out = writer.run(ctx)
    assert w_out.success
    ctx.questions = list(w_out.payload.questions)

    # 1. Projections: single-part has unprefixed question text
    gq = ctx.questions[0]
    assert len(gq.sub_parts) == 1
    assert not gq.question_text.startswith("(a)")
    assert "Explain the concept of virtual memory" in gq.question_text

    # 2. Evaluation pass
    e_out = evaluator.run(ctx)
    assert e_out.success
    ctx.audit_report = e_out.payload

    # 3. Checking pass
    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status in (STATUS_PASS, STATUS_DEGRADED)
    assert c_out.payload.status != STATUS_BLOCKED
    assert len(c_out.payload.blocked_reasons) == 0
    assert len(c_out.payload.unresolved_slots) == 0

    # 4. Legacy adaptation
    from core.generation.agents.orchestrator_contracts import OrchestratorResult
    orch_result = OrchestratorResult(
        success=True,
        qa_report=c_out.payload,
        plans=ctx.plans,
        questions=ctx.questions,
        audit_report=ctx.audit_report,
    )
    paper_parts, full_paper = to_legacy_shape(orch_result, ctx)
    assert len(full_paper) == 1
    assert full_paper[0]["marks"] == 10

    # 5. Physical DOCX export
    paper_dict = {
        "subject": "Operating Systems",
        "subject_code": "21CS42",
        "exam_type": "IAT1",
        "modules": paper_parts,
    }
    docx_io = render_docx_document(paper_dict)
    docx_bytes = docx_io.getvalue()
    assert docx_bytes.startswith(b"PK\x03\x04")
    assert len(docx_bytes) > 5000

    doc = docx.Document(io.BytesIO(docx_bytes))
    full_text = "\n".join(p.text for p in doc.paragraphs)
    assert "VISVESVARAYA TECHNOLOGICAL UNIVERSITY" in full_text
    # Verify question text is in the document table
    table_text = "\n".join(cell.text for t in doc.tables for row in t.rows for cell in row.cells)
    assert "virtual memory paging" in table_text.lower()


def test_scenario_02_two_part_6_4_question_with_different_topics():
    """Scenario 2: Two-part [6,4] question with independent topics, verified via Evaluation and DOCX."""
    topics = ["Dijkstra Shortest Path Algorithm", "Bellman-Ford Distance Vector Routing"]
    plan = make_planned_question([6, 4], slot_id="module_1_Q1", topics=topics)
    ctx = make_test_context(plans=[plan])

    resp_a = {
        "text": "Explain Dijkstra shortest path algorithm with greedy vertex selection.",
        "solution": "Dijkstra algorithm iteratively selects the unvisited vertex with minimal distance.",
        "marking_scheme": [
            {"criterion": "Algorithm description", "marks": 4},
            {"criterion": "Time complexity analysis", "marks": 2},
        ],
        "bloom_verb_used": "explain",
    }
    resp_b = {
        "text": "Discuss Bellman-Ford algorithm and negative-weight cycle detection.",
        "solution": "Bellman-Ford relaxes all edges |V|-1 times to detect negative cycles.",
        "marking_scheme": [
            {"criterion": "Algorithm description", "marks": 2},
            {"criterion": "Cycle detection condition", "marks": 2},
        ],
        "bloom_verb_used": "discuss",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    evaluator = EvaluationAgent()
    checker = CheckingAgent()

    w_out = writer.run(ctx)
    assert w_out.success
    ctx.questions = list(w_out.payload.questions)

    # 1. Verify two subparts with independent topics and marks
    gq = ctx.questions[0]
    assert len(gq.sub_parts) == 2
    assert gq.sub_parts[0].label == "a" and gq.sub_parts[0].marks == 6
    assert gq.sub_parts[1].label == "b" and gq.sub_parts[1].marks == 4
    assert gq.sub_parts[0].topic == topics[0]
    assert gq.sub_parts[1].topic == topics[1]

    # 2. Local evaluators pass sibling independence and criteria sums
    e_out = evaluator.run(ctx)
    assert e_out.success
    ctx.audit_report = e_out.payload

    # 3. Checking agent verifies no unresolved slots
    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status in (STATUS_PASS, STATUS_DEGRADED)
    assert c_out.payload.status != STATUS_BLOCKED
    assert len(c_out.payload.blocked_reasons) == 0
    assert len(c_out.payload.unresolved_slots) == 0

    # 4. Legacy adaptation separates sub-questions
    from core.generation.agents.orchestrator_contracts import OrchestratorResult
    orch_result = OrchestratorResult(
        success=True,
        qa_report=c_out.payload,
        plans=ctx.plans,
        questions=ctx.questions,
        audit_report=ctx.audit_report,
    )
    paper_parts, full_paper = to_legacy_shape(orch_result, ctx)
    assert len(full_paper) == 1
    sqs = full_paper[0].get("sub_questions", [])
    assert len(sqs) == 2
    assert sqs[0]["marks"] == 6 and sqs[0]["label"] == "a"
    assert sqs[1]["marks"] == 4 and sqs[1]["label"] == "b"

    # 5. Render DOCX
    docx_io = render_docx_document({
        "subject": "Computer Networks",
        "subject_code": "21CS52",
        "modules": paper_parts,
    })
    assert docx_io.getvalue().startswith(b"PK\x03\x04")


def test_scenario_03_two_part_5_5_question_independent_solutions():
    """Scenario 3: Two-part [5,5] question with independent solutions and marking schemes."""
    plan = make_planned_question([5, 5], slot_id="module_1_Q1", topics=["Paging Architecture", "Segmentation Model"])
    ctx = make_test_context(plans=[plan])

    resp_a = {
        "text": "Explain physical paging architecture and page fault handling.",
        "solution": "Paging divides logical address space into fixed-size pages mapped via page table.",
        "marking_scheme": [{"criterion": "Paging model", "marks": 5}],
        "bloom_verb_used": "explain",
    }
    resp_b = {
        "text": "Discuss segmentation memory model with segment table lookups.",
        "solution": "Segmentation provides variable-size logical segments defined by base and limit.",
        "marking_scheme": [{"criterion": "Segmentation model", "marks": 5}],
        "bloom_verb_used": "discuss",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    evaluator = EvaluationAgent()
    checker = CheckingAgent()

    w_out = writer.run(ctx)
    assert w_out.success
    ctx.questions = list(w_out.payload.questions)

    gq = ctx.questions[0]
    assert len(gq.sub_parts) == 2
    assert gq.sub_parts[0].marks == 5 and gq.sub_parts[1].marks == 5
    assert gq.sub_parts[0].solution != gq.sub_parts[1].solution

    e_out = evaluator.run(ctx)
    assert e_out.success
    ctx.audit_report = e_out.payload

    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status in (STATUS_PASS, STATUS_DEGRADED)
    assert c_out.payload.status != STATUS_BLOCKED
    assert len(c_out.payload.blocked_reasons) == 0
    assert len(c_out.payload.unresolved_slots) == 0


def test_scenario_04_registered_multipart_blueprint_resolution():
    """Scenario 4: Assessment registry lookup and blueprint partition adherence."""
    reg = AssessmentRegistry()
    config = reg.get(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
    )
    assert config is not None
    assert (5, 5) in config.permitted_partitions
    assert (4, 6) in config.permitted_partitions

    # Select registered partition (5, 5)
    plan = make_planned_question(
        [5, 5],
        slot_id="module_1_Q1",
        assessment_identity=("VTU_2022_NEP", "AEC_GENERAL_THEORY", "CIE_IAT", "VTU_AEC_ANNEXURE_II"),
    )
    assert plan.partition == [5, 5]
    assert plan.total_marks == 10

    llm = MockLLMCaller()
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    ctx = make_test_context(plans=[plan])
    w_out = writer.run(ctx)
    assert w_out.success
    gq = w_out.payload.questions[0]
    assert list(gq.partition) == [5, 5]


def test_scenario_05_insufficient_evidence_fails_closed():
    """Scenario 5: Insufficient evidence or ungrounded plan triggers fail-closed unresolved slot."""
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[10],
        total_marks=10,
        co="CO1",
        bloom="L2",
        sub_part_plans=[],
        status=PlanStatus.UNRESOLVED,
        failure_code=PlanFailureCode.INSUFFICIENT_TOPICS.value,
        failure_detail="No authoritative evidence blocks found for topic in module range",
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
        telemetry={
            "attempts": [
                {
                    "attempt_number": 1,
                    "strategy": "TOP_CONCEPTS",
                    "failure_code": PlanFailureCode.INSUFFICIENT_TOPICS.value,
                    "failure_detail": "No candidate topics found in module range",
                    "candidate_topics": [],
                }
            ]
        },
    )

    ctx = make_test_context(plans=[plan])
    llm = MockLLMCaller()
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    checker = CheckingAgent()

    # Writer bypasses unresolved plan
    w_out = writer.run(ctx)
    assert w_out.success
    assert len(w_out.payload.questions) == 0
    assert len(w_out.payload.failures) == 1
    assert w_out.payload.failures[0].slot_id == "module_1_Q1"

    # Orchestrator flow
    ctx.unresolved_slots = ["module_1_Q1"]
    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status == STATUS_BLOCKED
    assert BLOCK_REASON_UNRESOLVED_SLOTS in c_out.payload.blocked_reasons


def test_scenario_06_failed_subpart_after_retries_fails_closed_no_partial_redistribution():
    """Scenario 6: One failed subpart after retries fails whole slot with no marks redistribution."""
    plan = make_planned_question([6, 4], slot_id="module_1_Q1")
    ctx = make_test_context(plans=[plan])

    resp_a = {
        "text": "Explain shortest path routing principles.",
        "solution": "Shortest path routing chooses minimal weight paths.",
        "marking_scheme": [{"criterion": "Principles", "marks": 6}],
        "bloom_verb_used": "explain",
    }
    # Subpart b repeatedly fails with invalid marking sum
    resp_b_bad = {
        "text": "Discuss distance vector routing protocols.",
        "solution": "Distance vector protocols exchange vectors with neighbours.",
        "marking_scheme": [{"criterion": "Principles", "marks": 99}],  # bad marks
        "bloom_verb_used": "discuss",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b_bad, resp_b_bad, resp_b_bad])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)

    w_out = writer.run(ctx)
    assert w_out.success
    # Slot failed completely — zero partial questions returned
    assert len(w_out.payload.questions) == 0
    assert len(w_out.payload.failures) == 1
    f = w_out.payload.failures[0]
    assert f.slot_id == "module_1_Q1"
    assert f.failure_code == "SUBPART_GENERATION_FAILED"


def test_scenario_07_visual_question_object_based_figure():
    """Scenario 7: Visual question using an object-based figure routes to VLM and maps to parent."""
    fig_obj = MockFigureObject(id="fig_obj_1", image_path="/assets/diagram1.png", caption="OS Architecture", page=3)
    plan = make_planned_question([10], slot_id="module_1_Q1", visual_required=True, figure=fig_obj)
    ctx = make_test_context(plans=[plan])

    vlm_resp = {
        "text": "With reference to the figure, explain the layered operating system architecture.",
        "solution": "The layered architecture separates user space from kernel space.",
        "marking_scheme": [{"criterion": "Architecture layers", "marks": 10}],
        "references_image": True,
        "bloom_verb_used": "explain",
    }

    llm = MockLLMCaller()
    vlm = MockVLMCaller(responses=[vlm_resp])
    writer = WritingAgent(llm_caller=llm, vlm_caller=vlm, enable_irt=False)

    w_out = writer.run(ctx)
    assert w_out.success
    gq = w_out.payload.questions[0]

    assert gq.references_image is True
    assert gq.image_path == "/assets/diagram1.png"
    assert gq.figure_caption == "OS Architecture"
    assert gq.generation_source == "local_vlm"
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_path"] == "/assets/diagram1.png"


def test_scenario_08_visual_question_dictionary_based_figure():
    """Scenario 8: Visual question using a dictionary-based figure routes to VLM and maps to parent."""
    fig_dict = {
        "id": "fig_dict_1",
        "image_path": "/assets/cache_layout.png",
        "caption": "Cache Memory Layout",
        "page": 5,
    }
    plan = make_planned_question([10], slot_id="module_1_Q1", visual_required=True, figure=fig_dict)
    ctx = make_test_context(plans=[plan])

    vlm_resp = {
        "text": "Refer to the given figure and explain cache memory organization.",
        "solution": "Cache organization partitions blocks into sets and tags.",
        "marking_scheme": [{"criterion": "Organization explanation", "marks": 10}],
        "references_image": True,
        "bloom_verb_used": "explain",
    }

    llm = MockLLMCaller()
    vlm = MockVLMCaller(responses=[vlm_resp])
    writer = WritingAgent(llm_caller=llm, vlm_caller=vlm, enable_irt=False)

    w_out = writer.run(ctx)
    assert w_out.success
    gq = w_out.payload.questions[0]

    assert gq.references_image is True
    assert gq.image_path == "/assets/cache_layout.png"
    assert gq.figure_caption == "Cache Memory Layout"
    assert gq.generation_source == "local_vlm"
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_path"] == "/assets/cache_layout.png"


def test_scenario_09_successful_full_generation_through_actual_docx_export():
    """Scenario 9: Complete 5-agent pipeline execution through physical DOCX binary generation."""
    plan1 = make_planned_question([10], slot_id="module_1_Q1", global_q_idx=1, topics=["Process Synchronization"])
    plan2 = make_planned_question([6, 4], slot_id="module_1_Q2", global_q_idx=2, topics=["Deadlock Prevention", "Deadlock Avoidance"])
    ctx = make_test_context(plans=[plan1, plan2], module_count=1, questions_per_module=2)

    resp_1 = {
        "text": "Explain process synchronization using semaphores and mutex locks.",
        "solution": "Semaphores provide wait and signal primitives to enforce mutual exclusion.",
        "marking_scheme": [{"criterion": "Semaphore mechanics", "marks": 10}],
        "bloom_verb_used": "explain",
    }
    resp_2a = {
        "text": "Explain deadlock prevention conditions and mutual exclusion holding.",
        "solution": "Deadlock prevention eliminates at least one of the four necessary conditions.",
        "marking_scheme": [{"criterion": "Prevention conditions", "marks": 6}],
        "bloom_verb_used": "explain",
    }
    resp_2b = {
        "text": "Discuss Banker algorithm for deadlock avoidance in multi-resource systems.",
        "solution": "Banker algorithm verifies whether granting a request leaves the system in a safe state.",
        "marking_scheme": [{"criterion": "Banker algorithm", "marks": 4}],
        "bloom_verb_used": "discuss",
    }

    llm = MockLLMCaller(responses=[resp_1, resp_2a, resp_2b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    evaluator = EvaluationAgent()
    checker = CheckingAgent()

    # Step 1: Write
    w_out = writer.run(ctx)
    assert w_out.success
    ctx.questions = list(w_out.payload.questions)

    # Step 2: Evaluate
    e_out = evaluator.run(ctx)
    assert e_out.success
    ctx.audit_report = e_out.payload

    # Step 3: Check
    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status in (STATUS_PASS, STATUS_DEGRADED)
    assert c_out.payload.status != STATUS_BLOCKED
    assert len(c_out.payload.blocked_reasons) == 0
    assert len(c_out.payload.unresolved_slots) == 0

    # Step 4: Adapt to legacy paper dict
    from core.generation.agents.orchestrator_contracts import OrchestratorResult
    orch_result = OrchestratorResult(
        success=True,
        qa_report=c_out.payload,
        plans=ctx.plans,
        questions=ctx.questions,
        audit_report=ctx.audit_report,
    )
    paper_parts, full_paper = to_legacy_shape(orch_result, ctx)
    assert len(paper_parts) == 1
    assert len(full_paper) == 2

    # Step 5: Render physical DOCX
    paper_dict = {
        "title": "Operating Systems Internal Assessment Test",
        "subject": "Operating Systems",
        "subject_code": "21CS42",
        "exam_type": "IAT1",
        "department": "Computer Science & Engineering",
        "semester": "IV Semester",
        "max_marks": 20,
        "modules": paper_parts,
    }
    docx_io = render_docx_document(paper_dict)
    docx_bytes = docx_io.getvalue()

    # Verify physical document binary
    assert docx_bytes.startswith(b"PK\x03\x04")
    assert len(docx_bytes) > 8000

    doc = docx.Document(io.BytesIO(docx_bytes))
    paragraphs_text = "\n".join(p.text for p in doc.paragraphs)
    assert "VISVESVARAYA TECHNOLOGICAL UNIVERSITY" in paragraphs_text
    assert "Operating Systems" in paragraphs_text


def test_scenario_10_failed_generation_propagates_through_orchestrator_and_checker():
    """Scenario 10: Failed writing slot propagates to orchestrator, blocks checker, and raises PaperBlockedError."""
    plan = make_planned_question([10], slot_id="module_1_Q1")
    ctx = make_test_context(plans=[plan])

    # Model consistently returns unparseable garbage
    llm = MockLLMCaller(responses=["Not valid JSON 1", "Not valid JSON 2", "Not valid JSON 3"])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    checker = CheckingAgent()

    # Step 1: WritingAgent returns SlotFailure
    w_out = writer.run(ctx)
    assert w_out.success
    assert len(w_out.payload.questions) == 0
    assert len(w_out.payload.failures) == 1

    # Step 2: Orchestrator executes pipeline and records failure in context.unresolved_slots
    class MockPlanningAgent(PlanningAgent):
        def run(self, context):
            from core.generation.agents.base import AgentResult
            return AgentResult(success=True, payload=context.plans)

    orchestrator = AgentOrchestrator(
        planning_agent=MockPlanningAgent(),
        writing_agent=writer,
        checking_agent=checker,
    )

    result = orchestrator.run(ctx)
    assert result.success is True
    assert result.paper_blocked is True
    assert result.qa_report.status == STATUS_BLOCKED
    assert "module_1_Q1" in result.qa_report.unresolved_slots
    assert BLOCK_REASON_UNRESOLVED_SLOTS in result.qa_report.blocked_reasons

    # Step 3: run_or_raise triggers PaperBlockedError
    with pytest.raises(PaperBlockedError):
        orchestrator.run_or_raise(ctx)


# =============================================================================
# Stage G1-D.1: Full Five-Agent Pipeline & Refinement Preservation Integration
# =============================================================================


def test_scenario_11_full_five_agent_pipeline_success_orchestrator_planning_to_export():
    """
    Scenario 11 (G1-D.1): Genuine complete 5-agent pipeline execution through AgentOrchestrator.
    PlanningAgent -> WritingAgent -> EvaluationAgent -> RefinementAgent -> CheckingAgent -> DOCX.
    Plans are generated authoritatively by PlanningAgent from registered config and valid evidence blocks.
    """
    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10)
    blocks = [
        TextBlock(id="blk_1", text="Process scheduling selects candidate threads and controls CPU allocation.", page=1, block_role="BODY"),
        TextBlock(id="blk_2", text="CPU dispatching handles low-level context switching between threads.", page=2, block_role="BODY"),
    ]
    artifact = DocumentArtifact(
        source_pdf_sha256="test_sha_e2e",
        source_pdf_path="operating_systems.pdf",
        text_blocks=blocks,
    )
    seg = ModuleSegment(
        module_id=1,
        title="Module 1: Process and Memory Management",
        page_start=1,
        page_end=10,
        confidence=0.98,
        strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING,
    )

    ctx = AgentContext(
        paper_spec=spec,
        artifact=artifact,
        knowledge_graph=MockKnowledgeGraph(),
    )
    ctx.module_segments = [seg]
    ctx.request = {
        "scheme_id": "VTU_2022_NEP",
        "course_type": "AEC_GENERAL_THEORY",
        "assessment_type": "CIE_IAT",
        "blueprint_id": "VTU_AEC_ANNEXURE_II",
    }

    # Model returns compliant responses matching assigned Bloom verbs (L1 band: state, list)
    resp_a = {
        "text": "State the core process scheduling principles and define execution states.",
        "solution": "Process scheduling states include ready, running, and waiting.",
        "marking_scheme": [{"criterion": "Scheduling principles", "marks": 5}],
        "bloom_verb_used": "state",
    }
    resp_b = {
        "text": "List the CPU dispatching mechanics and define context switching latency.",
        "solution": "Dispatcher provides CPU control and saves program counter.",
        "marking_scheme": [{"criterion": "Dispatching mechanics", "marks": 5}],
        "bloom_verb_used": "list",
    }

    planner = PlanningAgent()
    writer = WritingAgent(llm_caller=MockLLMCaller(responses=[resp_a, resp_b]), enable_irt=False)
    evaluator = EvaluationAgent()
    refiner = RefinementAgent()
    checker = CheckingAgent()

    orchestrator = AgentOrchestrator(
        planning_agent=planner,
        writing_agent=writer,
        checking_agent=checker,
        evaluation_agent=evaluator,
        refinement_agent=refiner,
    )

    # Execute complete orchestrator pipeline
    result = orchestrator.run(ctx)

    # 1. Pipeline execution status
    assert result.success is True
    assert result.paper_blocked is False
    assert result.qa_report.status in (STATUS_PASS, STATUS_DEGRADED)
    assert len(result.qa_report.unresolved_slots) == 0
    assert len(result.qa_report.blocked_reasons) == 0

    # 2. Planning verification: real plans generated by PlanningAgent
    assert len(result.plans) == 1
    plan = result.plans[0]
    assert plan.status == PlanStatus.PLANNED
    assert plan.partition == [5, 5]
    assert plan.total_marks == 10
    assert len(plan.sub_part_plans) == 2
    assert plan.sub_part_plans[0].label == "a" and plan.sub_part_plans[0].marks == 5
    assert plan.sub_part_plans[1].label == "b" and plan.sub_part_plans[1].marks == 5
    assert len(plan.sub_part_plans[0].evidence_blocks) > 0
    assert len(plan.sub_part_plans[1].evidence_blocks) > 0

    # 3. Question structure and refinement preservation
    assert len(result.questions) == 1
    q = result.questions[0]
    assert len(q.sub_parts) == 2
    assert [sp.label for sp in q.sub_parts] == ["a", "b"]
    assert [sp.marks for sp in q.sub_parts] == [5, 5]
    assert list(q.partition) == [5, 5]
    assert q.marks == 10

    # 4. Downstream export integration through actual DOCX generation
    paper_parts, full_paper = to_legacy_shape(result, ctx)
    assert len(paper_parts) == 1
    assert len(full_paper) == 1
    assert len(full_paper[0]["sub_questions"]) == 2

    paper_dict = {
        "title": "Operating Systems Assessment",
        "subject": "Operating Systems",
        "subject_code": "21CS42",
        "exam_type": "IAT1",
        "department": "Computer Science & Engineering",
        "semester": "IV Semester",
        "max_marks": 10,
        "modules": paper_parts,
    }
    docx_io = render_docx_document(paper_dict)
    docx_bytes = docx_io.getvalue()

    assert docx_bytes.startswith(b"PK\x03\x04")
    assert len(docx_bytes) > 8000

    doc = docx.Document(io.BytesIO(docx_bytes))
    paragraphs_text = "\n".join(p.text for p in doc.paragraphs)
    assert "VISVESVARAYA TECHNOLOGICAL UNIVERSITY" in paragraphs_text
    table_text = "\n".join(cell.text for t in doc.tables for row in t.rows for cell in row.cells)
    assert "process scheduling" in table_text.lower()
    assert "cpu dispatching" in table_text.lower()


def test_scenario_12_full_pipeline_failure_propagation_planning_to_checker_blocking():
    """
    Scenario 12 (G1-D.1): Complete pipeline failure propagation from real PlanningAgent to CheckingAgent.
    Real PlanningAgent plans the slot -> WritingAgent fails -> AgentOrchestrator tracks unresolved slot ->
    CheckingAgent hard-blocks the paper -> run_or_raise raises PaperBlockedError.
    """
    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10)
    blocks = [
        TextBlock(id="blk_1", text="Process scheduling controls CPU allocation to threads.", page=1, block_role="BODY"),
    ]
    artifact = DocumentArtifact(source_pdf_sha256="test_sha", source_pdf_path="os.pdf", text_blocks=blocks)
    seg = ModuleSegment(
        module_id=1,
        title="Module 1",
        page_start=1,
        page_end=10,
        confidence=0.95,
        strategy_used=SegmentationStrategy.EXPLICIT_MODULE_HEADING,
    )

    ctx = AgentContext(
        paper_spec=spec,
        artifact=artifact,
        knowledge_graph=MockKnowledgeGraph(),
    )
    ctx.module_segments = [seg]
    ctx.request = {
        "scheme_id": "VTU_2022_NEP",
        "course_type": "AEC_GENERAL_THEORY",
        "assessment_type": "CIE_IAT",
        "blueprint_id": "VTU_AEC_ANNEXURE_II",
    }

    # Model consistently returns invalid JSON
    failing_llm = MockLLMCaller(responses=["garbage 1", "garbage 2", "garbage 3"])
    planner = PlanningAgent()
    writer = WritingAgent(llm_caller=failing_llm, enable_irt=False)
    checker = CheckingAgent()

    orchestrator = AgentOrchestrator(
        planning_agent=planner,
        writing_agent=writer,
        checking_agent=checker,
    )

    result = orchestrator.run(ctx)
    assert result.success is True
    assert result.paper_blocked is True
    assert result.qa_report.status == STATUS_BLOCKED
    assert "module_1_Q1" in result.qa_report.unresolved_slots
    assert BLOCK_REASON_UNRESOLVED_SLOTS in result.qa_report.blocked_reasons

    with pytest.raises(PaperBlockedError):
        orchestrator.run_or_raise(ctx)


def test_scenario_13_refinement_preservation_subpart_labels_marks_and_partitions_immutable_on_repair():
    """
    Scenario 13 (G1-D.1): Refinement preservation contract.
    When RefinementAgent repairs a question flagged by EvaluationAgent, the subpart labels,
    marks, partition, and total marks must remain strictly immutable.
    """
    class MockRepairLLM:
        def call(self, prompt, schema=None, seed=None):
            import json
            return json.dumps({
                "sub_parts": [
                    {
                        "label": "a",
                        "text": "State precisely the concepts of topic A.",
                        "solution": "Sol A",
                        "marking_scheme": [{"criterion": "Crit A", "marks": 6}],
                    },
                    {
                        "label": "b",
                        "text": "State precisely the concepts of topic B.",
                        "solution": "Sol B",
                        "marking_scheme": [{"criterion": "Crit B", "marks": 4}],
                    },
                ],
                "question_text": "State precisely the concepts of topic A and topic B.",
                "solution": "Refined comprehensive solution for topic A and B.",
                "marking_scheme": [
                    {"criterion": "Topic A explanation", "marks": 6},
                    {"criterion": "Topic B explanation", "marks": 4},
                ],
                "change_summary": "Updated question text and rubric criteria phrasing.",
            })

    orig_q = GeneratedQuestion(
        slot_id="module_1_Q1",
        module_id="module_1",
        global_q_idx=1,
        question_text="Explain topic A and topic B in detail.",
        solution="Original draft solution",
        marking_scheme=[
            {"criterion": "Topic A", "marks": 6},
            {"criterion": "Topic B", "marks": 4},
        ],
        marks=10,
        partition=[6, 4],
        sub_parts=[
            SubPart(label="a", marks=6, text="Explain topic A in detail.", topic="Topic A", solution="Sol A", marking_scheme=[{"criterion": "Crit A", "marks": 6}]),
            SubPart(label="b", marks=4, text="Explain topic B in detail.", topic="Topic B", solution="Sol B", marking_scheme=[{"criterion": "Crit B", "marks": 4}]),
        ],
    )

    verdict = SlotVerdict(
        slot_id="module_1_Q1",
        verdict="fail",
        source="local",
        reason_code=BLOOM_MISMATCH,
        detail="Bloom verb mismatch: question uses explain instead of state",
        suggested_fix="Change verb to state",
    )
    audit_report = AuditReport(verdicts=[verdict])
    ctx = AgentContext(questions=[orig_q], audit_report=audit_report)

    refiner = RefinementAgent(llm_caller=MockRepairLLM())
    res = refiner.run(ctx)
    assert res.success is True
    assert len(ctx.questions) == 1

    repaired_q = ctx.questions[0]

    # Verify that repair modified the question text as intended
    assert "State precisely" in repaired_q.question_text

    # Strict Refinement Preservation Invariants:
    # 1. Subpart count is preserved
    assert len(repaired_q.sub_parts) == len(orig_q.sub_parts)
    # 2. Subpart labels are preserved
    assert [sp.label for sp in repaired_q.sub_parts] == [sp.label for sp in orig_q.sub_parts] == ["a", "b"]
    # 3. Subpart marks are preserved
    assert [sp.marks for sp in repaired_q.sub_parts] == [sp.marks for sp in orig_q.sub_parts] == [6, 4]
    # 4. Partition is preserved
    assert list(repaired_q.partition) == list(orig_q.partition) == [6, 4]
    # 5. Total marks are preserved
    assert repaired_q.marks == orig_q.marks == 10
    # 6. Marks conservation
    assert sum(sp.marks for sp in repaired_q.sub_parts) == repaired_q.marks
