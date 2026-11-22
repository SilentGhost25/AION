import io
import os
import pytest
from typing import Any, Dict, List, Optional
import docx

from core.contracts.paper_spec import PaperSpec
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.base import AgentContext
from core.generation.agents.agent_orchestrator import AgentOrchestrator
from core.generation.agents.checking_agent import CheckingAgent
from core.generation.agents.checking_contracts import (
    STATUS_PASS,
    STATUS_DEGRADED,
    STATUS_BLOCKED,
)
from core.generation.agents.evaluation_contracts import AuditReport, SlotVerdict, BLOOM_DEPTH_MISMATCH
from core.generation.agents.evaluation_agent import EvaluationAgent
from core.generation.agents.generated_question import (
    GeneratedQuestion,
    SubPart,
    WritingAgentOutput,
)
from core.generation.agents.legacy_adapter import to_legacy_shape
from core.generation.agents.planning_agent import (
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
from core.generation.agents.local_evaluators import run_local_evaluators
from v0_1.docx_export import render_docx_document


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
        import json
        return json.dumps({
            "text": "Explain virtual memory paging architecture.",
            "solution": "Paging maps virtual pages to physical frames.",
            "marking_scheme": [
                {"criterion": "Architecture", "marks": 3},
                {"criterion": "Translation", "marks": 2},
            ],
            "bloom_verb_used": "explain",
        })


class MockAPICaller:
    """Mock APICaller for EvaluationAgent Phase 2 remote semantic audit."""
    def __init__(self):
        self.calls = []

    def call(self, prompt: str, schema: Optional[dict] = None) -> Any:
        self.calls.append({"prompt": prompt, "schema": schema})
        import json
        # Return valid batch audit verdicts
        return json.dumps({
            "verdicts": [
                {
                    "slot_id": "module_1_Q1",
                    "verdict": "pass",
                    "reason_code": "APPROVED",
                    "explanation": "Question exhibits strong alignment and independent subparts.",
                }
            ]
        })

    @property
    def last_provider_name(self) -> Optional[str]:
        return "mock_provider"

    def reset(self):
        pass


@pytest.fixture(autouse=True)
def setup_test_registry():
    AssessmentRegistry._reset_for_tests()
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


def test_scenario_g2_i1_all_10_local_evaluators_pass():
    """Scenario G2-I1: All local evaluators in EvaluationAgent pass on G2 generated structured questions."""
    sp_a = SubPartPlan(label="a", marks=6, topic="Cache Coherence", canonical_concept_id="c_a", bloom="L4", co="CO3")
    sp_b = SubPartPlan(label="b", marks=4, topic="Memory Consistency", canonical_concept_id="c_b", bloom="L4", co="CO3")
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[6, 4],
        total_marks=10,
        co="CO3",
        bloom="L4",
        topic="Computer Architecture",
        sub_part_plans=[sp_a, sp_b],
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )

    resp_a = {
        "text": "Analyze cache coherence protocols under write-invalidate snooping.",
        "solution": "MESI protocol maintains state transitions for Modified, Exclusive, Shared, Invalid.",
        "marking_scheme": [
            {"criterion": "Protocol states", "marks": 3},
            {"criterion": "Invalidation messages", "marks": 3},
        ],
        "bloom_verb_used": "analyze",
    }
    resp_b = {
        "text": "Compare sequential consistency and relaxed memory consistency models.",
        "solution": "Sequential consistency requires program order; relaxed consistency allows memory reordering.",
        "marking_scheme": [
            {"criterion": "Sequential model", "marks": 2},
            {"criterion": "Relaxed model", "marks": 2},
        ],
        "bloom_verb_used": "compare",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = writer._generate_one(plan)

    assert attempts == 1
    assert len(gq.sub_parts) == 2

    # Run all 10 local evaluators on the synthesized question
    verdict = run_local_evaluators(gq)
    assert verdict is None, f"Local evaluators failed unexpectedly: {verdict.message if verdict else ''}"


def test_scenario_g2_i2_remote_evaluator_admission_confirmed():
    """Scenario G2-I2: Remote evaluator admission (Phase 2) confirmed via recording API caller."""
    sp_a = SubPartPlan(label="a", marks=6, topic="Paging", canonical_concept_id="c_a", bloom="L2", co="CO1")
    sp_b = SubPartPlan(label="b", marks=4, topic="Segmentation", canonical_concept_id="c_b", bloom="L2", co="CO1")
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[6, 4],
        total_marks=10,
        co="CO1",
        bloom="L2",
        topic="Memory Management",
        sub_part_plans=[sp_a, sp_b],
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )

    resp_a = {
        "text": "Explain page table translation using page table base register.",
        "solution": "PTBR holds the starting physical address of the page table.",
        "marking_scheme": [{"criterion": "Translation", "marks": 6}],
        "bloom_verb_used": "explain",
    }
    resp_b = {
        "text": "Describe segmentation memory management with limit and base registers.",
        "solution": "Limit register checks segment offset bounds; base register calculates physical location.",
        "marking_scheme": [{"criterion": "Registers", "marks": 4}],
        "bloom_verb_used": "describe",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = writer._generate_one(plan)

    # Inject mock remote API caller into EvaluationAgent
    mock_api = MockAPICaller()
    evaluator = EvaluationAgent(api_caller=mock_api)

    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10)
    ctx = AgentContext(paper_spec=spec, questions=[gq], plans=[plan])

    result = evaluator.run(ctx)
    assert result.success
    report = result.payload
    assert not report.degraded_mode
    assert report.api_calls == 1

    # Confirm mock_api was called with prompt containing both subparts
    assert len(mock_api.calls) == 1
    call_prompt = mock_api.calls[0]["prompt"]
    assert "Explain page table translation" in call_prompt
    assert "Describe segmentation memory" in call_prompt


def test_scenario_g2_i3_full_pipeline_to_physical_docx():
    """Scenario G2-I3: Full 5-agent pipeline execution through physical DOCX binary generation."""
    sp_a = SubPartPlan(label="a", marks=5, topic="Process States", canonical_concept_id="c_a", bloom="L2", co="CO1")
    sp_b = SubPartPlan(label="b", marks=5, topic="Process Control Block", canonical_concept_id="c_b", bloom="L2", co="CO1")
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        co="CO1",
        bloom="L2",
        topic="Processes",
        sub_part_plans=[sp_a, sp_b],
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )

    resp_a = {
        "text": "Explain process state transitions between ready, running, and blocked.",
        "solution": "Processes transition to ready on dispatch, running on CPU grant, blocked on I/O.",
        "marking_scheme": [{"criterion": "States", "marks": 5}],
        "bloom_verb_used": "explain",
    }
    resp_b = {
        "text": "Describe the contents of a Process Control Block in the Linux kernel.",
        "solution": "PCB includes task_struct, PID, process state, CPU registers, and scheduling parameters.",
        "marking_scheme": [{"criterion": "PCB fields", "marks": 5}],
        "bloom_verb_used": "describe",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    mock_api = MockAPICaller()

    spec = PaperSpec(exam_type="IAT1", module_count=1, questions_per_module=1, marks_per_question=10)
    ctx = AgentContext(paper_spec=spec, plans=[plan])

    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    evaluator = EvaluationAgent(api_caller=mock_api)
    refiner = RefinementAgent()
    checker = CheckingAgent()

    # Step 1: Writer
    w_out = writer.run(ctx)
    assert w_out.success
    ctx.questions = list(w_out.payload.questions)

    # Step 2: Evaluator
    e_out = evaluator.run(ctx)
    assert e_out.success
    ctx.audit_report = e_out.payload

    # Step 3: Refinement (no repairs needed)
    r_out = refiner.run(ctx)
    assert r_out.success

    # Step 4: Checker
    c_out = checker.run(ctx)
    assert c_out.success
    assert c_out.payload.status in ("OK", STATUS_PASS)

    # Step 5: Export to DOCX
    from core.generation.agents.orchestrator_contracts import OrchestratorResult
    orch_result = OrchestratorResult(
        success=True,
        qa_report=c_out.payload,
        plans=ctx.plans,
        questions=ctx.questions,
        audit_report=ctx.audit_report,
    )
    paper_parts, full_paper = to_legacy_shape(orch_result, ctx)
    paper_dict = {
        "title": "Operating Systems Internal Assessment Test",
        "subject": "Operating Systems",
        "subject_code": "21CS42",
        "exam_type": "IAT1",
        "department": "Computer Science & Engineering",
        "semester": "IV Semester",
        "max_marks": 10,
        "modules": paper_parts,
    }

    docx_io = render_docx_document(paper_dict)
    docx_bytes = docx_io.getvalue() if hasattr(docx_io, "getvalue") else docx_io
    assert isinstance(docx_bytes, (bytes, bytearray))
    assert len(docx_bytes) > 2000

    # Inspect physical DOCX contents
    doc = docx.Document(io.BytesIO(docx_bytes))
    all_table_text = " ".join([cell.text for t in doc.tables for row in t.rows for cell in row.cells])
    all_para_text = " ".join([p.text for p in doc.paragraphs])
    full_doc_text = f"{all_para_text} {all_table_text}"

    # Verify both subparts rendered with correct labels and text
    assert "(a)" in full_doc_text
    assert "(b)" in full_doc_text
    assert "Explain process state transitions" in full_doc_text
    assert "Describe the contents of a Process Control Block" in full_doc_text


def test_scenario_g2_i4_mixed_level_sibling_subpart_behavior_documented():
    """Scenario G2-I4: Mixed-level sibling plan behavior documented across WritingAgent and EvaluationAgent."""
    # Subpart (a) has Bloom L4 (6 marks, CO3), Subpart (b) has Bloom L2 (4 marks, CO1)
    sp_a = SubPartPlan(label="a", marks=6, topic="Cache Coherence", canonical_concept_id="c_a", bloom="L4", co="CO3")
    sp_b = SubPartPlan(label="b", marks=4, topic="Memory Consistency", canonical_concept_id="c_b", bloom="L2", co="CO1")
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[6, 4],
        total_marks=10,
        co="CO3",
        bloom="L4",  # Parent slot is rated L4
        topic="Computer Architecture",
        sub_part_plans=[sp_a, sp_b],
        constraints={
            "subpart_blooms": {"b": "L2"},
            "subpart_cos": {"b": "CO1"},
        },
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )

    resp_a = {
        "text": "Analyze snooping protocols in shared-memory multiprocessor architectures.",
        "solution": "Snooping monitors bus transactions to invalidate or update stale cache lines.",
        "marking_scheme": [{"criterion": "Analysis of snooping", "marks": 6}],
        "bloom_verb_used": "analyze",
    }
    resp_b = {
        "text": "Explain directory-based cache coherence structures and pointer storage.",
        "solution": "Directory controllers track sharing vectors across nodes to reduce broadcast traffic.",
        "marking_scheme": [{"criterion": "Explanation of directory", "marks": 4}],
        "bloom_verb_used": "explain",
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)

    # 1. WritingAgent validates each subpart according to its OWN SubPartPlan.bloom:
    # Subpart (a) validates against L4 (opening with 'analyze' - valid).
    # Subpart (b) validates against L2 (opening with 'explain' - valid).
    gq, attempts = writer._generate_one(plan)
    assert attempts == 1
    assert len(gq.sub_parts) == 2
    assert gq.sub_parts[0].text.startswith("Analyze snooping")
    assert gq.sub_parts[1].text.startswith("Explain directory-based")

    # 2. Document downstream EvaluationAgent behavior:
    # Parent question has question.bloom == "L4".
    # EvaluationAgent's bloom_content_evaluator checks all subparts of an L4 question.
    # Because subpart (b) used an L1/L2 verb ('explain') under an L4 parent rating,
    # EvaluationAgent flags BLOOM_DEPTH_MISMATCH.
    verdict = run_local_evaluators(gq)
    assert verdict is not None
    assert verdict.reason_code == BLOOM_DEPTH_MISMATCH
    assert "uses lower-order verb 'explain' (L1/L2) in a slot marked L4" in verdict.detail
