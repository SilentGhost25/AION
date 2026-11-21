"""
Acceptance Test Suite for Stage G1-C:
WritingAgent Structured SubPart Generation & Bounded Retries.

Covers all 35 specified tests across Groups 1 through 7:
1. Contract Adherence (Tests 1–5)
2. Single-part & Multi-part Formats (Tests 6–10)
3. Immutable Plan Inputs & Pedagogical Bounds (Tests 11–15)
4. Strict Criteria & Response Validation (Tests 16–20)
5. Retry Isolation & Bounded Retries (Tests 21–25)
6. Projection Determinism & Visual Mapping (Tests 26–30)
7. End-to-End Orchestrator & Plan Status Gating (Tests 31–35)
"""

import copy
import pytest
from dataclasses import replace
from typing import Any, Dict, List, Optional

from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.base import AgentContext
from core.generation.agents.checking_contracts import (
    BLOCK_REASON_UNRESOLVED_SLOTS,
    STATUS_BLOCKED,
)
from core.generation.agents.generated_question import (
    GeneratedQuestion,
    SlotFailure,
    SubPart,
    WritingAgentOutput,
)
from core.generation.agents.planning_agent import (
    PlanFailureCode,
    PlanStatus,
    QuestionPlan,
    SubPartPlan,
)
from core.generation.agents.prompts import (
    BLOOM_VERBS,
    build_subpart_prompt,
    build_task_section,
)
from core.generation.agents.writing_agent import (
    GenerationFailed,
    WritingAgent,
)


# =============================================================================
# Test Mocks & Fixtures
# =============================================================================


class MockFigure:
    def __init__(self, id: str = "fig1", image_path: str = "/images/fig1.png", caption: str = "Test Figure", page: int = 1):
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
            return resp
        # Default valid response
        return {
            "text": "Explain the role of the operating system scheduler.",
            "solution": "The scheduler allocates CPU time to ready processes.",
            "marking_scheme": [
                {"criterion": "Role definition", "marks": 2},
                {"criterion": "Explanation of dispatching", "marks": 3},
            ],
            "diagram_request": None,
            "references_image": False,
            "bloom_verb_used": "Explain",
        }


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
            return resp
        return {
            "text": "With the aid of the given diagram, explain the state transition model.",
            "solution": "Processes transition between Ready, Running, and Blocked states.",
            "marking_scheme": [
                {"criterion": "Diagram identification", "marks": 2},
                {"criterion": "Transition explanation", "marks": 3},
            ],
            "diagram_request": {"diagram_type": "state_diagram", "description": "Process state transitions"},
            "references_image": True,
            "bloom_verb_used": "Explain",
        }


@pytest.fixture(autouse=True)
def setup_test_registry():
    """Register synthetic assessment configs for unit tests."""
    AssessmentRegistry._reset_for_tests()
    cfg_10 = AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="CIE_IAT",
        blueprint_id="BP_10M",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4), (10,)),
        max_subparts=2,
        reference_source="Test Specs 10M",
    )
    cfg_20 = AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="SEE_EXAM",
        blueprint_id="BP_20M",
        total_marks_per_question=20,
        permitted_partitions=((8, 6, 6), (10, 10), (20,)),
        max_subparts=3,
        reference_source="Test Specs 20M",
    )
    AssessmentRegistry.register(cfg_10)
    AssessmentRegistry.register(cfg_20)
    yield
    AssessmentRegistry._reset_for_tests()


def create_planned_question(
    partition: List[int] = [5, 5],
    total_marks: int = 10,
    blueprint_id: str = "BP_10M",
    co: str = "CO1",
    bloom: str = "L2",
    assessment_type: str = "CIE_IAT",
) -> QuestionPlan:
    """Helper to create a fully valid QuestionPlan with synthetic subparts."""
    subparts = []
    for i, marks in enumerate(partition):
        lbl = chr(ord('a') + i)
        subparts.append(SubPartPlan(
            label=lbl,
            marks=marks,
            topic=f"Concept {lbl.upper()}",
            canonical_concept_id=f"concept_{lbl}",
            bloom=bloom,
            co=co,
        ))

    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=list(partition),
        total_marks=total_marks,
        co=co,
        bloom=bloom,
        topic="Operating Systems Overview",
        sub_part_plans=subparts,
        status=PlanStatus.PLANNED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", assessment_type, blueprint_id),
    )
    valid, code, err = plan.validation_result()
    assert valid, f"Test setup fixture plan is invalid: {err} ({code})"
    return plan


# =============================================================================
# Group 1: Contract Adherence (Tests 1–5)
# =============================================================================

def test_01_subpart_uses_text_field_name():
    """Test 1: SubPart instances use 'text' and do not accept 'question_text'."""
    plan = create_planned_question([5, 5])
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(gq.sub_parts) == 2
    for sp in gq.sub_parts:
        assert isinstance(sp, SubPart)
        assert hasattr(sp, "text")
        assert not hasattr(sp, "question_text")
        assert isinstance(sp.text, str)
        assert sp.text.strip() != ""


def test_02_marking_scheme_criteria_sum_strictly_enforced():
    """Test 2: Marking scheme criteria sum mismatch triggers retry and succeeds on fix."""
    plan = create_planned_question([5, 5])
    # Subpart a: Attempt 1 has sum 4 != 5, Attempt 2 has sum 5 == 5
    resp_bad_sum = {
        "text": "Explain process states.",
        "solution": "Processes are New, Ready, Running, Waiting, Terminated.",
        "marking_scheme": [{"criterion": "States", "marks": 4}],
    }
    resp_good_sum_a = {
        "text": "Explain process states.",
        "solution": "Processes are New, Ready, Running, Waiting, Terminated.",
        "marking_scheme": [{"criterion": "States", "marks": 5}],
    }
    resp_good_b = {
        "text": "Describe PCB fields.",
        "solution": "PCB contains PID, PC, registers, memory limits.",
        "marking_scheme": [{"criterion": "PCB fields", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_bad_sum, resp_good_sum_a, resp_good_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].marks == 5
    assert sum(c["marks"] for c in gq.sub_parts[0].marking_scheme) == 5
    assert gq.telemetry["subpart_attempts"]["a"] == 2
    assert gq.telemetry["subpart_attempts"]["b"] == 1


def test_03_empty_text_or_solution_rejected():
    """Test 3: Empty text or empty solution triggers retry."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    resp_empty_text = {
        "text": "   ",
        "solution": "Valid solution.",
        "marking_scheme": [{"criterion": "c1", "marks": 10}],
    }
    resp_valid = {
        "text": "Explain virtual memory.",
        "solution": "Virtual memory separates user logical memory from physical storage.",
        "marking_scheme": [{"criterion": "Virtual memory", "marks": 10}],
    }

    llm = MockLLMCaller(responses=[resp_empty_text, resp_valid])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].text == "Explain virtual memory."
    assert gq.telemetry["subpart_attempts"]["a"] == 2


def test_04_marks_exact_int_type():
    """Test 4: Marks on SubPart and criteria must be exact positive int (rejecting bool)."""
    sp = SubPart(
        label="a",
        marks=5,
        text="Explain paging.",
        solution="Paging divides memory into frames.",
        marking_scheme=[{"criterion": "Explanation", "marks": 5}],
    )
    assert type(sp.marks) is int
    assert not isinstance(sp.marks, bool)

    with pytest.raises(TypeError):
        SubPart(label="a", marks=True, text="Explain paging.")

    with pytest.raises(TypeError):
        SubPart(label="a", marks=5.0, text="Explain paging.")


def test_05_field_25_subparts_populated():
    """Test 5: Structured generation populates Field 25 (sub_parts) on GeneratedQuestion."""
    plan = create_planned_question([5, 5])
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert hasattr(gq, "sub_parts")
    assert isinstance(gq.sub_parts, list)
    assert len(gq.sub_parts) == 2
    assert all(isinstance(sp, SubPart) for sp in gq.sub_parts)


# =============================================================================
# Group 2: Single-part & Multi-part Formats (Tests 6–10)
# =============================================================================

def test_06_single_part_10_marks_unprefixed_projection():
    """Test 6: Single-part [10] question has un-prefixed question_text (no '(a)')."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    llm = MockLLMCaller(responses=[{
        "text": "Explain memory management unit functions.",
        "solution": "MMU translates virtual addresses to physical addresses.",
        "marking_scheme": [{"criterion": "MMU explanation", "marks": 10}],
    }])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert not gq.is_multi_part
    assert len(gq.sub_parts) == 1
    assert gq.question_text == "Explain memory management unit functions."
    assert not gq.question_text.startswith("(a)")


def test_07_multipart_6_4_marks_labeled_projections():
    """Test 7: Multi-part [6, 4] produces labeled '(a)' and '(b)' question_text."""
    plan = create_planned_question([6, 4], blueprint_id="BP_10M")
    resp_a = {
        "text": "Explain page replacement algorithms.",
        "solution": "FIFO, LRU, Optimal.",
        "marking_scheme": [{"criterion": "Algorithms", "marks": 6}],
    }
    resp_b = {
        "text": "Calculate page faults for the given reference string.",
        "solution": "Total page faults = 9.",
        "marking_scheme": [{"criterion": "Calculation", "marks": 4}],
    }
    llm = MockLLMCaller(responses=[resp_a, resp_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.is_multi_part
    assert len(gq.sub_parts) == 2
    assert gq.question_text == "(a) Explain page replacement algorithms.\n\n(b) Calculate page faults for the given reference string."
    assert gq.question_text.startswith("(a)")
    assert "\n\n(b)" in gq.question_text


def test_08_multipart_8_6_6_marks_labeled_projections():
    """Test 8: Multi-part [8, 6, 6] produces (a), (b), (c) subparts and projections."""
    plan = create_planned_question([8, 6, 6], total_marks=20, blueprint_id="BP_20M", assessment_type="SEE_EXAM")
    resp_a = {
        "text": "Explain CPU scheduling criteria.",
        "solution": "CPU utilization, throughput, turnaround time.",
        "marking_scheme": [{"criterion": "Criteria", "marks": 8}],
    }
    resp_b = {
        "text": "Explain deadlock prevention methods.",
        "solution": "Eliminate mutual exclusion, hold and wait.",
        "marking_scheme": [{"criterion": "Deadlock methods", "marks": 6}],
    }
    resp_c = {
        "text": "Describe semaphore implementation.",
        "solution": "Wait and signal atomic operations.",
        "marking_scheme": [{"criterion": "Semaphores", "marks": 6}],
    }
    llm = MockLLMCaller(responses=[resp_a, resp_b, resp_c])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(gq.sub_parts) == 3
    assert [sp.label for sp in gq.sub_parts] == ["a", "b", "c"]
    assert [sp.marks for sp in gq.sub_parts] == [8, 6, 6]
    assert gq.question_text.startswith("(a)")
    assert "\n\n(b)" in gq.question_text
    assert "\n\n(c)" in gq.question_text


def test_09_solution_materialization_multi_part():
    """Test 9: Parent solution materializes from subparts with '(a)' and '(b)' labels."""
    plan = create_planned_question([5, 5])
    resp_a = {
        "text": "Explain threads.",
        "solution": "Thread is a lightweight unit of CPU execution.",
        "marking_scheme": [{"criterion": "Threads", "marks": 5}],
    }
    resp_b = {
        "text": "Describe kernel modules.",
        "solution": "Kernel modules allow dynamic driver loading.",
        "marking_scheme": [{"criterion": "Modules", "marks": 5}],
    }
    llm = MockLLMCaller(responses=[resp_a, resp_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.solution == "(a) Thread is a lightweight unit of CPU execution.\n\n(b) Kernel modules allow dynamic driver loading."


def test_10_marking_scheme_materialization_multi_part():
    """Test 10: Parent marking_scheme synthesizes criteria with subpart labels."""
    plan = create_planned_question([6, 4], blueprint_id="BP_10M")
    resp_a = {
        "text": "Explain TLB.",
        "solution": "TLB caches page table lookups.",
        "marking_scheme": [
            {"criterion": "TLB definition", "marks": 3},
            {"criterion": "Hit/miss handling", "marks": 3},
        ],
    }
    resp_b = {
        "text": "Explain thrashing.",
        "solution": "High paging activity.",
        "marking_scheme": [{"criterion": "Thrashing explanation", "marks": 4}],
    }
    llm = MockLLMCaller(responses=[resp_a, resp_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(gq.marking_scheme) == 3
    assert gq.marking_scheme[0]["criterion"] == "(a) TLB definition"
    assert gq.marking_scheme[0]["subpart"] == "a"
    assert gq.marking_scheme[1]["criterion"] == "(a) Hit/miss handling"
    assert gq.marking_scheme[1]["subpart"] == "a"
    assert gq.marking_scheme[2]["criterion"] == "(b) Thrashing explanation"
    assert gq.marking_scheme[2]["subpart"] == "b"


# =============================================================================
# Group 3: Immutable Plan Inputs & Pedagogical Bounds (Tests 11–15)
# =============================================================================

def test_11_subpart_topics_match_plan():
    """Test 11: SubPart.topic matches SubPartPlan.topic exactly."""
    plan = create_planned_question([5, 5])
    plan.sub_part_plans[0].topic = "Process Scheduling Mechanics"
    plan.sub_part_plans[1].topic = "Virtual Memory Paging Schemes"

    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].topic == "Process Scheduling Mechanics"
    assert gq.sub_parts[1].topic == "Virtual Memory Paging Schemes"


def test_12_subpart_marks_match_plan():
    """Test 12: SubPart.marks matches SubPartPlan.marks exactly."""
    plan = create_planned_question([6, 4], blueprint_id="BP_10M")
    resp_a = {
        "text": "Explain CPU scheduling.",
        "solution": "...",
        "marking_scheme": [{"criterion": "c1", "marks": 6}],
    }
    resp_b = {
        "text": "Explain memory allocation.",
        "solution": "...",
        "marking_scheme": [{"criterion": "c2", "marks": 4}],
    }
    llm = MockLLMCaller(responses=[resp_a, resp_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].marks == 6
    assert gq.sub_parts[1].marks == 4


def test_13_subpart_labels_match_plan_authoritatively():
    """Test 13: SubPart.label is strictly assigned from SubPartPlan.label."""
    plan = create_planned_question([5, 5])
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].label == "a"
    assert gq.sub_parts[1].label == "b"


def test_14_co_and_bloom_preserved_in_prompt_and_parent():
    """Test 14: Verify CO/Bloom constraints in prompt formatting and parent GeneratedQuestion."""
    plan = create_planned_question([5, 5], co="CO2", bloom="L3")
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    # Prompt sent to model must explicitly reflect L3 and CO2
    assert len(llm.calls) == 2
    for call in llm.calls:
        prompt_text = call["prompt"]
        assert "Bloom level: L3" in prompt_text
        assert "Course Outcome: CO2" in prompt_text

    # Parent GeneratedQuestion preserves pedagogical bounds
    assert gq.bloom == "L3"
    assert gq.co == "CO2"

    # SubPart dataclass itself does NOT have bloom or co fields
    assert not hasattr(gq.sub_parts[0], "bloom")
    assert not hasattr(gq.sub_parts[0], "co")


def test_15_unknown_bloom_level_rejected_with_value_error():
    """Test 15: Reject unknown Bloom levels with explicit ValueError instead of defaulting to L2."""
    plan = create_planned_question([5, 5])
    sp_plan = plan.sub_part_plans[0]
    sp_plan.bloom = "L7"  # Invalid bloom level

    with pytest.raises(ValueError) as excinfo:
        build_subpart_prompt(plan, sp_plan)
    assert "Unknown bloom level 'L7'" in str(excinfo.value)

    plan.bloom = "L7"
    with pytest.raises(ValueError) as excinfo_parent:
        build_task_section(plan)
    assert "Unknown bloom level 'L7'" in str(excinfo_parent.value)


# =============================================================================
# Group 4: Strict Criteria & Response Validation (Tests 16–20)
# =============================================================================

def test_16_criteria_count_1_to_4_accepted():
    """Test 16: Marking schemes with 1 to 4 criteria are accepted."""
    plan = create_planned_question([5, 5])
    # Subpart a: 1 criterion of 5 marks
    resp_1_crit = {
        "text": "Define system call.",
        "solution": "System call is interface between process and OS.",
        "marking_scheme": [{"criterion": "Definition", "marks": 5}],
    }
    # Subpart b: 4 criteria summing to 5 marks
    resp_4_crit = {
        "text": "Explain interrupt handling steps.",
        "solution": "Steps 1 to 4.",
        "marking_scheme": [
            {"criterion": "Save state", "marks": 2},
            {"criterion": "Execute ISR", "marks": 1},
            {"criterion": "Clear interrupt", "marks": 1},
            {"criterion": "Restore state", "marks": 1},
        ],
    }
    llm = MockLLMCaller(responses=[resp_1_crit, resp_4_crit])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(gq.sub_parts[0].marking_scheme) == 1
    assert len(gq.sub_parts[1].marking_scheme) == 4


def test_17_criteria_count_0_or_over_4_rejected():
    """Test 17: Marking schemes with 0 criteria or >4 criteria trigger retry."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    resp_0_crit = {
        "text": "Explain virtual memory.",
        "solution": "...",
        "marking_scheme": [],
    }
    resp_5_crit = {
        "text": "Explain virtual memory.",
        "solution": "...",
        "marking_scheme": [{"criterion": f"c{i}", "marks": 2} for i in range(5)],
    }
    resp_valid = {
        "text": "Explain virtual memory.",
        "solution": "...",
        "marking_scheme": [{"criterion": "c1", "marks": 5}, {"criterion": "c2", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_0_crit, resp_5_crit, resp_valid])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(gq.sub_parts[0].marking_scheme) == 2
    assert gq.telemetry["subpart_attempts"]["a"] == 3


def test_18_malformed_json_triggers_retry():
    """Test 18: Malformed non-JSON response triggers retry with strict_json directive."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    raw_garbage = "I cannot fulfill this in json. Here is text."
    resp_valid = {
        "text": "Explain paging hardware.",
        "solution": "Paging hardware uses page tables.",
        "marking_scheme": [{"criterion": "Hardware", "marks": 10}],
    }

    llm = MockLLMCaller(responses=[raw_garbage, resp_valid])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert len(llm.calls) == 2
    assert "Your previous response could not be parsed as JSON" in llm.calls[1]["prompt"]
    assert gq.sub_parts[0].text == "Explain paging hardware."


def test_19_missing_required_keys_triggers_retry():
    """Test 19: Response missing required subpart key triggers retry."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    resp_missing_sol = {
        "text": "Explain deadlocks.",
        "marking_scheme": [{"criterion": "c1", "marks": 10}],
    }
    resp_valid = {
        "text": "Explain deadlocks.",
        "solution": "Deadlock is mutual blocking.",
        "marking_scheme": [{"criterion": "c1", "marks": 10}],
    }

    llm = MockLLMCaller(responses=[resp_missing_sol, resp_valid])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].solution == "Deadlock is mutual blocking."
    assert gq.telemetry["subpart_attempts"]["a"] == 2


def test_20_model_supplied_wrong_label_ignored_not_retried():
    """Test 20: Model-supplied wrong label is ignored without triggering retry."""
    plan = create_planned_question([5, 5])
    # Model incorrectly returns "label": "z" for subpart a, and "label": "wrong" for subpart b
    resp_a = {
        "label": "z",
        "text": "Explain process scheduling.",
        "solution": "Selects process to run.",
        "marking_scheme": [{"criterion": "Scheduling", "marks": 5}],
    }
    resp_b = {
        "label": "wrong",
        "text": "Explain memory management.",
        "solution": "Manages physical frames.",
        "marking_scheme": [{"criterion": "Memory", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    # Label is set authoritatively from sp_plan; no retry triggered
    assert gq.sub_parts[0].label == "a"
    assert gq.sub_parts[1].label == "b"
    assert gq.telemetry["subpart_attempts"]["a"] == 1
    assert gq.telemetry["subpart_attempts"]["b"] == 1


# =============================================================================
# Group 5: Retry Isolation & Bounded Retries (Tests 21–25)
# =============================================================================

def test_21_retry_isolation_failing_subpart_retried_alone():
    """Test 21: Retry isolation: failing subpart retried alone, successful sibling not recalled."""
    plan = create_planned_question([5, 5])
    resp_a_success = {
        "text": "Explain semaphores.",
        "solution": "Synchronization primitive.",
        "marking_scheme": [{"criterion": "Semaphores", "marks": 5}],
    }
    resp_b_fail = {
        "text": "Explain monitors.",
        "solution": "Synchronization construct.",
        "marking_scheme": [{"criterion": "Monitors", "marks": 4}],  # bad sum
    }
    resp_b_success = {
        "text": "Explain monitors.",
        "solution": "Synchronization construct.",
        "marking_scheme": [{"criterion": "Monitors", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a_success, resp_b_fail, resp_b_success])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    # 3 total calls: 1 for 'a', 2 for 'b'
    assert len(llm.calls) == 3
    assert "Subpart: (a)" in llm.calls[0]["prompt"]
    assert "Subpart: (b)" in llm.calls[1]["prompt"]
    assert "Subpart: (b)" in llm.calls[2]["prompt"]


def test_22_successful_sibling_preserved_across_retries():
    """Test 22: Successful sibling subpart is preserved across retry of another subpart."""
    plan = create_planned_question([5, 5])
    resp_a = {
        "text": "Explain CPU scheduling.",
        "solution": "Sol a.",
        "marking_scheme": [{"criterion": "Criteria", "marks": 5}],
    }
    resp_b_fail = "Not JSON"
    resp_b_success = {
        "text": "Explain disk scheduling.",
        "solution": "Sol b.",
        "marking_scheme": [{"criterion": "Criteria", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b_fail, resp_b_success])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.sub_parts[0].text == "Explain CPU scheduling."
    assert gq.sub_parts[1].text == "Explain disk scheduling."


def test_23_exhausted_retries_fails_entire_slot():
    """Test 23: Exhausting retries on any subpart fails the entire slot without partial return."""
    plan = create_planned_question([5, 5])
    resp_a = {
        "text": "Explain CPU scheduling.",
        "solution": "Sol a.",
        "marking_scheme": [{"criterion": "Criteria", "marks": 5}],
    }
    # Subpart b fails 3 times
    llm = MockLLMCaller(responses=[resp_a, "Fail 1", "Fail 2", "Fail 3"])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    with pytest.raises(GenerationFailed) as excinfo:
        agent._generate_one(plan)

    assert excinfo.value.code == "SUBPART_GENERATION_FAILED"
    assert "Subpart (b) failed after 3 attempts" in excinfo.value.detail


def test_24_no_marks_redistribution_on_failure():
    """Test 24: Failed subparts never cause mark redistribution to remaining parts."""
    plan = create_planned_question([6, 4], blueprint_id="BP_10M")
    resp_a = {
        "text": "Explain scheduling.",
        "solution": "...",
        "marking_scheme": [{"criterion": "c1", "marks": 6}],
    }
    llm = MockLLMCaller(responses=[resp_a, "Bad", "Bad", "Bad"])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    try:
        agent._generate_one(plan)
        pytest.fail("Should have raised GenerationFailed")
    except GenerationFailed:
        # Verify plan itself was not mutated
        assert plan.partition == [6, 4]
        assert plan.sub_part_plans[0].marks == 6
        assert plan.sub_part_plans[1].marks == 4


def test_25_caller_timeout_preserved_as_timeout_code():
    """Test 25: Caller timeout exception is preserved as TIMEOUT failure code."""
    plan = create_planned_question([10], blueprint_id="BP_10M")
    llm = MockLLMCaller(responses=[
        TimeoutError("Connection timed out"),
        TimeoutError("Connection timed out"),
        TimeoutError("Connection timed out"),
    ])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    with pytest.raises(GenerationFailed) as excinfo:
        agent._generate_one(plan)

    assert excinfo.value.code == "TIMEOUT"
    assert "Subpart (a) failed" in excinfo.value.detail


# =============================================================================
# Group 6: Projection Determinism & Visual Mapping (Tests 26–30)
# =============================================================================

def test_26_sync_materialized_projections_idempotent():
    """Test 26: sync_materialized_projections() is strictly idempotent."""
    plan = create_planned_question([5, 5])
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    t1 = gq.question_text
    s1 = gq.solution
    m1 = copy.deepcopy(gq.marking_scheme)

    gq.sync_materialized_projections()
    gq.sync_materialized_projections()

    assert gq.question_text == t1
    assert gq.solution == s1
    assert gq.marking_scheme == m1


def test_27_verify_projection_consistency_detects_drift():
    """Test 27: verify_projection_consistency() detects drift if parent text is mutated."""
    plan = create_planned_question([5, 5])
    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    valid, _ = gq.verify_projection_consistency()
    assert valid

    # Introduce drift
    gq.question_text = "Drifted question text."
    valid_tampered, err = gq.verify_projection_consistency()
    assert not valid_tampered
    assert "question_text does not match" in err


def test_28_visual_mapping_to_parent_fields():
    """Test 28: Visual subpart maps references_image, image_path, caption, and diagram_request to parent."""
    plan = create_planned_question([5, 5])
    fig = MockFigure(image_path="/path/figure.png", caption="Process State Diagram")
    plan.sub_part_plans[0].visual_required = True
    plan.sub_part_plans[0].figure = fig

    resp_a = {
        "text": "With the aid of the given diagram, describe process transitions.",
        "solution": "Model solution.",
        "marking_scheme": [{"criterion": "Visual interpretation", "marks": 5}],
        "diagram_request": {"diagram_type": "state_machine", "description": "Transitions"},
        "references_image": True,
    }
    resp_b = {
        "text": "Explain context switching overhead.",
        "solution": "Context switching takes time.",
        "marking_scheme": [{"criterion": "Overhead", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_b])
    vlm = MockVLMCaller(responses=[resp_a])
    agent = WritingAgent(llm_caller=llm, vlm_caller=vlm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.references_image is True
    assert gq.image_path == "/path/figure.png"
    assert gq.figure_caption == "Process State Diagram"
    assert gq.diagram_request == {"diagram_type": "state_machine", "description": "Transitions"}


def test_29_vlm_source_recorded_when_visual_used():
    """Test 29: generation_source is 'local_vlm' when VLM is used for visual subpart."""
    plan = create_planned_question([5, 5])
    fig = MockFigure(image_path="/path/figure.png", caption="Process State Diagram")
    plan.sub_part_plans[1].visual_required = True
    plan.sub_part_plans[1].figure = fig

    resp_a = {
        "text": "Explain memory management.",
        "solution": "Memory.",
        "marking_scheme": [{"criterion": "Memory", "marks": 5}],
    }
    resp_b = {
        "text": "Refer to the given figure to calculate page offset.",
        "solution": "Offset.",
        "marking_scheme": [{"criterion": "Offset", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a])
    vlm = MockVLMCaller(responses=[resp_b])
    agent = WritingAgent(llm_caller=llm, vlm_caller=vlm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.generation_source == "local_vlm"


def test_30_attempt_accounting_preserves_int_and_telemetry():
    """Test 30: Preserves attempts as int (max across subparts) and records breakdown in telemetry."""
    plan = create_planned_question([5, 5])
    resp_a_success = {
        "text": "Explain semaphores.",
        "solution": "Synchronization primitive.",
        "marking_scheme": [{"criterion": "Semaphores", "marks": 5}],
    }
    resp_b_fail = {
        "text": "Explain monitors.",
        "solution": "Synchronization construct.",
        "marking_scheme": [{"criterion": "Monitors", "marks": 4}],
    }
    resp_b_success = {
        "text": "Explain monitors.",
        "solution": "Synchronization construct.",
        "marking_scheme": [{"criterion": "Monitors", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a_success, resp_b_fail, resp_b_success])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert isinstance(gq.attempts, int)
    assert gq.attempts == 2
    assert gq.telemetry["subpart_attempts"] == {"a": 1, "b": 2}


# =============================================================================
# Group 7: End-to-End Orchestrator & Plan Status Gating (Tests 31–35)
# =============================================================================

def test_31_unresolved_plan_bypasses_writer_with_zero_calls():
    """Test 31: QuestionPlan with UNRESOLVED status bypasses model with 0 LLM calls."""
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        status=PlanStatus.UNRESOLVED,
        failure_code="SIBLING_COLLISION",
        failure_detail="Concept overlap between subparts",
        telemetry={
            "attempts": [{
                "attempt_number": 1,
                "strategy": "base",
                "failure_code": "SIBLING_COLLISION",
                "failure_detail": "Concept overlap between subparts",
                "candidate_topics": ["c1", "c2"],
            }]
        },
    )

    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    with pytest.raises(GenerationFailed) as excinfo:
        agent._generate_one(plan)

    assert excinfo.value.code == "SIBLING_COLLISION"
    assert "Concept overlap" in excinfo.value.detail
    assert len(llm.calls) == 0


def test_32_planned_status_with_missing_subpart_plans_fails_closed():
    """Test 32: PLANNED status with missing sub_part_plans must fail closed, never fall back to legacy."""
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        status=PlanStatus.LEGACY_UNVALIDATED,
        assessment_identity=("SYNTH", "SYNTHETIC_TEST", "CIE_IAT", "BP_10M"),
    )
    # Mutate to PLANNED without sub_part_plans
    plan.status = PlanStatus.PLANNED
    plan.sub_part_plans = []

    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    with pytest.raises(GenerationFailed) as excinfo:
        agent._generate_one(plan)

    assert excinfo.value.code == PlanFailureCode.INVALID_CONFIG.value
    assert len(llm.calls) == 0


def test_33_planned_status_with_invalid_subpart_plans_fails_closed():
    """Test 33: PLANNED status with invalid sub_part_plans fails closed at Writer boundary."""
    plan = create_planned_question([5, 5])
    # Corrupt subpart marks so partition sum mismatch occurs
    plan.sub_part_plans[0].marks = 7  # 7 + 5 = 12 != 10

    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)

    with pytest.raises(GenerationFailed) as excinfo:
        agent._generate_one(plan)

    assert excinfo.value.code == PlanFailureCode.MARKS_MISMATCH.value
    assert len(llm.calls) == 0


def test_34_failure_propagation_to_orchestrator_and_checking_agent():
    """Test 34: WritingAgent SlotFailure propagates to unresolved_slots and blocks CheckingAgent."""
    plan = QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[5, 5],
        total_marks=10,
        status=PlanStatus.UNRESOLVED,
        failure_code="INVALID_CONFIG",
        failure_detail="Test plan invalid config",
        telemetry={"attempts": []},
    )

    llm = MockLLMCaller()
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    ctx = AgentContext(plans=[plan])

    result = agent.run(ctx)
    assert result.success is True
    output: WritingAgentOutput = result.payload
    assert len(output.failures) == 1
    assert output.failures[0].slot_id == "module_1_Q1"
    assert output.failures[0].failure_code == "INVALID_CONFIG"

    # Simulate orchestrator unresolved_slots propagation
    for f in output.failures:
        if f.slot_id not in ctx.unresolved_slots:
            ctx.unresolved_slots.append(f.slot_id)

    assert "module_1_Q1" in ctx.unresolved_slots

    # Verify checking agent contract: unresolved slots trigger block
    from core.generation.agents.checking_contracts import QAReport
    from core.generation.agents.checking_agent import CheckingAgent

    chk_agent = CheckingAgent()
    chk_result = chk_agent.run(ctx)
    assert chk_result.success is True
    report: QAReport = chk_result.payload

    assert report.status == STATUS_BLOCKED
    assert BLOCK_REASON_UNRESOLVED_SLOTS in report.blocked_reasons


def test_35_legacy_unvalidated_plan_uses_legacy_generator():
    """Test 35: LEGACY_UNVALIDATED plan uses legacy generator and preserves backward compatibility."""
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
        topic="Legacy Topic",
        status=PlanStatus.LEGACY_UNVALIDATED,
    )

    llm = MockLLMCaller(responses=[{
        "question_text": "Explain monolithic operating systems.",
        "solution": "All OS services run in kernel space.",
        "marking_scheme": [{"criterion": "Explanation", "marks": 10}],
    }])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert gq.question_text == "Explain monolithic operating systems."
    assert gq.solution == "All OS services run in kernel space."
    assert len(gq.sub_parts) == 0  # Legacy raw generation leaves sub_parts empty
