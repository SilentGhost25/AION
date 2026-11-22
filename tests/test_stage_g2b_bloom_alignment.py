"""
Stage G2-B Test Suite: Canonical VTU Bloom Taxonomy Alignment & Cognitive-Demand Compliance
===========================================================================================

Verifies:
  1. Canonical Classification (56 verbs across L1-L6, case, punctuation, variants, substring protection).
  2. 6 x 6 Exact Slot Compliance Matrix (zero adjacency tolerance).
  3. Semantic Cognitive Demand Evaluation (PASS, FAIL, INDETERMINATE fail-closed across all 6 levels).
  4. Cross-Layer Consistency (prompts, local_evaluators, and canonical map synchronized).
  5. WritingAgent Generation-Time Integration, Retry Feedback, and Sibling Preservation.
"""

import pytest
from core.validation.bloom_validator import (
    BLOOM_VERB_LEVEL_MAP,
    classify_verb,
    get_all_canonical_verbs,
    get_canonical_verbs,
    is_exact_slot_compliant,
)
from core.generation.agents.local_evaluators import (
    _first_meaningful_word,
    ALL_BLOOM_VERBS,
    BLOOM_ADJACENCY,
    evaluate_cognitive_demand,
    SemanticDemandOutcome,
    bloom_verb_evaluator,
    bloom_content_evaluator,
)
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.prompts import BLOOM_VERBS
from core.generation.agents.writing_agent import WritingAgent, GenerationFailed
from core.generation.agents.generated_question import GeneratedQuestion, SubPart
from core.generation.agents.planning_agent import PlanStatus, QuestionPlan, SubPartPlan
from tests.test_writing_agent_subparts import MockLLMCaller, create_planned_question


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


# =============================================================================
# 1. Canonical Classification Tests
# =============================================================================

def test_g2b_all_56_canonical_verbs_correctly_classified():
    """All 56 canonical verbs across L1-L6 map to their designated level."""
    expected_counts = {
        "L1": 9,
        "L2": 9,
        "L3": 11,
        "L4": 10,
        "L5": 8,
        "L6": 9,
    }
    total_verbs = sum(len(verbs) for verbs in BLOOM_VERB_LEVEL_MAP.values())
    assert total_verbs == 56

    for level, verbs in BLOOM_VERB_LEVEL_MAP.items():
        assert len(verbs) == expected_counts[level]
        for verb in verbs:
            detected = classify_verb(verb)
            assert detected == level, f"Verb '{verb}' classified as {detected}, expected {level}"


def test_g2b_case_and_punctuation_handling():
    """Classification is case-insensitive and ignores trailing punctuation."""
    variations = [
        ("ANALYZE", "L4"),
        ("Explain,", "L2"),
        ("Calculate:", "L3"),
        ("dEfiNe.", "L1"),
        ("EVALUATE;", "L5"),
        ("design!", "L6"),
    ]
    for text, expected in variations:
        assert classify_verb(text) == expected


def test_g2b_british_american_spelling_variants():
    """Spelling variants (analyse/analyze, summarise/summarize, etc.) map to canonical level."""
    pairs = [
        ("analyse", "analyze", "L4"),
        ("summarise", "summarize", "L2"),
        ("recognise", "recognize", "L1"),
        ("categorise", "categorize", "L4"),
    ]
    for uk, us, level in pairs:
        assert classify_verb(uk) == level
        assert classify_verb(us) == level


def test_g2b_unknown_verbs_return_none():
    """Non-canonical / arbitrary verbs are not assigned an RBT level."""
    unknowns = ["ponder", "giggle", "run", "jump", "contemplate", "guess", "imagine"]
    for verb in unknowns:
        assert classify_verb(verb) is None


def test_g2b_substring_collision_protection():
    """Words containing canonical verbs as substrings are not mistakenly classified as that verb."""
    non_verbs = [
        ("listen", None),       # contains 'list'
        ("statement", None),    # contains 'state'
        ("format", None),       # contains 'form'
        ("recalled_item", None),
        ("undefined", None),
    ]
    for word, expected in non_verbs:
        assert classify_verb(word) == expected


# =============================================================================
# 2. 6 x 6 Exact Slot Compliance Matrix Tests
# =============================================================================

@pytest.mark.parametrize("slot_level", ["L1", "L2", "L3", "L4", "L5", "L6"])
def test_g2b_exact_slot_compliance_6x6_matrix(slot_level):
    """
    Every canonical verb belonging to slot_level passes.
    Verbs from any of the other 5 levels fail with exact diagnostic reason.
    """
    # 1. Matching verbs must pass
    for verb in BLOOM_VERB_LEVEL_MAP[slot_level]:
        compliant, detected, reason = is_exact_slot_compliant(verb, slot_level)
        assert compliant is True, f"Verb '{verb}' failed compliance for its own level {slot_level}"
        assert detected == slot_level
        assert reason is None

    # 2. Non-matching verbs from all other 5 levels must fail
    other_levels = [lvl for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"] if lvl != slot_level]
    for other_lvl in other_levels:
        for verb in BLOOM_VERB_LEVEL_MAP[other_lvl]:
            compliant, detected, reason = is_exact_slot_compliant(verb, slot_level)
            assert compliant is False, f"Verb '{verb}' from {other_lvl} incorrectly passed for slot {slot_level}"
            assert detected == other_lvl
            assert reason is not None
            assert f"classified as {other_lvl}" in reason
            assert f"not in approved verbs for {slot_level}" in reason

    # 3. Unknown verb must fail
    compliant, detected, reason = is_exact_slot_compliant("unsupportedverb", slot_level)
    assert compliant is False
    assert detected is None
    assert "not a recognized canonical Bloom action verb" in reason


# =============================================================================
# 3. Semantic Demand Validation (PASS / FAIL / INDETERMINATE)
# =============================================================================

@pytest.mark.parametrize("level,text", [
    ("L1", "Define deadlock and state the four necessary conditions."),
    ("L1", "List the major registers in the CPU architecture."),
    ("L1", "Identify the components of a process control block."),
    ("L2", "Explain how registers support CPU execution cycles."),
    ("L2", "Describe the process state transition model using a diagram."),
    ("L2", "Discuss the advantages and disadvantages of monolithic operating system kernels."),
    ("L3", "Apply the given formula to calculate the effective memory access time."),
    ("L3", "Calculate the average waiting time for the given processes using FCFS scheduling."),
    ("L3", "Determine the number of page faults using the LRU algorithm for reference string 7, 0, 1, 2, 0, 3."),
    ("L4", "Analyze the relationship between page size and internal fragmentation."),
    ("L4", "Compare and contrast direct-mapped and set-associative cache placement policies."),
    ("L4", "Differentiate between preemptive and non-preemptive scheduling with respect to turnaround time."),
    ("L5", "Evaluate the proposed normalization strategy against the stated performance requirements."),
    ("L5", "Assess the trade-offs between user-level and kernel-level threads for real-time applications."),
    ("L5", "Justify the selection of the Banker algorithm for deadlock avoidance in safety-critical systems."),
    ("L6", "Design a custom memory allocator satisfying the given spatial constraints."),
    ("L6", "Develop a synchronization protocol using semaphores to eliminate race conditions."),
    ("L6", "Formulate an algorithm to detect cycles in a resource allocation graph within O(V+E) time."),
])
def test_g2b_semantic_demand_positive_pass_cases(level, text):
    """Positive valid questions across all 6 levels return SemanticDemandOutcome.PASS."""
    outcome, reason = evaluate_cognitive_demand(text, level)
    assert outcome == SemanticDemandOutcome.PASS, f"Expected PASS for '{text}', got {outcome} with reason: {reason}"
    assert reason is None


@pytest.mark.parametrize("level,text,expected_reason_substr", [
    ("L1", "Define the trade-offs and advantages and disadvantages of paging.", "analytical comparison"),
    ("L2", "Explain the definition of a register in hardware.", "Definitional veneer"),
    ("L3", "Apply the definition of a formula to memory.", "Definitional veneer"),
    ("L4", "Analyze what paging means in operating systems.", "Definitional veneer"),
    ("L4", "Analyze the definition of a deadlock in operating systems.", "Definitional veneer"),
    ("L4", "Explain the trade-offs between paging and segmentation.", "lower-order verb"),
    ("L5", "Evaluate the definition of database normalization.", "Definitional veneer"),
    ("L5", "Evaluate the role of demand paging in modern operating systems.", "Descriptive veneer"),
    ("L5", "Calculate the average turnaround time for the given workload.", "procedural verb"),
    ("L5", "Evaluate the page replacement by calculating page faults.", "Procedural veneer"),
    ("L6", "Design the definition of a data structure.", "Definitional veneer"),
    ("L6", "Explain how to build a synchronization protocol.", "lower-order verb"),
])
def test_g2b_semantic_demand_negative_fail_cases(level, text, expected_reason_substr):
    """Veneer phrases and cognitive mismatches across levels return SemanticDemandOutcome.FAIL."""
    outcome, reason = evaluate_cognitive_demand(text, level)
    assert outcome == SemanticDemandOutcome.FAIL, f"Expected FAIL for '{text}', got {outcome}"
    assert reason is not None
    assert expected_reason_substr.lower() in reason.lower()


@pytest.mark.parametrize("level,text", [
    ("L3", "Calculate throughput."),
    ("L3", "Determine the latency."),
    ("L4", "Analyze paging."),
    ("L4", "Examine cache memory."),
    ("L5", "Evaluate the operating system."),
    ("L5", "Assess virtual memory."),
    ("L6", "Design a cache."),
    ("L6", "Develop a data structure."),
])
def test_g2b_semantic_demand_indeterminate_fail_closed_cases(level, text):
    """Borderline/naked statements lacking explicit criteria return INDETERMINATE."""
    outcome, reason = evaluate_cognitive_demand(text, level)
    assert outcome == SemanticDemandOutcome.INDETERMINATE, f"Expected INDETERMINATE for '{text}', got {outcome}"
    assert reason is not None
    assert "lacks explicit" in reason or "lacks concrete" in reason


# =============================================================================
# 4. Cross-Layer Consistency Tests
# =============================================================================

def test_g2b_prompts_verbs_derived_from_canonical_map():
    """prompts.BLOOM_VERBS exactly mirrors core.validation.bloom_validator.BLOOM_VERB_LEVEL_MAP."""
    assert set(BLOOM_VERBS.keys()) == set(BLOOM_VERB_LEVEL_MAP.keys())
    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        assert set(BLOOM_VERBS[lvl]) == BLOOM_VERB_LEVEL_MAP[lvl]


def test_g2b_local_evaluators_all_bloom_verbs_matches_canonical():
    """local_evaluators.ALL_BLOOM_VERBS matches canonical get_all_canonical_verbs()."""
    assert ALL_BLOOM_VERBS == get_all_canonical_verbs()
    assert len(ALL_BLOOM_VERBS) == 56


def test_g2b_bloom_adjacency_zero_leakage():
    """BLOOM_ADJACENCY dictionary enforces exact level mapping with zero cross-level leakage."""
    for lvl in ["L1", "L2", "L3", "L4", "L5", "L6"]:
        assert BLOOM_ADJACENCY[lvl] == {lvl}


def test_g2b_bloom_verb_evaluator_enforces_exact_slot():
    """bloom_verb_evaluator in local_evaluators rejects adjacent-level verbs and accepts exact-level verbs."""
    class DummyQuestion:
        def __init__(self, text: str, bloom: str):
            self.question_text = text
            self.bloom = bloom
            self.slot_id = "test_slot"

    # L3 slot with L4 verb 'analyze' must fail (no adjacency tolerance)
    q_adj = DummyQuestion("Analyze cache performance.", "L3")
    verdict_adj = bloom_verb_evaluator(q_adj)
    assert verdict_adj is not None
    assert verdict_adj.verdict == "fail"
    assert "classified as L4" in verdict_adj.detail

    # L3 slot with exact L3 verb 'calculate' must pass
    q_exact = DummyQuestion("Calculate the effective access time.", "L3")
    verdict_exact = bloom_verb_evaluator(q_exact)
    assert verdict_exact is None


def test_g2b_bloom_content_evaluator_fails_indeterminate():
    """bloom_content_evaluator treats INDETERMINATE fail-closed as BLOOM_DEPTH_MISMATCH."""
    class DummyQuestion:
        def __init__(self, text: str, bloom: str):
            self.question_text = text
            self.bloom = bloom
            self.slot_id = "test_slot"

    # 'Analyze paging.' is INDETERMINATE -> must fail closed
    q = DummyQuestion("Analyze paging.", "L4")
    verdict = bloom_content_evaluator(q)
    assert verdict is not None
    assert verdict.verdict == "fail"
    assert "lacks explicit" in verdict.detail


# =============================================================================
# 5. WritingAgent Integration & Retry Feedback Tests
# =============================================================================

def test_g2b_writing_agent_exact_level_retry_and_recovery():
    """WritingAgent rejects adjacent or wrong-level verb and recovers with corrective feedback."""
    plan = create_planned_question([10], blueprint_id="BP_10M", co="CO2", bloom="L3")

    # Attempt 1: L2 verb 'explain' on L3 plan -> rejected by exact slot compliance
    resp_l2 = {
        "text": "Explain process scheduling algorithms.",
        "solution": "FCFS and SJF scheduling.",
        "marking_scheme": [{"criterion": "Explanation", "marks": 10}],
    }
    # Attempt 2: L3 verb 'calculate' with concrete numbers -> accepted
    resp_l3 = {
        "text": "Calculate turnaround times for the given workload with arrival times [0, 2, 4].",
        "solution": "Turnaround time = completion time - arrival time.",
        "marking_scheme": [{"criterion": "Calculation", "marks": 10}],
    }

    llm = MockLLMCaller(responses=[resp_l2, resp_l3])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert attempts == 2
    assert gq.telemetry["subpart_attempts"]["a"] == 2
    # Verify corrective guidance was injected
    retry_prompt = llm.calls[1]["prompt"]
    assert "[CORRECTIVE GUIDANCE FOR RETRY]" in retry_prompt
    assert "classified as L2" in retry_prompt
    assert "not in approved verbs for L3" in retry_prompt


def test_g2b_writing_agent_indeterminate_demand_triggers_fail_closed_retry():
    """WritingAgent catches INDETERMINATE cognitive demand, injects guidance, and succeeds on concrete framing."""
    plan = create_planned_question([10], blueprint_id="BP_10M", co="CO3", bloom="L4")

    # Attempt 1: L4 verb 'analyze' but naked without analytical context (INDETERMINATE)
    resp_indet = {
        "text": "Analyze paging.",
        "solution": "Paging is a memory management scheme.",
        "marking_scheme": [{"criterion": "Paging", "marks": 10}],
    }
    # Attempt 2: L4 verb with comparative context (PASS)
    resp_pass = {
        "text": "Analyze the trade-offs between paging and segmentation regarding fragmentation.",
        "solution": "Paging suffers from internal fragmentation; segmentation suffers from external fragmentation.",
        "marking_scheme": [{"criterion": "Trade-off analysis", "marks": 10}],
    }

    llm = MockLLMCaller(responses=[resp_indet, resp_pass])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    assert attempts == 2
    assert gq.telemetry["subpart_attempts"]["a"] == 2
    retry_prompt = llm.calls[1]["prompt"]
    assert "Cognitive demand is indeterminate for L4" in retry_prompt


def test_g2b_writing_agent_sibling_preservation_on_subpart_b_retry():
    """When subpart (b) fails Bloom compliance and retries, accepted subpart (a) is preserved intact."""
    plan = create_planned_question([5, 5], co="CO3", bloom="L4")

    resp_a = {
        "text": "Analyze race conditions in preemptive kernel scheduling.",
        "solution": "Race conditions occur when multiple threads access shared resources concurrently.",
        "marking_scheme": [{"criterion": "Race conditions", "marks": 5}],
    }
    # Subpart b Attempt 1: Wrong verb (L2 'explain' in L4 plan)
    resp_b_bad = {
        "text": "Explain deadlock detection algorithms.",
        "solution": "Cycle detection identifies deadlock.",
        "marking_scheme": [{"criterion": "Deadlock", "marks": 5}],
    }
    # Subpart b Attempt 2: Valid L4 verb and comparative context
    resp_b_good = {
        "text": "Compare spinlocks and sleep-locks in multi-core operating systems.",
        "solution": "Spinlocks busy-wait; sleep-locks put thread to sleep.",
        "marking_scheme": [{"criterion": "Comparison", "marks": 5}],
    }

    llm = MockLLMCaller(responses=[resp_a, resp_b_bad, resp_b_good])
    agent = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = agent._generate_one(plan)

    # Exactly 3 calls: 1 for a, 2 for b
    assert len(llm.calls) == 3
    assert gq.sub_parts[0].text.startswith("Analyze race conditions")
    assert gq.sub_parts[1].text.startswith("Compare spinlocks")
    assert gq.telemetry["subpart_attempts"]["a"] == 1
    assert gq.telemetry["subpart_attempts"]["b"] == 2
