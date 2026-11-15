"""
Stage 2A Characterization Suite: Refinement Failure Lifecycle, Evaluator Evidence,
and Diagnostic Telemetry Preservation.

Verifies:
1. SlotVerdict constructor backward compatibility (positional and keyword).
2. SlotVerdict serialization via asdict and jsonable.
3. Evaluator structured affected_subparts propagation (Bloom depth, sibling independence, whole question).
4. Unaliased deep snapshot of original question before repair.
5. Append-only multi-finding and multi-attempt telemetry preservation.
6. Conservative aggregation rules for unresolved scope, subparts, and failure codes.
7. Authentic text preservation on failed repair (zero placeholder substitution).
8. Clean metadata state on successful repair.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

import pytest

from core.generation.agents.base import AgentContext
from core.generation.agents.evaluation_contracts import (
    BLOOM_DEPTH_MISMATCH,
    BLOOM_MISMATCH,
    QUESTION_TOO_SHORT,
    SIBLING_INTERCONNECTED,
    UNGROUNDED,
    AuditReport,
    SlotVerdict,
)
from core.generation.agents.generated_question import GeneratedQuestion
from core.generation.agents.local_evaluators import (
    _fail,
    bloom_content_evaluator,
    bloom_verb_evaluator,
    question_length_evaluator,
    sibling_independence_evaluator,
)
from core.generation.agents.refinement_agent import (
    RefinementAgent,
    _aggregate_unresolved_metadata,
)
from core.serialization import jsonable


# =============================================================================
# 1. SlotVerdict Constructor Compatibility & Serialization
# =============================================================================

def test_slot_verdict_legacy_positional_construction():
    """Verify legacy positional construction with 3 to 6 arguments succeeds."""
    # 3 args: slot_id, verdict, source
    v3 = SlotVerdict("slot_1", "pass", "local")
    assert v3.slot_id == "slot_1"
    assert v3.verdict == "pass"
    assert v3.source == "local"
    assert v3.reason_code is None
    assert v3.detail == ""
    assert v3.suggested_fix is None
    assert v3.affected_subparts == ()

    # 6 args: slot_id, verdict, source, reason_code, detail, suggested_fix
    v6 = SlotVerdict("slot_2", "fail", "local", BLOOM_MISMATCH, "Bad verb", "Fix verb")
    assert v6.reason_code == BLOOM_MISMATCH
    assert v6.detail == "Bad verb"
    assert v6.suggested_fix == "Fix verb"
    assert v6.affected_subparts == ()


def test_slot_verdict_full_positional_and_keyword_construction():
    """Verify 7-arg positional and keyword construction with affected_subparts."""
    # 7 positional args
    v7 = SlotVerdict(
        "slot_3", "fail", "local", BLOOM_DEPTH_MISMATCH, "L2 verb in L4", "Elevate verb", ("b",)
    )
    assert v7.affected_subparts == ("b",)

    # Keyword construction
    vk = SlotVerdict(
        slot_id="slot_4",
        verdict="fail",
        source="local",
        reason_code=SIBLING_INTERCONNECTED,
        detail="Chaining detected",
        affected_subparts=("a", "b"),
    )
    assert vk.affected_subparts == ("a", "b")


def test_slot_verdict_asdict_and_jsonable_serialization():
    """Verify SlotVerdict serializes cleanly to dict and JSON primitives."""
    v = SlotVerdict(
        slot_id="slot_v",
        verdict="fail",
        source="local",
        reason_code=BLOOM_DEPTH_MISMATCH,
        detail="Depth mismatch",
        affected_subparts=("b",),
    )
    d = asdict(v)
    assert d["affected_subparts"] == ("b",)
    assert len(d) == 7

    j = jsonable(v, _strict=True)
    assert isinstance(j["affected_subparts"], list)
    assert j["affected_subparts"] == ["b"]


# =============================================================================
# 2. Evaluator Structured affected_subparts Propagation
# =============================================================================

def test_fail_helper_normalizes_and_propagates_subparts():
    """Verify _fail helper formats subpart strings to clean lowercase tuples."""
    q = GeneratedQuestion("s1", "m1", 1, "text", "sol", [])
    v1 = _fail(q, BLOOM_DEPTH_MISMATCH, "detail", affected_subparts=(" (B) ", "a."))
    assert v1.affected_subparts == ("b", "a")

    v_empty = _fail(q, QUESTION_TOO_SHORT, "detail")
    assert v_empty.affected_subparts == ()


def test_bloom_content_evaluator_identifies_affected_subpart_b():
    """Verify bloom_content_evaluator emits affected_subparts=('b',) on subpart failure."""
    q = GeneratedQuestion(
        slot_id="slot_bloom_l4",
        module_id="module_3",
        global_q_idx=6,
        question_text="(a) Analyze the cache coherence protocols. (b) Describe the bus snooping mechanism.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L4",
    )
    verdict = bloom_content_evaluator(q)
    assert verdict is not None
    assert verdict.reason_code == BLOOM_DEPTH_MISMATCH
    assert verdict.affected_subparts == ("b",)


def test_sibling_independence_evaluator_identifies_pairwise_subparts():
    """Verify sibling_independence_evaluator emits affected_subparts=('a', 'b') on chaining."""
    q_chain = GeneratedQuestion(
        slot_id="slot_chain",
        module_id="module_4",
        global_q_idx=8,
        question_text="(a) Explain the role of the system call table. (b) With reference to part (a), explain interrupt dispatching.",
        solution="Model solution.",
        marking_scheme=[],
        marks=10,
        partition=[6, 4],
        bloom="L2",
    )
    verdict_chain = sibling_independence_evaluator(q_chain)
    assert verdict_chain is not None
    assert verdict_chain.reason_code == SIBLING_INTERCONNECTED
    assert verdict_chain.affected_subparts == ("a", "b")

    q_pronoun = GeneratedQuestion(
        slot_id="slot_pronoun",
        module_id="module_4",
        global_q_idx=8,
        question_text="(a) Explain page table structures. (b) It is used to map virtual addresses to physical pages.",
        solution="Model solution.",
        marking_scheme=[],
        marks=10,
        partition=[6, 4],
        bloom="L2",
    )
    verdict_pronoun = sibling_independence_evaluator(q_pronoun)
    assert verdict_pronoun is not None
    assert verdict_pronoun.reason_code == SIBLING_INTERCONNECTED
    assert verdict_pronoun.affected_subparts == ("a", "b")


def test_whole_question_evaluators_emit_empty_affected_subparts():
    """Verify whole-question evaluators emit empty affected_subparts."""
    # bloom_verb_evaluator checks the opening word of the whole slot
    q_bloom_verb = GeneratedQuestion(
        slot_id="slot_verb",
        module_id="module_1",
        global_q_idx=1,
        question_text="Summarize the OS components.",  # 'Summarize' is not approved for L4
        solution="Sol",
        marking_scheme=[],
        marks=10,
        bloom="L4",
    )
    v_verb = bloom_verb_evaluator(q_bloom_verb)
    assert v_verb is not None
    assert v_verb.reason_code == BLOOM_MISMATCH
    assert v_verb.affected_subparts == ()

    # question_length_evaluator
    q_short = GeneratedQuestion("s_short", "m1", 1, "Short?", "Sol", [], marks=10)
    v_short = question_length_evaluator(q_short)
    assert v_short is not None
    assert v_short.reason_code == QUESTION_TOO_SHORT
    assert v_short.affected_subparts == ()


# =============================================================================
# 3. Conservative Metadata Aggregation
# =============================================================================

def test_aggregate_metadata_subquestion_scope():
    """Verify aggregation yields subquestion scope when all findings specify subparts."""
    v1 = SlotVerdict("s1", "fail", "local", BLOOM_DEPTH_MISMATCH, "Verb L2 in (b)", affected_subparts=("b",))
    scope, subparts, code, detail = _aggregate_unresolved_metadata([v1])
    assert scope == "subquestion"
    assert subparts == ("b",)
    assert code == BLOOM_DEPTH_MISMATCH
    assert detail == "Verb L2 in (b)"


def test_aggregate_metadata_pairwise_dependency():
    """Verify pairwise findings result in ('a', 'b') subparts."""
    v1 = SlotVerdict("s1", "fail", "local", SIBLING_INTERCONNECTED, "Chaining between (a) and (b)", affected_subparts=("a", "b"))
    scope, subparts, code, detail = _aggregate_unresolved_metadata([v1])
    assert scope == "subquestion"
    assert subparts == ("a", "b")
    assert code == SIBLING_INTERCONNECTED


def test_aggregate_metadata_conservative_whole_question_precedence():
    """Verify that any question-level or ambiguous finding forces final scope to 'question' and subparts to ()."""
    v_sub = SlotVerdict("s1", "fail", "local", BLOOM_DEPTH_MISMATCH, "Part b verb mismatch", affected_subparts=("b",))
    v_whole = SlotVerdict("s1", "fail", "local", UNGROUNDED, "Whole slot ungrounded", affected_subparts=())

    scope, subparts, code, detail = _aggregate_unresolved_metadata([v_sub, v_whole])
    assert scope == "question"
    assert subparts == ()
    assert code == f"{BLOOM_DEPTH_MISMATCH};{UNGROUNDED}"
    assert "Part b verb mismatch; Whole slot ungrounded" == detail


def test_aggregate_metadata_empty_verdicts():
    """Verify safe fallback on empty verdict list."""
    scope, subparts, code, detail = _aggregate_unresolved_metadata([])
    assert scope == "question"
    assert subparts == ()
    assert code == "UNKNOWN"


# =============================================================================
# 4. RefinementAgent Lifecycle & Telemetry Snapshot Guarantees
# =============================================================================

def test_diagnostic_snapshot_captured_once_pre_repair():
    """
    Verify RefinementAgent captures an unaliased deep snapshot of the original question
    before attempting repair, and retains it in telemetry["diagnostic_unresolved"].
    """
    original = GeneratedQuestion(
        slot_id="slot_snap",
        module_id="module_2",
        global_q_idx=3,
        question_text="(a) Explain CPU scheduling. (b) Explain CPU scheduling.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO2",
    )

    verdict = SlotVerdict(
        slot_id="slot_snap",
        verdict="fail",
        source="local",
        reason_code=SIBLING_INTERCONNECTED,
        detail="Chaining between (a) and (b)",
        affected_subparts=("a", "b"),
    )

    context = AgentContext(
        questions=[original],
        audit_report=AuditReport(verdicts=[verdict]),
    )

    # RefinementAgent with no LLM caller (deterministic repair fails for duplicate ask without template fabrication)
    agent = RefinementAgent(llm_caller=None)
    result = agent.run(context)

    assert result.success is True
    diag_registry = context.telemetry.get("diagnostic_unresolved", {})
    assert "slot_snap" in diag_registry

    snapshot = diag_registry["slot_snap"]
    orig_data = snapshot["original_question"]

    # Verify original question content is preserved
    assert orig_data["question_text"] == original.question_text
    assert orig_data["solution"] == original.solution
    assert orig_data["marking_scheme"] == original.marking_scheme

    # Verify deep copy unaliasing: mutating original does not alter snapshot
    original.marking_scheme[0]["marks"] = 999
    assert orig_data["marking_scheme"][0]["marks"] == 6

    # Verify findings list recorded
    assert len(snapshot["findings"]) == 1
    assert snapshot["findings"][0]["reason_code"] == SIBLING_INTERCONNECTED
    assert snapshot["findings"][0]["affected_subparts"] == ["a", "b"]

    # Verify repair attempts recorded
    assert len(snapshot["repair_attempts"]) == 1
    assert snapshot["repair_attempts"][0]["resolved"] is False
    assert snapshot["repair_attempts"][0]["reason_out"] == "NO_LLM_CALLER"


def test_failed_repair_populates_active_question_metadata_without_text_mutation():
    """
    Verify failed repair sets typed metadata on GeneratedQuestion in context.questions,
    while leaving question_text, solution, and marking_scheme 100% authentic (no placeholders).
    """
    raw_text = "(a) Explain paging. (b) Outline paging."
    original = GeneratedQuestion(
        slot_id="slot_failed",
        module_id="module_3",
        global_q_idx=5,
        question_text=raw_text,
        solution="Model solution text",
        marking_scheme=[{"criterion": "Part a", "marks": 5}, {"criterion": "Part b", "marks": 5}],
        marks=10,
        partition=[5, 5],
        bloom="L4",
    )

    verdict = SlotVerdict(
        slot_id="slot_failed",
        verdict="fail",
        source="local",
        reason_code=BLOOM_DEPTH_MISMATCH,
        detail="Subpart (b) uses lower-order verb 'outline' for L4 slot",
        affected_subparts=("b",),
    )

    context = AgentContext(
        questions=[original],
        audit_report=AuditReport(verdicts=[verdict]),
    )

    agent = RefinementAgent(llm_caller=None)
    agent.run(context)

    # Question in context.questions is the active question
    active_q = context.questions[0]
    assert active_q.slot_id == "slot_failed"
    assert active_q.is_unresolved is True
    assert active_q.unresolved_scope == "subquestion"
    assert active_q.unresolved_subparts == ("b",)
    assert active_q.failure_code == BLOOM_DEPTH_MISMATCH
    assert "outline" in active_q.failure_detail

    # Authentic text strictly preserved; zero placeholders
    assert active_q.question_text == raw_text
    assert "[UNRESOLVED" not in active_q.question_text
    assert active_q.solution == "Model solution text"

    # Slot recorded in context.unresolved_slots
    assert "slot_failed" in context.unresolved_slots


def test_multi_finding_accumulation_in_telemetry_and_aggregation():
    """
    Verify multiple findings on a single slot accumulate in telemetry['diagnostic_unresolved'],
    and active question reflects conservative aggregation.
    """
    original = GeneratedQuestion(
        slot_id="slot_multi",
        module_id="module_2",
        global_q_idx=4,
        question_text="(a) Explain CPU scheduling criteria. (b) Derive the throughput equations for this system.",
        solution="Model solution.",
        marking_scheme=[],
        marks=10,
        partition=[6, 4],
        bloom="L4",
    )

    # Finding 1: subpart b verb mismatch
    v1 = SlotVerdict(
        slot_id="slot_multi",
        verdict="fail",
        source="local",
        reason_code=BLOOM_DEPTH_MISMATCH,
        detail="Subpart (b) uses L1 verb 'define' for L4",
        affected_subparts=("b",),
    )
    # Finding 2: sibling dependency between (a) and (b)
    v2 = SlotVerdict(
        slot_id="slot_multi",
        verdict="fail",
        source="local",
        reason_code=SIBLING_INTERCONNECTED,
        detail="Subpart (b) references part (a)",
        affected_subparts=("a", "b"),
    )

    context = AgentContext(
        questions=[original],
        audit_report=AuditReport(verdicts=[v1, v2]),
    )

    agent = RefinementAgent(llm_caller=None)
    agent.run(context)

    diag = context.telemetry["diagnostic_unresolved"]["slot_multi"]
    assert len(diag["findings"]) == 2
    assert diag["findings"][0]["reason_code"] == BLOOM_DEPTH_MISMATCH
    assert diag["findings"][1]["reason_code"] == SIBLING_INTERCONNECTED

    # Both repair attempts recorded
    assert len(diag["repair_attempts"]) == 2

    # Active question aggregation: union of subparts ('b') | ('a', 'b') -> ('a', 'b')
    active_q = context.questions[0]
    assert active_q.is_unresolved is True
    assert active_q.unresolved_scope == "subquestion"
    assert active_q.unresolved_subparts == ("a", "b")
    assert active_q.failure_code == f"{BLOOM_DEPTH_MISMATCH};{SIBLING_INTERCONNECTED}"


def test_successful_repair_clears_active_unresolved_metadata():
    """Verify that when a deterministic repair succeeds, active unresolved metadata is clean."""
    # Forbidden preamble repair: deterministic repair succeeds
    original = GeneratedQuestion(
        slot_id="slot_preamble",
        module_id="module_1",
        global_q_idx=1,
        question_text="Based on the provided evidence, explain virtual memory management.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "All", "marks": 10}],
        marks=10,
        partition=[10],
        bloom="L2",
        # Stale metadata simulating earlier pass
        is_unresolved=True,
        unresolved_scope="question",
        unresolved_subparts=("a",),
        failure_code="OLD_FAILURE",
        failure_detail="Old detail",
    )

    verdict = SlotVerdict(
        slot_id="slot_preamble",
        verdict="fail",
        source="local",
        reason_code="FORBIDDEN_PREAMBLE",
        detail="Starts with forbidden preamble",
        affected_subparts=(),
    )

    context = AgentContext(
        questions=[original],
        audit_report=AuditReport(verdicts=[verdict]),
    )

    agent = RefinementAgent(llm_caller=None)
    result = agent.run(context)

    assert result.success is True
    assert "slot_preamble" not in context.unresolved_slots

    repaired_q = context.questions[0]
    assert repaired_q.question_text == "Explain virtual memory management."
    assert repaired_q.is_unresolved is False
    assert repaired_q.unresolved_scope is None
    assert repaired_q.unresolved_subparts == ()
    assert repaired_q.failure_code is None
    assert repaired_q.failure_detail is None


if __name__ == "__main__":
    pytest.main(["-v", __file__])
