"""
Unit tests for NumericalVerifier and SymPy verification integration.

Tests:
    - Normalization of math expressions and unit stripping
    - Equality chain verification
    - Formula and parameter substitution checks
    - False rejection prevention on qualitative / unparseable content
    - Full RefinementAgent integration rejecting flawed numerical repairs
"""

from typing import Any, Dict, List
import pytest

from core.generation.agents import (
    AgentContext,
    AuditReport,
    FACTUAL_ERROR,
    GeneratedQuestion,
    RefinementAgent,
    RefinementOutcome,
    SlotVerdict,
)
from core.generation.agents.dependency_factory import default_sympy_verifier_factory
from core.numerical_engine import NumericalVerifier


# -----------------------------------------------------------------------------
# Fixtures & Helpers
# -----------------------------------------------------------------------------


@pytest.fixture
def verifier():
    return NumericalVerifier(tolerance_pct=5.0)


class MockLLMCaller:
    """Mock LLM that returns a specific repaired payload."""
    def __init__(self, response_dict: dict):
        self.response_dict = response_dict
        self.calls: List[str] = []

    def call(self, prompt: str, schema=None, image_path=None, seed=None):
        self.calls.append(prompt)
        return self.response_dict


def make_question(slot_id: str, question_text: str, solution: str) -> GeneratedQuestion:
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id="module_1",
        global_q_idx=1,
        question_text=question_text,
        solution=solution,
        marking_scheme=[{"criterion": "correct calculation", "marks": 10}],
        marks=10,
        partition=[10],
        bloom="L3",
        co="CO1",
        topic="Circuits",
    )


def fail_verdict(slot_id: str, reason_code: str, detail: str = "issue") -> SlotVerdict:
    return SlotVerdict(
        slot_id=slot_id,
        verdict="fail",
        source="api",
        reason_code=reason_code,
        detail=detail,
    )


# -----------------------------------------------------------------------------
# Math Normalization Tests
# -----------------------------------------------------------------------------


def test_clean_math_preserves_single_letter_variables(verifier):
    text = "Given V = I * R with V = 25 V and I = 2 A."
    cleaned = verifier.clean_math(text)
    assert "V = I * R" in cleaned
    assert "V = 25" in cleaned
    assert "I = 2" in cleaned
    # Ensure 'V' and 'A' units immediately following numbers were stripped
    assert "25 V" not in cleaned
    assert "2 A" not in cleaned


def test_clean_math_converts_latex_and_unicode_symbols(verifier):
    text = r"f = \frac{1}{2\pi \sqrt{L \times C}} \approx 1000 \text{ Hz}"
    cleaned = verifier.clean_math(text)
    assert "sqrt" in cleaned
    assert "=" in cleaned
    assert "*" in cleaned


# -----------------------------------------------------------------------------
# Equality Chain Verification Tests
# -----------------------------------------------------------------------------


def test_verify_consistent_equality_chain(verifier):
    # sqrt(3.986e5 / 7000) is ~7.546 km/s
    q = "Calculate orbital velocity for r = 7000 km."
    s = "Using v = sqrt(3.986e5 / 7000) = 7.55 km/s."
    assert verifier.verify(q, s) is True


def test_verify_inconsistent_equality_chain(verifier):
    q = "Calculate orbital velocity for r = 7000 km."
    s = "Using v = sqrt(3.986e5 / 7000) = 999 km/s."
    assert verifier.verify(q, s) is False


def test_verify_simple_arithmetic_chain(verifier):
    assert verifier.verify("What is the power?", "P = 10 * 5 = 50 W") is True
    assert verifier.verify("What is the power?", "P = 10 * 5 = 65 W") is False


# -----------------------------------------------------------------------------
# Variable Substitution & Formula Verification Tests
# -----------------------------------------------------------------------------


def test_verify_formula_substitution_valid(verifier):
    q = "An electrical circuit operates with current I = 2 A and resistance R = 10 ohms."
    s = "Using Ohm's law V = I * R, the voltage across the component is V = 20 V."
    assert verifier.verify(q, s) is True


def test_verify_formula_substitution_invalid(verifier):
    q = "An electrical circuit operates with current I = 2 A and resistance R = 10 ohms."
    s = "Using Ohm's law V = I * R, the voltage across the component is V = 25 V."
    assert verifier.verify(q, s) is False


# -----------------------------------------------------------------------------
# Edge Cases & Qualitative Safety Tests
# -----------------------------------------------------------------------------


def test_verify_qualitative_text_fails_open(verifier):
    q = "Explain the fundamental operating principle of satellite transponders."
    s = "A satellite transponder receives an uplink signal, translates frequency, and amplifies."
    assert verifier.verify(q, s) is True


def test_verify_empty_strings(verifier):
    assert verifier.verify("", "") is True


def test_default_sympy_verifier_factory_returns_verifier():
    inst = default_sympy_verifier_factory()
    assert inst is not None
    assert isinstance(inst, NumericalVerifier)


# -----------------------------------------------------------------------------
# RefinementAgent Integration Tests with Real NumericalVerifier
# -----------------------------------------------------------------------------


def test_refinement_agent_rejects_mathematically_flawed_repair():
    """When LLM repairs a question with wrong arithmetic, NumericalVerifier rejects it."""
    orig_q = make_question(
        slot_id="module_1_Q1",
        question_text="Calculate the voltage for current I = 3 A and resistance R = 4 ohms.",
        solution="V = 10 V (initial flawed solution)",
    )

    # Repaired response by LLM has an arithmetic contradiction: V = 3 * 4 = 18 V
    flawed_repair = {
        "question_text": "Calculate the voltage when current I = 3 A and resistance R = 4 ohms.",
        "solution": "V = I * R = 3 * 4 = 18 V.",
        "marking_scheme": [{"criterion": "correct calculation", "marks": 10}],
        "change_summary": "Attempted calculation repair",
    }
    llm = MockLLMCaller(flawed_repair)
    verifier = NumericalVerifier()
    agent = RefinementAgent(llm_caller=llm, sympy_verifier=verifier)

    audit = AuditReport(
        verdicts=[fail_verdict("module_1_Q1", FACTUAL_ERROR, "Calculation incorrect")]
    )
    context = AgentContext(questions=[orig_q], audit_report=audit)

    result = agent.run(context)
    report = result.payload

    # The mathematically flawed repair was rejected by SymPy
    assert report.sympy_rejections >= 1
    assert "module_1_Q1" in report.unresolved_slots
    assert report.outcome_for("module_1_Q1").reason_out == "SYMPY_VERIFICATION_FAILED"


def test_refinement_agent_accepts_mathematically_sound_repair():
    """When LLM repairs a question with sound arithmetic, NumericalVerifier accepts it."""
    orig_q = make_question(
        slot_id="module_1_Q1",
        question_text="Calculate the voltage for current I = 3 A and resistance R = 4 ohms.",
        solution="V = 10 V (initial flawed solution)",
    )

    # Repaired response by LLM is mathematically sound: V = 3 * 4 = 12 V
    sound_repair = {
        "question_text": "Calculate the voltage when current I = 3 A and resistance R = 4 ohms.",
        "solution": "V = I * R = 3 * 4 = 12 V.",
        "marking_scheme": [{"criterion": "correct calculation", "marks": 10}],
        "change_summary": "Fixed arithmetic",
    }
    llm = MockLLMCaller(sound_repair)
    verifier = NumericalVerifier()
    agent = RefinementAgent(llm_caller=llm, sympy_verifier=verifier)

    audit = AuditReport(
        verdicts=[fail_verdict("module_1_Q1", FACTUAL_ERROR, "Calculation incorrect")]
    )
    context = AgentContext(questions=[orig_q], audit_report=audit)

    result = agent.run(context)
    report = result.payload

    assert report.sympy_rejections == 0
    assert "module_1_Q1" not in report.unresolved_slots
    outcome = report.outcome_for("module_1_Q1")
    assert outcome.resolved is True
    assert outcome.tier_used == "llm"
