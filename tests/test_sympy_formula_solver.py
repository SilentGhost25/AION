"""
Tests for Phase 2: 3-Tier Subject-Agnostic Numeric Synthesis
===========================================================
Verifies SymPy Universal Formula Solver, Algorithmic State Tracer,
Dual-vLLM double-blind verification, and automatic archetype downgrade.
"""

import pytest
from core.numerical_engine import (
    UniversalFormulaSolver,
    AlgorithmicStateTracer,
    DualVLLMVerifier,
    NumericalEngine,
)


def test_universal_formula_solver_ohms_law():
    """Verify SymPy parses V = I * R with proper symbol binding and calculates exact ground truth."""
    solver = UniversalFormulaSolver()
    result = solver.solve_equation("V = I * R", seed=42)

    assert result is not None
    assert result["target"] == "V"
    assert "I" in result["parameters"]
    assert "R" in result["parameters"]
    # Check mathematical correctness: V == I * R
    expected_ans = round(result["parameters"]["I"] * result["parameters"]["R"], 4)
    assert abs(result["answer"] - expected_ans) < 1e-4
    assert result["target_unit"] == "V"


def test_universal_formula_solver_multiletter_acronyms():
    """Verify SymPy solves satellite link budget with multi-letter symbols without multiplying letters."""
    solver = UniversalFormulaSolver()
    # EIRP = Pt + Gt - Lf
    result = solver.solve_equation("EIRP = Pt + Gt - Lf", seed=42)

    assert result is not None
    assert result["target"] == "EIRP"
    pt = result["parameters"]["Pt"]
    gt = result["parameters"]["Gt"]
    lf = result["parameters"]["Lf"]
    expected = round(pt + gt - lf, 4)
    assert abs(result["answer"] - expected) < 1e-4


def test_universal_formula_scheme_of_valuation():
    """Verify generated NumericalTemplate produces 4-part university marking scheme."""
    solver = UniversalFormulaSolver()
    template = solver.generate_template(
        topic="Transformer Efficiency",
        marks=10,
        context_text="The efficiency of an electrical device is given by: P_out = V * I * cos_phi.",
        seed=123
    )

    assert template is not None
    assert template.bloom_level == "L3"
    assert "governing formula" in template.template
    assert "Substitute" in template.template
    assert "Calculate" in template.template
    assert "Final Answer" in template.solution_hint


def test_algorithmic_state_tracer():
    """Verify AlgorithmicStateTracer produces discrete sequence and trace rubric."""
    tracer = AlgorithmicStateTracer()
    template = tracer.generate_template(topic="Merge Sort Partitioning", marks=8, seed=99)

    assert template.domain == "algorithmic_trace"
    assert "sequence" in template.params
    assert len(template.params["sequence"]) == 7
    assert "comparisons" in template.template


def test_dual_vllm_verifier_tolerance():
    """Verify DualVLLMVerifier enforces ±3% relative error tolerance."""
    verifier = DualVLLMVerifier()

    # 100.0 vs 101.5 (1.5% diff) -> PASS
    assert verifier.verify(100.0, 101.5, tolerance_pct=3.0) is True

    # 100.0 vs 105.0 (5% diff) -> FAIL
    assert verifier.verify(100.0, 105.0, tolerance_pct=3.0) is False

    # Near-zero values
    assert verifier.verify(0.0, 0.0, tolerance_pct=3.0) is True


def test_numerical_engine_subject_agnostic_detection():
    """Verify NumericalEngine detects formulas in non-catalog engineering topics."""
    engine = NumericalEngine()

    biotech_text = "Enzyme catalytic rate follows V = (Vmax * S) / (Km + S) where S is substrate concentration."
    domain = engine.detect_domain(biotech_text)
    assert domain == "universal_formula"

    # Verify template generation from chunks
    chunks = [{"text": biotech_text}]
    template = engine.generate_from_chunks(chunks, marks=10, seed=42)
    assert template is not None
    assert template.domain == "universal_formula"


def test_numerical_engine_clean_downgrade_for_qualitative_law():
    """Verify NumericalEngine cleanly returns None for non-mathematical subjects to trigger archetype downgrade."""
    engine = NumericalEngine()

    law_text = """
    Under Section 10 of the Indian Contract Act, all agreements are contracts if they are made
    by the free consent of parties competent to contract, for a lawful consideration.
    """
    domain = engine.detect_domain(law_text)
    assert domain is None

    chunks = [{"text": law_text}]
    template = engine.generate_from_chunks(chunks, marks=10)
    assert template is None  # Triggers automatic qualitative downgrade in orchestrator


def test_dual_vllm_auditor_output_schema():
    """Verify AuditorNumericOutput produces expected schema and parse_auditor_output handles all formats."""
    from core.numerical_engine import AuditorNumericOutput
    schema = DualVLLMVerifier.get_auditor_schema()
    assert "properties" in schema
    assert "final_value" in schema["properties"]
    assert "unit" in schema["properties"]

    # Test dict input
    val = DualVLLMVerifier.parse_auditor_output({"final_value": 42.5, "unit": "V"})
    assert val == 42.5

    # Test object input
    obj = AuditorNumericOutput(final_value=12.34, unit="A")
    assert DualVLLMVerifier.parse_auditor_output(obj) == 12.34

    # Test float input
    assert DualVLLMVerifier.parse_auditor_output(99.9) == 99.9

    # Test malformed / missing
    assert DualVLLMVerifier.parse_auditor_output({"invalid": "data"}) is None
    assert DualVLLMVerifier.parse_auditor_output(None) is None
