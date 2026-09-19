"""
Tests for Phase 1: Zero-Hardcoding Domain Refactoring
=====================================================
Verifies that the DynamicPedagogicalProfiler, DocumentKnowledgeGraph,
and generalized domain integrity gates handle non-CS courses seamlessly
without defaulting to CSE, satellite communications, or automotive heuristics.
"""

import pytest
from core.domain.subject_detector import (
    DynamicPedagogicalProfiler,
    DocumentKnowledgeGraph,
    SubjectDetector,
    SubjectProfile,
)
from core.domain.integrity_gate import DomainIntegrityGate
from core.validation.slot_remapper import infer_co
from v0_1.module_alignment import ModuleAlignmentValidator


def test_dynamic_profiler_biotechnology():
    """Verify biotechnology notes classify as EMPIRICAL_ANALYTICAL with extracted bio terms."""
    bio_text = """
    Subject: Biotechnology and Genomics
    Course Code: 21BT52
    Module 1: Cellular and Molecular Biology
    Recombinant DNA technology involves the use of restriction enzymes and DNA ligase to insert
    target genes into cloning vectors. Polymerase chain reaction (PCR) amplifies specific DNA sequences.
    Enzyme kinetics follow the Michaelis-Menten model where Vmax and Km determine catalytic efficiency.
    """
    profiler = DynamicPedagogicalProfiler()
    dkg = profiler.profile(bio_text)

    assert dkg.subject_code == "21BT52"
    assert "Biotechnology" in dkg.subject_name or "Genomics" in dkg.subject_name
    assert dkg.discipline_type == "EMPIRICAL_ANALYTICAL"
    assert "enzymes" in dkg.vocabulary or "enzyme" in dkg.vocabulary
    assert "recombinant" in dkg.vocabulary or "polymerase" in dkg.vocabulary

    profile = profiler.create_subject_profile(dkg)
    assert profile.code == "21BT52"
    assert "empirical deduction" in profile.reasoning_styles or "data interpretation" in profile.reasoning_styles


def test_dynamic_profiler_corporate_law():
    """Verify corporate law notes classify as THEORETICAL_QUALITATIVE without engineering artifacts."""
    law_text = """
    Course Title: Corporate and Commercial Law
    Code: LAW401
    Module 1: Principles of Contract Formation
    An agreement enforceable by law is a contract under Section 2(h) of the Indian Contract Act.
    Essential elements include offer, acceptance, lawful consideration, capacity of parties,
    free consent, and lawful object. Breach of contract entitles the aggrieved party to damages
    under the rule of Hadley v Baxendale.
    """
    profiler = DynamicPedagogicalProfiler()
    dkg = profiler.profile(law_text)

    assert dkg.subject_code == "LAW401"
    assert dkg.discipline_type == "THEORETICAL_QUALITATIVE"
    assert "contract" in dkg.vocabulary
    assert "damages" in dkg.vocabulary

    profile = profiler.create_subject_profile(dkg)
    assert len(profile.numerical_patterns) == 0  # Zero artificial numerical patterns for qualitative law


def test_dynamic_profiler_linear_algebra():
    """Verify mathematical formal texts classify as MATHEMATICAL_FORMAL with derivation styles."""
    math_text = """
    Course: Advanced Linear Algebra and Calculus
    Sub Code: 21MAT31
    Module 1: Vector Spaces and Eigenvalues
    Let V be a vector space over field F. A linear transformation T: V -> V has eigenvalues
    satisfying the characteristic equation det(A - lambda*I) = 0.
    The Cayley-Hamilton theorem states that every square matrix satisfies its own characteristic equation.
    """
    profiler = DynamicPedagogicalProfiler()
    dkg = profiler.profile(math_text)

    assert dkg.subject_code == "21MAT31"
    assert dkg.discipline_type == "MATHEMATICAL_FORMAL"
    assert "characteristic" in dkg.vocabulary or "eigenvalues" in dkg.vocabulary

    profile = profiler.create_subject_profile(dkg)
    assert "formal derivation" in profile.reasoning_styles or "algebraic proof" in profile.reasoning_styles


def test_subject_detector_does_not_default_unknown_to_cse():
    """Verify SubjectDetector dynamically profiles unrecognized courses rather than defaulting to CSE."""
    detector = SubjectDetector()
    philosophy_text = """
    Course Title: Classical Moral Philosophy and Ethics
    Examine the moral categorical imperative formulated by Immanuel Kant. Compare utilitarianism
    and deontological ethical frameworks in resolving ethical dilemmas in governance.
    """
    profile = detector.detect(philosophy_text)
    # Must NOT default to CSE
    assert profile.code != "CSE"
    assert profile.name != "Computer Science & Engineering"
    assert "ethical" in profile.permitted_vocabulary or "utilitarianism" in profile.permitted_vocabulary


def test_domain_integrity_gate_allows_scenario_framing():
    """Verify DomainIntegrityGate allows standard application scenarios without flagging unseen errors."""
    gate = DomainIntegrityGate()
    question = "In an autonomous hospital robot navigation system, explain how sensors track mobile obstacles."
    ku_concepts = {"sensors", "navigation", "obstacles"}
    evidence = "Mobile robot navigation uses ultrasonic sensors to detect local obstacles in indoor environments."

    result = gate.check(question, ku_concepts, evidence)
    assert result.passed is True
    assert len(result.violations) == 0


def test_infer_co_preserves_module_outcome():
    """Verify infer_co preserves module-based Course Outcome rather than overriding with satellite terms."""
    # Question mentions orbit/velocity but belongs to Module 4 (CO4)
    q_text = "Calculate the orbital velocity and transfer energy for a remote sensing satellite."
    # With default_co = 4, it should preserve CO4
    co = infer_co(q_text, default_co=4)
    assert co == 4


def test_module_alignment_subject_agnostic():
    """Verify ModuleAlignmentValidator passes non-CS questions when no custom syllabus map is provided."""
    validator = ModuleAlignmentValidator()
    # A biology question tested against module 1
    q_text = "Explain the mechanism of enzyme inhibition with competitive and non-competitive examples."
    res = validator.validate(q_text, target_module=1, custom_syllabus_map=None)
    assert res.passed is True
    assert res.reason == "COMPLIANT"
