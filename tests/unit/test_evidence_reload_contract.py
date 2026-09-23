"""
Unit tests for SafeEvidencePack, evidence reloading contract, and question quality firewall.
Verifies the fix for AttributeError: 'NoneType' object has no attribute 'split' and quality guards.
"""

import pytest
import re
from unittest.mock import MagicMock, patch

from core.contracts.question_slot import QuestionSlot
from core.contracts.task_signature import TaskSignature
from core.generation.orchestrator import SlotOrchestrator, SafeEvidencePack
from core.validators.question_quality_firewall import QuestionQualityFirewall


@pytest.fixture
def sample_slot():
    return QuestionSlot(
        slot_id="module_1_Q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="module_1_OR_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="Understand",
        co="CO1",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Automatic Irrigation Systems",
        evidence_ids=("chunk_1",),
        answer_budget=150,
        question_budget=40,
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "THEORY"),
        math_required=False,
        visual_required=False,
        generation_seed=42,
    )


def test_safe_evidence_pack_invariants():
    """Verify SafeEvidencePack contract guarantees."""
    # Default values
    pack = SafeEvidencePack()
    assert pack.combined_text == ""
    assert pack.math_artifacts == "none"
    assert str(pack) == ""

    # None coercion
    pack_none = SafeEvidencePack(combined_text=None, math_artifacts=None)
    assert pack_none.combined_text == ""
    assert pack_none.math_artifacts == "none"
    assert str(pack_none) == ""

    # Standard values
    pack_data = SafeEvidencePack("Drip irrigation operates via emitters.", "none")
    assert pack_data.combined_text == "Drip irrigation operates via emitters."
    assert str(pack_data) == "Drip irrigation operates via emitters."


def test_reload_evidence_tier2_fallback_on_builder_failure(sample_slot, caplog):
    """When EvidencePackBuilder fails, _reload_evidence preserves current_pack (Tier 2)."""
    orchestrator = SlotOrchestrator(subject="Test Subject")
    orchestrator.artifact = MagicMock()

    current_pack = SafeEvidencePack(combined_text="Original module evidence text.")

    mock_builder_mod = MagicMock()
    mock_builder_mod.EvidencePackBuilder.build.side_effect = RuntimeError("Builder broken")
    with patch.dict("sys.modules", {"core.evidence.pack_builder": mock_builder_mod}):
        import logging
        with caplog.at_level(logging.DEBUG):
            result = orchestrator._reload_evidence(
                slot=sample_slot,
                excluded_concepts={"irrigation"},
                current_pack=current_pack
            )

    assert result is not None
    assert getattr(result, "combined_text", "") == "Original module evidence text."
    assert "EvidencePackBuilder unavailable or failed" in caplog.text


def test_reload_evidence_tier3_fallback_to_artifact_modules(sample_slot):
    """When current_pack is None, _reload_evidence extracts content from artifact.modules (Tier 3)."""
    orchestrator = SlotOrchestrator(subject="Test Subject")
    
    class MockArtifact:
        combined_text = None
        text = None
        def __init__(self, modules):
            self.modules = modules

    class MockModule:
        def __init__(self, idx, content):
            self.module_index = idx
            self.content = content

    mock_mod = MockModule(1, "Module 1 extracted text from notes.")
    orchestrator.artifact = MockArtifact([mock_mod])

    # Builder fails / not found
    with patch.dict("sys.modules", {"core.evidence.pack_builder": None}):
        result = orchestrator._reload_evidence(
            slot=sample_slot,
            excluded_concepts=set(),
            current_pack=None
        )

    assert result is not None
    assert result.combined_text == "Module 1 extracted text from notes."


def test_reload_evidence_tier4_total_fallback(sample_slot):
    """When no artifact or current_pack is available, Tier 4 returns grounded SafeEvidencePack."""
    orchestrator = SlotOrchestrator(subject="Test Subject")
    orchestrator.artifact = None

    result = orchestrator._reload_evidence(
        slot=sample_slot,
        excluded_concepts=set(),
        current_pack=None
    )

    assert isinstance(result, SafeEvidencePack)
    assert "Automatic Irrigation Systems" in result.combined_text
    assert len(result.combined_text) > 0


def test_format_prompt_handles_none_combined_text(sample_slot):
    """Verify _format_prompt never crashes if combined_text is None."""
    orchestrator = SlotOrchestrator(subject="Test Subject")

    # Broken object where combined_text is explicitly None
    class BrokenEvidencePack:
        combined_text = None
        math_artifacts = "none"

    broken_pack = BrokenEvidencePack()

    # Must NOT raise AttributeError: 'NoneType' object has no attribute 'split'
    prompt = orchestrator._format_prompt(
        slot=sample_slot,
        evidence_pack=broken_pack,
        extra_hints=""
    )

    assert isinstance(prompt, str)
    assert len(prompt) > 0


def test_question_quality_firewall_junk_placeholders():
    """Verify firewall catches exercise headers and template junk."""
    junk_questions = [
        "Discuss the fundamental principles governing Draw Figure 2.1.",
        "Reproduce Figure 3.2 concerning soil moisture sensor interfacing.",
        "Review the diagram and formula revision list for greenhouse systems.",
        "Outline the last-minute revision points for smart irrigation.",
        "Summarize the formula revision list for sensor calibration.",
        "Describe the revision checklist for telemetry links.",
    ]

    for jq in junk_questions:
        decision = QuestionQualityFirewall.validate(jq)
        assert not decision.passed, f"Expected junk question to fail: {jq}"
        assert decision.code == "JUNK_PLACEHOLDER_LEAK"


def test_question_quality_firewall_calculation_without_givens():
    """Verify firewall rejects calculation questions without numerical values."""
    unparameterized = "Calculate the advantage of drip irrigation over sprinkler irrigation."
    decision = QuestionQualityFirewall.validate(unparameterized)
    assert not decision.passed
    assert decision.code == "CALCULATION_WITHOUT_NUMERICAL_GIVENS"

    # With concrete numerical givens, it must pass
    parameterized = "Calculate the irrigation requirement when soil moisture is 22% and field capacity is 35%."
    decision_ok = QuestionQualityFirewall.validate(parameterized)
    assert decision_ok.passed


def test_v0_1_topic_filter_regex():
    """Verify the invalid_topic_line regex rejects junk book headers and preserves topics."""
    invalid_topic_line = re.compile(
        r'^(?:otherwise|if|when|suppose|consider|assume|note|let|for example|where|table|figure|fig|p\.)\b|'
        r'\b(?:draw|reproduce)\s+figure\b|'
        r'\b(?:diagram\s+and\s+formula|formula|last[- ]minute|exam(?:ination)?[- ]oriented)\s+revision\b|'
        r'\b(?:revision\s+(?:list|points?|checklist))\b|'
        r'\b(?:expected\s+learning\s+outcomes?|additional\s+knowledge)\b|'
        r'[?!\.:;]$|'
        r'[|∀∃⇔⊆⇒∧∨¬={}]|'
        r'^(?:[ivx]+\.|\d+[\.\)])\s+|'
        r'\.(?:pdf|docx?|txt|md)\b',
        re.IGNORECASE
    )

    # Must be recognized as invalid
    assert invalid_topic_line.search("Draw Figure 2.1: Block diagram")
    assert invalid_topic_line.search("DRAW   FIGURE")
    assert invalid_topic_line.search("reproduce\tfigure")
    assert invalid_topic_line.search("Diagram and formula revision list")
    assert invalid_topic_line.search("Last-minute revision points")
    assert invalid_topic_line.search("Formula revision list")
    assert invalid_topic_line.search("Revision checklist")

    # Authentic engineering topics must be valid (not matched by regex)
    assert not invalid_topic_line.search("Automatic Irrigation System")
    assert not invalid_topic_line.search("Soil Moisture Sensor Interfacing")
    assert not invalid_topic_line.search("Greenhouse Climate Control")
