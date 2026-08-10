"""
AION Question Quality Firewall & Quality Gate Unit Test Suite
============================================================
Verifies QuestionQualityFirewall rules and Quality Gate Hard-Stop enforcement.
"""

import pytest
from core.validators.question_quality_firewall import QuestionQualityFirewall, FirewallDecision
from v0_1.paper_validator import PaperValidator


def test_firewall_pdf_binary_contamination():
    res = QuestionQualityFirewall.validate("Explain the concept of neural networks /FontFile2 endobj")
    assert res.passed is False
    assert res.code == "PDF_BINARY_CONTAMINATION"
    assert res.hard_failure is True


def test_firewall_incomplete_sentence():
    res = QuestionQualityFirewall.validate("Explain the architecture of transformers using")
    assert res.passed is False
    assert res.code == "INCOMPLETE_SENTENCE"
    assert res.repairable is True


def test_firewall_dangling_operator():
    res = QuestionQualityFirewall.validate("Calculate the loss function value L =")
    assert res.passed is False
    assert res.code == "DANGLING_OPERATOR"
    assert res.repairable is True


def test_firewall_unbalanced_delimiters():
    res = QuestionQualityFirewall.validate("Explain the function f(x) = (ax + b")
    assert res.passed is False
    assert res.code == "UNBALANCED_DELIMITERS"


def test_firewall_invalid_bloom_grammar():
    res = QuestionQualityFirewall.validate("Create between the structural components of an agent.")
    assert res.passed is False
    assert res.code == "INVALID_BLOOM_GRAMMAR"


def test_firewall_prompt_leakage():
    res = QuestionQualityFirewall.validate("Question: Explain the difference between search algorithms.")
    assert res.passed is False
    assert res.code == "PROMPT_LEAKAGE_DETECTED"


def test_firewall_valid_question():
    res = QuestionQualityFirewall.validate("Explain the principle of operation of a simple reflex agent with a neat diagram.")
    assert res.passed is True
    assert res.code == "OK"


def test_paper_validator_or_parity_fail():
    validator = PaperValidator()
    paper = {
        "totalMarks": 50,
        "modules": [
            {
                "module_index": 1,
                "questions": [
                    {"mqIndex": 1, "subQuestions": [{"marks": 6}, {"marks": 4}]},
                    {"mqIndex": 2, "subQuestions": [{"marks": 5}, {"marks": 4}]}, # 9 != 10
                ]
            }
        ]
    }
    report = validator.validate(paper, exam_type="IA")
    assert report.passed is False
    assert any(i.code in ("OR_PARITY", "OR_PARTITION_MISMATCH") for i in report.errors())
