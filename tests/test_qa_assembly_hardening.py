"""
Tests for QA & Assembly Hardening Patch
==========================================
Verifies:
 1. Attemptable marks = 50 for IA (OR choices taking max per group).
 2. OR sub-question mark parity.
 3. Strict module-locked retrieval rejecting cross-module chunks.
 4. Final Quality Gate status harmonization (score 0 or hard errors cannot pass).
 5. Question Quality Firewall expansion ('Create how', 'Create between', 'Apply why', etc. rejected).
 6. OR-pair semantic similarity gate (>= 0.85 rejected, < 0.85 passed).
"""

import pytest
from v0_1.paper_validator import PaperValidator, PaperValidationReport
from v0_1.retriever import GroundedRetriever
from v0_1.unified_pipeline import FinalPaper, ExamType
from core.validators.question_quality_firewall import QuestionQualityFirewall


def test_attemptable_marks_50_for_ia():
    validator = PaperValidator()
    # 5 modules, each has 2 main questions (Q1 and Q2 in OR pair), each main question = 10 marks (6+4)
    modules = []
    for mod_num in range(1, 6):
        co_str = f"CO{mod_num}"
        modules.append({
            "module_num": mod_num,
            "questions": [
                {
                    "mqIndex": (mod_num - 1) * 2 + 1,
                    "subQuestions": [
                        {"letter": "a", "marks": 6, "text": f"Define module {mod_num} core concepts.", "co": co_str},
                        {"letter": "b", "marks": 4, "text": f"Explain module {mod_num} applications.", "co": co_str}
                    ],
                    "totalMarks": 10,
                },
                {
                    "mqIndex": (mod_num - 1) * 2 + 2,
                    "subQuestions": [
                        {"letter": "a", "marks": 6, "text": f"Describe module {mod_num} alternative approach.", "co": co_str},
                        {"letter": "b", "marks": 4, "text": f"Summarize module {mod_num} trade-offs.", "co": co_str}
                    ],
                    "totalMarks": 10,
                },
            ]
        })

    paper_payload = {"modules": modules, "totalMarks": 50}
    report = validator.validate(paper_payload, exam_type="IA")

    # The attemptable marks MUST be 50, NOT 100
    actual_marks = validator._compute_attemptable_marks(modules)
    assert actual_marks == 50
    assert report.checklist["total_marks"] is True
    assert report.passed is True


def test_or_marks_not_double_counted():
    validator = PaperValidator()
    modules = [
        {
            "module_num": 1,
            "questions": [
                {"mqIndex": 1, "subQuestions": [{"marks": 10}]},
                {"mqIndex": 2, "subQuestions": [{"marks": 10}]},
            ]
        }
    ]
    # Attemptable marks for 1 module with OR pair = 10, not 20
    assert validator._compute_attemptable_marks(modules) == 10


def test_or_subquestion_mark_parity():
    validator = PaperValidator()
    modules = [
        {
            "module_num": 1,
            "questions": [
                {"mqIndex": 1, "subQuestions": [{"marks": 6}, {"marks": 4}]},
                {"mqIndex": 2, "subQuestions": [{"marks": 5}, {"marks": 5}]}, # [6,4] != [5,5]
            ]
        }
    ]
    report = validator.validate({"modules": modules, "totalMarks": 10}, exam_type="IA")
    assert report.checklist["or_parity"] is False
    assert any(i.code == "OR_PARTITION_MISMATCH" for i in report.errors())


def test_validate_final_paper_contract():
    from aion_api import validate_final_paper_contract
    modules = []
    for mod_num in range(1, 6):
        modules.append({
            "module_index": mod_num,
            "questions": [
                {"mqIndex": (mod_num - 1) * 2 + 1, "subQuestions": [{"marks": 6}, {"marks": 4}]},
                {"mqIndex": (mod_num - 1) * 2 + 2, "subQuestions": [{"marks": 6}, {"marks": 4}]},
            ]
        })
    paper = {"modules": modules}
    assert validate_final_paper_contract(paper, "IA") is True


def test_validate_final_paper_contract_partition_mismatch():
    from aion_api import validate_final_paper_contract
    modules = []
    for mod_num in range(1, 6):
        modules.append({
            "module_index": mod_num,
            "questions": [
                {"mqIndex": (mod_num - 1) * 2 + 1, "subQuestions": [{"marks": 6}, {"marks": 4}]},
                {"mqIndex": (mod_num - 1) * 2 + 2, "subQuestions": [{"marks": 5}, {"marks": 5}]}, # Mismatch
            ]
        })
    paper = {"modules": modules}
    with pytest.raises(ValueError, match="ContractViolation: Module 1 OR pair partition mismatch"):
        validate_final_paper_contract(paper, "IA")


def test_module_locked_retrieval():
    retriever = GroundedRetriever()
    chunks = [
        "Machine learning fundamentals chunk for Module 1",
        "Neural network deep learning research from 1986 chunk for Module 3",
    ]
    metas = [
        {"module_id": "1"},
        {"module_id": "3"},
    ]

    # Retrieving for Module 1 should strictly exclude Module 3 chunk
    res_m1 = retriever.retrieve("machine learning", chunks, metas, module_id="1")
    assert len(res_m1) == 1
    assert res_m1[0].meta["module_id"] == "1"

    # Retrieving for Module 3 should strictly exclude Module 1 chunk
    res_m3 = retriever.retrieve("neural network", chunks, metas, module_id="3")
    assert len(res_m3) == 1
    assert res_m3[0].meta["module_id"] == "3"


def test_firewall_invalid_bloom_grammar_expansion():
    invalid_examples = [
        "Create how the definitions of AI differ.",
        "Create between the structural components of the system.",
        "Create the effectiveness of using a critic element.",
        "Create the implications of learning agents.",
        "Apply why the agent operates in an environment.",
        "Evaluate why the model failed.",
        "List how humans think.",
        "Describe between simple reflex and goal-based agents.",
        "Explain between agent architectures.",
        "Evaluate between vacuum cleaner environments.",
        "Analyze between search strategies.",
        "Justify why the rule stating that the Evaluate of language works.",
    ]

    for ex in invalid_examples:
        decision = QuestionQualityFirewall.validate(ex)
        assert decision.passed is False, f"Expected '{ex}' to be rejected by firewall"
        assert decision.code == "INVALID_BLOOM_GRAMMAR"


def test_firewall_valid_bloom_grammar_passes():
    valid_examples = [
        "Construct a state space representation for the 8-puzzle problem.",
        "Create a neural network architecture for image classification.",
        "Evaluate the performance of Dijkstra's algorithm.",
        "Analyze the time complexity of QuickSort.",
        "List the four types of agent programs.",
        "Describe the vacuum cleaner world environment.",
    ]

    for ex in valid_examples:
        decision = QuestionQualityFirewall.validate(ex)
        assert decision.passed is True, f"Expected '{ex}' to pass firewall but got {decision.reason}"


def test_or_similarity_high_rejected():
    validator = PaperValidator()
    q1_text = "Justify how Computational Intelligence and AI differ in fundamental principles."
    q2_text = "Create how Computational Intelligence and AI differ in fundamental principles."

    modules = [
        {
            "module_num": 1,
            "questions": [
                {"mqIndex": 1, "subQuestions": [{"letter": "a", "text": q1_text, "marks": 10}]},
                {"mqIndex": 2, "subQuestions": [{"letter": "a", "text": q2_text, "marks": 10}]},
            ]
        }
    ]

    report = validator.validate({"modules": modules, "totalMarks": 10}, exam_type="IA")
    assert report.checklist["no_duplicates"] is False
    assert any(i.code == "OR_SIMILARITY_DUPLICATE" for i in report.warnings())


def test_or_similarity_differentiated_passes():
    validator = PaperValidator()
    q1_text = "Explain the architecture of simple reflex agents with a block diagram."
    q2_text = "Differentiate between goal-based agents and utility-based agents."

    modules = [
        {
            "module_num": 1,
            "questions": [
                {"mqIndex": 1, "subQuestions": [{"letter": "a", "text": q1_text, "marks": 10}]},
                {"mqIndex": 2, "subQuestions": [{"letter": "a", "text": q2_text, "marks": 10}]},
            ]
        }
    ]

    report = validator.validate({"modules": modules, "totalMarks": 10}, exam_type="IA")
    assert report.checklist["no_duplicates"] is True
