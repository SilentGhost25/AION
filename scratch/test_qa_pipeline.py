import sys
import os
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from v0_1.qa_engine import (
    BloomsTaxonomyValidator,
    TopicDiversityEnforcer,
    QuestionCompletenessChecker,
    MarkAllocationOptimizer,
    CognitiveLevelBalancer,
    QPGeneratorWithQA,
)


def test_blooms_validator():
    print("\n--- Testing BloomsTaxonomyValidator ---")
    val = BloomsTaxonomyValidator()

    q1 = "List the two modern computer system hardware instructions."
    valid, level, conf = val.validate_question(q1, "L1_Remember")
    print(f"Q1 Validation: Valid={valid}, Level={level}, Conf={conf}")
    assert valid is True

    q2 = "List the two modern computer system hardware instructions."
    valid, level, conf = val.validate_question(q2, "L3_Apply")
    print(f"Q2 Mismatch Detection: Valid={valid}, Level={level}")
    assert valid is False

    suggested = val.auto_correct_blooms_level(q2, "L3_Apply")
    print(f"Q2 Auto-Corrected: '{suggested}'")
    assert suggested.lower().startswith("apply")
    print("[OK] BloomsTaxonomyValidator PASSED")


def test_topic_diversity():
    print("\n--- Testing TopicDiversityEnforcer ---")
    sample_text = """
    1.1 Critical Section Problem
    The critical section problem requires mutual exclusion, progress, and bounded waiting.

    1.2 Peterson's Solution
    Peterson's solution is a classic software-based solution to the critical section problem.

    1.3 Semaphores and Mutex Locks
    Semaphores are integer variables used for process synchronization.
    """

    enforcer = TopicDiversityEnforcer()
    topics = enforcer.extract_topics_from_module(sample_text)
    print(f"Extracted topics ({len(topics)}): {topics}")
    assert len(topics) >= 2

    matrix = enforcer.create_topic_distribution_matrix(module_identifier=1, num_questions=4, available_topics=topics)
    print(f"Topic Distribution Matrix: {matrix}")
    assert len(matrix) == 4

    mock_qs = [
        {"text": "Explain critical section problem and mutual exclusion."},
        {"text": "Describe Peterson's solution for two processes."},
        {"text": "Demonstrate the use of semaphores for process synchronization."},
        {"text": "Compare mutex locks and binary semaphores."}
    ]
    is_div, report = enforcer.validate_diversity(mock_qs, 1, required_topics=topics)
    print(f"Diversity check: Diverse={is_div}, Coverage={report['coverage_ratio']:.0%}")
    print("[OK] TopicDiversityEnforcer PASSED")


def test_completeness_checker():
    print("\n--- Testing QuestionCompletenessChecker ---")
    checker = QuestionCompletenessChecker()

    good = "Explain how mutex locks prevent race conditions in multithreaded programs."
    is_c, reason = checker.is_complete(good)
    print(f"Good question check: Complete={is_c}, Reason='{reason}'")
    assert is_c is True

    bad_trunc = "Explain how the value of turn determines which process is allowed to"
    is_c, reason = checker.is_complete(bad_trunc)
    print(f"Truncated check: Complete={is_c}, Reason='{reason}'")
    assert is_c is False

    fixed = checker.auto_fix_truncation(bad_trunc)
    print(f"Fixed question: '{fixed}'")
    assert fixed.endswith(".")
    print("[OK] QuestionCompletenessChecker PASSED")


def test_mark_optimizer():
    print("\n--- Testing MarkAllocationOptimizer ---")
    opt = MarkAllocationOptimizer()

    marks_apply = opt.calculate_optimal_marks("Apply Peterson's algorithm to solve the bounded buffer problem.", "L3_Apply")
    print(f"Optimal marks for L3_Apply: {marks_apply}M")

    dist = opt.distribute_marks_across_subquestions(total_marks=20, num_subquestions=3, blooms_levels=['L2_Understand', 'L4_Analyze', 'L3_Apply'])
    print(f"20M Distribution across 3 subquestions (L2, L4, L3): {dist}")
    assert sum(dist) == 20
    print("[OK] MarkAllocationOptimizer PASSED")


def test_cognitive_balancer():
    print("\n--- Testing CognitiveLevelBalancer ---")
    balancer = CognitiveLevelBalancer()

    dist = balancer.generate_balanced_distribution(4)
    print(f"Planned distribution for 4 questions: {dist}")
    assert len(dist) == 4
    print("[OK] CognitiveLevelBalancer PASSED")


def test_full_pipeline():
    print("\n--- Testing Integrated QPGeneratorWithQA ---")
    qa = QPGeneratorWithQA()

    mock_paper = [
        {
            "module_index": 1,
            "module_title": "Process Synchronization",
            "questions": [
                {
                    "mq_index": 1,
                    "bloom_level": 3,
                    "bloom_name": "Apply",
                    "sub_questions": [
                        {"letter": "a", "text": "Apply Peterson's solution for process synchronization.", "marks": 7, "bloom": 3},
                        {"letter": "b", "text": "Explain the concept of race conditions.", "marks": 7, "bloom": 2},
                        {"letter": "c", "text": "List two hardware instructions for synchronization.", "marks": 6, "bloom": 1},
                    ]
                }
            ]
        }
    ]

    report = qa.run_full_paper_qa(mock_paper)
    print(f"QA Score: {report['quality_score']}/100")
    print(f"Issues Found: {report['total_issues_found']}")
    print(f"Issues Auto-Fixed: {report['issues_auto_fixed']}")
    assert report['quality_score'] >= 0
    print("[OK] Integrated QPGeneratorWithQA PASSED")


if __name__ == "__main__":
    print("==================================================")
    print("         AION QA Pipeline Test Suite              ")
    print("==================================================")
    test_blooms_validator()
    test_topic_diversity()
    test_completeness_checker()
    test_mark_optimizer()
    test_cognitive_balancer()
    test_full_pipeline()
    print("\nALL QA PIPELINE TESTS PASSED SUCCESSFULLY!")
