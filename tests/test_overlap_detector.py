"""
Characterization Test Suite for Subquestion Overlap Detector (Stage 1A).

Verifies strict mathematical and behavioral equivalence between the newly
extracted `core.generation.agents.overlap_detector` and a frozen test-only
reference copy of the original baseline implementation (commit 9034521b).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple

import pytest

from core.generation.agents.overlap_detector import (
    _specific_overlap,
    detect_subquestion_overlap,
)
from core.generation.agents.legacy_adapter import (
    _to_question_dict,
    _specific_overlap as adapter_specific_overlap,
    detect_subquestion_overlap as adapter_detect_overlap,
)
from core.generation.agents.generated_question import GeneratedQuestion


# =============================================================================
# Frozen Baseline Reference Implementation (Commit 9034521b verbatim)
# =============================================================================

def _frozen_baseline_specific_overlap(s1: str, s2: str) -> tuple:
    generic = {
        'the', 'of', 'in', 'by', 'which', 'a', 'an', 'and', 'for', 'to', 'is', 'are', 'with',
        'describe', 'explain', 'outline', 'role', 'method', 'methods', 'way', 'ways', 'how', 
        'ensure', 'ensures', 'use', 'using', 'detail', 'system', 'systems', 'computer', 'computers', 
        'following', 'state', 'list', 'define', 'respect', 'provide', 'briefly'
    }
    def get_terms(s):
        w = [t for t in re.findall(r'[a-zA-Z]+', s.lower()) if t not in generic]
        res = set()
        for x in w:
            for suf in ['ingly', 'ously', 'fully', 'ment', 'tion', 'sion', 'ness', 'able', 'ible', 'ing', 'ied', 'ies', 'est', 'ers', 'ely', 'ity', 'ive', 'ous', 'ful', 'ed', 'er', 'ly', 'es', 's']:
                if len(x) > len(suf) + 2 and x.endswith(suf):
                    x = x[:-len(suf)]
                    break
            if x not in generic:
                res.add(x)
        return res
    t1 = get_terms(s1)
    t2 = get_terms(s2)
    common = t1 & t2
    min_len = min(len(t1), len(t2))
    containment = len(common) / min_len if min_len > 0 else 0.0
    return common, containment


def _frozen_baseline_detect_overlap(sub_texts: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    letters = ["a", "b", "c", "d", "e"]
    overlap_detected = False
    overlap_details: List[Dict[str, Any]] = []

    if len(sub_texts) > 1:
        for i in range(len(sub_texts)):
            for j in range(i):
                w_i = set(re.findall(r"\w+", sub_texts[i].lower()))
                w_j = set(re.findall(r"\w+", sub_texts[j].lower()))
                jaccard = len(w_i & w_j) / max(1, len(w_i | w_j))
                common, containment = _frozen_baseline_specific_overlap(sub_texts[i], sub_texts[j])
                if sub_texts[i].strip().lower() == sub_texts[j].strip().lower() or jaccard >= 0.40 or (containment >= 0.6 and len(common) >= 2):
                    overlap_detected = True
                    overlap_details.append({
                        "sub_i": letters[i] if i < len(letters) else f"sub_{i+1}",
                        "sub_j": letters[j] if j < len(letters) else f"sub_{j+1}",
                        "jaccard": round(jaccard, 3),
                        "containment": round(containment, 3),
                    })

    return overlap_detected, overlap_details


# =============================================================================
# Test Dataset: Comprehensive Equivalence Corpus
# =============================================================================

EQUIVALENCE_CORPUS = [
    # 1. Exact duplicates
    ("Explain the architecture of computer systems.", "Explain the architecture of computer systems."),
    # 2. Near duplicates with casing/punctuation
    ("Describe the role of OS in memory management.", "Describe the roles of OS in memory management!"),
    # 3. Paraphrased duplicates (common concepts)
    ("Explain CPU scheduling algorithms in modern operating systems.", "Discuss the algorithms used for CPU scheduling in an operating system."),
    # 4. Related but distinct concepts
    ("Explain components of a computer system.", "Discuss two types of system calls."),
    # 5. Shared technical terminology on distinct concepts
    ("Calculate paging memory overhead for a 32-bit address space.", "Explain the role of TLB hit ratio in paging translation."),
    # 6. Short questions affected by stopwords
    ("Define a thread.", "Define a process."),
    # 7. Different instructional verbs, same topic
    ("Explain the Many-to-One multithreading model.", "Analyze the Many-to-One multithreading model."),
    # 8. Empty strings
    ("", ""),
    # 9. One empty string
    ("Explain process synchronization.", ""),
    # 10. Whitespace-only strings
    ("   ", " \t\n "),
    # 11. Punctuation variations
    ("Hello world!", "Hello world???"),
    # 12. Repeated words
    ("scheduling scheduling scheduling algorithm", "scheduling algorithm"),
    # 13. One-word inputs
    ("scheduling", "paging"),
    # 14. Identical single generic word
    ("explain", "explain"),
    # 15. Real audit defect Q2(a) vs Q2(b)
    ("Describe the role of the OS in managing hardware resources efficiently.", "Outline the methods by which the OS ensures efficient use of computer hardware."),
    # 16. Real audit defect Q4(a) vs Q4(b)
    ("Calculate the speedup of a system using Amdahl's Law.", "Explain the critical trade-offs of multiprocessor parallelism in detail."),
]


# =============================================================================
# Equivalence Tests: Extracted vs Frozen Baseline
# =============================================================================

@pytest.mark.parametrize("s1,s2", EQUIVALENCE_CORPUS)
def test_specific_overlap_exact_equivalence(s1: str, s2: str):
    """Assert extracted _specific_overlap is bit-for-bit identical to frozen baseline."""
    act_common, act_containment = _specific_overlap(s1, s2)
    exp_common, exp_containment = _frozen_baseline_specific_overlap(s1, s2)

    assert act_common == exp_common, f"Common tokens mismatch for: {s1!r} vs {s2!r}"
    assert abs(act_containment - exp_containment) < 1e-9, f"Containment mismatch for: {s1!r} vs {s2!r}"


@pytest.mark.parametrize("s1,s2", EQUIVALENCE_CORPUS)
def test_detect_subquestion_overlap_exact_equivalence(s1: str, s2: str):
    """Assert extracted detect_subquestion_overlap is bit-for-bit identical to frozen baseline."""
    sub_texts = [s1, s2]
    act_detected, act_details = detect_subquestion_overlap(sub_texts)
    exp_detected, exp_details = _frozen_baseline_detect_overlap(sub_texts)

    assert act_detected == exp_detected, f"Detection flag mismatch for: {sub_texts}"
    assert act_details == exp_details, f"Overlap details mismatch for: {sub_texts}"


def test_multi_subquestion_list_equivalence():
    """Verify equivalence across multi-part questions (3, 4, and 5 subparts)."""
    sub_lists = [
        [
            "Explain memory hierarchy.",
            "Describe cache memory organization.",
            "Explain memory hierarchy in detail.",
        ],
        [
            "Define semaphore.",
            "Explain mutex lock.",
            "Compare semaphore and mutex.",
            "Define monitor in operating systems.",
        ],
        ["Only one subquestion"],
        [],
    ]
    for sub_texts in sub_lists:
        act_detected, act_details = detect_subquestion_overlap(sub_texts)
        exp_detected, exp_details = _frozen_baseline_detect_overlap(sub_texts)
        assert act_detected == exp_detected
        assert act_details == exp_details


# =============================================================================
# Empirical Baseline Scores: The 7 Documented Matrix Categories
# =============================================================================

def test_documented_7_category_baseline_scores():
    """
    Verify the 7 documented baseline matrix categories retain their exact
    empirically calculated scores and outcomes.
    """
    # 1. Exact Duplicate
    res1, det1 = detect_subquestion_overlap([
        "Explain the architecture of computer systems.",
        "Explain the architecture of computer systems.",
    ])
    assert res1 is True
    assert det1[0]["jaccard"] == 1.000
    assert det1[0]["containment"] == 1.000

    # 2. Near-Duplicate
    res2, det2 = detect_subquestion_overlap([
        "Describe the role of OS in memory management.",
        "Describe the roles of OS in memory management!",
    ])
    assert res2 is True
    assert det2[0]["jaccard"] == 0.778
    assert det2[0]["containment"] == 1.000

    # 3. Paraphrased Duplicate
    res3, det3 = detect_subquestion_overlap([
        "Explain CPU scheduling algorithms in modern operating systems.",
        "Discuss the algorithms used for CPU scheduling in an operating system.",
    ])
    assert res3 is True
    assert det3[0]["jaccard"] == 0.357
    assert det3[0]["containment"] == 0.8

    # 4. Related but Distinct
    res4, det4 = detect_subquestion_overlap([
        "Explain components of a computer system.",
        "Discuss two types of system calls.",
    ])
    assert res4 is False
    assert len(det4) == 0

    # 5. Shared Terminology (Distinct)
    res5, det5 = detect_subquestion_overlap([
        "Calculate paging memory overhead for a 32-bit address space.",
        "Explain the role of TLB hit ratio in paging translation.",
    ])
    assert res5 is False
    assert len(det5) == 0

    # 6. Short Questions (Demonstrating stopword inflation on baseline Jaccard)
    res6, det6 = detect_subquestion_overlap([
        "Define a thread.",
        "Define a process.",
    ])
    assert res6 is True
    assert det6[0]["jaccard"] == 0.500
    assert det6[0]["containment"] == 0.000

    # 7. Different Verbs, Same Query
    res7, det7 = detect_subquestion_overlap([
        "Explain the Many-to-One multithreading model.",
        "Analyze the Many-to-One multithreading model.",
    ])
    assert res7 is True
    assert det7[0]["jaccard"] == 0.750
    assert det7[0]["containment"] == 1.000


# =============================================================================
# Legacy Adapter Re-export and Integration Equivalence
# =============================================================================

def test_legacy_adapter_re_exports():
    """Verify legacy_adapter re-exports _specific_overlap and detect_subquestion_overlap."""
    assert adapter_specific_overlap is _specific_overlap
    assert adapter_detect_overlap is detect_subquestion_overlap


def test_legacy_adapter_to_question_dict_overlap_integration():
    """Verify _to_question_dict produces identical telemetry and overlap keys."""
    # Overlapping question
    q_dup = GeneratedQuestion(
        slot_id="slot_dup",
        module_id="module_1",
        global_q_idx=1,
        question_text="(a) Describe the role of OS in hardware management. (b) Describe the role of OS in hardware management.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
    )
    d_dup = _to_question_dict(q_dup)
    assert d_dup["subquestion_overlap"] is True
    assert d_dup["telemetry"]["subquestion_overlap"] is True
    assert len(d_dup["overlap_details"]) == 1
    assert d_dup["overlap_details"][0]["sub_i"] == "b"
    assert d_dup["overlap_details"][0]["sub_j"] == "a"

    # Distinct question
    q_clean = GeneratedQuestion(
        slot_id="slot_clean",
        module_id="module_1",
        global_q_idx=2,
        question_text="(a) Explain components of a computer system. (b) Discuss two types of system calls.",
        solution="Model solution.",
        marking_scheme=[{"criterion": "Part a", "marks": 6}, {"criterion": "Part b", "marks": 4}],
        marks=10,
        partition=[6, 4],
        bloom="L2",
        co="CO1",
    )
    d_clean = _to_question_dict(q_clean)
    assert d_clean["subquestion_overlap"] is False
    assert "subquestion_overlap" not in d_clean["telemetry"]
    assert len(d_clean["overlap_details"]) == 0


if __name__ == "__main__":
    pytest.main(["-v", __file__])
