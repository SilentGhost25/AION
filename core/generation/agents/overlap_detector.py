"""
Canonical Subquestion Overlap Detector.
Extracted verbatim from legacy_adapter.py:208-260 (baseline 9034521b).

Preserves exact tokenization, stemming, stopword filtering, Jaccard and
containment calculations, thresholds, and return metadata.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set, Tuple


def _specific_overlap(s1: str, s2: str) -> tuple:
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


def detect_subquestion_overlap(sub_texts: List[str]) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Detect subquestion overlap/duplicates and return structured validation telemetry.
    Preserves exact baseline logic and thresholds from legacy_adapter.py:243-260.
    """
    letters = ["a", "b", "c", "d", "e"]
    overlap_detected = False
    overlap_details: List[Dict[str, Any]] = []

    if len(sub_texts) > 1:
        for i in range(len(sub_texts)):
            for j in range(i):
                w_i = set(re.findall(r"\w+", sub_texts[i].lower()))
                w_j = set(re.findall(r"\w+", sub_texts[j].lower()))
                jaccard = len(w_i & w_j) / max(1, len(w_i | w_j))
                common, containment = _specific_overlap(sub_texts[i], sub_texts[j])
                if sub_texts[i].strip().lower() == sub_texts[j].strip().lower() or jaccard >= 0.40 or (containment >= 0.6 and len(common) >= 2):
                    overlap_detected = True
                    overlap_details.append({
                        "sub_i": letters[i] if i < len(letters) else f"sub_{i+1}",
                        "sub_j": letters[j] if j < len(letters) else f"sub_{j+1}",
                        "jaccard": round(jaccard, 3),
                        "containment": round(containment, 3),
                    })

    return overlap_detected, overlap_details
