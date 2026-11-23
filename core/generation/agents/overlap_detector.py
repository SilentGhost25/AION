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
        'following', 'state', 'list', 'define', 'respect', 'provide', 'briefly',
        'discuss', 'illustrate', 'compare', 'contrast', 'differentiate', 'distinguish',
        'write', 'short', 'note', 'notes', 'advantages', 'disadvantages', 'suitable',
        'example', 'examples', 'diagram', 'neat', 'sketch', 'sketches', 'principles',
        'characteristics', 'give', 'various', 'brief'
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


# Deliberate Policy Thresholds for Cross-Module Deduplication:
# 1. QUESTION_JACCARD_THRESHOLD (0.65): In academic exam generation, questions sharing identical topics
#    and requirements under different wording have token Jaccard >= 0.65 (after stopword filtering).
# 2. QUESTION_CONTAINMENT_THRESHOLD (0.75): Detects when a question in one module is a substantial subset
#    of a question in another module with at least 4 common substantive root terms.
# 3. SUBPART_JACCARD_THRESHOLD (0.70): Subparts are shorter clauses (typically 10-25 words); higher threshold
#    prevents false positives while flagging illicit cross-module topic clause collisions.
# 4. SUBPART_CONTAINMENT_THRESHOLD (0.75): Substantive containment threshold on subparts with at least 3 common terms.
QUESTION_JACCARD_THRESHOLD = 0.65
QUESTION_CONTAINMENT_THRESHOLD = 0.75
QUESTION_MIN_COMMON_TERMS = 4

SUBPART_JACCARD_THRESHOLD = 0.70
SUBPART_CONTAINMENT_THRESHOLD = 0.75
SUBPART_MIN_COMMON_TERMS = 3


def audit_cross_module_duplicates(questions: List[Any]) -> List[Dict[str, Any]]:
    """
    Audits cross-module question pairs for duplicates (Stage G5-C).
    Compares every distinct pair (q_i, q_j) where module_id_i != module_id_j.
    Distinguishes question-level duplicates from subpart-level duplicates.
    Reports each pair at most once under canonical sorting, prioritizing question-level over subpart-level.
    Order-invariant: producing identical results regardless of the permutation of questions.
    """
    findings: List[Dict[str, Any]] = []
    seen_pairs: Set[Tuple[str, str]] = set()
    n = len(questions)

    for i in range(n):
        for j in range(i + 1, n):
            q1 = questions[i]
            q2 = questions[j]
            mod1 = getattr(q1, "module_id", "") or ""
            mod2 = getattr(q2, "module_id", "") or ""
            if not mod1 or not mod2 or mod1 == mod2:
                continue

            slot1_raw = getattr(q1, "slot_id", f"slot_{i}")
            slot2_raw = getattr(q2, "slot_id", f"slot_{j}")
            slot1 = str(slot1_raw)
            slot2 = str(slot2_raw)
            t1 = (getattr(q1, "question_text", "") or "").strip()
            t2 = (getattr(q2, "question_text", "") or "").strip()
            if not t1 or not t2:
                continue

            # Enforce canonical ordering (slot_a <= slot_b) for strict order-invariance
            if slot1 <= slot2:
                can_slot_a, can_slot_b = slot1, slot2
                can_mod_a, can_mod_b = str(mod1), str(mod2)
                can_t_a, can_t_b = t1, t2
                can_sps_a = getattr(q1, "sub_parts", None) or []
                can_sps_b = getattr(q2, "sub_parts", None) or []
            else:
                can_slot_a, can_slot_b = slot2, slot1
                can_mod_a, can_mod_b = str(mod2), str(mod1)
                can_t_a, can_t_b = t2, t1
                can_sps_a = getattr(q2, "sub_parts", None) or []
                can_sps_b = getattr(q1, "sub_parts", None) or []

            pair_key = (can_slot_a, can_slot_b)
            if pair_key in seen_pairs:
                continue

            has_multi_subparts = len(can_sps_a) > 1 and len(can_sps_b) > 1

            # 1. Exact question match
            if can_t_a.lower() == can_t_b.lower():
                seen_pairs.add(pair_key)
                findings.append({
                    "slot_a": can_slot_a,
                    "slot_b": can_slot_b,
                    "module_a": can_mod_a,
                    "module_b": can_mod_b,
                    "severity": "question_level",
                    "match_type": "exact_question",
                    "similarity": 1.0,
                    "containment": 1.0,
                    "common_terms": [],
                    "detail": f"Exact identical question text between {can_slot_a} ({can_mod_a}) and {can_slot_b} ({can_mod_b})",
                })
                continue

            # 2. Check for subpart collisions first if multi-part questions
            if has_multi_subparts:
                matched_subpart_pairs = []
                for sp_a in can_sps_a:
                    sp_text_a = (getattr(sp_a, "text", "") or "").strip()
                    if not sp_text_a:
                        continue
                    for sp_b in can_sps_b:
                        sp_text_b = (getattr(sp_b, "text", "") or "").strip()
                        if not sp_text_b:
                            continue

                        # Subpart exact match
                        if sp_text_a.lower() == sp_text_b.lower():
                            matched_subpart_pairs.append({
                                "sp_a": getattr(sp_a, "label", "?"),
                                "sp_b": getattr(sp_b, "label", "?"),
                                "match_type": "exact_subpart",
                                "similarity": 1.0,
                                "common_terms": [],
                            })
                            continue

                        # Subpart near-duplicate
                        sp_w1 = set(re.findall(r"\w+", sp_text_a.lower()))
                        sp_w2 = set(re.findall(r"\w+", sp_text_b.lower()))
                        sp_jaccard = len(sp_w1 & sp_w2) / max(1, len(sp_w1 | sp_w2))
                        sp_common, sp_containment = _specific_overlap(sp_text_a, sp_text_b)

                        if (
                            (sp_jaccard >= SUBPART_JACCARD_THRESHOLD or sp_containment >= SUBPART_CONTAINMENT_THRESHOLD)
                            and len(sp_common) >= SUBPART_MIN_COMMON_TERMS
                        ):
                            matched_subpart_pairs.append({
                                "sp_a": getattr(sp_a, "label", "?"),
                                "sp_b": getattr(sp_b, "label", "?"),
                                "match_type": "near_duplicate_subpart",
                                "similarity": round(sp_jaccard, 3),
                                "common_terms": sorted(list(sp_common)),
                            })

                if matched_subpart_pairs:
                    seen_pairs.add(pair_key)
                    # If all subparts match -> question_level
                    if len(matched_subpart_pairs) >= min(len(can_sps_a), len(can_sps_b)):
                        findings.append({
                            "slot_a": can_slot_a,
                            "slot_b": can_slot_b,
                            "module_a": can_mod_a,
                            "module_b": can_mod_b,
                            "severity": "question_level",
                            "match_type": "near_duplicate_question",
                            "similarity": round(sum(p["similarity"] for p in matched_subpart_pairs) / len(matched_subpart_pairs), 3),
                            "common_terms": [t for p in matched_subpart_pairs for t in p["common_terms"]],
                            "detail": f"Whole question duplicate across all subparts between {can_slot_a} and {can_slot_b}",
                        })
                    else:
                        # Subset of subparts matched -> subpart_level!
                        first_match = matched_subpart_pairs[0]
                        findings.append({
                            "slot_a": can_slot_a,
                            "slot_b": can_slot_b,
                            "module_a": can_mod_a,
                            "module_b": can_mod_b,
                            "severity": "subpart_level",
                            "match_type": first_match["match_type"],
                            "subpart_a": first_match["sp_a"],
                            "subpart_b": first_match["sp_b"],
                            "similarity": first_match["similarity"],
                            "common_terms": first_match["common_terms"],
                            "detail": f"Subpart collision between {can_slot_a}({first_match['sp_a']}) and {can_slot_b}({first_match['sp_b']})",
                        })
                    continue

            # 3. Whole question near-duplicate (single-part questions or high Jaccard)
            w1 = set(re.findall(r"\w+", can_t_a.lower()))
            w2 = set(re.findall(r"\w+", can_t_b.lower()))
            jaccard = len(w1 & w2) / max(1, len(w1 | w2))
            common, containment = _specific_overlap(can_t_a, can_t_b)

            if (
                (jaccard >= QUESTION_JACCARD_THRESHOLD or containment >= QUESTION_CONTAINMENT_THRESHOLD)
                and len(common) >= QUESTION_MIN_COMMON_TERMS
            ):
                seen_pairs.add(pair_key)
                findings.append({
                    "slot_a": can_slot_a,
                    "slot_b": can_slot_b,
                    "module_a": can_mod_a,
                    "module_b": can_mod_b,
                    "severity": "question_level",
                    "match_type": "near_duplicate_question",
                    "similarity": round(jaccard, 3),
                    "containment": round(containment, 3),
                    "common_terms": sorted(list(common)),
                    "detail": f"Near-duplicate question (Jaccard={round(jaccard, 3)}) between {can_slot_a} and {can_slot_b}",
                })

    findings.sort(key=lambda f: (f["slot_a"], f["slot_b"]))
    return findings
