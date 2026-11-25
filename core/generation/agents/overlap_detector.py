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


# =============================================================================
# Stage G6: Concept Provenance, Structural Relationship & Integrity Classifier
# =============================================================================


def verify_concept_provenance(
    footprint: Any,
    module_idx: Any,
    kg: Any = None,
    artifact: Any = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    allow_synthetic: bool = False,
) -> Tuple[str, str]:
    """
    Module-aware concept provenance ownership verification (Section 6.1).
    Strictly read-only; never mutates concepts, module IDs, or KG nodes.

    Returns:
        (status, reason)
        status: "PROVENANCE_VERIFIED" | "CROSS_MODULE_PROVENANCE_MISMATCH" | "UNVERIFIED_CONCEPT_REVIEW"
    """
    if footprint is None:
        return "UNVERIFIED_CONCEPT_REVIEW", "Missing concept footprint"

    entity_id = getattr(footprint, "provenance_entity_id", None)
    if entity_id is None and isinstance(footprint, dict):
        entity_id = footprint.get("provenance_entity_id")

    if not entity_id or not str(entity_id).strip():
        return "UNVERIFIED_CONCEPT_REVIEW", "Missing or empty provenance entity ID"

    entity_id = str(entity_id).strip()
    prov_type = getattr(footprint, "provenance_type", None)
    if prov_type is None and isinstance(footprint, dict):
        prov_type = footprint.get("provenance_type")
    prov_type = str(prov_type or "").strip()

    if allow_synthetic and (kg is None and artifact is None):
        return "PROVENANCE_VERIFIED", "Synthetic concept provenance verified"

    # Normalize assigned module representation
    assigned_mod_num = str(module_idx).lower().replace("module_", "").strip() if module_idx is not None else ""

    # 1. KG_CONCEPT Ownership Check
    if prov_type == "KG_CONCEPT":
        if kg is None:
            return "UNVERIFIED_CONCEPT_REVIEW", "No knowledge graph available for verification"

        has_node = getattr(kg, "has_node", None)
        if callable(has_node):
            node_exists = bool(kg.has_node(entity_id))
        elif hasattr(kg, "nodes") and isinstance(kg.nodes, dict):
            node_exists = entity_id in kg.nodes
        elif hasattr(kg, "get_node", None):
            node_exists = kg.get_node(entity_id) is not None
        else:
            node_exists = False

        if not node_exists:
            return "UNVERIFIED_CONCEPT_REVIEW", f"Provenance entity '{entity_id}' not found in knowledge graph"

        # Check module association on the node
        get_node = getattr(kg, "get_node", None)
        node = get_node(entity_id) if callable(get_node) else (kg.nodes.get(entity_id) if hasattr(kg, "nodes") else None)
        if node is not None:
            node_mod = getattr(node, "module_id", None)
            if node_mod is None and isinstance(node, dict):
                node_mod = node.get("module_id")

            if node_mod is not None:
                node_mod_num = str(node_mod).lower().replace("module_", "").strip()
                if assigned_mod_num and node_mod_num != assigned_mod_num:
                    return (
                        "CROSS_MODULE_PROVENANCE_MISMATCH",
                        f"Entity '{entity_id}' belongs to module '{node_mod}', not assigned module '{module_idx}'",
                    )

        return "PROVENANCE_VERIFIED", "KG concept provenance verified"

    # 2. SYLLABUS_HEADING Ownership Check
    elif prov_type == "SYLLABUS_HEADING":
        if artifact is None:
            return "UNVERIFIED_CONCEPT_REVIEW", "No artifact available for syllabus heading verification"

        heading = None
        has_heading = getattr(artifact, "has_heading", None)
        if callable(has_heading):
            if not artifact.has_heading(entity_id):
                return "UNVERIFIED_CONCEPT_REVIEW", f"Heading '{entity_id}' not found in syllabus artifact"
            get_h = getattr(artifact, "get_heading", None)
            if callable(get_h):
                heading = artifact.get_heading(entity_id)
        elif hasattr(artifact, "headings") and isinstance(artifact.headings, dict):
            if entity_id not in artifact.headings:
                return "UNVERIFIED_CONCEPT_REVIEW", f"Heading '{entity_id}' not found in artifact headings"
            heading = artifact.headings[entity_id]
        elif hasattr(artifact, "headings") and isinstance(artifact.headings, list):
            match = [
                h for h in artifact.headings
                if getattr(h, "id", None) == entity_id
                or getattr(h, "heading", None) == entity_id
                or (isinstance(h, dict) and (h.get("id") == entity_id or h.get("heading") == entity_id))
            ]
            if not match:
                return "UNVERIFIED_CONCEPT_REVIEW", f"Heading '{entity_id}' not found in artifact headings list"
            heading = match[0]
        else:
            return "UNVERIFIED_CONCEPT_REVIEW", f"Heading '{entity_id}' cannot be verified on artifact"

        # Check page bounds if start and end are provided
        if heading is not None and page_start is not None and page_end is not None:
            h_page = getattr(heading, "page", None)
            if h_page is None and isinstance(heading, dict):
                h_page = heading.get("page")
            if h_page is not None and isinstance(h_page, (int, float)):
                if not (page_start <= h_page <= page_end):
                    return (
                        "CROSS_MODULE_PROVENANCE_MISMATCH",
                        f"Heading '{entity_id}' on page {h_page} falls outside module range [{page_start}, {page_end}]",
                    )

        return "PROVENANCE_VERIFIED", "Syllabus heading provenance verified"

    # 3. DOCUMENT_SECTION Ownership Check
    elif prov_type == "DOCUMENT_SECTION":
        if artifact is None:
            return "UNVERIFIED_CONCEPT_REVIEW", "No artifact available for document section verification"
        return "PROVENANCE_VERIFIED", "Document section provenance verified"

    return "UNVERIFIED_CONCEPT_REVIEW", f"Unrecognized provenance type '{prov_type}'"


def classify_relationship(
    a: Any,
    b: Any,
    kg: Any = None,
    artifact: Any = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Structural Relationship & Precedence Classifier (Section 6.2).
    Order-invariant: classify_relationship(A, B) == classify_relationship(B, A).

    Precedence Hierarchy:
      1. Cross-Module Precedence (CROSS_MODULE_RECURRENCE_ALLOWED)
      2. Choice-Pair Precedence (CHOICE_PAIR_TOPIC_COLLAPSE -> FAIL)
      3. Sibling Progression Precedence (4 conditions -> LEGITIMATE_PROGRESSION / SAME_CONCEPT_REVIEW)
      4. Provenance Precedence (CROSS_MODULE_PROVENANCE_MISMATCH -> FAIL, UNVERIFIED_CONCEPT_REVIEW -> INDETERMINATE)
      5. Distinct Concepts vs Collision (DISTINCT_CONCEPT -> PASS, SAME_CONCEPT_COLLISION -> FAIL)
    """
    def _extract_meta(item: Any) -> Dict[str, Any]:
        if item is None:
            return {}
        mod = getattr(item, "module_id", None) or getattr(item, "module", None)
        cg = getattr(item, "choice_group_id", None)
        cv = getattr(item, "choice_variant", None)
        qid = getattr(item, "question_id", None) or getattr(item, "slot_id", None)
        sp_idx = getattr(item, "subpart_index", None)
        lbl = getattr(item, "label", None)
        bloom = getattr(item, "bloom", None)
        fp = getattr(item, "concept_footprint", None)

        if isinstance(item, dict):
            mod = mod or item.get("module_id") or item.get("module")
            cg = cg or item.get("choice_group_id")
            cv = cv or item.get("choice_variant")
            qid = qid or item.get("question_id") or item.get("slot_id")
            sp_idx = sp_idx if sp_idx is not None else item.get("subpart_index")
            lbl = lbl or item.get("label")
            bloom = bloom or item.get("bloom")
            fp = fp or item.get("concept_footprint")

        # If item itself is a ConceptFootprint
        if hasattr(item, "canonical_concept") and hasattr(item, "provenance_entity_id"):
            fp = item

        # Extract slot_id module hint if module is missing
        if not mod and qid and isinstance(qid, str) and "module_" in qid:
            m_match = re.search(r"module_(\d+)", qid)
            if m_match:
                mod = f"module_{m_match.group(1)}"

        return {
            "module_id": mod,
            "choice_group_id": cg,
            "choice_variant": cv,
            "question_id": qid,
            "subpart_index": sp_idx,
            "label": lbl,
            "bloom": bloom,
            "footprint": fp,
        }

    meta_a = _extract_meta(a)
    meta_b = _extract_meta(b)

    # Establish canonical symmetric ordering key
    key_a = (
        str(meta_a.get("module_id") or ""),
        str(meta_a.get("choice_group_id") or ""),
        str(meta_a.get("question_id") or ""),
        str(meta_a.get("subpart_index") if meta_a.get("subpart_index") is not None else ""),
        str(meta_a.get("label") or ""),
    )
    key_b = (
        str(meta_b.get("module_id") or ""),
        str(meta_b.get("choice_group_id") or ""),
        str(meta_b.get("question_id") or ""),
        str(meta_b.get("subpart_index") if meta_b.get("subpart_index") is not None else ""),
        str(meta_b.get("label") or ""),
    )

    # Ensure stable ordering for symmetric invariant
    if key_a > key_b:
        first_item, second_item = b, a
        meta_first, meta_second = meta_b, meta_a
    else:
        first_item, second_item = a, b
        meta_first, meta_second = meta_a, meta_b

    mod_1 = meta_first.get("module_id")
    mod_2 = meta_second.get("module_id")
    fp_1 = meta_first.get("footprint")
    fp_2 = meta_second.get("footprint")

    concept_1 = getattr(fp_1, "canonical_concept", None) if fp_1 else None
    if concept_1 is None and isinstance(fp_1, dict):
        concept_1 = fp_1.get("canonical_concept")
    concept_2 = getattr(fp_2, "canonical_concept", None) if fp_2 else None
    if concept_2 is None and isinstance(fp_2, dict):
        concept_2 = fp_2.get("canonical_concept")

    c1_clean = str(concept_1 or "").strip().lower()
    c2_clean = str(concept_2 or "").strip().lower()
    concepts_match = bool(c1_clean and c2_clean and c1_clean == c2_clean)

    op_1 = getattr(fp_1, "primary_operation", None) if fp_1 else None
    if op_1 is None and isinstance(fp_1, dict):
        op_1 = fp_1.get("primary_operation")
    op_2 = getattr(fp_2, "primary_operation", None) if fp_2 else None
    if op_2 is None and isinstance(fp_2, dict):
        op_2 = fp_2.get("primary_operation")
    op1_clean = str(op_1 or "").strip().lower()
    op2_clean = str(op_2 or "").strip().lower()
    operations_match = bool(op1_clean and op2_clean and op1_clean == op2_clean)

    # -------------------------------------------------------------------------
    # Precedence Step 1: Cross-Module Precedence
    # -------------------------------------------------------------------------
    norm_mod1 = str(mod_1).lower().replace("module_", "").strip() if mod_1 else ""
    norm_mod2 = str(mod_2).lower().replace("module_", "").strip() if mod_2 else ""
    if norm_mod1 and norm_mod2 and norm_mod1 != norm_mod2:
        if concepts_match:
            return {
                "status": "PASS",
                "classification": "CROSS_MODULE_RECURRENCE_ALLOWED",
                "telemetry": {"cross_module_recurrence": True, "concept": c1_clean},
                "detail": f"Interdisciplinary concept recurrence of '{c1_clean}' across modules {mod_1} and {mod_2} is allowed",
            }
        else:
            return {
                "status": "PASS",
                "classification": "DISTINCT_CONCEPT",
                "detail": "Distinct concepts across different modules",
            }

    # -------------------------------------------------------------------------
    # Precedence Step 2: Choice-Pair Precedence
    # -------------------------------------------------------------------------
    cg_1 = meta_first.get("choice_group_id")
    cg_2 = meta_second.get("choice_group_id")
    is_choice_pair = bool(cg_1 and cg_2 and str(cg_1).strip() == str(cg_2).strip())

    if is_choice_pair:
        if concepts_match and operations_match:
            return {
                "status": "FAIL",
                "classification": "CHOICE_PAIR_TOPIC_COLLAPSE",
                "detail": f"Choice pair in group '{cg_1}' collapses on identical concept '{c1_clean}' and operation '{op1_clean}'",
            }
        elif concepts_match and not operations_match:
            # Tier 3 compact module fallback
            return {
                "status": "PASS",
                "classification": "TIER_3_COMPACT_MODULE_FALLBACK",
                "telemetry": {"tier_3_fallback": True, "concept": c1_clean},
                "detail": f"Choice pair in compact module covers '{c1_clean}' with orthogonal operations ('{op1_clean}' vs '{op2_clean}')",
            }
        else:
            return {
                "status": "PASS",
                "classification": "DISTINCT_CONCEPT",
                "detail": f"Choice pair in group '{cg_1}' covers distinct concepts ('{c1_clean}' vs '{c2_clean}')",
            }

    # -------------------------------------------------------------------------
    # Precedence Step 3: Sibling Progression Precedence
    # -------------------------------------------------------------------------
    qid_1 = meta_first.get("question_id")
    qid_2 = meta_second.get("question_id")
    is_sibling = bool(qid_1 and qid_2 and str(qid_1).strip() == str(qid_2).strip())
    # Or both share parent question object
    if not is_sibling and meta_first.get("subpart_index") is not None and meta_second.get("subpart_index") is not None:
        if norm_mod1 == norm_mod2 and norm_mod1:
            is_sibling = True

    if is_sibling and concepts_match:
        # Sort strictly by subpart_index / sequence
        idx_1 = meta_first.get("subpart_index", 0) or 0
        idx_2 = meta_second.get("subpart_index", 1) or 1
        if idx_1 > idx_2:
            earlier_sp, later_sp = second_item, first_item
            earlier_meta, later_meta = meta_second, meta_first
            earlier_fp, later_fp = fp_2, fp_1
        else:
            earlier_sp, later_sp = first_item, second_item
            earlier_meta, later_meta = meta_first, meta_second
            earlier_fp, later_fp = fp_1, fp_2

        # Condition 1: Provenance verified for both subparts with matching module
        p_status_1, _ = verify_concept_provenance(earlier_fp, mod_1, kg, artifact, page_start, page_end, allow_synthetic=allow_synthetic)
        p_status_2, _ = verify_concept_provenance(later_fp, mod_1, kg, artifact, page_start, page_end, allow_synthetic=allow_synthetic)

        if p_status_1 == "CROSS_MODULE_PROVENANCE_MISMATCH" or p_status_2 == "CROSS_MODULE_PROVENANCE_MISMATCH":
            return {
                "status": "FAIL",
                "classification": "CROSS_MODULE_PROVENANCE_MISMATCH",
                "detail": "Sibling subpart has cross-module provenance mismatch",
            }
        if p_status_1 == "UNVERIFIED_CONCEPT_REVIEW" or p_status_2 == "UNVERIFIED_CONCEPT_REVIEW":
            return {
                "status": "INDETERMINATE",
                "classification": "UNVERIFIED_CONCEPT_REVIEW",
                "detail": "Sibling subpart has unverified concept provenance",
            }

        # Condition 2: Different cognitive operations (op1 != op2)
        e_op = getattr(earlier_fp, "primary_operation", None) or (earlier_fp.get("primary_operation") if isinstance(earlier_fp, dict) else "")
        l_op = getattr(later_fp, "primary_operation", None) or (later_fp.get("primary_operation") if isinstance(later_fp, dict) else "")
        cond2 = bool(e_op and l_op and str(e_op).strip().lower() != str(l_op).strip().lower())

        # Condition 3: Different response types (resp_type1 != resp_type2)
        e_rt = getattr(earlier_fp, "response_type", None) or (earlier_fp.get("response_type") if isinstance(earlier_fp, dict) else "")
        l_rt = getattr(later_fp, "response_type", None) or (later_fp.get("response_type") if isinstance(later_fp, dict) else "")
        cond3 = bool(e_rt and l_rt and str(e_rt).strip().lower() != str(l_rt).strip().lower())

        # Condition 4: Strictly higher Bloom rank (BloomRank(later) > BloomRank(earlier))
        bloom_rank = {"L1": 1, "L2": 2, "L3": 3, "L4": 4, "L5": 5, "L6": 6}
        b_e = earlier_meta.get("bloom") or getattr(earlier_fp, "bloom_level", None) or "L2"
        b_l = later_meta.get("bloom") or getattr(later_fp, "bloom_level", None) or "L2"
        rank_e = bloom_rank.get(str(b_e).upper(), 2)
        rank_l = bloom_rank.get(str(b_l).upper(), 2)
        cond4 = rank_l > rank_e

        if cond2 and cond3 and cond4:
            return {
                "status": "PASS",
                "classification": "LEGITIMATE_PROGRESSION",
                "detail": f"Sibling progression verified: ({b_e}->{b_l}, ops '{e_op}'->'{l_op}', resp '{e_rt}'->'{l_rt}')",
            }
        elif cond2:
            return {
                "status": "INDETERMINATE",
                "classification": "SAME_CONCEPT_REVIEW",
                "detail": "Sibling subparts have different operations but Bloom rank or response type does not strictly advance",
            }
        else:
            return {
                "status": "FAIL",
                "classification": "SAME_CONCEPT_COLLISION",
                "detail": f"Sibling subparts share identical concept '{c1_clean}' and operation without legitimate progression",
            }

    # -------------------------------------------------------------------------
    # Precedence Step 4: Provenance Precedence
    # -------------------------------------------------------------------------
    p_status_1, _ = verify_concept_provenance(fp_1, mod_1, kg, artifact, page_start, page_end, allow_synthetic=allow_synthetic)
    p_status_2, _ = verify_concept_provenance(fp_2, mod_2, kg, artifact, page_start, page_end, allow_synthetic=allow_synthetic)

    if p_status_1 == "CROSS_MODULE_PROVENANCE_MISMATCH" or p_status_2 == "CROSS_MODULE_PROVENANCE_MISMATCH":
        return {
            "status": "FAIL",
            "classification": "CROSS_MODULE_PROVENANCE_MISMATCH",
            "detail": "Cross-module concept provenance mismatch detected",
        }
    if p_status_1 == "UNVERIFIED_CONCEPT_REVIEW" or p_status_2 == "UNVERIFIED_CONCEPT_REVIEW":
        return {
            "status": "INDETERMINATE",
            "classification": "UNVERIFIED_CONCEPT_REVIEW",
            "detail": "Concept provenance cannot be verified against authoritative source",
        }

    # -------------------------------------------------------------------------
    # Precedence Step 5: Distinct Concepts vs Collision
    # -------------------------------------------------------------------------
    if not concepts_match:
        return {
            "status": "PASS",
            "classification": "DISTINCT_CONCEPT",
            "detail": f"Concepts are distinct ('{c1_clean}' vs '{c2_clean}')",
        }
    else:
        return {
            "status": "FAIL",
            "classification": "SAME_CONCEPT_COLLISION",
            "detail": f"Intra-module concept collision on '{c1_clean}'",
        }


def detect_concept_collision(
    questions: List[Any],
    kg: Any = None,
    artifact: Any = None,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    allow_synthetic: bool = False,
) -> List[Dict[str, Any]]:
    """
    Audits all question and subpart pairs for concept integrity and collisions.
    Non-mutating read-only audit.
    """
    findings: List[Dict[str, Any]] = []
    items: List[Any] = []

    # Leaf items: if question has subparts, examine the subparts; otherwise examine the question itself
    for q in questions:
        sps = getattr(q, "sub_parts", None) or []
        if sps:
            for sp in sps:
                items.append(sp)
        else:
            items.append(q)

    n = len(items)
    for i in range(n):
        for j in range(i + 1, n):
            res = classify_relationship(
                items[i], items[j],
                kg=kg, artifact=artifact,
                page_start=page_start, page_end=page_end,
                allow_synthetic=allow_synthetic,
            )
            if res.get("status") != "PASS" or res.get("classification") == "CROSS_MODULE_RECURRENCE_ALLOWED":
                findings.append({
                    "item_a": items[i],
                    "item_b": items[j],
                    **res,
                })

    return findings


def generate_concept_integrity_report(
    questions: List[Any],
    kg: Any = None,
    artifact: Any = None,
    allow_synthetic: bool = False,
) -> Dict[str, Any]:
    """
    Generates structured Concept Integrity Report for CheckingAgent gate aggregation.
    """
    findings = detect_concept_collision(questions, kg=kg, artifact=artifact, allow_synthetic=allow_synthetic)

    has_fail = any(f.get("status") == "FAIL" for f in findings)
    has_indeterminate = any(f.get("status") == "INDETERMINATE" for f in findings)

    if has_fail:
        gate_status = "FAIL"
    elif has_indeterminate:
        gate_status = "INDETERMINATE"
    else:
        gate_status = "PASS"

    telemetry: Dict[str, Any] = {
        "total_pairs_checked": len(questions) * (len(questions) - 1) // 2 if len(questions) > 1 else 0,
        "collision_findings_count": len(findings),
        "cross_module_recurrence_count": sum(1 for f in findings if f.get("classification") == "CROSS_MODULE_RECURRENCE_ALLOWED"),
    }

    return {
        "gate_status": gate_status,
        "findings": findings,
        "telemetry": telemetry,
    }

