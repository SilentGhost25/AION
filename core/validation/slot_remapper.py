# core/validation/slot_remapper.py
"""
AION Slot Remapper
==================
Post-generation pass that corrects:
  1. Bloom Level (BL)   — inferred from the first verb of the question
  2. Marks assignment   — enforced by BL + slot position rules
  3. CO assignment      — inferred from topic keywords in question text
  4. CO coverage table  — computed from final assignments

Does NOT modify question text or generation logic.
Runs after orchestrator, before PDF rendering.
"""

from __future__ import annotations

import re
from typing import List, Dict, Tuple, Optional

# ─────────────────────────────────────────────────────────────────────────────
# 1. BLOOM'S VERB → LEVEL MAP
# ─────────────────────────────────────────────────────────────────────────────

# Each verb maps to its canonical Bloom level (1–6)
BLOOM_VERB_MAP: Dict[str, int] = {
    # L1 — Remember
    "define":       1, "list":        1, "recall":      1, "name":        1,
    "state":        1, "identify":    1, "label":       1, "recognize":   1,
    "reproduce":    1, "memorize":    1, "describe":    1, "outline":     1,
    "mention":      1, "write":       1, "what":        1,

    # L2 — Understand
    "explain":      2, "summarize":   2, "paraphrase":  2, "classify":    2,
    "discuss":      2, "interpret":   2, "illustrate":  2, "translate":   2,
    "give":         2, "show":        2, "review":      2,

    # L3 — Apply
    "apply":        3, "calculate":   3, "compute":     3, "solve":       3,
    "use":          3, "demonstrate": 3, "implement":   3, "construct":   3,
    "determine":    3, "find":        3, "plot":        3, "sketch":      3,
    "model":        3, "simulate":    3,

    # L4 — Analyze
    "analyze":      4, "analyse":     4, "compare":     4, "contrast":    4,
    "differentiate":4, "examine":     4, "distinguish": 4, "investigate": 4,
    "break":        4, "relate":      4, "test":        4, "infer":       4,

    # L5 — Evaluate
    "evaluate":     5, "assess":      5, "critique":    5, "judge":       5,
    "justify":      5, "defend":      5, "prioritize":  5, "rate":        5,
    "select":       5, "argue":       5, "formulate":   5,

    # L6 — Create
    "create":       6, "design":      6, "develop":     6, "generate":    6,
    "plan":         6, "produce":     6, "construct":   6, "invent":      6,
    "compose":      6, "build":       6, "devise":      6,
}

BLOOM_LEVEL_LABELS = {
    1: "L1", 2: "L2", 3: "L3",
    4: "L4", 5: "L5", 6: "L6",
}


def infer_bloom_level(question_text: str) -> int:
    """
    Extract the first meaningful verb from the question and map it
    to a Bloom level. Returns the inferred level (1–6).
    Defaults to L2 if no verb is matched.
    """
    # Normalize and extract first sentence
    text = question_text.strip().lower()
    first_sentence = re.split(r"[,\.;]", text)[0]
    words = re.findall(r"\b[a-z]+\b", first_sentence)

    for word in words[:6]:  # only check first 6 words
        if word in BLOOM_VERB_MAP:
            return BLOOM_VERB_MAP[word]

    # Broader scan if first 6 words don't match
    for word in words:
        if word in BLOOM_VERB_MAP:
            return BLOOM_VERB_MAP[word]

    return 2  # safe default: L2 Understand


# ─────────────────────────────────────────────────────────────────────────────
# 2. MARKS RULES
# ─────────────────────────────────────────────────────────────────────────────

# IAT1 structure: Q(a) = 6 marks, Q(b) = 4 marks
# BL expectations per slot:
#   6-mark (a) slot: L1–L3  (knowledge, understanding, basic application)
#   4-mark (b) slot: L4–L6  (analysis, evaluation, creation)

MARKS_FOR_BLOOM: Dict[int, int] = {
    1: 6,   # L1 Remember       → 6 marks (a-slot)
    2: 6,   # L2 Understand     → 6 marks (a-slot)
    3: 6,   # L3 Apply          → 6 marks (a-slot)
    4: 4,   # L4 Analyze        → 4 marks (b-slot)
    5: 4,   # L5 Evaluate       → 4 marks (b-slot)
    6: 4,   # L6 Create         → 4 marks (b-slot)
}

# If BL clashes with slot position, enforce slot rule
# slot_type: "a" → must be 6 marks, "b" → must be 4 marks
SLOT_MARKS: Dict[str, int] = {"a": 6, "b": 4}

# Expected BL range per slot
SLOT_BLOOM_RANGE: Dict[str, Tuple[int, int]] = {
    "a": (1, 3),   # a-slot: L1–L3
    "b": (4, 6),   # b-slot: L4–L6
}


def correct_marks(bloom_level: int, slot_type: str) -> int:
    """
    Return correct marks based on slot type.
    Slot type overrides bloom-inferred marks (slot position is authoritative).
    """
    return SLOT_MARKS.get(slot_type, MARKS_FOR_BLOOM.get(bloom_level, 6))


def correct_bloom_for_slot(bloom_level: int, slot_type: str) -> int:
    """
    If bloom level is wrong for the slot, clamp it to valid range.
    e.g. L5 question in an 'a' slot → clamp to L3
         L1 question in a  'b' slot → clamp to L4
    """
    lo, hi = SLOT_BLOOM_RANGE.get(slot_type, (1, 6))
    return max(lo, min(hi, bloom_level))


# ─────────────────────────────────────────────────────────────────────────────
# 3. CO KEYWORD MAP
# ─────────────────────────────────────────────────────────────────────────────

# Map topic keywords → CO number
# Extend this dict with subject-specific terms
CO_KEYWORD_MAP: Dict[str, int] = {
    # CO1 — Fundamentals / Theory
    "geostationary":    1, "orbit":           1, "kepler":          1,
    "semi-major":       1, "anomaly":         1, "inclination":     1,
    "perigee":          1, "apogee":          1, "velocity":        1,
    "propagation":      1, "delay":           1, "spin":            1,
    "stabiliz":         1, "three-axis":      1, "axis":            1,
    "antenna":          1, "solar panel":     1, "orientation":     1,

    # CO2 — Analysis / Application
    "injection":        2, "circular orbit":  2, "elliptical":      2,
    "ascending node":   2, "true anomaly":    2, "cosmic velocity": 2,
    "landsat":          2, "eclipse":         2, "tracking":        2,
    "battery":          2, "thermal":         2, "radiation":       2,

    # CO3 — Implementation / Systems
    "parking orbit":    3, "sun-synchronous": 3, "remote sensing":  3,
    "dual spinner":     3, "simple spinner":  3, "launcher":        3,
    "transfer orbit":   3, "hohmann":         3, "ground station":  3,
    "pointing":         3, "accuracy":        3,

    # CO4 — Performance / Design
    "tradeoff":         4, "performance":     4, "design":          4,
    "bandwidth":        4, "power budget":    4, "link budget":     4,
    "modulation":       4, "noise":           4, "ber":             4,

    # CO5 — Evaluation / Specification
    "evaluate":         5, "specification":   5, "quality":         5,
    "optimize":         5, "mission":         5, "planning":        5,
}


def infer_co(question_text: str, default_co: int = 1) -> int:
    """
    Scan question text for CO-indicative keywords.
    Returns the CO number with the most keyword hits.
    Falls back to default_co if nothing matches.
    """
    text = question_text.lower()
    co_scores: Dict[int, int] = {}

    for keyword, co_num in CO_KEYWORD_MAP.items():
        if keyword in text:
            co_scores[co_num] = co_scores.get(co_num, 0) + 1

    if not co_scores:
        return default_co

    return max(co_scores, key=lambda k: co_scores[k])


# ─────────────────────────────────────────────────────────────────────────────
# 4. MAIN REMAPPER
# ─────────────────────────────────────────────────────────────────────────────

def remap_slot(
    question_text: str,
    slot_id: str,
    original_marks: int,
    original_bl: str,
    original_co: int,
) -> Dict:
    """
    Given a single question slot, return corrected metadata.

    Args:
        question_text   : The full question string
        slot_id         : e.g. "module_1_Q1_a" or "module_1_Q1_b"
        original_marks  : What the generator assigned
        original_bl     : What the generator assigned e.g. "L1"
        original_co     : What the generator assigned e.g. 1

    Returns dict with keys:
        marks, bloom_level, bloom_label, co, corrections (list of what changed)
    """
    corrections = []

    # Determine slot type from slot_id
    slot_type = "b" if slot_id.endswith("_b") else "a"

    # ── Step 1: Infer BL from question verb ──────────────────────────────────
    inferred_bl = infer_bloom_level(question_text)

    # ── Step 2: Clamp BL to slot expectations ────────────────────────────────
    corrected_bl = correct_bloom_for_slot(inferred_bl, slot_type)
    original_bl_int = int(original_bl.replace("L", "")) if isinstance(original_bl, str) else original_bl

    if corrected_bl != original_bl_int:
        corrections.append(
            f"BL: {BLOOM_LEVEL_LABELS[original_bl_int]} → {BLOOM_LEVEL_LABELS[corrected_bl]}"
            f" (verb-inferred={BLOOM_LEVEL_LABELS[inferred_bl]}, slot={slot_type})"
        )

    # ── Step 3: Correct marks from slot type ─────────────────────────────────
    corrected_marks = correct_marks(corrected_bl, slot_type)
    if corrected_marks != original_marks:
        corrections.append(f"Marks: {original_marks} → {corrected_marks}")

    # ── Step 4: Infer CO from question content ────────────────────────────────
    corrected_co = infer_co(question_text, default_co=original_co)
    if corrected_co != original_co:
        corrections.append(f"CO: CO{original_co} → CO{corrected_co}")

    return {
        "marks":       corrected_marks,
        "bloom_level": corrected_bl,
        "bloom_label": BLOOM_LEVEL_LABELS[corrected_bl],
        "co":          corrected_co,
        "corrections": corrections,
    }


def remap_paper(questions: List[Dict]) -> Tuple[List[Dict], Dict]:
    """
    Remap an entire paper's questions.

    Args:
        questions: List of question dicts, each must have:
            {
              "slot_id":    str,   e.g. "module_1_Q1_a"
              "text":       str,   question text
              "marks":      int,
              "bloom":      str,   e.g. "L1"
              "co":         int,   e.g. 1
            }

    Returns:
        (remapped_questions, co_coverage_table)
        
        co_coverage_table: {1: 40, 2: 30, 3: 30, 4: 0, 5: 0}  (percentages)
    """
    remapped = []
    co_marks_total: Dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    grand_total_marks = 0

    for q in questions:
        result = remap_slot(
            question_text=q["text"],
            slot_id=q["slot_id"],
            original_marks=q["marks"],
            original_bl=q["bloom"],
            original_co=q["co"],
        )

        if result["corrections"]:
            print(f"[REMAPPER] {q['slot_id']}: {' | '.join(result['corrections'])}")

        remapped_q = {**q}  # copy original
        remapped_q["marks"] = result["marks"]
        remapped_q["bloom"] = result["bloom_label"]
        remapped_q["co"]    = result["co"]
        remapped_q["_remap_corrections"] = result["corrections"]
        remapped.append(remapped_q)

        co_marks_total[result["co"]] = (
            co_marks_total.get(result["co"], 0) + result["marks"]
        )
        grand_total_marks += result["marks"]

    # ── CO Coverage Table ─────────────────────────────────────────────────────
    co_coverage: Dict[int, int] = {}
    for co_num in range(1, 6):
        if grand_total_marks > 0:
            pct = round((co_marks_total.get(co_num, 0) / grand_total_marks) * 100)
        else:
            pct = 0
        co_coverage[co_num] = pct

    return remapped, co_coverage
