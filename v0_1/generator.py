"""
AION Module: Question Generator
Maturity:    v0.1 — RAG² (REVERSE ASSESSMENT GENERATION) ENGINE
Contract:    concept: Concept -> GeneratedQuestion (see schemas.py)
"""

from __future__ import annotations

import os
import re
import random
import time
from typing import Optional, List, Tuple
from .schemas import Concept, GeneratedQuestion
from .content_validator import validate_chunk, clean_chunk
from .llm import get_llm

# ─────────────────────────────────────────────────────────────
# VTU Standard Partitions (Ensures min >= 4 and max <= 10)
# ─────────────────────────────────────────────────────────────
IA_PARTITIONS = [
    [10],
    [5, 5],
    [4, 6],
    [6, 4]
]

SEE_PARTITIONS = [
    [10, 10],
    [8, 8, 4],
    [6, 6, 8],
    [5, 5, 10],
    [4, 8, 8],
    [5, 5, 5, 5],
    [4, 4, 6, 6]
]

# ─────────────────────────────────────────────────────────────
# Prompt Templates (Stripped of meta references)
# ─────────────────────────────────────────────────────────────
_TURBO_PROMPTS = [
    """\
You are a senior VTU university examiner setting a semester exam paper.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Generate ONE complete, standalone exam question worth {marks} marks at Bloom Level {bloom}.

MANDATORY RULES:
- Start immediately with ONE of these verbs: Explain, Compare, Derive, Analyse, Illustrate, Describe, Define, Discuss, Evaluate, Justify, Design.
- Do NOT use phrases like "as described in the text", "per the notes", "from the source", "in the textbook".
- Do NOT mention any author, researcher, or book names.
- Output ONLY the question text. No answers, no notes, no markdown bold (**).

Question:""",

    """\
You are a VTU exam paper setter.

SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

Write ONE descriptive exam question worth {marks} marks at Bloom Level {bloom} based on the concept.

MANDATORY RULES:
- Start with a strong verb: Differentiate, Compare, Analyse, Discuss, Design, Evaluate.
- Write a clean, self-contained question for an exam answer sheet.
- Do NOT reference any external source, notes, or chapter.
- No meta text, no warnings, no markdown formatting. Output ONLY the raw question.

Question:"""
]

_TURBO_STOP = [
    "Ideal Answer", "Ideal answer", "ideal answer",
    "Marking Scheme", "marking scheme",
    "Note:", "note:", "Answer:", "Explanation:",
    "Here is", "Here's", "here is", "here's",
    "Q2)", "Q2.", "Q3)", "---", "===", "```",
    "as described in", "from the material", "according to the"
]

# ─────────────────────────────────────────────────────────────
# Domain & Quality Guards
# ─────────────────────────────────────────────────────────────
_OFF_DOMAIN_TERMS = re.compile(
    r"\b(transformer|resistor|capacitor|inductor|voltage|current|watt|ampere|"
    r"kva|kw|kwh|ohm|circuit|diode|transistor|rectifier|inverter|lathe|forge)\b",
    re.I
)

def _is_valid_vtu_question(q: str) -> bool:
    """Fast validation to protect against hallucinations and off-domain drift."""
    words = q.split()
    if len(words) < 5 or len(words) > 150:
        return False
    if _OFF_DOMAIN_TERMS.search(q):
        return False
    return True

def _post_clean(text: str) -> str:
    """Post-generation cleanser to ensure pristine formatting."""
    t = text.strip()

    # Cut off answers or notes
    for pat in [r"\n.*?Ideal Answer.*", r"\n.*?Marking Scheme.*", r"\n.*?Note\s*:.*", r"\n.*?Answer\s*:.*", r"\n.*?Explanation\s*:.*", r"\n.*?---.*"]:
        t = re.sub(pat, "", t, flags=re.S|re.I).strip()

    # Remove academic meta-references
    source_refs = [
        r",?\s*as (described|outlined|mentioned|stated|discussed|defined|shown|noted|explained) in the (source|given|provided|above|following)?\s*(material|text|passage|document|textbook|notes|reading|context)",
        r",?\s*per the (source|material|text|notes|textbook|document)",
        r",?\s*from the (source|material|text|notes|textbook|document)",
        r",?\s*\(refer to the (source|material|text|notes|textbook)\)",
        r",?\s*as (described|outlined|shown) above",
        # Strips author names completely
        r",?\s*by\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*",
        r"\b(Russell|Norvig|Knuth|Brooks|Turing|Dijkstra|Cormen|Sedgewick|Balagurusamy|Reema)'?s?\b"
    ]
    for pat in source_refs:
        t = re.sub(pat, "", t, flags=re.I).strip()

    # Strip clean punctuation
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([.,;:?!])", r"\1", t)
    t = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", t)
    t = re.sub(r"\*+", "", t)

    t = t.strip().rstrip(",:;")
    if t and t[-1] not in ".?!":
        t += "."
    return t

# ─────────────────────────────────────────────────────────────
# Dynamic Marks & Bloom Mapper
# ─────────────────────────────────────────────────────────────
def get_bloom_level_name(level: int) -> str:
    return {
        1: "Remember",
        2: "Understand",
        3: "Apply",
        4: "Analyze",
        5: "Evaluate",
        6: "Create"
    }.get(level, "Understand")

BLOOM_VERBS = {
    1: ["Define", "List", "State", "Recall", "Identify", "Name"],
    2: ["Explain", "Describe", "Summarize", "Interpret", "Classify"],
    3: ["Apply", "Illustrate", "Demonstrate", "Solve", "Implement"],
    4: ["Analyze", "Compare", "Differentiate", "Examine", "Breakdown"],
    5: ["Evaluate", "Justify", "Assess", "Critique", "Argue"],
    6: ["Design", "Create", "Develop", "Formulate", "Construct"],
}

def get_vtu_vibe_question(
    chunk:       str,
    marks:       int,
    bloom:       int,
    _used_verbs: Optional[set] = None
) -> str:
    """Generates a single question matching target marks and Bloom level."""
    available_verbs = BLOOM_VERBS.get(bloom, ["Explain"])
    if _used_verbs:
        fresh = [v for v in available_verbs if v not in _used_verbs]
        verb_hint = fresh[0] if fresh else available_verbs[0]
    else:
        verb_hint = random.choice(available_verbs)

    prompt = f"""\
You are a senior VTU university examiner.

SOURCE MATERIAL:
\"\"\"{chunk[:1200]}\"\"\"

Generate ONE exam question worth {marks} marks at Bloom Level {bloom}.

STRICT RULES:
- Start with the verb: {verb_hint}
- Focus on a SPECIFIC concept from the source material above
- Do NOT write a generic or vague question
- Do NOT use phrases like "as described", "from the text", "per the notes"
- Output ONLY the question. No answers, no notes, no markdown.

Question:"""

    options = {
        "temperature": round(random.uniform(0.7, 0.9), 2),
        "num_predict": 120,
        "num_ctx":     2048,
        "top_p":       0.92,
        "stop":        _TURBO_STOP,
        "seed":        int(time.time() * 1000) % 2**31,
    }

    raw = ""
    try:
        raw = get_llm().generate(prompt, options=options)
    except Exception as e:
        print(f"[GENERATOR] LLM error: {e}")

    if not raw or len(raw.split()) < 5:
        raw = f"{verb_hint} the key concepts and principles covered in this module."

    return _post_clean(raw)

def generate_turbo(concept, marks: int = 5) -> GeneratedQuestion:
    """Legacy backward compatibility wrapper."""
    chunk = concept.get("content") if isinstance(concept, dict) else getattr(concept, "content", "")
    q = get_vtu_vibe_question(chunk, marks, 2)
    return GeneratedQuestion(
        concept_id="legacy",
        ideal_answer=None,
        question_text=q,
        marks=marks,
        bloom_level=2
    )

def generate(concept: Concept, mode: str = "balanced") -> GeneratedQuestion:
    """Legacy RAG² generator placeholder."""
    return generate_turbo(concept, marks=10)
