"""
AION Module: Question Generator
Integrates: difficulty levels, formula inclusion,
            visual questions, custom modelfile.
Max 3 sub-questions per main question.
"""

from __future__ import annotations
from core.generation.marks_partitioner import get_user_split

import os
import re
import random
import time
from typing import Optional, List

from .schemas           import Concept, GeneratedQuestion
from .content_validator import validate_chunk, clean_chunk
from .llm               import get_llm
from .difficulty        import DifficultyManager, DifficultyLevel
from .formula_extractor import find_formulas_in_chunk, format_formula_for_prompt
from .semantic_enforcer import SemanticEnforcer, get_enforcer

_enforcer = get_enforcer(strict=False)  # auto-repair mode

# -------------------------------------------------------------
# VTU Standard Partitions — MAX 3 sub-questions
# -------------------------------------------------------------

# IA = 10 marks total
IA_PARTITIONS = [
    [10],          # 1 sub-question
    [5, 5],        # 2 sub-questions
    [4, 6],        # 2 sub-questions
    [6, 4],        # 2 sub-questions
    [4, 3, 3],     # 3 sub-questions
    [3, 3, 4],     # 3 sub-questions
    [5, 3, 2],     # 3 sub-questions
    [2, 4, 4],     # 3 sub-questions
]

# SEE = 20 marks total
SEE_PARTITIONS = [
    [10, 10],      # 2 sub-questions
    [8, 8, 4],     # 3 sub-questions
    [6, 6, 8],     # 3 sub-questions
    [5, 5, 10],    # 3 sub-questions
    [4, 8, 8],     # 3 sub-questions
    [7, 7, 6],     # 3 sub-questions
    [6, 7, 7],     # 3 sub-questions
    [10, 6, 4],    # 3 sub-questions
    [8, 6, 6],     # 3 sub-questions
    [5, 7, 8],     # 3 sub-questions
]


# -- Bloom Distribution Controller --------------------------------------------
# Ensures a paper has the correct spread of cognitive levels.
# Called by the pipeline orchestrator when selecting Bloom levels per question.

BLOOM_TARGETS_IA = {
    1: 10,   # Remember   — 10% of marks
    2: 20,   # Understand — 20%
    3: 30,   # Apply      — 30%
    4: 20,   # Analyze    — 20%
    5: 15,   # Evaluate   — 15%
    6: 5,    # Create     —  5%
}

BLOOM_TARGETS_SEE = {
    1: 5,
    2: 15,
    3: 30,
    4: 25,
    5: 15,
    6: 10,
}


class BloomDistributionController:
    """
    Tracks Bloom level usage and recommends next level
    so the final paper matches target distribution.
    """

    def __init__(self, exam_type: str = "IA", total_marks: int = 50):
        self.total_marks = total_marks
        self.targets     = BLOOM_TARGETS_IA if exam_type in ("IA", "IAT1", "IAT2") else BLOOM_TARGETS_SEE
        self.used        = {b: 0 for b in range(1, 7)}

    def recommend(self) -> int:
        """Return the Bloom level most needed right now."""
        target_marks = {
            b: round(pct / 100 * self.total_marks)
            for b, pct in self.targets.items()
        }
        deficit = {
            b: target_marks[b] - self.used[b]
            for b in range(1, 7)
        }
        return max(deficit, key=deficit.get)

    def record(self, bloom: int, marks: int):
        """Record that marks were assigned to a bloom level."""
        self.used[bloom] = self.used.get(bloom, 0) + marks

    def summary(self) -> dict:
        total = max(1, sum(self.used.values()))
        return {b: round(m / total * 100) for b, m in self.used.items()}

# -------------------------------------------------------------
# Prompt Templates
# -------------------------------------------------------------

_TEXT_PROMPT = """\
SOURCE MATERIAL:
\"\"\"{chunk}\"\"\"

{formula_section}
TASK: Generate ONE exam question worth {marks} marks.
Bloom Level: {bloom} ({bloom_name})
Difficulty: {difficulty}
Difficulty hint: {difficulty_hint}

Start with the verb: {verb}

Rules:
- Output ONLY the question text
- No answers, notes, or explanations
- No source references or author names
- Include formula/expression in question if provided above

Question:"""


_VISUAL_PROMPT = """\
A figure is provided showing:
{facts}

Visual type: {visual_type}
{formula_section}

TASK: Generate ONE exam question worth {marks} marks \
that REQUIRES examining the provided figure.
Bloom Level: {bloom} ({bloom_name})
Difficulty: {difficulty}

Start with the verb: {verb}
Include the phrase "with reference to the given figure" or \
"using the figure shown" in the question.

Output ONLY the question text.

Question:"""


# -------------------------------------------------------------
# Stop sequences
# -------------------------------------------------------------

_STOP = [
    "Ideal Answer", "ideal answer", "Answer:", "answer:",
    "Marking Scheme", "Note:", "Explanation:",
    "Here is", "Here's", "Q2)", "---", "===", "```",
    "as described in", "from the material", "according to",
]

# -------------------------------------------------------------
# Off-domain guard
# -------------------------------------------------------------

_OFF_DOMAIN = re.compile(
    r"\b(resistor|capacitor|inductor|voltage|watt|"
    r"ampere|ohm|diode|rectifier|lathe|forge)\b",
    re.I,
)


def _is_valid(q: str) -> bool:
    if len(q.split()) < 6 or len(q.split()) > 160:
        return False
    if _OFF_DOMAIN.search(q):
        return False
    try:
        from core.validators.question_quality_firewall import QuestionQualityFirewall
        fw = QuestionQualityFirewall.validate(q)
        if not fw.passed:
            return False
    except Exception:
        pass
    return True


def _clean(text: str) -> str:
    t = text.strip()
    for pat in [
        r"\n.*?Ideal Answer.*", r"\n.*?Note\s*:.*",
        r",?\s*as (described|outlined|mentioned) in the.*",
        r",?\s*per the (source|notes|textbook).*",
        r",?\s*from the (source|notes|material).*",
        # Strip leaked meta-instructions the LLM echoes back
        r"\.?\s*Include the (term|phrase|word)[^.]*in your answer[^.]*",
        r"\.?\s*Use the (term|phrase|word)[^.]*in your answer[^.]*",
        r"\.?\s*Start (your answer |the question )?with the verb[^.]*",
        r"\.?\s*Output ONLY the question text[^.]*",
        r"\.?\s*No answers?,? notes? or explanations?[^.]*",
        r"\.?\s*Begin (your )?(question |answer )?with[^.]{0,60}",
    ]:
        t = re.sub(pat, "", t, flags=re.S | re.I).strip()

    t = re.sub(r"\*+", "", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([.,;:?!])", r"\1", t)
    t = t.strip().rstrip()
    if t and t[-1] not in ".?!":
        t += "."
    return t


def get_bloom_level_name(level: int) -> str:
    return {
        1: "Remember", 2: "Understand", 3: "Apply",
        4: "Analyse",  5: "Evaluate",   6: "Create",
    }.get(level, "Understand")


# -------------------------------------------------------------
# Core question generators
# -------------------------------------------------------------

def get_vtu_vibe_question(
    chunk:       str,
    marks:       int,
    bloom:       int,
    difficulty:  DifficultyLevel = "medium",
    diff_manager: Optional[DifficultyManager] = None,
) -> str:
    """
    Generate a single text-based question.
    Automatically detects and includes formulas from chunk.
    """
    dm   = diff_manager or DifficultyManager.from_string(difficulty)
    verb = dm.get_verb(difficulty, bloom)
    hint = dm.get_hint(difficulty)

    formulas = find_formulas_in_chunk(chunk)
    formula_section = ""
    if formulas:
        best = max(formulas, key=lambda f: len(f.raw))
        formula_section = (
            f"FORMULA/EXPRESSION TO INCLUDE:\n"
            f"{format_formula_for_prompt(best)}\n\n"
        )

    prompt = _TEXT_PROMPT.format(
        chunk            = chunk[:1000],
        formula_section  = formula_section,
        marks            = marks,
        bloom            = bloom,
        bloom_name       = get_bloom_level_name(bloom),
        difficulty       = difficulty.upper(),
        difficulty_hint  = hint,
        verb             = verb,
    )

    options = {
        "temperature": _temp_for_difficulty(difficulty),
        "num_predict": _tokens_for_marks(marks),
        "num_ctx":     2048,
        "top_p":       0.92,
        "stop":        _STOP,
        "seed":        int(time.time() * 1000) % 2**31,
    }

    raw = ""
    try:
        raw = get_llm().generate(prompt, options=options)
    except Exception as e:
        print(f"[GEN] LLM error: {e}")

    if not raw or len(raw.split()) < 5:
        raw = f"{verb} the key principles and applications of the given concept."

    cleaned = _clean(raw)

    try:
        from .qa_engine import QuestionCompletenessChecker, BloomsTaxonomyValidator
        completeness = QuestionCompletenessChecker()
        blooms_val   = BloomsTaxonomyValidator()

        cleaned = completeness.auto_fix_truncation(cleaned)
        cleaned = blooms_val.auto_correct_blooms_level(cleaned, bloom)
    except Exception as e:
        print(f"[GEN] QA inline check warning: {e}")

    return cleaned


def get_visual_question(
    card,                          # FigureCard
    marks:       int,
    bloom:       int,
    difficulty:  DifficultyLevel  = "medium",
    diff_manager: Optional[DifficultyManager] = None,
    chunk:       str = "",         # Source text for formula detection
) -> Optional[str]:
    """
    Generate a question that requires examining a figure.
    Returns None if generation fails verification.
    """
    dm   = diff_manager or DifficultyManager.from_string(difficulty)
    verb = dm.get_verb(difficulty, bloom)

    facts_text = "\n".join(
        f"- {f.text}"
        for f in card.facts
        if f.confidence >= 0.45
    )
    if not facts_text:
        return None

    formula_section = ""
    if chunk:
        formulas = find_formulas_in_chunk(chunk)
        if formulas:
            best = max(formulas, key=lambda f: len(f.raw))
            formula_section = (
                f"RELATED FORMULA:\n"
                f"{format_formula_for_prompt(best)}\n"
            )

    prompt = _VISUAL_PROMPT.format(
        facts           = facts_text,
        visual_type     = card.visual_type,
        formula_section = formula_section,
        marks           = marks,
        bloom           = bloom,
        bloom_name      = get_bloom_level_name(bloom),
        difficulty      = difficulty.upper(),
        verb            = verb,
    )

    options = {
        "temperature": 0.5,
        "num_predict": _tokens_for_marks(marks),
        "num_ctx":     2048,
        "stop":        _STOP,
    }

    raw = ""
    try:
        raw = get_llm().generate(
            prompt,
            system=(
                "You are a VTU exam setter. "
                "Generate questions that require examining the figure."
            ),
            options=options,
        )
    except Exception as e:
        print(f"[GEN-VIS] LLM error: {e}")

    if not raw or len(raw.split()) < 6:
        return None

    cleaned = _clean(raw)

    fig_refs = re.compile(
        r"\b(figure|diagram|given|shown|above|refer|image|chart|graph)\b",
        re.I
    )
    if not fig_refs.search(cleaned):
        cleaned = f"With reference to the given figure, {cleaned[0].lower()}{cleaned[1:]}"

    return cleaned


def _temp_for_difficulty(d: DifficultyLevel) -> float:
    return {"easy": 0.55, "medium": 0.72, "hard": 0.85}.get(d, 0.72)


def _tokens_for_marks(marks: int) -> int:
    """Token budget per question. Never below 200 or questions truncate."""
    if marks <= 3:
        return 300
    elif marks <= 5:
        return 500
    elif marks <= 8:
        return 700
    else:
        return 900


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
    """Full question object generator."""
    chunk = getattr(concept, "content", "") or getattr(concept, "canonical_definition", "")
    q = get_vtu_vibe_question(chunk, 10, 3)
    return GeneratedQuestion(
        concept_id=getattr(concept, "concept_id", "gen_1"),
        ideal_answer=None,
        question_text=q,
        marks=10,
        bloom_level=3
    )

