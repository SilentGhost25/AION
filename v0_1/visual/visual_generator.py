"""
AION Visual: Visual Question Generator
========================================
Core principle:
  Image is ALWAYS selected BEFORE the LLM runs.
  LLM only generates text — never selects or identifies images.
  Application code attaches the image deterministically.

Fail-safe design:
  Every stage returns None on failure.
  Caller always falls back to text-only question.
"""

from __future__ import annotations

import re
import time
import random
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .figure_card import FigureCard
    from .verifier import VisualVerifier

from .figure_card import FigureCard, VisualFact
from .verifier import VisualVerifier


# -------------------------------------------------------------
# Bloom verb pools
# -------------------------------------------------------------

_BLOOM_VERBS: dict[int, list[str]] = {
    1: ["Identify",   "List",      "Name",        "State",     "Label"],
    2: ["Describe",   "Explain",   "Summarise",   "Interpret", "Outline"],
    3: ["Apply",      "Illustrate","Demonstrate", "Solve",     "Use"],
    4: ["Analyse",    "Compare",   "Differentiate","Examine",  "Contrast"],
    5: ["Evaluate",   "Justify",   "Assess",      "Critique",  "Argue"],
    6: ["Design",     "Construct", "Formulate",   "Develop",   "Propose"],
}

# -------------------------------------------------------------
# Prompt templates — one per visual type
# -------------------------------------------------------------

_PROMPTS: dict[str, str] = {

    "flowchart": """\
A flowchart or process diagram is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require examining the flowchart.
Include the phrase "using the given flowchart" or "with reference to the process diagram".
Output only the question text. No explanation.""",

    "architecture": """\
A system or software architecture diagram is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require examining the architecture diagram.
Include the phrase "with reference to the given diagram" or "using the architecture shown".
Output only the question text. No explanation.""",

    "graph": """\
A graph or chart is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require reading or interpreting the graph.
Include the phrase "from the given graph" or "using the chart provided".
Output only the question text. No explanation.""",

    "circuit": """\
A circuit diagram is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require analysing the circuit diagram.
Include the phrase "using the given circuit" or "with reference to the circuit shown".
Output only the question text. No explanation.""",

    "table": """\
A table or data matrix is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require interpreting the table.
Include the phrase "from the given table" or "using the data shown".
Output only the question text. No explanation.""",

    "equation": """\
A mathematical equation or formula is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require using or deriving the given expression.
Include the phrase "using the given expression" or "with reference to the formula shown".
Output only the question text. No explanation.""",

    "photo": """\
A photograph or diagram is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require examining the provided figure.
Include the phrase "with reference to the given figure" or "using the image shown".
Output only the question text. No explanation.""",

    "default": """\
A figure is provided.

Verified visible facts:
{facts}

Write ONE exam question worth {marks} marks at Bloom Level {bloom}.
Start with: {verb}
The question must require examining the provided figure.
Include the phrase "with reference to the given figure" or "using the figure shown".
Output only the question text. No explanation.""",
}

# -------------------------------------------------------------
# Stop sequences
# -------------------------------------------------------------

_STOP = [
    "Answer:", "answer:", "Ideal Answer", "ideal answer",
    "Marking Scheme", "Note:", "Explanation:",
    "Here is", "Here's", "Q2)", "---", "===", "```",
    "as described", "from the material", "according to",
]

# -------------------------------------------------------------
# Figure-reference pattern — used in verifier
# -------------------------------------------------------------

_FIG_REF = re.compile(
    r"\b(figure|diagram|flowchart|circuit|graph|chart|"
    r"table|image|shown|given|provided|above|refer|"
    r"illustration|drawing|plot)\b",
    re.I,
)


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _clean(text: str) -> str:
    """Strip artefacts from LLM output."""
    t = text.strip()
    t = re.sub(r"\*+", "", t)
    t = re.sub(r"_{2,}", "", t)
    t = re.sub(
        r",?\s*as (described|outlined|mentioned|shown) in the.*",
        "", t, flags=re.I | re.S
    )
    t = re.sub(r",?\s*per the (source|notes|textbook).*", "", t, flags=re.I)
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([.,;:?!])", r"\1", t)
    t = t.strip().rstrip(",:;")
    if t and t[-1] not in ".?!":
        t += "."
    return t


def _has_figure_reference(text: str) -> bool:
    """Check the generated question mentions 'figure', 'diagram', etc."""
    return bool(_FIG_REF.search(text))


def _inject_figure_reference(text: str, visual_type: str) -> str:
    """
    If the model forgot to reference the figure,
    inject a reference at the start.
    Never alters the technical content.
    """
    refs = {
        "flowchart":    "With reference to the given flowchart, ",
        "architecture": "With reference to the given diagram, ",
        "graph":        "Using the graph provided, ",
        "circuit":      "With reference to the given circuit, ",
        "table":        "Using the table provided, ",
        "equation":     "Using the given expression, ",
        "photo":        "With reference to the given figure, ",
        "default":      "With reference to the given figure, ",
    }
    prefix = refs.get(visual_type, "With reference to the given figure, ")
    if text:
        text = text[0].lower() + text[1:]
    return prefix + text


def _build_facts_text(card: FigureCard) -> str:
    """
    Build a clean fact list from a FigureCard.
    Only includes facts with sufficient confidence.
    """
    facts = [
        f for f in card.facts
        if f.confidence >= 0.45 and len(f.text.split()) >= 3
    ]

    if not facts:
        parts = []
        if card.caption:
            parts.append(card.caption)
        if card.ocr_text:
            parts.append(card.ocr_text[:200])
        if card.preceding_text:
            parts.append(card.preceding_text[:200])
        return "\n".join(f"- {p}" for p in parts if p) or "No facts available."

    return "\n".join(f"- {f.text}" for f in facts[:6])


def _call_llm_with_timeout(
    prompt:  str,
    options: dict,
    timeout: int = 60,
) -> str:
    """
    Call LLM in a thread with hard timeout.
    Returns empty string on timeout — never raises.
    """
    from ..llm import get_llm

    result = [""]
    error  = [None]

    def _run():
        try:
            result[0] = get_llm().generate(prompt, options=options)
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        print(f"[VISUAL-GEN] LLM timed out after {timeout}s")
        return ""

    if error[0]:
        print(f"[VISUAL-GEN] LLM error: {error[0]}")
        return ""

    return result[0]


# -------------------------------------------------------------
# Main class
# -------------------------------------------------------------

class VisualQuestionGenerator:
    """
    Generates exam questions that require examining a figure.
    """

    def __init__(
        self,
        verifier:    Optional[VisualVerifier] = None,
        llm_timeout: int = 60,
    ):
        self.verifier    = verifier or VisualVerifier()
        self.llm_timeout = llm_timeout
        self._used_verbs: set[str] = set()

    def generate(
        self,
        card:        FigureCard,
        marks:       int,
        bloom:       int,
        module_id:   str,
        difficulty:  str = "medium",
        diff_manager = None,
    ) -> Optional[dict]:
        """
        Generate one visual question from a verified FigureCard.
        Returns dict on success, None on failure.
        """

        if not card or not card.eligible:
            print(f"[VISUAL-GEN] Card ineligible: {getattr(card, 'skip_reason', '?')}")
            return None

        facts_text = _build_facts_text(card)
        if not facts_text or facts_text == "No facts available.":
            print(f"[VISUAL-GEN] No usable facts for {card.id}")
            return None

        if card.module_id and card.module_id != module_id:
            print(
                f"[VISUAL-GEN] Module mismatch: "
                f"card={card.module_id} question={module_id}"
            )
            return None

        question_text = self._generate_question(
            card       = card,
            facts_text = facts_text,
            marks      = marks,
            bloom      = bloom,
            difficulty = difficulty,
        )

        if not question_text:
            print(f"[VISUAL-GEN] Generation failed for {card.id}")
            return None

        passed, reason = self.verifier.verify(card, question_text, module_id)

        if not passed:
            print(f"[VISUAL-GEN] Verifier fail: {reason} — retrying")
            question_text = self._generate_question(
                card       = card,
                facts_text = facts_text,
                marks      = marks,
                bloom      = bloom,
                difficulty = difficulty,
                retry      = True,
            )
            if question_text:
                passed, reason = self.verifier.verify(
                    card, question_text, module_id
                )

        if not passed:
            print(f"[VISUAL-GEN] Both attempts failed: {reason}")
            return None

        if not _has_figure_reference(question_text):
            question_text = _inject_figure_reference(
                question_text, card.visual_type
            )

        if card.explicit_refs:
            lane = "A"
        elif card.provenance_score >= 0.50:
            lane = "B"
        else:
            lane = "C"

        bloom_names = {
            1: "Remember", 2: "Understand", 3: "Apply",
            4: "Analyse",  5: "Evaluate",   6: "Create",
        }

        print(
            f"[VISUAL-GEN] Verified: {card.id} | "
            f"lane={lane} | "
            f"type={card.visual_type} | "
            f"bloom={bloom} | "
            f"marks={marks} | "
            f"conf={card.provenance_score:.2f}"
        )

        return {
            "text":        question_text,
            "marks":       marks,
            "bloom_level": bloom,
            "bloom_name":  bloom_names.get(bloom, "Understand"),
            "difficulty":  difficulty,
            "lane":        lane,
            "confidence":  card.provenance_score,
            "image": {
                "id":           card.id,
                "url":          card.image_url,
                "caption":      card.caption,
                "visual_type":  card.visual_type,
                "page":         card.page,
                "confidence":   card.provenance_score,
                "facts_used":   len(card.facts),
            },
        }

    def generate_question(
        self,
        card:   FigureCard,
        marks:  int = 10,
        bloom:  int = 4,
    ) -> Optional[dict]:
        """Backward compatibility wrapper method."""
        return self.generate(card, marks, bloom, card.module_id)

    def _generate_question(
        self,
        card:       FigureCard,
        facts_text: str,
        marks:      int,
        bloom:      int,
        difficulty: str,
        retry:      bool = False,
    ) -> Optional[str]:

        template = _PROMPTS.get(
            card.visual_type,
            _PROMPTS["default"]
        )

        verb = self._pick_verb(bloom, retry=retry)

        prompt = template.format(
            facts  = facts_text,
            marks  = marks,
            bloom  = bloom,
            verb   = verb,
        )

        options = {
            "temperature": self._temp_for_difficulty(difficulty, retry),
            "num_predict": self._tokens_for_marks(marks),
            "num_ctx":     2048,
            "top_p":       0.90,
            "stop":        _STOP,
            "seed":        int(time.time() * 1000) % 2**31,
        }

        raw = _call_llm_with_timeout(
            prompt  = prompt,
            options = options,
            timeout = self.llm_timeout,
        )

        if not raw or len(raw.split()) < 6:
            return None

        cleaned = _clean(raw)

        if len(cleaned.split()) < 6:
            return None

        return cleaned

    def _pick_verb(self, bloom: int, retry: bool = False) -> str:
        pool  = _BLOOM_VERBS.get(bloom, ["Explain"])
        fresh = [v for v in pool if v not in self._used_verbs]

        if not fresh:
            self._used_verbs.clear()
            fresh = pool

        if retry:
            candidates = [v for v in fresh if v not in self._used_verbs]
            verb = random.choice(candidates) if candidates else random.choice(pool)
        else:
            verb = random.choice(fresh)

        self._used_verbs.add(verb)

        if len(self._used_verbs) > 12:
            self._used_verbs.clear()

        return verb

    @staticmethod
    def _temp_for_difficulty(difficulty: str, retry: bool = False) -> float:
        base = {
            "easy":   0.45,
            "medium": 0.60,
            "hard":   0.75,
        }.get(difficulty, 0.60)

        return max(0.30, base - 0.10) if retry else base

    @staticmethod
    def _tokens_for_marks(marks: int) -> int:
        if marks <= 4:
            return 80
        elif marks <= 6:
            return 100
        elif marks <= 8:
            return 130
        elif marks <= 10:
            return 150
        else:
            return 180


# -------------------------------------------------------------
# Batch generator — one visual per module
# -------------------------------------------------------------

class ModuleVisualPlanner:
    """
    Plans ONE visual question per module.
    Ensures each figure is used at most once.
    Ensures module isolation.
    """

    def __init__(
        self,
        registry_cards: list[FigureCard],
        generator:      Optional[VisualQuestionGenerator] = None,
    ):
        self.cards     = registry_cards
        self.generator = generator or VisualQuestionGenerator()
        self._used_ids: set[str] = set()

    def plan_for_module(
        self,
        module_id:  str,
        marks:      int,
        bloom:      int,
        difficulty: str = "medium",
    ) -> Optional[dict]:
        candidates = [
            c for c in self.cards
            if c.eligible
            and (not c.module_id or c.module_id == module_id)
            and c.id not in self._used_ids
        ]

        if not candidates:
            print(f"[PLANNER] No candidates for {module_id}")
            return None

        candidates.sort(key=lambda c: c.provenance_score, reverse=True)

        for card in candidates[:3]:
            # Temporarily tag module_id if empty
            if not card.module_id:
                card.module_id = module_id

            result = self.generator.generate(
                card       = card,
                marks      = marks,
                bloom      = bloom,
                module_id  = module_id,
                difficulty = difficulty,
            )

            if result:
                self._used_ids.add(card.id)
                print(
                    f"[PLANNER] Verified: {module_id} -> "
                    f"{card.id} (provenance={card.provenance_score:.2f})"
                )
                return result

        print(f"[PLANNER] All candidates failed for {module_id}")
        return None

    def summary(self) -> dict:
        return {
            "total_cards":   len(self.cards),
            "eligible":      sum(1 for c in self.cards if c.eligible),
            "used":          len(self._used_ids),
            "remaining":     sum(
                1 for c in self.cards
                if c.eligible and c.id not in self._used_ids
            ),
        }
