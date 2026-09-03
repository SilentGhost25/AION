"""
AION Auto-Healer
================
Programmatically fixes common generation failures WITHOUT user intervention.
Called by the orchestrator before retrying a failed slot.

Unlike recovery hints (which just tell the LLM what went wrong),
the auto-healer DIRECTLY MODIFIES the output or slot to fix the problem.

Failure -> Auto-Fix mapping:
- BLOOM_VERB_NOT_AT_START     -> rewrite instruction to start with correct verb
- SIBLING_SIMILARITY          -> force different chunk + topic on retry
- ANSWERABILITY_FAILURE       -> switch to a different evidence chunk
- DISALLOWED_SECONDARY_TASK   -> strip disallowed verbs from instruction
- ANSWER_LEAK                 -> remove answer patterns from question text
- META_LANGUAGE               -> strip source references from text
- INSUFFICIENT_DECLARED_DIMS  -> inject missing dimensions into instruction
- COMPARISON_NOT_DECLARED     -> inject comparison language
- JUSTIFICATION_NOT_DECLARED  -> inject justification language
- CALCULATION_NOT_DECLARED    -> inject calculation verb at start
"""

from __future__ import annotations

import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.generation.output_schema import QuestionOutput
    from core.contracts.question_slot import QuestionSlot


# -- Patterns to strip from question text --------------------------------------
ANSWER_LEAK_PATTERNS = [
    r"answer\s*:.*",
    r"solution\s*:.*",
    r"the answer is.*",
    r"correct answer.*",
    r"model answer.*",
    r"therefore,\s*the correct.*",
    r"hence,\s*the result.*",
    r"thus,\s*we find.*",
    r"thus,\s*the required.*",
    r"hence the output will be.*",
    r"therefore,\s*the output is.*",
]

META_LANGUAGE_PATTERNS = [
    r"from the source[^.]*\.",
    r"from the notes[^.]*\.",
    r"provided notes[^.]*\.",
    r"uploaded document[^.]*\.",
    r"source material[^.]*\.",
    r"based on the provided[^.]*\.",
    r"according to the notes[^.]*\.",
    r"as mentioned in[^.]*\.",
    r"as stated in[^.]*\.",
    r"refer to[^.]*\.",
]

DISALLOWED_VERB_PATTERN = re.compile(
    r"\b(design|create|develop|construct|formulate|propose|invent|build)\b",
    re.IGNORECASE
)

SAFE_REPLACEMENT_VERBS = {
    "design":    "analyze",
    "create":    "explain",
    "develop":   "describe",
    "construct": "determine",
    "formulate": "evaluate",
    "propose":   "justify",
    "invent":    "examine",
    "build":     "calculate",
}


class AutoHealer:
    """
    Programmatic auto-healer for common generation failures.
    All methods are safe — they return the original if healing fails.
    """

    @classmethod
    def heal(
        cls,
        failure_code: str,
        output: "QuestionOutput",
        slot: "QuestionSlot",
        failure_message: str = "",
    ) -> "QuestionOutput":
        """
        Main entry point.
        Applies the appropriate healing strategy for the failure code.
        Returns modified output (or original if healing not applicable).
        """
        try:
            if failure_code == "BLOOM_VERB_NOT_AT_START":
                return cls._fix_bloom_verb_start(output, slot)

            elif failure_code == "DISALLOWED_SECONDARY_TASK":
                return cls._fix_disallowed_verbs(output, slot)

            elif failure_code == "ANSWER_LEAK":
                return cls._fix_answer_leak(output)

            elif failure_code == "META_LANGUAGE":
                return cls._fix_meta_language(output)

            elif failure_code == "COMPARISON_NOT_DECLARED":
                return cls._fix_add_comparison(output, slot)

            elif failure_code == "JUSTIFICATION_NOT_DECLARED":
                return cls._fix_add_justification(output, slot)

            elif failure_code == "CALCULATION_NOT_DECLARED":
                return cls._fix_add_calculation(output, slot)

            elif failure_code == "INSUFFICIENT_DECLARED_DIMENSIONS":
                return cls._fix_add_dimensions(output, slot, failure_message)

            elif failure_code == "SIBLING_SIMILARITY":
                return cls._fix_sibling_similarity(output, slot)

            # Note: DOMAIN_INTEGRITY_VIOLATION requires full LLM regeneration with
            # recovery hint rather than regex truncation which leaves broken grammar.

        except Exception as e:
            print(f"[AUTO-HEALER] Healing failed for {failure_code}: {e}")

        return output  # return original if healing not applicable

    # -- Individual healers ----------------------------------------------------

    @classmethod
    def _fix_bloom_verb_start(cls, output, slot) -> "QuestionOutput":
        """Rewrites instruction and question_text to start with the correct Bloom verb."""
        verb = slot.bloom_verb

        # Fix instruction
        instr = output.instruction.strip()
        words = instr.split()
        if words and words[0].lower().rstrip(".,;:") != verb.lower():
            # Remove first word if it is a wrong verb, prepend correct one
            first = words[0].lower().rstrip(".,;:")
            bloom_verbs = {
                "define", "list", "identify", "state", "explain", "describe",
                "summarize", "discuss", "calculate", "determine", "solve",
                "apply", "analyze", "compare", "differentiate", "examine",
                "evaluate", "justify", "assess", "critique", "design",
                "propose", "formulate", "develop", "name", "recall",
                "construct", "illustrate", "demonstrate",
            }
            if first in bloom_verbs:
                words[0] = verb
                instr = " ".join(words)
            else:
                instr = f"{verb} {instr}"

        output.instruction = instr

        # Fix question_text similarly
        qt = output.question_text.strip()
        qt_words = qt.split()
        if qt_words and qt_words[0].lower().rstrip(".,;:") != verb.lower():
            first_qt = qt_words[0].lower().rstrip(".,;:")
            if first_qt in {"define", "list", "explain", "calculate", "analyze",
                            "evaluate", "design", "name", "describe", "discuss",
                            "construct", "determine", "compare", "justify"}:
                qt_words[0] = verb
                qt = " ".join(qt_words)
            else:
                qt = f"{verb} {qt}"
        output.question_text = qt

        # Sync bloom_level metadata to match the healed verb / slot contract
        if hasattr(output, "bloom_level"):
            from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP
            slot_level = getattr(slot, "bloom_level", None)
            if slot_level:
                output.bloom_level = slot_level
            else:
                v_lower = verb.lower()
                for lvl, verbs in BLOOM_VERB_LEVEL_MAP.items():
                    if v_lower in verbs:
                        output.bloom_level = lvl
                        break

        print(f"[AUTO-HEALER] Fixed BLOOM_VERB_NOT_AT_START -> starts with '{verb}' (level: {getattr(output, 'bloom_level', 'unknown')})")
        return output

    @classmethod
    def _fix_disallowed_verbs(cls, output, slot) -> "QuestionOutput":
        """Replaces disallowed higher-order verbs with bloom-appropriate ones."""
        bloom_verb = slot.bloom_verb.lower()

        def replace_verb(text: str) -> str:
            def _replace(m):
                original = m.group(0).lower()
                replacement = SAFE_REPLACEMENT_VERBS.get(original, bloom_verb)
                # Preserve original capitalisation
                if m.group(0)[0].isupper():
                    return replacement.capitalize()
                return replacement
            return DISALLOWED_VERB_PATTERN.sub(_replace, text)

        # Don't replace the primary bloom verb if it IS one of the disallowed ones
        if slot.bloom_verb.lower() not in SAFE_REPLACEMENT_VERBS:
            output.instruction   = replace_verb(output.instruction)
            output.question_text = replace_verb(output.question_text)

        print(f"[AUTO-HEALER] Fixed DISALLOWED_SECONDARY_TASK — replaced disallowed verbs")
        return output

    @classmethod
    def _fix_answer_leak(cls, output) -> "QuestionOutput":
        """Strips answer/solution leakage patterns from question text."""
        text = output.question_text
        for pattern in ANSWER_LEAK_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)
        output.question_text = text.strip()

        instr = output.instruction
        for pattern in ANSWER_LEAK_PATTERNS:
            instr = re.sub(pattern, "", instr, flags=re.IGNORECASE | re.DOTALL)
        output.instruction = instr.strip()

        print(f"[AUTO-HEALER] Fixed ANSWER_LEAK — stripped answer patterns")
        return output

    @classmethod
    def _fix_meta_language(cls, output) -> "QuestionOutput":
        """Strips source/document reference language from question text."""
        text = output.question_text
        for pattern in META_LANGUAGE_PATTERNS:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        output.question_text = text.strip()

        instr = output.instruction
        for pattern in META_LANGUAGE_PATTERNS:
            instr = re.sub(pattern, "", instr, flags=re.IGNORECASE)
        output.instruction = instr.strip()

        print(f"[AUTO-HEALER] Fixed META_LANGUAGE — stripped source references")
        return output

    @classmethod
    def _fix_add_comparison(cls, output, slot) -> "QuestionOutput":
        """Injects comparison language into instruction if missing."""
        if not any(w in output.instruction.lower() or w in output.question_text.lower() for w in
                   ["compare", "contrast", "difference", "distinguish", "versus",
                    "analyze", "analyse", "examine", "evaluate", "between",
                    "unlike", "whereas", "while", "however", "assess", "justify"]):
            output.instruction = output.instruction.rstrip(".") +                 ", and compare the key differences."
            output.question_text = output.question_text.rstrip(".") +                 ", comparing their key characteristics."
            print(f"[AUTO-HEALER] Fixed COMPARISON_NOT_DECLARED — injected comparison")
        return output

    @classmethod
    def _fix_add_justification(cls, output, slot) -> "QuestionOutput":
        """Injects justification language into instruction if missing."""
        if not any(w in output.instruction.lower() for w in
                   ["justify", "why", "reason", "because", "critique", "evaluate"]):
            output.instruction = output.instruction.rstrip(".") +                 ", and justify your answer with appropriate reasoning."
            output.question_text = output.question_text.rstrip(".") +                 ", justifying your response."
            print(f"[AUTO-HEALER] Fixed JUSTIFICATION_NOT_DECLARED — injected justification")
        return output

    @classmethod
    def _fix_add_calculation(cls, output, slot) -> "QuestionOutput":
        """Ensures a calculation verb starts the instruction for numerical slots."""
        calc_verbs = {"calculate", "compute", "determine", "find", "solve", "derive"}
        first_word = output.instruction.strip().split()[0].lower().rstrip(".,;:") if output.instruction.strip() else ""
        if first_word not in calc_verbs:
            output.instruction   = f"Calculate {output.instruction[0].lower()}{output.instruction[1:]}"
            output.question_text = f"Calculate {output.question_text[0].lower()}{output.question_text[1:]}"
            print(f"[AUTO-HEALER] Fixed CALCULATION_NOT_DECLARED — prepended Calculate")
        return output

    @classmethod
    def _fix_add_dimensions(cls, output, slot, failure_message: str) -> "QuestionOutput":
        """Adds missing analytical dimensions to instruction."""
        # Extract required count from failure message
        import re as _re
        m = _re.search(r"requires at least (\d+) dimensions", failure_message)
        required = int(m.group(1)) if m else 2
        current = output.instruction

        additions = [
            ", explaining the underlying principles",
            ", analyzing the key factors involved",
            ", evaluating the practical implications",
            ", comparing with alternative approaches",
        ]

        # Count existing dimensions (clauses separated by and/or/while)
        existing = len(_re.split(r"\b(?:and|or|while|whereas|as well as)\b", current))
        needed = required - existing

        for i in range(min(needed, len(additions))):
            current = current.rstrip(".") + additions[i]

        output.instruction = current + "."
        print(f"[AUTO-HEALER] Fixed INSUFFICIENT_DECLARED_DIMENSIONS — added {needed} dimensions")
        return output

    @classmethod
    def _fix_sibling_similarity(cls, output, slot) -> "QuestionOutput":
        """
        When a question is too similar to a sibling, inject a topic-shift directive.
        This doesn't change the topic (that requires a new chunk) but forces
        a different angle/framing on the same evidence.
        """
        angles = [
            " Focus on the mathematical derivation rather than conceptual explanation.",
            " Approach this from a system-design perspective.",
            " Focus on the practical applications rather than theoretical aspects.",
            " Emphasize the comparison with alternative methods.",
            " Focus on the limitations and edge cases.",
        ]
        import hashlib
        angle_idx = int(hashlib.md5(slot.slot_id.encode()).hexdigest(), 16) % len(angles)
        directive = angles[angle_idx]

        output.instruction   = output.instruction.rstrip(".") + directive
        output.question_text = output.question_text.rstrip(".") + directive
        print(f"[AUTO-HEALER] Fixed SIBLING_SIMILARITY — injected topic-shift angle")
        return output

