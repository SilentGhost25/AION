# AION-Trainer/ese/language_realizer.py
"""
Language Realizer — Step 6 of the ESE.

Responsible for neural language generation and post-processing.
Cleans up questions, ensures academic grammar, removes formatting artifacts,
and formats them for examination presentation.
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

logger = logging.getLogger("aion.ese.realizer")


class LanguageRealizer:
    def __init__(self, llm_client=None):
        self.llm = llm_client

    def realize(
        self,
        question_text: str,
        bloom_level: str,
        marks: int,
    ) -> str:
        text = question_text.strip()
        if not text:
            return ""

        # 1. Apply deterministic cleanups first
        text = self._basic_cleanup(text)

        # 2. Neural realization if LLM client is available
        if self.llm:
            try:
                refined = self._neural_refine(text, bloom_level, marks)
                if refined and len(refined.split()) >= 4:
                    text = self._basic_cleanup(refined)
            except Exception as e:
                logger.error(f"[LanguageRealizer] Neural refinement failed: {e}. Falling back to basic cleanup.")

        # 3. Ensure proper final punctuation based on the type of question
        text = self._enforce_punctuation(text)

        # 4. Standardise marks tag (e.g. adding (10 Marks) or similar)
        # We keep the core question text separate and append marks at the ESE presentation level if needed.
        return text

    def _basic_cleanup(self, text: str) -> str:
        # Remove markdown symbols
        text = text.replace("**", "").replace("*", "").replace("`", "")
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove duplicate punctuation
        text = re.sub(r"\?+", "?", text)
        text = re.sub(r"\.+", ".", text)
        # Capitalise first letter
        if text:
            text = text[0].upper() + text[1:]
        return text.strip()

    def _enforce_punctuation(self, text: str) -> str:
        if not text:
            return text
        
        # If starts with question word, must end with question mark
        question_words = ["what", "why", "how", "when", "where", "which", "who", "differentiate", "distinguish"]
        first_word = text.split()[0].lower().rstrip(",:.")
        
        if first_word in question_words:
            if not text.endswith("?"):
                text = text.rstrip(".") + "?"
        else:
            if not text.endswith(".") and not text.endswith("?"):
                text = text + "."
        return text

    def _neural_refine(self, text: str, bloom_level: str, marks: int) -> str:
        prompt = (
            f"You are a proofreader for university examination papers.\n"
            f"Your job is to rewrite the draft question below into a professional, clear, "
            f"and grammatically perfect VTU university exam question.\n\n"
            f"Draft Question: \"{text}\"\n"
            f"Bloom Cognitive Level: {bloom_level}\n"
            f"Marks: {marks}\n\n"
            f"Instructions:\n"
            f"1. Fix any spelling or grammatical errors.\n"
            f"2. Ensure it starts with a proper uppercase action verb (e.g., Explain, Describe, Define, Differentiate).\n"
            f"3. Do not change the underlying meaning or the concept being assessed.\n"
            f"4. Do not include any introductory text or instructions — output ONLY the final corrected question.\n\n"
            f"Refined Question:"
        )
        refined = self.llm.generate(prompt, temperature=0.2, max_tokens=100)
        return refined.strip().strip('"')
