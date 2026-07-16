# AION-Trainer/ese/grammar_validator.py
"""
Grammar Validator — deterministic rules checking grammatical compliance
and rejecting formatting errors or conversational outputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class GrammarIssue:
    rule_name: str
    message: str
    severity: str  # "error" | "warning"


class GrammarValidator:
    def __init__(self):
        pass

    def validate(self, text: str) -> List[GrammarIssue]:
        issues = []
        text_strip = text.strip()

        if not text_strip:
            issues.append(GrammarIssue(
                rule_name="empty_text",
                message="Question text is empty.",
                severity="error"
            ))
            return issues

        # 1. Capitalization check
        if not text_strip[0].isupper():
            issues.append(GrammarIssue(
                rule_name="capitalization",
                message="Question must start with an uppercase letter.",
                severity="error"
            ))

        # 2. Terminal Punctuation
        if not text_strip[-1] in (".", "?", "!"):
            issues.append(GrammarIssue(
                rule_name="punctuation",
                message="Question must end with terminal punctuation (period or question mark).",
                severity="error"
            ))

        # 3. Slang and informal language checks
        informal_words = ["wanna", "gonna", "gotta", "like", "okay", "hey", "hello", "please"]
        words = [w.lower().rstrip(",.?:!\"'") for w in text_strip.split()]
        for idx, w in enumerate(words):
            if w in informal_words:
                issues.append(GrammarIssue(
                    rule_name="slang_detected",
                    message=f"Informal/conversational word '{w}' detected.",
                    severity="error"
                ))

        # 4. LLM Conversational Prompts/Intros
        intro_patterns = [
            r"here is", r"sure", r"i have generated", r"the question is",
            r"draft", r"question \d+:", r"q\d+:"
        ]
        text_lower = text_strip.lower()
        for pattern in intro_patterns:
            if re.match(r"^" + pattern, text_lower):
                issues.append(GrammarIssue(
                    rule_name="conversational_intro",
                    message="Question has a conversational intro/prefix.",
                    severity="error"
                ))

        # 5. Length Check
        word_count = len(text_strip.split())
        if word_count < 4:
            issues.append(GrammarIssue(
                rule_name="too_short",
                message=f"Question is too short ({word_count} words). Min required: 4 words.",
                severity="error"
            ))
        elif word_count > 60:
            issues.append(GrammarIssue(
                rule_name="too_long",
                message=f"Question is too long ({word_count} words). Max recommended: 60 words.",
                severity="warning"
            ))

        return issues
