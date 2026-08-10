"""
AION Question Quality Firewall — Quality & Grammar Rejection Gate
===================================================================
Pre-export firewall enforcing 6 strict quality rules on generated question text:
  1. Incomplete sentence detection
  2. Dangling operator detection
  3. Unbalanced delimiter & math delimiter detection
  4. PDF / binary corruption detection (\ufffd, /FontFile, /ToUnicode)
  5. Bloom grammar sanity check ("create between", "evaluate between")
  6. Prompt / model leakage detection ("Question:", "Answer:", "As an AI:")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple


INCOMPLETE_ENDINGS: Set[str] = {
    "and", "or", "but", "because", "which", "that", "using", "by", "with",
    "from", "for", "to", "of", "in", "on", "as", "such as", "based on",
    "according to", "considering", "including", "between", "than", "where",
}

DANGLING_OPERATORS: Tuple[str, ...] = (
    "=", "+", "-", "*", "/", "×", "÷", "±", "<", ">", "≤", "≥", "\\",
)

HARD_FAILURE_PATTERNS: Tuple[str, ...] = (
    "\ufffd", "/FontFile", "/ToUnicode", "/Contents", "/FlateDecode",
    "endobj", "stream", "xref", "trailer", "%PDF-",
)

INVALID_GRAMMAR_PATTERNS: Tuple[str, ...] = (
    r"\bcreate\s+between\b",
    r"\bevaluate\s+between\b",
    r"\banalyze\s+between\b",
    r"\bapply\s+why\b",
    r"\bdescribe\s+why\s+by\b",
    r"\blist\s+how\b",
    r"\bexplain\s+between\b",
)

PROMPT_LEAKAGE_PATTERNS: Tuple[str, ...] = (
    r"^\s*question\s*:\s*",
    r"^\s*answer\s*:\s*",
    r"^\s*expected\s+answer\s*:\s*",
    r"^\s*explanation\s*:\s*",
    r"^\s*as\s+an\s+ai\b",
    r"^\s*here\s+is\s+the\s+question\b",
    r"^\s*sure\s*,\s*here\b",
)


@dataclass
class FirewallDecision:
    passed       : bool
    code         : str = "OK"
    reason       : str = "Question passed quality firewall."
    repairable   : bool = False
    hard_failure : bool = False
    details      : List[str] = field(default_factory=list)


class QuestionQualityFirewall:
    """Quality & grammar firewall for generated question text."""

    @classmethod
    def validate(cls, text: str) -> FirewallDecision:
        if not text or not text.strip():
            return FirewallDecision(
                passed=False,
                code="EMPTY_QUESTION_TEXT",
                reason="Question text is empty or whitespace.",
                hard_failure=True
            )

        cleaned = text.strip()
        details: List[str] = []

        # Rule 1: PDF / Binary Contamination (Hard Stop)
        for pat in HARD_FAILURE_PATTERNS:
            if pat in cleaned:
                return FirewallDecision(
                    passed=False,
                    code="PDF_BINARY_CONTAMINATION",
                    reason=f"Binary/PDF artifact '{pat}' detected in question text.",
                    hard_failure=True
                )

        # Rule 2: Prompt / Model Leakage
        for pat in PROMPT_LEAKAGE_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                details.append(f"Prompt leakage pattern: '{pat}'")

        if details:
            return FirewallDecision(
                passed=False,
                code="PROMPT_LEAKAGE_DETECTED",
                reason="Prompt/model leakage detected in question text.",
                repairable=True,
                details=details
            )

        # Rule 3: Bloom Grammar Sanity Check
        for pat in INVALID_GRAMMAR_PATTERNS:
            if re.search(pat, cleaned, re.IGNORECASE):
                return FirewallDecision(
                    passed=False,
                    code="INVALID_BLOOM_GRAMMAR",
                    reason=f"Grammatically invalid construction matching '{pat}'.",
                    repairable=True,
                    details=[f"Matched invalid pattern: {pat}"]
                )

        # Rule 4: Dangling Operator
        if cleaned.rstrip(".").endswith(DANGLING_OPERATORS):
            return FirewallDecision(
                passed=False,
                code="DANGLING_OPERATOR",
                reason="Question text ends with a dangling operator or symbol.",
                repairable=True
            )

        # Rule 5: Incomplete Sentence Check
        words = re.sub(r"[^\w\s]", "", cleaned.lower()).split()
        if words and words[-1] in INCOMPLETE_ENDINGS:
            return FirewallDecision(
                passed=False,
                code="INCOMPLETE_SENTENCE",
                reason=f"Question ends in incomplete phrase word '{words[-1]}'.",
                repairable=True
            )

        # Rule 6: Unbalanced Delimiters & Math Delimiters
        delim_ok, delim_msg = cls._check_delimiters(cleaned)
        if not delim_ok:
            return FirewallDecision(
                passed=False,
                code="UNBALANCED_DELIMITERS",
                reason=f"Unbalanced delimiters in question: {delim_msg}",
                repairable=True
            )

        return FirewallDecision(passed=True)

    @classmethod
    def _check_delimiters(cls, text: str) -> Tuple[bool, str]:
        # Check standard brackets
        stack = []
        pairs = {"(": ")", "[": "]", "{": "}"}
        for char in text:
            if char in pairs:
                stack.append(pairs[char])
            elif char in pairs.values():
                if not stack or stack.pop() != char:
                    return False, "Mismatched bracket pair."
        if stack:
            return False, "Unclosed bracket."

        # Check LaTeX delimiters
        if text.count("$") % 2 != 0:
            return False, "Unmatched '$' math delimiter."
        if text.count(r"\(") != text.count(r"\)"):
            return False, r"Unmatched '\(' and '\)' math delimiters."
        if text.count(r"\[") != text.count(r"\]"):
            return False, r"Unmatched '\[' and '\]' math delimiters."

        return True, "OK"
