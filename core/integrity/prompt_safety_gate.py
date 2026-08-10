"""
AION Core Integrity — Prompt Safety Gate
==========================================
Scans generated question output and source evidence for prompt injection attempts,
meta-instruction leakage, format artifacts, system prompt leakage, and foreign language instruction leaks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class SafetyReport:
    status   : str          # "CLEAN" | "INJECTION_DETECTED"
    patterns : List[str] = field(default_factory=list)
    action   : str = "PASS" # "PASS" | "REJECT_AND_REGENERATE" | "QUARANTINE"


# ── INJECTION PATTERNS ────────────────────────────────────────────────────────

PROMPT_INJECTION_PATTERNS = [

    # ── META-INSTRUCTION PATTERNS ─────────────────────────────────────────────
    r"(?i)ignore previous instructions",
    r"(?i)ignore all previous",
    r"(?i)disregard (?:the )?(?:above|previous|prior)",
    r"(?i)you are (?:now )?(?:a |an )?(?:AI|GPT|ChatGPT|language model)",
    r"(?i)act as (?:a |an )?(?:AI|GPT|assistant|language model)",
    r"(?i)system prompt",
    r"(?i)your (?:new )?(?:role|task|purpose|instruction) is",
    r"(?i)from now on",
    r"(?i)respond only (?:with|in)",
    r"(?i)do not (?:include|add|mention|write)",
    r"(?i)provide only",
    r"(?i)never (?:include|add|mention|say)",

    # ── TEMPLATE/FORMAT LEAKAGE ───────────────────────────────────────────────
    r"(?i)^question:\s",
    r"(?i)^answer:\s",
    r"(?i)^instruction:\s",
    r"(?i)^prompt:\s",
    r"(?i)^context:\s",
    r"(?i)^system:\s",
    r"(?i)\[INST\]",
    r"(?i)<\|im_start\|>",
    r"(?i)<\|im_end\|>",
    r"(?i)<<SYS>>",
    r"\[/INST\]",

    # ── GENERATION ARTIFACT LEAKAGE ───────────────────────────────────────────
    r"(?i)generate (?:a |an )?question",
    r"(?i)write (?:a |an )?question",
    r"(?i)create (?:a |an )?question for",
    r"(?i)question should be",
    r"(?i)the question (?:must|should|will) (?:test|assess|evaluate)",
    r"(?i)bloom's taxonomy",
    r"(?i)learning outcome",

    # ── FOREIGN LANGUAGE META-INSTRUCTIONS ───────────────────────────────────
    # Lithuanian (from spec example)
    r"(?i)turi būti tik klausimas",
    r"(?i)be jokių atsakymų",
    # Common instruction-pattern prefixes in non-English prompts
    r"(?i)jame turi",
    r"(?i)naudojant",
]


class PromptSafetyGate:
    """Scans generated output and source evidence for prompt injections and leaks."""

    @classmethod
    def scan(cls, text: str) -> SafetyReport:
        if not text or not text.strip():
            return SafetyReport(status="CLEAN", patterns=[], action="PASS")

        triggered: List[str] = []

        # STEP 1: PATTERN MATCHING
        for pattern in PROMPT_INJECTION_PATTERNS:
            if re.search(pattern, text, re.MULTILINE):
                # Format friendly pattern name
                clean_name = pattern.replace("(?i)", "").replace("\\s", " ").strip("^$")
                triggered.append(f"PATTERN:{clean_name[:40]}")

        # STEP 2: STRUCTURAL ANOMALY CHECK
        if "Question:" in text and text.index("Question:") > 50:
            triggered.append("MID_TEXT_QUESTION_LABEL")

        if text.count("\n\n") > 5:
            triggered.append("EXCESS_BLANK_LINES")

        # STEP 3: VERDICT
        if len(triggered) > 0:
            return SafetyReport(
                status   = "INJECTION_DETECTED",
                patterns = triggered,
                action   = "REJECT_AND_REGENERATE"
            )

        return SafetyReport(status="CLEAN", patterns=[], action="PASS")

    @classmethod
    def scan_source(cls, text: str) -> SafetyReport:
        """Scan raw source evidence chunk before feeding to Qwen."""
        report = cls.scan(text)
        if report.status == "INJECTION_DETECTED":
            report.action = "QUARANTINE"
        return report
