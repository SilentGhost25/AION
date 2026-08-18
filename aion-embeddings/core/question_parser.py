# core/question_parser.py

import re
import hashlib
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Bloom's taxonomy keywords for auto-detection
BLOOM_KEYWORDS = {
    "remember": [
        "define", "list", "state", "name", "identify", "recall",
        "recognize", "label", "match", "select", "what is", "who",
        "when", "where", "enumerate", "mention", "write"
    ],
    "understand": [
        "explain", "describe", "discuss", "summarize", "interpret",
        "classify", "compare", "contrast", "distinguish", "illustrate",
        "paraphrase", "differentiate", "elaborate"
    ],
    "apply": [
        "apply", "solve", "calculate", "compute", "determine",
        "demonstrate", "use", "implement", "execute", "find",
        "show that", "construct", "draw", "sketch", "plot"
    ],
    "analyze": [
        "analyze", "examine", "investigate", "break down", "categorize",
        "compare and contrast", "differentiate", "distinguish between",
        "relate", "why", "how does", "what causes"
    ],
    "evaluate": [
        "evaluate", "justify", "assess", "argue", "critique",
        "judge", "defend", "support", "validate", "recommend",
        "which is better", "pros and cons"
    ],
    "create": [
        "design", "create", "develop", "propose", "formulate",
        "construct", "plan", "devise", "invent", "compose",
        "write a program", "build", "derive"
    ]
}

# Question type indicators
TYPE_INDICATORS = {
    "mcq": [
        r"\(a\)", r"\(b\)", r"\(c\)", r"\(d\)",
        r"\ba\)", r"\bb\)", r"\bc\)", r"\bd\)",
        r"A\.", r"B\.", r"C\.", r"D\.",
        "choose the correct", "select the", "which of the following",
        "pick the correct"
    ],
    "numerical": [
        "calculate", "compute", "find the value", "solve",
        "determine the", "evaluate the expression",
        "how many", "how much"
    ],
    "short": [
        "short note", "brief", "in one line",
        "define", "state", "list", "name",
        "write short", "mention"
    ],
    "long": [
        "explain in detail", "discuss at length", "elaborate",
        "describe with", "write in detail", "detailed explanation",
        "explain with examples", "discuss various"
    ]
}


class QuestionParser:
    """
    Parses .txt question paper files into structured question objects.
    Handles multiple formats and messy input gracefully.
    """

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.detect_marks = self.config.get("detect_marks", True)
        self.detect_bloom = self.config.get("detect_bloom", True)
        self.detect_type = self.config.get("detect_type", True)
        self.min_length = self.config.get("min_question_length", 10)

    def parse_file(self, filepath: str, subject: str = "general") -> List[Dict]:
        """Parse a .txt file and return structured questions."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            raw_text = f.read()

        # Try structured parsing first, fall back to intelligent splitting
        questions = self._try_structured_parse(raw_text)

        if not questions:
            questions = self._try_numbered_parse(raw_text)

        if not questions:
            questions = self._try_line_by_line_parse(raw_text)

        if not questions:
            questions = self._fallback_paragraph_parse(raw_text)

        # Enrich each question with metadata
        enriched = []
        for q in questions:
            if len(q.get("question_text", "").strip()) < self.min_length:
                continue

            q["subject"] = subject
            q["id"] = hashlib.md5(
                f"{subject}_{q['question_text'][:100]}".encode()
            ).hexdigest()

            if self.detect_marks:
                q["marks"] = q.get("marks") or self._extract_marks(q["raw_line"])

            if self.detect_bloom:
                q["bloom_level"] = self._detect_bloom_level(q["question_text"])

            if self.detect_type:
                q["question_type"] = q.get("question_type") or self._detect_question_type(
                    q["question_text"]
                )

            q["difficulty"] = self._estimate_difficulty(q)
            enriched.append(q)

        return enriched

    # -- Parsing strategies (ordered by reliability) -----------------------

    def _try_structured_parse(self, text: str) -> List[Dict]:
        """
        Try to parse Q: ... A: ... format or similar structured formats.
        """
        questions = []

        # Pattern: Q1. or Q.1 or Question 1: etc
        pattern = r'(?:Q(?:uestion)?[\s.:]*(\d+)[\s.:]*)(.*?)(?=Q(?:uestion)?[\s.:]*\d+|$)'
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)

        for num, body in matches:
            body = body.strip()
            if len(body) < self.min_length:
                continue

            # Check if answer is embedded
            answer = None
            ans_match = re.split(
                r'\n\s*(?:A(?:ns(?:wer)?)?[\s.:]*)',
                body, maxsplit=1, flags=re.IGNORECASE
            )
            if len(ans_match) > 1:
                body = ans_match[0].strip()
                answer = ans_match[1].strip()

            questions.append({
                "question_text": body,
                "answer_text": answer,
                "raw_line": body[:200],
                "question_type": None,
                "marks": None
            })

        return questions

    def _try_numbered_parse(self, text: str) -> List[Dict]:
        """
        Parse numbered lists: 1. or 1) or (1) or i. or a.
        """
        questions = []
        lines = text.split("\n")
        current_question = []
        current_raw = ""

        # Patterns that indicate a new question
        new_q_pattern = re.compile(
            r'^\s*(?:'
            r'\d+[\.\)]\s+'                        # 1. or 1)
            r'|\(\d+\)\s+'                          # (1)
            r'|[ivxlc]+[\.\)]\s+'                   # i. or ii)
            r'|[a-e][\.\)]\s+'                      # a. or b)
            r'|(?:Q|Que|Question)\s*\.?\s*\d+'      # Q1, Que.2, Question 3
            r')',
            re.IGNORECASE
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                if current_question:
                    current_question.append("")
                continue

            if new_q_pattern.match(stripped):
                # Save previous question
                if current_question:
                    q_text = " ".join(current_question).strip()
                    # Remove the leading number/letter
                    q_text = re.sub(
                        r'^\s*(?:\d+[\.\)]|\(\d+\)|[ivxlc]+[\.\)]|[a-e][\.\)]|(?:Q|Que|Question)\s*\.?\s*\d+[\.\):]?)\s*',
                        '', q_text, flags=re.IGNORECASE
                    ).strip()

                    if len(q_text) >= self.min_length:
                        questions.append({
                            "question_text": q_text,
                            "answer_text": None,
                            "raw_line": current_raw,
                            "question_type": None,
                            "marks": None
                        })

                current_question = [stripped]
                current_raw = stripped
            else:
                current_question.append(stripped)

        # Don't forget the last question
        if current_question:
            q_text = " ".join(current_question).strip()
            q_text = re.sub(
                r'^\s*(?:\d+[\.\)]|\(\d+\)|[ivxlc]+[\.\)]|[a-e][\.\)]|(?:Q|Que|Question)\s*\.?\s*\d+[\.\):]?)\s*',
                '', q_text, flags=re.IGNORECASE
            ).strip()
            if len(q_text) >= self.min_length:
                questions.append({
                    "question_text": q_text,
                    "answer_text": None,
                    "raw_line": current_raw,
                    "question_type": None,
                    "marks": None
                })

        return questions

    def _try_line_by_line_parse(self, text: str) -> List[Dict]:
        """
        Each non-empty line that looks like a question is treated as one.
        Uses question indicators (?, action verbs, etc.)
        """
        questions = []
        lines = text.split("\n")

        question_indicators = [
            "?", "explain", "define", "describe", "list", "what",
            "how", "why", "compare", "discuss", "write", "state",
            "derive", "prove", "solve", "calculate", "differentiate",
            "illustrate", "evaluate", "design", "construct", "analyze"
        ]

        for line in lines:
            stripped = line.strip()
            if len(stripped) < self.min_length:
                continue

            lower = stripped.lower()
            is_question = any(ind in lower for ind in question_indicators)

            if is_question:
                # Clean leading numbering
                cleaned = re.sub(r'^\s*[\d]+[\.\)]\s*', '', stripped).strip()
                questions.append({
                    "question_text": cleaned,
                    "answer_text": None,
                    "raw_line": stripped,
                    "question_type": None,
                    "marks": None
                })

        return questions

    def _fallback_paragraph_parse(self, text: str) -> List[Dict]:
        """
        Last resort: split by double newlines and treat each block as a question.
        Filter by question-likeness.
        """
        questions = []
        blocks = re.split(r'\n\s*\n', text)

        for block in blocks:
            block = block.strip()
            if len(block) < self.min_length:
                continue

            # Only keep blocks that look like questions
            lower = block.lower()
            if "?" in block or any(
                word in lower for word in ["explain", "define", "describe", "discuss", "write"]
            ):
                questions.append({
                    "question_text": block,
                    "answer_text": None,
                    "raw_line": block[:200],
                    "question_type": None,
                    "marks": None
                })

        return questions

    # -- Metadata extraction -----------------------------------------------

    def _extract_marks(self, raw_line: str) -> Optional[int]:
        """Extract marks from patterns like [5], (5 marks), 5M, etc."""
        if not raw_line:
            return None

        patterns = [
            r'\[(\d+)\]',                       # [5]
            r'\((\d+)\s*(?:marks?|pts?|points?)\)',  # (5 marks)
            r'(\d+)\s*(?:marks?|pts?|points?)',      # 5 marks
            r'\[(\d+)\s*[mM]\]',                # [5M]
            r'(\d+)\s*[mM]\b',                  # 5M
        ]

        for pattern in patterns:
            match = re.search(pattern, raw_line, re.IGNORECASE)
            if match:
                marks = int(match.group(1))
                if 1 <= marks <= 20:  # sanity check
                    return marks

        return None

    def _detect_bloom_level(self, question_text: str) -> str:
        """Detect Bloom's taxonomy level from question text."""
        lower = question_text.lower()

        # Check from highest to lowest (more specific first)
        for level in ["create", "evaluate", "analyze", "apply", "understand", "remember"]:
            for keyword in BLOOM_KEYWORDS[level]:
                if keyword in lower:
                    return level

        return "understand"  # default

    def _detect_question_type(self, question_text: str) -> str:
        """Detect whether question is MCQ, short, long, numerical, etc."""
        lower = question_text.lower()

        for qtype, indicators in TYPE_INDICATORS.items():
            for indicator in indicators:
                if isinstance(indicator, str) and indicator in lower:
                    return qtype
                elif re.search(indicator, question_text, re.IGNORECASE):
                    return qtype

        # Length-based heuristic
        word_count = len(question_text.split())
        if word_count < 10:
            return "short"
        elif word_count > 30:
            return "long"

        return "descriptive"

    def _estimate_difficulty(self, question: Dict) -> str:
        """Estimate difficulty from bloom level and marks."""
        bloom = question.get("bloom_level", "understand")
        marks = question.get("marks", 5)

        bloom_difficulty = {
            "remember": 1, "understand": 2, "apply": 3,
            "analyze": 4, "evaluate": 5, "create": 5
        }

        score = bloom_difficulty.get(bloom, 2)
        if marks and marks > 8:
            score += 1
        if marks and marks > 12:
            score += 1

        if score <= 2:
            return "easy"
        elif score <= 4:
            return "medium"
        else:
            return "hard"
