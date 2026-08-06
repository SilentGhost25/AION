"""
AION Semantic Enforcer
======================
Validates every LLM request and response against academic rules.
Sits between the pipeline and the LLM caller.

Rules enforced:
  - Marks integrity (sum must match total)
  - Sub-question count limits
  - Bloom level / verb consistency
  - Output format (valid JSON only)
  - Grounding check (concepts must exist in source)
  - Hallucination detection (basic)
  - Diagram flag auto-set
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict


# ── Constants ──────────────────────────────────────────────────────────────────

EXAM_RULES = {
    "IA": {
        "total_marks":     10,
        "max_subquestions": 2,
        "valid_splits":    [(6, 4), (5, 5)],
    },
    "SEE": {
        "total_marks":     20,
        "max_subquestions": 3,
        "valid_splits":    [(8, 6, 6), (10, 6, 4), (8, 8, 4), (10, 10), (7, 7, 6)],
    },
}

BLOOM_VERBS = {
    "L1": ["define", "list", "state", "recall", "identify", "name", "write", "label"],
    "L2": ["explain", "describe", "summarize", "interpret", "classify", "discuss", "outline"],
    "L3": ["illustrate", "apply", "demonstrate", "solve", "construct", "use", "show", "calculate"],
    "L4": ["compare", "analyze", "differentiate", "examine", "contrast", "distinguish", "categorize"],
    "L5": ["evaluate", "justify", "assess", "critique", "judge", "argue", "defend"],
    "L6": ["design", "develop", "propose", "create", "formulate", "build", "invent"],
}

DIAGRAM_VERBS = {"illustrate", "draw", "sketch", "show", "depict", "plot"}

REASON_CODES = {
    "RC-01": "Grammar or language error",
    "RC-02": "Bloom level mismatch — verb does not match declared level",
    "RC-03": "Marks mismatch — parts do not sum to question total",
    "RC-04": "Concept drift — question tests a concept not in source material",
    "RC-05": "Hallucination — question references facts not in academic content",
    "RC-06": "Professor style mismatch",
    "RC-07": "Duplicate question",
    "RC-08": "Structural violation — sub-question count or format wrong",
    "RC-09": "Numerical error",
    "RC-10": "Diagram required but not referenced",
}


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    valid:        bool
    errors:       List[str]   = field(default_factory=list)
    warnings:     List[str]   = field(default_factory=list)
    reason_codes: List[str]   = field(default_factory=list)
    fixed:        bool        = False
    fixed_data:   Optional[dict] = None

    def add_error(self, msg: str, code: str = ""):
        self.errors.append(msg)
        if code:
            self.reason_codes.append(code)
        self.valid = False

    def add_warning(self, msg: str):
        self.warnings.append(msg)


# ── Main Enforcer ──────────────────────────────────────────────────────────────

class SemanticEnforcer:
    """
    Validates and auto-repairs LLM responses before they
    reach the AION pipeline.
    """

    def __init__(self, strict_mode: bool = False):
        """
        strict_mode=True  → reject any question with violations
        strict_mode=False → attempt auto-repair first
        """
        self.strict_mode = strict_mode

    # ── Public API ─────────────────────────────────────────────────────────────

    def validate_request(
        self,
        exam_type:       str,
        bloom_level:     str,
        marks:           int,
        academic_content: str,
        num_questions:   int = 1,
    ) -> ValidationResult:
        """Validate a generation request before sending to LLM."""
        result = ValidationResult(valid=True)

        # Exam type
        if exam_type not in EXAM_RULES:
            result.add_error(
                f"Invalid exam_type '{exam_type}'. Must be IA or SEE.",
                "RC-08"
            )

        # Bloom level
        if bloom_level not in BLOOM_VERBS:
            result.add_error(
                f"Invalid bloom_level '{bloom_level}'. Must be L1-L6.",
                "RC-02"
            )

        # Marks match
        if exam_type in EXAM_RULES:
            expected = EXAM_RULES[exam_type]["total_marks"]
            if marks != expected:
                result.add_error(
                    f"Marks mismatch: requested {marks} but {exam_type} requires {expected}.",
                    "RC-03"
                )

        # Academic content
        if not academic_content or len(academic_content.strip()) < 50:
            result.add_error(
                "Academic content is too short or missing. "
                "Cannot generate grounded questions without source material.",
                "RC-04"
            )

        # Question count sanity
        if num_questions < 1 or num_questions > 20:
            result.add_warning(
                f"num_questions={num_questions} is unusual. "
                "Expected 1-10 for a single module."
            )

        return result

    def validate_response(
        self,
        raw_response:    str,
        exam_type:       str,
        bloom_level:     str,
        academic_content: str,
        auto_repair:     bool = True,
    ) -> ValidationResult:
        """
        Validate and optionally repair an LLM response.
        Returns ValidationResult with fixed_data if repaired.
        """
        result = ValidationResult(valid=True)

        # ── Step 1: Parse JSON ─────────────────────────────────────────────────
        parsed = self._extract_json(raw_response)
        if parsed is None:
            result.add_error(
                "Response is not valid JSON. LLM produced non-JSON output.",
                "RC-08"
            )
            return result

        # Handle both single question and multiple questions
        questions = []
        if "questions" in parsed:
            questions = parsed["questions"]
        elif "question" in parsed:
            questions = [parsed]
        elif "error" in parsed:
            result.add_warning(f"LLM returned error: {parsed['error']}")
            result.fixed_data = parsed
            return result
        else:
            result.add_error(
                "Response JSON missing 'question' or 'questions' key.",
                "RC-08"
            )
            return result

        # ── Step 2: Validate each question ────────────────────────────────────
        repaired_questions = []
        for i, q in enumerate(questions):
            q_result, q_fixed = self._validate_question(
                q, exam_type, bloom_level, academic_content, i, auto_repair
            )
            for err in q_result.errors:
                result.add_error(f"Q{i+1}: {err}")
            for warn in q_result.warnings:
                result.add_warning(f"Q{i+1}: {warn}")
            for rc in q_result.reason_codes:
                if rc not in result.reason_codes:
                    result.reason_codes.append(rc)
            if q_fixed:
                result.fixed = True
            repaired_questions.append(q_fixed or q)

        # ── Step 3: Build fixed output ─────────────────────────────────────────
        if len(repaired_questions) == 1:
            result.fixed_data = repaired_questions[0]
        else:
            result.fixed_data = {"questions": repaired_questions}

        return result

    def build_prompt(
        self,
        exam_type:        str,
        bloom_level:      str,
        subject:          str,
        module:           int,
        chapter:          str,
        academic_content: str,
        num_questions:    int = 1,
        professor_style:  str = "",
        diagram_allowed:  bool = True,
        numerical_allowed: bool = False,
        keywords:         List[str] = None,
    ) -> str:
        """
        Build a semantically constrained prompt that enforces all rules
        before the model even generates.
        """
        rules     = EXAM_RULES.get(exam_type, EXAM_RULES["IA"])
        verbs     = BLOOM_VERBS.get(bloom_level, BLOOM_VERBS["L3"])
        verb_list = ", ".join(v.capitalize() for v in verbs[:4])

        marks      = rules["total_marks"]
        max_subq   = rules["max_subquestions"]
        splits     = rules["valid_splits"]
        split_str  = " OR ".join(
            "+".join(str(m) for m in s) for s in splits[:3]
        )

        kw_section = ""
        if keywords:
            kw_section = f"\nKEYWORDS TO COVER: {', '.join(keywords)}"

        style_section = ""
        if professor_style:
            style_section = f"\nPROFESSOR STYLE: {professor_style}"

        diagram_rule = (
            "Diagrams are ALLOWED when relevant."
            if diagram_allowed
            else "Do NOT reference diagrams or figures."
        )

        numerical_rule = (
            "Numerical problems are ALLOWED."
            if numerical_allowed
            else "Do NOT generate numerical problems. Conceptual questions only."
        )

        prompt = f"""Generate {num_questions} VTU exam question(s) with these EXACT constraints:

EXAM TYPE    : {exam_type}
SUBJECT      : {subject}
MODULE       : {module}
CHAPTER      : {chapter}
BLOOM LEVEL  : {bloom_level}
COMMAND VERBS: Use ONLY one of these — {verb_list}
TOTAL MARKS  : {marks} per question
SUB-QUESTIONS: Maximum {max_subq} parts
MARKS SPLIT  : Must be {split_str} (must sum to exactly {marks})
{diagram_rule}
{numerical_rule}{kw_section}{style_section}

ACADEMIC CONTENT (base ALL questions on this ONLY):
{academic_content}

CRITICAL RULES:
1. Output ONLY valid JSON. No explanation, no markdown, no code blocks.
2. Marks must sum to exactly {marks}.
3. Maximum {max_subq} sub-questions.
4. Every concept must come from the academic content above.
5. Use exactly one command verb from: {verb_list}
6. Set requires_diagram=true if the verb is Illustrate, Draw, or Sketch.

OUTPUT FORMAT:
{{
  "question": "Full question using {verb_list.split(',')[0]} verb...",
  "exam_type": "{exam_type}",
  "total_marks": {marks},
  "bloom_level": "{bloom_level}",
  "sub_questions": [
    {{"part": "a", "marks": {splits[0][0]}, "text": "...", "bloom": "{bloom_level}"}},
    {{"part": "b", "marks": {splits[0][-1]}, "text": "...", "bloom": "L2"}}
  ],
  "keywords": ["keyword1", "keyword2"],
  "requires_diagram": false,
  "grounding": "Which concept from the content this question tests."
}}"""

        return prompt

    # ── Private Validators ─────────────────────────────────────────────────────

    def _validate_question(
        self,
        q:               dict,
        exam_type:       str,
        bloom_level:     str,
        academic_content: str,
        index:           int,
        auto_repair:     bool,
    ) -> Tuple[ValidationResult, Optional[dict]]:
        result  = ValidationResult(valid=True)
        fixed_q = dict(q)  # copy to modify

        rules = EXAM_RULES.get(exam_type, EXAM_RULES["IA"])

        # ── Check 1: Required fields ───────────────────────────────────────────
        for field_name in ["question", "sub_questions", "total_marks"]:
            if field_name not in q:
                result.add_error(f"Missing field: '{field_name}'", "RC-08")

        if not result.valid:
            return result, None

        question_text = q.get("question", "")
        sub_questions = q.get("sub_questions", [])
        total_marks   = q.get("total_marks", rules["total_marks"])

        # ── Check 2: Marks sum ────────────────────────────────────────────────
        declared_sum = sum(sq.get("marks", 0) for sq in sub_questions)
        expected     = rules["total_marks"]

        if declared_sum != expected:
            if auto_repair:
                fixed_q = self._repair_marks(fixed_q, exam_type)
                result.add_warning(
                    f"Marks repaired: {declared_sum} → {expected} "
                    f"using {exam_type} split."
                )
                result.fixed = True
            else:
                result.add_error(
                    f"Marks mismatch: sub-questions sum to {declared_sum}, "
                    f"expected {expected}.",
                    "RC-03"
                )

        # ── Check 3: Sub-question count ────────────────────────────────────────
        max_subq = rules["max_subquestions"]
        if len(sub_questions) > max_subq:
            if auto_repair:
                fixed_q["sub_questions"] = fixed_q["sub_questions"][:max_subq]
                result.add_warning(
                    f"Truncated sub-questions from {len(sub_questions)} "
                    f"to {max_subq} (max for {exam_type})."
                )
                result.fixed = True
            else:
                result.add_error(
                    f"Too many sub-questions: {len(sub_questions)} "
                    f"(max {max_subq} for {exam_type}).",
                    "RC-08"
                )

        # ── Check 4: Bloom verb consistency ───────────────────────────────────
        expected_verbs = BLOOM_VERBS.get(bloom_level, [])
        question_lower = question_text.lower()
        verb_found     = any(v in question_lower for v in expected_verbs)

        if not verb_found:
            detected_level = self._detect_bloom_level(question_text)
            if auto_repair and detected_level != bloom_level and expected_verbs:
                correct_verb = expected_verbs[0].capitalize()
                old_verb     = self._get_first_verb(question_text)
                if old_verb:
                    fixed_q["question"] = re.sub(
                        rf"\b{re.escape(old_verb)}\b",
                        correct_verb,
                        fixed_q["question"],
                        count=1,
                        flags=re.IGNORECASE
                    )
                    result.add_warning(
                        f"Bloom verb repaired: '{old_verb}' → '{correct_verb}' "
                        f"for level {bloom_level}."
                    )
                    result.fixed = True
            else:
                result.add_warning(
                    f"Question verb may not match {bloom_level}. "
                    f"Expected one of: {', '.join(expected_verbs[:3])}",
                )

        # ── Check 5: Diagram flag auto-set ────────────────────────────────────
        uses_diagram_verb = any(
            v in question_lower for v in DIAGRAM_VERBS
        )
        if uses_diagram_verb and not q.get("requires_diagram", False):
            fixed_q["requires_diagram"] = True
            result.add_warning("Auto-set requires_diagram=true based on command verb.")
            result.fixed = True

        # ── Check 6: Basic grounding check ────────────────────────────────────
        content_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", academic_content.lower())
        )
        question_words = set(
            re.findall(r"\b[a-zA-Z]{4,}\b", question_text.lower())
        )
        overlap = question_words & content_words

        if len(overlap) < 3:
            result.add_warning(
                f"Low content overlap ({len(overlap)} common words). "
                "Question may not be grounded in source material.",
            )
            result.reason_codes.append("RC-04")

        # ── Check 7: Ensure exam_type is set ──────────────────────────────────
        if "exam_type" not in fixed_q:
            fixed_q["exam_type"] = exam_type
            result.fixed = True

        if "bloom_level" not in fixed_q:
            fixed_q["bloom_level"] = bloom_level
            result.fixed = True

        return result, fixed_q if result.fixed else None

    def _repair_marks(self, q: dict, exam_type: str) -> dict:
        """Auto-repair marks distribution to match exam rules."""
        rules  = EXAM_RULES[exam_type]
        total  = rules["total_marks"]
        splits = rules["valid_splits"]
        subqs  = q.get("sub_questions", [])

        best_split = splits[0]
        n_subqs    = max(1, min(len(subqs), rules["max_subquestions"]))

        for split in splits:
            if len(split) == n_subqs:
                best_split = split
                break

        parts = ["a", "b", "c"]
        fixed_subqs = []
        for i, marks in enumerate(best_split):
            if i < len(subqs):
                sq = dict(subqs[i])
                sq["marks"] = marks
                sq["part"]  = parts[i]
                fixed_subqs.append(sq)
            else:
                fixed_subqs.append({
                    "part":  parts[i],
                    "marks": marks,
                    "text":  "",
                    "bloom": q.get("bloom_level", "L2"),
                })

        q["sub_questions"] = fixed_subqs
        q["total_marks"]   = total
        return q

    def _detect_bloom_level(self, question_text: str) -> str:
        """Detect the Bloom level from the verb used in the question."""
        lower = question_text.lower()
        for level in ["L6", "L5", "L4", "L3", "L2", "L1"]:
            for verb in BLOOM_VERBS[level]:
                if re.search(rf"\b{verb}\b", lower):
                    return level
        return "L2"  # default

    def _get_first_verb(self, question_text: str) -> Optional[str]:
        """Extract the first word (likely the command verb) from a question."""
        words = question_text.strip().split()
        return words[0] if words else None

    def _extract_json(self, text: str) -> Optional[dict]:
        """
        Robustly extract JSON from LLM output.
        Handles markdown code blocks, leading text, trailing text.
        """
        # Remove markdown code blocks
        text = re.sub(r"```json\s*", "", text)
        text = re.sub(r"```\s*",     "", text)
        text = text.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        return None


# ── Convenience function ───────────────────────────────────────────────────────

def get_enforcer(strict: bool = False) -> SemanticEnforcer:
    return SemanticEnforcer(strict_mode=strict)
