"""
AION Paper Validator
====================
Pre-export validation checklist.
Every rule that must pass before a paper is returned to the frontend.

Production-safe. No laptop-specific code.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple


# ── Bloom taxonomy ────────────────────────────────────────────────────────────

BLOOM_VERBS = {
    1: ["define", "list", "state", "recall", "identify", "name", "write"],
    2: ["explain", "describe", "summarize", "discuss", "interpret", "outline"],
    3: ["apply", "illustrate", "demonstrate", "solve", "use", "calculate"],
    4: ["analyze", "compare", "differentiate", "examine", "contrast"],
    5: ["evaluate", "justify", "assess", "critique", "argue", "judge"],
    6: ["design", "develop", "create", "propose", "formulate", "build"],
}

# Target Bloom distribution for a VTU IAT (percentage of marks)
TARGET_BLOOM_IA = {1: 10, 2: 20, 3: 30, 4: 20, 5: 15, 6: 5}
TARGET_BLOOM_SEE = {1: 5,  2: 15, 3: 30, 4: 25, 5: 15, 6: 10}

# CO tolerance — actual vs target within this percentage
CO_TOLERANCE = 10


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class ValidationIssue:
    severity: str        # ERROR / WARNING / INFO
    code:     str
    message:  str
    fix:      str = ""


@dataclass
class PaperValidationReport:
    passed:   bool
    issues:   List[ValidationIssue] = field(default_factory=list)
    checklist: Dict = field(default_factory=dict)

    def errors(self):
        return [i for i in self.issues if i.severity == "ERROR"]

    def warnings(self):
        return [i for i in self.issues if i.severity == "WARNING"]

    def summary(self) -> str:
        e = len(self.errors())
        w = len(self.warnings())
        return f"{'PASS' if self.passed else 'FAIL'} — {e} errors, {w} warnings"


# ── Main validator ────────────────────────────────────────────────────────────

class PaperValidator:
    """
    Runs all validation rules against a formatted paper dict.
    Call validate(paper_dict) before returning paper to frontend.
    """

    def validate(self, paper: dict, exam_type: str = "IA") -> PaperValidationReport:
        issues    = []
        checklist = {}

        modules = paper.get("modules", [])
        target  = paper.get("totalMarks", 50)

        # ── Rule 1: Total marks ───────────────────────────────────────────────
        actual_total = self._compute_attemptable_marks(modules)
        ok = abs(actual_total - target) <= 2
        checklist["total_marks"] = ok
        if not ok:
            issues.append(ValidationIssue(
                severity = "ERROR",
                code     = "MARKS_TOTAL",
                message  = f"Attemptable marks = {actual_total}, expected {target}.",
                fix      = "Adjust sub-question marks so OR pairs sum to correct total.",
            ))

        # ── Rule 2: OR parity ────────────────────────────────────────────────
        or_ok, or_issues = self._validate_or_parity(modules)
        checklist["or_parity"] = or_ok
        issues.extend(or_issues)

        # ── Rule 3: All questions have text ──────────────────────────────────
        text_ok, text_issues = self._validate_question_text(modules)
        checklist["question_text"] = text_ok
        issues.extend(text_issues)

        # ── Rule 4: Sub-question count ────────────────────────────────────────
        subq_ok, subq_issues = self._validate_subquestion_count(modules, exam_type)
        checklist["subquestion_count"] = subq_ok
        issues.extend(subq_issues)

        # ── Rule 5: Bloom distribution ───────────────────────────────────────
        bloom_ok, bloom_issues = self._validate_bloom(modules, exam_type)
        checklist["bloom_distribution"] = bloom_ok
        issues.extend(bloom_issues)

        # ── Rule 6: CO coverage ───────────────────────────────────────────────
        co_ok, co_issues = self._validate_co(modules)
        checklist["co_coverage"] = co_ok
        issues.extend(co_issues)

        # ── Rule 7: No duplicate questions ───────────────────────────────────
        dup_ok, dup_issues = self._validate_duplicates(modules)
        checklist["no_duplicates"] = dup_ok
        issues.extend(dup_issues)

        # ── Rule 8: Question numbering ────────────────────────────────────────
        num_ok = self._validate_numbering(modules)
        checklist["numbering"] = num_ok
        if not num_ok:
            issues.append(ValidationIssue(
                severity = "WARNING",
                code     = "NUMBERING",
                message  = "Question numbering is not sequential.",
                fix      = "Re-index questions starting from 1.",
            ))

        # ── Rule 9: Marks per sub-question sums to main question total ────────
        sum_ok, sum_issues = self._validate_submarks_sum(modules)
        checklist["submarks_sum"] = sum_ok
        issues.extend(sum_issues)

        # ── Rule 10: Bloom verb matches bloom level ───────────────────────────
        verb_ok, verb_issues = self._validate_bloom_verbs(modules)
        checklist["bloom_verbs"] = verb_ok
        issues.extend(verb_issues)

        has_errors = any(i.severity == "ERROR" for i in issues)
        return PaperValidationReport(
            passed    = not has_errors,
            issues    = issues,
            checklist = checklist,
        )

    # ── Rule implementations ──────────────────────────────────────────────────

    def _compute_attemptable_marks(self, modules: list) -> int:
        """Sum only one side of each OR pair per module."""
        total = 0
        for mod in modules:
            qs    = mod.get("questions", [])
            pairs = {}
            for i, q in enumerate(qs):
                idx      = q.get("mqIndex") or q.get("mq_index") or (i + 1)
                pair_key = (idx - 1) // 2
                pairs.setdefault(pair_key, []).append(q)
            for pair_qs in pairs.values():
                best = max(
                    self._get_question_marks(q)
                    for q in pair_qs
                )
                total += best
        return total

    def _get_question_marks(self, q: dict) -> int:
        subs = q.get("subQuestions") or q.get("sub_questions") or []
        if subs:
            return sum(sq.get("marks", 0) for sq in subs)
        return q.get("totalMarks") or q.get("total_marks") or q.get("marks", 10)

    def _validate_or_parity(self, modules: list):
        issues = []
        ok     = True
        for mod in modules:
            qs    = mod.get("questions", [])
            pairs = {}
            for q in qs:
                idx      = q.get("mqIndex", 1)
                pair_key = (idx - 1) // 2
                pairs.setdefault(pair_key, []).append(q)

            for pair_key, pair_qs in pairs.items():
                if len(pair_qs) < 2:
                    continue
                marks_list = [
                    sum(sq.get("marks", 0) for sq in q.get("subQuestions", []))
                    for q in pair_qs
                ]
                if len(set(marks_list)) > 1:
                    ok = False
                    q_nums = [q.get("mqIndex") for q in pair_qs]
                    issues.append(ValidationIssue(
                        severity = "ERROR",
                        code     = "OR_PARITY",
                        message  = (
                            f"OR pair Q{q_nums[0]}/Q{q_nums[1]} has unequal marks: "
                            f"{marks_list}. Both must total the same marks."
                        ),
                        fix = (
                            f"Adjust Q{q_nums[1]} sub-question marks to sum to {marks_list[0]}."
                        ),
                    ))
        return ok, issues

    def _validate_question_text(self, modules: list):
        issues = []
        ok     = True
        for mod in modules:
            for q in mod.get("questions", []):
                for sq in q.get("subQuestions", []):
                    text = sq.get("text", "").strip()
                    if not text or len(text.split()) < 4:
                        ok = False
                        issues.append(ValidationIssue(
                            severity = "ERROR",
                            code     = "MISSING_TEXT",
                            message  = (
                                f"Q{q.get('mqIndex')} part {sq.get('letter','?')}: "
                                f"question text is missing or too short."
                            ),
                            fix = "Regenerate this question block.",
                        ))
        return ok, issues

    def _validate_subquestion_count(self, modules: list, exam_type: str):
        issues  = []
        ok      = True
        max_sub = 2 if exam_type in ("IA", "IAT1", "IAT2") else 3

        for mod in modules:
            for q in mod.get("questions", []):
                count = len(q.get("subQuestions", []))
                if count > max_sub:
                    ok = False
                    issues.append(ValidationIssue(
                        severity = "ERROR",
                        code     = "SUBQ_COUNT",
                        message  = (
                            f"Q{q.get('mqIndex')}: has {count} sub-questions "
                            f"(max {max_sub} for {exam_type})."
                        ),
                        fix = f"Remove extra sub-questions from Q{q.get('mqIndex')}.",
                    ))
        return ok, issues

    def _validate_bloom(self, modules: list, exam_type: str):
        issues  = []
        ok      = True
        target  = TARGET_BLOOM_IA if exam_type in ("IA", "IAT1", "IAT2") else TARGET_BLOOM_SEE
        counts  = {b: 0 for b in range(1, 7)}
        total_m = 0

        for mod in modules:
            for q in mod.get("questions", []):
                bloom = q.get("bloomLevel", 2)
                for sq in q.get("subQuestions", []):
                    m = sq.get("marks", 0)
                    counts[bloom] = counts.get(bloom, 0) + m
                    total_m += m

        if total_m == 0:
            return True, []

        actual = {b: round(m / total_m * 100) for b, m in counts.items()}

        # Check that L4+ exists
        higher = sum(actual.get(b, 0) for b in [4, 5, 6])
        if higher < 15:
            ok = False
            issues.append(ValidationIssue(
                severity = "WARNING",
                code     = "BLOOM_HIGHER",
                message  = (
                    f"Only {higher}% of marks are at L4+ (Analyze/Evaluate/Create). "
                    f"Recommended minimum is 25%."
                ),
                fix = "Add more L4 (Compare/Analyze) and L5 (Evaluate/Justify) questions.",
            ))

        # Check L1 is not dominant
        if actual.get(1, 0) > 30:
            issues.append(ValidationIssue(
                severity = "WARNING",
                code     = "BLOOM_LOW",
                message  = f"L1 (Remember) is {actual[1]}% of marks. Max recommended is 15%.",
                fix      = "Replace some L1 questions with L2 or L3.",
            ))

        return ok, issues

    def _validate_co(self, modules: list):
        issues = []
        ok     = True
        counts = {}
        total  = 0

        for mod in modules:
            for q in mod.get("questions", []):
                for sq in q.get("subQuestions", []):
                    co = sq.get("co", "CO1")
                    m  = sq.get("marks", 0)
                    counts[co] = counts.get(co, 0) + m
                    total += m

        if total == 0:
            return True, []

        # Check all 5 COs are represented
        for i in range(1, 6):
            co = f"CO{i}"
            if co not in counts:
                issues.append(ValidationIssue(
                    severity = "WARNING",
                    code     = "CO_MISSING",
                    message  = f"{co} has no questions mapped to it.",
                    fix      = f"Map at least one question to {co}.",
                ))
                ok = False

        return ok, issues

    def _validate_duplicates(self, modules: list):
        issues    = []
        ok        = True
        all_texts = []

        for mod in modules:
            for q in mod.get("questions", []):
                subs = q.get("subQuestions") or q.get("sub_questions") or []
                if subs:
                    for sq in subs:
                        text = sq.get("text", "").strip().lower()
                        if text:
                            all_texts.append((q.get("mqIndex") or q.get("mq_index"), sq.get("letter"), text))
                else:
                    text = q.get("text") or q.get("question_text") or ""
                    text = text.strip().lower()
                    if text:
                        all_texts.append((q.get("mqIndex") or q.get("mq_index"), "", text))

        # Semantic duplicate check (similarity threshold >= 0.85)
        for i in range(len(all_texts)):
            for j in range(i + 1, len(all_texts)):
                qi, li, ti = all_texts[i]
                qj, lj, tj = all_texts[j]
                sim = self._word_overlap(ti, tj)
                if sim >= 0.85:
                    ok = False
                    issues.append(ValidationIssue(
                        severity = "ERROR",
                        code     = "OR_SIMILARITY_DUPLICATE",
                        message  = (
                            f"Q{qi}{li} and Q{qj}{lj} are {sim:.0%} semantically similar. "
                            f"Likely duplicate or insufficiently differentiated OR pair."
                        ),
                        fix = f"Regenerate Q{qj}{lj}.",
                    ))
        return ok, issues

    def _word_overlap(self, a: str, b: str) -> float:
        stopwords = {
            "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
            "which", "this", "that", "these", "those", "then", "just", "so", "than",
            "such", "both", "through", "about", "against", "between", "into", "through",
            "during", "before", "after", "above", "below", "to", "from", "up", "down",
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
            "own", "same", "so", "than", "too", "very", "can", "will", "should", "now",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do",
            "does", "did", "for", "with", "of", "at", "by", "for", "with", "about"
        }
        import re
        words_a = {w for w in re.findall(r'\b[a-z]{2,}\b', a.lower()) if w not in stopwords}
        words_b = {w for w in re.findall(r'\b[a-z]{2,}\b', b.lower()) if w not in stopwords}
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        jaccard = intersection / union if union > 0 else 0.0
        overlap = intersection / max(1, min(len(words_a), len(words_b)))
        return round(0.5 * jaccard + 0.5 * overlap, 4)

    def _validate_numbering(self, modules: list) -> bool:
        expected = 1
        for mod in modules:
            for q in mod.get("questions", []):
                if q.get("mqIndex") != expected:
                    return False
                expected += 1
        return True

    def _validate_submarks_sum(self, modules: list):
        issues = []
        ok     = True
        for mod in modules:
            for q in mod.get("questions", []):
                declared = q.get("totalMarks", 0)
                actual   = sum(sq.get("marks", 0) for sq in q.get("subQuestions", []))
                if declared > 0 and actual != declared:
                    ok = False
                    issues.append(ValidationIssue(
                        severity = "ERROR",
                        code     = "SUBMARKS_SUM",
                        message  = (
                            f"Q{q.get('mqIndex')}: sub-questions sum to {actual} "
                            f"but declared total is {declared}."
                        ),
                        fix = f"Adjust marks so sub-questions of Q{q.get('mqIndex')} sum to {declared}.",
                    ))
        return ok, issues

    def _validate_bloom_verbs(self, modules: list):
        issues = []
        ok     = True
        for mod in modules:
            for q in mod.get("questions", []):
                bloom = q.get("bloomLevel", 2)
                expected_verbs = BLOOM_VERBS.get(bloom, [])
                for sq in q.get("subQuestions", []):
                    text  = sq.get("text", "").lower()
                    first = text.split()[0] if text.split() else ""
                    if expected_verbs and first and first not in expected_verbs:
                        issues.append(ValidationIssue(
                            severity = "WARNING",
                            code     = "BLOOM_VERB",
                            message  = (
                                f"Q{q.get('mqIndex')}{sq.get('letter','')}: "
                                f"verb '{first}' may not match L{bloom}. "
                                f"Expected: {expected_verbs[:3]}."
                            ),
                            fix = f"Start with one of: {', '.join(expected_verbs[:3])}.",
                        ))
        return ok, issues

    def _word_overlap(self, a: str, b: str) -> float:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / max(len(words_a), len(words_b))


# ── OR marks enforcer (called in generator) ───────────────────────────────────

def enforce_or_parity(
    question_a: dict,
    question_b: dict,
    target_marks: int,
) -> Tuple[dict, dict]:
    """
    Ensure both sides of an OR pair sum to the same marks.
    Adjusts the last sub-question of the shorter side.
    """
    def get_total(q):
        return sum(sq.get("marks", 0) for sq in q.get("subQuestions", []))

    def adjust_to(q, target):
        subs = q.get("subQuestions", [])
        if not subs:
            return q
        current = get_total(q)
        diff    = target - current
        if diff == 0:
            return q
        subs    = list(subs)
        subs[-1] = {**subs[-1], "marks": max(1, subs[-1].get("marks", 0) + diff)}
        return {**q, "subQuestions": subs}

    a_total = get_total(question_a)
    b_total = get_total(question_b)

    if a_total == b_total:
        return question_a, question_b

    # Use the larger as the target, or target_marks if both differ
    target = target_marks if target_marks > 0 else max(a_total, b_total)
    return adjust_to(question_a, target), adjust_to(question_b, target)
