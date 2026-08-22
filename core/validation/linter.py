# core/validation/linter.py
from __future__ import annotations          # ← ADD THIS AS LINE 2 (after the comment)


import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from core.validation.common import CheckResult, RetryAction
from core.validation.demand_validator import DemandValidator
from core.validation.math_validator import validate_math_consistency, validate_math_block_with_render

if TYPE_CHECKING:
    from core.contracts.question_slot import QuestionSlot, QuestionContract
    from core.generation.output_schema import QuestionOutput
    from core.contracts.question import GeneratedQuestion

def _jaccard_similarity(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity between two strings."""
    set_a = set(re.findall(r"\b[a-zA-Z]{3,}\b", a.lower()))
    set_b = set(re.findall(r"\b[a-zA-Z]{3,}\b", b.lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def check_sibling_uniqueness(
    question: "GeneratedQuestion",
    sibling_texts: list,
    threshold: float = 0.65,
) -> "CheckResult":
    """
    Rejects a question if it is too similar to any sibling
    sub-question or OR alternative already generated.
    Similarity is measured by Jaccard overlap on meaningful words.
    """
    q_text = question.question_text.lower()
    for i, sib in enumerate(sibling_texts):
        sim = _jaccard_similarity(q_text, sib.lower())
        if sim >= threshold:
            return CheckResult.fail(
                "SIBLING_SIMILARITY",
                f"Question is too similar to sibling slot {i+1} "
                f"(Jaccard={sim:.2f} >= threshold={threshold}). "
                f"Generate a conceptually distinct question.",
                action=RetryAction.REGENERATE,
            )
    return CheckResult.pass_()



BLOOM_VERB_MAP = {
    "L1": ["Define", "List", "Identify", "Name", "State", "Recall"],
    "L2": ["Explain", "Describe", "Summarize", "Summarise", "Illustrate", "Classify", "Interpret"],
    "L3": ["Calculate", "Apply", "Demonstrate", "Determine", "Solve"],
    "L4": ["Analyze", "Analyse", "Compare", "Examine", "Differentiate"],
    "L5": ["Evaluate", "Critique", "Justify", "Assess"],
    "L6": ["Design", "Develop", "Construct", "Propose", "Formulate"],
}

VERB_OPERATION_MAP = {
    # L1
    "define": "REMEMBER", "list": "REMEMBER", "identify": "REMEMBER", "name": "REMEMBER", "state": "REMEMBER", "recall": "REMEMBER",
    # L2
    "explain": "UNDERSTAND", "describe": "UNDERSTAND", "summarize": "UNDERSTAND", "summarise": "UNDERSTAND", "illustrate": "UNDERSTAND",
    "classify": "UNDERSTAND", "interpret": "UNDERSTAND",
    # L3
    "calculate": "CALCULATE", "solve": "CALCULATE", "determine": "CALCULATE", "apply": "APPLY", "demonstrate": "APPLY",
    # L4
    "analyze": "ANALYZE", "analyse": "ANALYZE", "examine": "ANALYZE", "differentiate": "ANALYZE", "compare": "COMPARE",
    # L5
    "evaluate": "EVALUATE", "critique": "EVALUATE", "assess": "EVALUATE", "justify": "JUSTIFY",
    # L6
    "design": "CREATE", "develop": "CREATE", "construct": "CREATE", "propose": "CREATE", "formulate": "CREATE",
}

MULTI_SLOT_PATTERNS = [
    r"^\s*\([a-z]\)\s+",       # (a) ...
    r"^\s*[a-z]\)\s+",         # a) ...
    r"^\s*Q(?:uestion)?\s*\d+", # Q1 / Question 1
    r"^\s*OR\s*$",
    r"^\s*OR\s+(?:QUESTION|Q)\s*\d+",
]


class LintReport:
    """Contains linter results per check for a single slot question candidate."""
    def __init__(self, slot_id: str, checks: Dict[str, CheckResult]):
        self.slot_id = slot_id
        self.checks = checks

    @property
    def passed(self) -> bool:
        return all((getattr(r, "passed", r[0] if isinstance(r, (tuple, list)) else bool(r))) for r in self.checks.values())

    def get_failure(self) -> Optional[CheckResult]:
        for r in self.checks.values():
            if not (getattr(r, "passed", r[0] if isinstance(r, (tuple, list)) else bool(r))):
                return r
        return None


def find_bloom_verbs_in_clause(clause: str) -> List[Tuple[str, str, str]]:
    """Scan clause for action verbs and return list of (verb, level, operation)."""
    found = []
    clause_lower = clause.lower()
    words = re.findall(r'\b[a-zA-Z]+\b', clause_lower)
    for word in words:
        if word in VERB_OPERATION_MAP:
            level = "L1"
            for lvl, verbs in BLOOM_VERB_MAP.items():
                if any(v.lower() == word for v in verbs):
                    level = lvl
                    break
            found.append((word, level, VERB_OPERATION_MAP[word]))
    return found


def check_bloom_verb_at_start(instruction: str, slot: QuestionSlot) -> CheckResult:
    """H8 — Verifies instruction starts with the expected Bloom verb (case-insensitive)."""
    verb = slot.bloom_verb.lower()
    words = instruction.strip().split()
    first_word = words[0].lower().rstrip(".,;:") if words else ""
    if first_word != verb:
        return CheckResult.fail(
            "BLOOM_VERB_NOT_AT_START",
            f"Instruction must start with expected Bloom verb '{slot.bloom_verb}'. Got: '{first_word}'",
            action=RetryAction.REGENERATE_WITH_BLOOM_HINT
        )
    return CheckResult.pass_()


def check_task_count_from_instruction(instruction: str, slot: QuestionSlot) -> CheckResult:
    """H8 — Scans instruction field only and enforces operation boundaries."""
    sig = slot.task_signature
    allowed = {sig.primary_operation} | set(sig.allowed_secondary_operations)
    
    verbs_found = find_bloom_verbs_in_clause(instruction.lower())
    disallowed = [
        (verb, op) for verb, level, op in verbs_found
        if op not in allowed and verb != slot.bloom_verb.lower()
    ]

    if disallowed:
        return CheckResult.fail(
            "DISALLOWED_SECONDARY_TASK",
            f"Instruction contains disallowed operations: {[v for v, _ in disallowed]}. "
            f"Allowed: {allowed}",
            action=RetryAction.REGENERATE
        )
    return CheckResult.pass_()


def check_no_answer_leakage(text: str) -> CheckResult:
    """Rejects questions containing model solutions or contextual answers."""
    PATTERNS = [
        "answer:", "solution:", "the answer is", "correct answer",
        "model answer", "expected answer", "therefore, the correct",
        "hence, the result", "thus, we find", "the correct answer is",
        "thus, the required result", "hence the output will be",
        "therefore, the output is"
    ]
    text_lower = text.lower()
    for p in PATTERNS:
        if p in text_lower:
            return CheckResult.fail(
                "ANSWER_LEAK",
                f"Contextual answer/leak pattern detected: '{p}'",
                action=RetryAction.REGENERATE
            )
    return CheckResult.pass_()


def check_no_meta_language(text: str) -> CheckResult:
    """Rejects prompt leaking or reference-acknowledgment text."""
    PATTERNS = [
        "from the source", "from the notes", "provided notes",
        "uploaded document", "source material",
        "based on the provided", "according to the notes",
        "/fontfile", "flatdecode", "endobj", "ideal answer",
        "marking scheme", "grading rubrics"
    ]
    text_lower = text.lower()
    for p in PATTERNS:
        if p in text_lower:
            return CheckResult.fail(
                "META_LANGUAGE",
                f"Meta-language reference detected: '{p}'",
                action=RetryAction.REGENERATE
            )
    return CheckResult.pass_()


def check_unicode_integrity(text: str) -> CheckResult:
    """Rejects corrupted characters or null bytes."""
    if "\ufffd" in text or "\x00" in text:
        return CheckResult.critical(
            "UNICODE_FAILURE",
            "Text contains corrupted replacement or null characters."
        )
    return CheckResult.pass_()


def check_all_math_with_render(output: QuestionOutput) -> CheckResult:
    """Validates and compiles all math blocks via KaTeX availability gate."""
    for block in output.math_blocks:
        res = validate_math_block_with_render(block)
        if not res.passed:
            return res
    return CheckResult.pass_()


def check_visual_policy(question: GeneratedQuestion, slot: QuestionSlot) -> CheckResult:
    """Enforces slot visual required policy constraints."""
    has_diagram = question.diagram_request is not None
    if slot.visual_required and not has_diagram:
        return CheckResult.fail(
            "VISUAL_POLICY_VIOLATION",
            "Visual policy is REQUIRED, but candidate question did not declare a diagram request.",
            action=RetryAction.REGENERATE
        )
    elif not slot.visual_required and has_diagram:
        return CheckResult.fail(
            "VISUAL_POLICY_VIOLATION",
            "Visual policy is FORBIDDEN, but candidate question contains a diagram request.",
            action=RetryAction.REGENERATE
        )
    return CheckResult.pass_()


def check_math_policy(question: GeneratedQuestion, slot: QuestionSlot) -> CheckResult:
    """Enforces slot math required policy constraints."""
    num_blocks = len(question.math_blocks)
    if slot.math_required and num_blocks == 0:
        return CheckResult.fail(
            "MATH_POLICY_VIOLATION",
            "Math policy is REQUIRED, but candidate contains zero MathBlocks.",
            action=RetryAction.REGENERATE
        )
    elif not slot.math_required and num_blocks > 0:
        return CheckResult.fail(
            "MATH_POLICY_VIOLATION",
            f"Math policy is FORBIDDEN, but candidate contains {num_blocks} MathBlocks.",
            action=RetryAction.REGENERATE
        )
    return CheckResult.pass_()


def check_multi_slot_contamination(text: str) -> CheckResult:
    """Protects against multi-subquestion fusion by rejecting structural label patterns."""
    for pattern in MULTI_SLOT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE | re.MULTILINE):
            return CheckResult.fail(
                "MULTI_SLOT_CONTAMINATION",
                f"Multi-slot contamination structural label matched: '{pattern}'",
                action=RetryAction.REGENERATE
            )
    return CheckResult.pass_()


def check_answerability(question: GeneratedQuestion, slot: QuestionSlot, evidence_text: str = "") -> CheckResult:
    """
    Evaluates whether the question contains a clearly identifiable task,
    meets marks-aware length demands, and is fully supported by the evidence (🔴 11).
    """
    text = question.question_text.lower()
    # Check for presence of command words or standard verbs
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    has_action_verb = any(w in VERB_OPERATION_MAP for w in words)
    if not has_action_verb:
        return CheckResult.fail(
            "ANSWERABILITY_FAILURE",
            "Question text has no clearly identifiable task or action verb.",
            action=RetryAction.REGENERATE
        )

    # Marks-aware scope check
    # High marks require longer instructions/dimension complexity
    if slot.marks >= 9 and len(text.split()) < 20:
        return CheckResult.fail(
            "ANSWERABILITY_FAILURE",
            f"Question is too brief ({len(text.split())} words) for a high-marks ({slot.marks}M) task.",
            action=RetryAction.REGENERATE
        )

    # Grounding Support Check (🔴 11)
    if evidence_text:
        evidence_lower = evidence_text.lower()
        question_terms = set(re.findall(r'\b[a-z]{5,}\b', text))
        
        # Exclude common query/English words
        stopwords = {"explain", "describe", "compare", "calculate", "question", "following", "between", "detail", "principles", "characteristics"}
        topic_terms = question_terms - stopwords
        
        if topic_terms:
            matched_terms = sum(1 for term in topic_terms if term in evidence_lower)
            match_ratio = matched_terms / len(topic_terms)
            
            if match_ratio < 0.10:
                # UNSUPPORTED: Block and regenerate
                return CheckResult.fail(
                    "ANSWERABILITY_FAILURE",
                    f"Question is UNSUPPORTED by the evidence (match ratio {match_ratio:.2f}). "
                    f"Terms: {topic_terms} do not appear in the source chunk.",
                    action=RetryAction.REGENERATE
                )
            elif match_ratio < 0.25 and slot.bloom_level in ("L4", "L5", "L6"):
                # PARTIALLY SUPPORTED: Reject/retry for higher cognitive levels
                return CheckResult.fail(
                    "ANSWERABILITY_FAILURE",
                    f"Question is only PARTIALLY SUPPORTED by the evidence (match ratio {match_ratio:.2f}). "
                    "For high-order questions (L4+), please choose more grounded evidence.",
                    action=RetryAction.REGENERATE
                )
                
    return CheckResult.pass_()


def check_contract_integrity(question: GeneratedQuestion, slot: QuestionSlot) -> CheckResult:
    """ContractIntegrityGate: Reconciles GeneratedQuestion parameters with slot contracts."""
    if question.marks != slot.marks:
        return CheckResult.critical("CONTRACT_FAILURE", f"Marks mismatch: expected {slot.marks}, got {question.marks}")
    if question.bloom_level != slot.bloom_level:
        return CheckResult.critical("CONTRACT_FAILURE", f"Bloom mismatch: expected {slot.bloom_level}, got {question.bloom_level}")
    if question.co != slot.co:
        return CheckResult.critical("CONTRACT_FAILURE", f"CO mismatch: expected {slot.co}, got {question.co}")
    if question.module_id != slot.module_id:
        return CheckResult.critical("CONTRACT_FAILURE", f"Module ID mismatch: expected {slot.module_id}, got {question.module_id}")
    return CheckResult.pass_()


def check_evidence_binding(question: GeneratedQuestion, slot: QuestionSlot) -> CheckResult:
    """Verifies that the generated evidence bindings exactly match slot allocations."""
    if tuple(question.evidence_ids) != tuple(slot.evidence_ids):
        return CheckResult.fail(
            "EVIDENCE_FAILURE",
            f"Evidence bindings mismatch: expected {slot.evidence_ids}, got {question.evidence_ids}",
            action=RetryAction.REGENERATE
        )
    return CheckResult.pass_()


def check_module_isolation(question: GeneratedQuestion, slot: QuestionSlot) -> CheckResult:
    """Module Isolation Gate: Rejects cross-module evidence binding (🔴 10)."""
    for chunk_id in question.evidence_ids:
        # Check if the chunk ID contains reference to another module (e.g. "m4" in a module 3 slot)
        match = re.search(r'm(?:odule)?_?(\d+)', chunk_id.lower())
        if match:
            chunk_mod = int(match.group(1))
            if chunk_mod != slot.module_id:
                return CheckResult.fail(
                    "EVIDENCE_FAILURE",
                    f"Cross-module evidence detected: chunk {chunk_id} belongs to Module {chunk_mod}, "
                    f"but target slot module is Module {slot.module_id}.",
                    action=RetryAction.REGENERATE
                )
    return CheckResult.pass_()


def run_linter(
    output   : QuestionOutput,
    question : GeneratedQuestion,
    slot     : QuestionSlot,
    contract : QuestionContract,
    evidence_text: str = "",
    sibling_texts: list = None,
) -> LintReport:
    """
    Complete deterministic linter.
    Runs all v5 validation checks.
    sibling_texts: list of already-generated question texts in the same OR pair / module
    """
    from core.validation.teacher_suitability_gate import TeacherSuitabilityGate
    checks = {
        "bloom_verb_start"   : check_bloom_verb_at_start(output.instruction, slot),
        "answer_demand"      : DemandValidator.validate(output, contract),
        "task_count"         : check_task_count_from_instruction(output.instruction, slot),
        "answer_leak"        : check_no_answer_leakage(output.question_text),
        "meta_language"      : check_no_meta_language(output.question_text),
        "unicode_text"       : check_unicode_integrity(output.question_text),
        "unicode_instruction": check_unicode_integrity(output.instruction),
        "math_consistency"   : validate_math_consistency(output),
        "sibling_uniqueness" : check_sibling_uniqueness(question, sibling_texts or [], threshold=0.55),
        "math_render"        : check_all_math_with_render(output),
        "visual_policy"      : check_visual_policy(question, slot),
        "math_policy"        : check_math_policy(question, slot),
        "multi_slot"         : check_multi_slot_contamination(output.question_text),
        "answerability"      : check_answerability(question, slot, evidence_text=evidence_text),
        "contract_integrity" : check_contract_integrity(question, slot),
        "evidence_binding"   : check_evidence_binding(question, slot),
        "module_isolation"   : check_module_isolation(question, slot),
        "teacher_suitability": TeacherSuitabilityGate.validate(question, slot, evidence_text=evidence_text),
    }
    return LintReport(slot.slot_id, checks)
