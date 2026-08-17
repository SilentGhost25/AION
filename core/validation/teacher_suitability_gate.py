# core/validation/teacher_suitability_gate.py

import re
from core.contracts.question_slot import QuestionSlot
from core.contracts.question import GeneratedQuestion
from core.validation.common import CheckResult, RetryAction


class TeacherSuitabilityGate:
    """
    TeacherSuitabilityGate (P1)
    Performs final pedagogical checks on:
    - Syllabus alignment & core depth
    - Specialization level (under core emphasis)
    - Numerical solvability (contains numerical data if numerical)
    - Math block invariants (every block must declare id, latex, and source/evidence) (🔴 5)
    """

    @classmethod
    def validate(cls, question: GeneratedQuestion, slot: QuestionSlot, evidence_text: str = "") -> CheckResult:
        text = question.question_text.lower()
        inst = getattr(question.output, "instruction", getattr(question, "instruction", "")) or ""
        inst = inst.lower()
        combined = f"{inst} {text}"

        # 1. Multi-factor Specialization Scoring (🔴 6)
        words = combined.split()
        if not words:
            tech_density = 0.0
        else:
            tech_tokens = 0
            for w in words:
                # Acronyms (e.g. RFC, DMA, TCP)
                if w.isupper() and len(w) >= 2 and w.isalpha():
                    tech_tokens += 1
                # Numbers/hex values
                elif any(c.isdigit() for c in w):
                    tech_tokens += 1
                # Math/code operators or symbols
                elif any(c in w for c in ["_", "->", "==", "=", "\\", "$", "[", "]", "{", "}"]):
                    tech_tokens += 1
            tech_density = tech_tokens / len(words)

        # Rarity score: count density of rare/jargon words
        jargon_words = ["register", "offset", "address", "hex", "byte", "bit", "port", "instruction", "opcode", "dma", "interrupt", "pointer", "struct", "class", "function", "array", "null", "ptr"]
        jargon_count = sum(1 for w in words if any(j in w.lower() for j in jargon_words))
        rarity_score = jargon_count / len(words) if words else 0.0

        # Syllabus distance factor (based on default module concepts keyword matching)
        syllabus_dist = 0.0
        match = re.search(r'module_(\d+)', slot.slot_id or "")
        if match:
            mod_idx = int(match.group(1))
            from v0_1.module_alignment import DEFAULT_MODULE_CONCEPTS
            core_concepts = DEFAULT_MODULE_CONCEPTS.get(mod_idx, [])
            matched_concepts = sum(1 for concept in core_concepts if concept.lower() in combined)
            if matched_concepts == 0:
                syllabus_dist = 0.4  # Outside core module syllabus concepts
            elif matched_concepts == 1:
                syllabus_dist = 0.2
            else:
                syllabus_dist = 0.0
        else:
            syllabus_dist = 0.2

        # Combined specialization score (weights: 40% density, 30% rarity, 30% distance)
        spec_score = (tech_density * 0.4) + (rarity_score * 0.3) + (syllabus_dist * 0.3)
        
        # Hard penalty thresholds
        if spec_score > 0.65:
            return CheckResult.fail(
                "TEACHER_SUITABILITY_FAILURE",
                f"Question is excessively specialized (specialization score: {spec_score:.2f}). "
                "Please simplify and emphasize core syllabus concepts.",
                action=RetryAction.REGENERATE
            )

        # 2. Numerical Solvability Check
        if slot.question_type == "NUMERICAL":
            numbers_found = re.findall(r'\b\d+(?:\.\d+)?\b', combined)
            math_symbols = any(sym in combined for sym in ["=", "+", "-", "*", "/", "\\frac", "$"])
            
            # If numerical but doesn't provide solvable inputs/numbers, fail suitability
            if not numbers_found and not math_symbols:
                return CheckResult.fail(
                    "TEACHER_SUITABILITY_FAILURE",
                    "Question is marked as NUMERICAL but does not contain solvable inputs, equations, or parameters.",
                    action=RetryAction.REGENERATE
                )

            # Input variable and parameter grounding verification (🔴 1)
            if evidence_text:
                gen_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', combined))
                ev_numbers = set(re.findall(r'\b\d+(?:\.\d+)?\b', evidence_text.lower()))
                
                # Exclude standard page numbers, question/subquestion markers, and common constants
                structural_numbers = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "20"}
                gen_inputs = gen_numbers - structural_numbers
                ev_inputs = ev_numbers - structural_numbers
                
                ungrounded_inputs = gen_inputs - ev_inputs
                if ungrounded_inputs:
                    return CheckResult.fail(
                        "TEACHER_SUITABILITY_FAILURE",
                        f"Question introduces ungrounded numerical inputs/values: {ungrounded_inputs}. "
                        "All numeric inputs must be strictly sourced from the evidence chunk.",
                        action=RetryAction.REGENERATE
                    )

        # 3. Math Block Invariant Check (🔴 5)
        if question.math_blocks:
            for idx, block in enumerate(question.math_blocks):
                # Ensure id/block_id is present and non-empty
                bid = getattr(block, "block_id", None) or getattr(block, "id", None)
                if not bid or not str(bid).strip():
                    return CheckResult.fail(
                        "TEACHER_SUITABILITY_FAILURE",
                        f"Math block at index {idx} is missing a valid 'id' or 'block_id'.",
                        action=RetryAction.REGENERATE
                    )
                # Ensure latex is present and non-empty
                latex_val = getattr(block, "latex", None)
                if not latex_val or not str(latex_val).strip():
                    return CheckResult.fail(
                        "TEACHER_SUITABILITY_FAILURE",
                        f"Math block '{bid}' is missing a valid 'latex' expression.",
                        action=RetryAction.REGENERATE
                    )
                # Ensure source / evidence reference is present and non-empty
                source_val = getattr(block, "source", None)
                if not source_val or not str(source_val).strip():
                    return CheckResult.fail(
                        "TEACHER_SUITABILITY_FAILURE",
                        f"Math block '{bid}' is missing a valid 'source/evidence' reference.",
                        action=RetryAction.REGENERATE
                    )

        return CheckResult.pass_()
