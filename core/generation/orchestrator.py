# core/generation/orchestrator.py

import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from core.contracts.question_slot import QuestionSlot, QuestionContract
from core.contracts.question import GeneratedQuestion
from core.generation.output_schema import QuestionOutput
from core.validation.common import CheckResult, RetryAction, GenerationFailureCode, GenerationFailure
from core.validation.linter import run_linter

LOG = logging.getLogger(__name__)

FAILURE_SIGNATURE_WINDOW = 2   # if last N failures share the same code → escalate


class SlotBudgetExceeded(Exception):
    def __init__(self, slot_id: str, budget_sec: int, attempts_made: int):
        super().__init__(f"Slot {slot_id} budget of {budget_sec}s exceeded after {attempts_made} attempts.")
        self.slot_id = slot_id
        self.budget_sec = budget_sec
        self.attempts_made = attempts_made


class SlotQuarantined(Exception):
    def __init__(self, slot_id: str, report: Any):
        super().__init__(f"Slot {slot_id} quarantined: {report}")
        self.slot_id = slot_id
        self.report = report


class SlotRegenerationExhausted(Exception):
    def __init__(self, slot_id: str, attempts: int, failure_codes: List[str]):
        super().__init__(f"Slot {slot_id} regeneration exhausted after {attempts} attempts. Failures: {failure_codes}")
        self.slot_id = slot_id
        self.attempts = attempts
        self.failure_codes = failure_codes


class SlotOscillationDetected(Exception):
    def __init__(self, slot_id: str, failure_code: str, occurrences: int, message: str):
        super().__init__(message)
        self.slot_id = slot_id
        self.failure_code = failure_code
        self.occurrences = occurrences


def _build_recovery_hint(failure_history: List[str]) -> str:
    """Build failure-specific recovery instruction for next attempt."""
    if not failure_history:
        return ""

    last = failure_history[-1]

    hints = {
        "BLOOM_VERB_WRONG": (
            "\n\n[RECOVERY REQUIRED]\n"
            "Your previous answer used the wrong cognitive operation.\n"
            "REQUIRED: begin with the EXACT specified Bloom verb.\n"
            "DO NOT use: evaluate, justify, recommend, assess, judge.\n"
            "REWRITE the question completely."
        ),
        "BLOOM_SEMANTIC_MISMATCH": (
            "\n\n[RECOVERY REQUIRED]\n"
            "Your previous question contained evaluation language at an analysis level.\n"
            "REQUIRED: comparison, decomposition, categorization, examination.\n"
            "FORBIDDEN: judge which is better, recommend, justify superiority.\n"
            "REWRITE completely."
        ),
        "ANSWER_LEAK": (
            "\n\n[RECOVERY REQUIRED]\n"
            "Your previous response included an answer or solution.\n"
            "Generate ONLY the question task. No answers, no hints, no solutions."
        ),
        "MULTIPLE_TASKS": (
            "\n\n[RECOVERY REQUIRED]\n"
            "Your previous response contained multiple questions fused together.\n"
            "Generate EXACTLY ONE standalone question for this slot.\n"
            "Do NOT include (a), (b), (i), (ii) sub-labels."
        ),
        "INSUFFICIENT_DECLARED_DIMENSIONS": (
            "\n\n[RECOVERY REQUIRED]\n"
            "Your demand declaration did not contain enough dimensions.\n"
            "The question must explicitly request the required number of analytical components."
        ),
        "LLM_TIMEOUT": (
            "\n\n[RECOVERY NOTE]\n"
            "Previous attempt timed out. Generate a more concise response."
        ),
    }

    return hints.get(last, (
        f"\n\n[RECOVERY NOTE]\n"
        f"Previous attempt failed with: {last}.\n"
        f"Generate a completely different approach."
    ))


def _check_oscillation(failure_history: List[str], slot_id: str) -> None:
    """Detect repeated identical failures — prevent oscillation."""
    window = failure_history[-FAILURE_SIGNATURE_WINDOW:]
    if len(window) >= FAILURE_SIGNATURE_WINDOW and len(set(window)) == 1:
        raise SlotOscillationDetected(
            slot_id       = slot_id,
            failure_code  = window[0],
            occurrences   = len(window),
            message       = (
                f"Slot {slot_id} failed with {window[0]} "
                f"{len(window)} consecutive times. "
                f"Recovery not converging — blocking slot."
            )
        )


class SlotOrchestrator:
    """
    Drives sub-question generation and validation using a failure-specific 
    bounded retry state machine.
    """
    def __init__(self, llm_client=None, rng=None, artifact=None, profile=None):
        self.llm_client = llm_client
        self.rng = rng
        self.artifact = artifact
        self.profile = profile
        self.session_log: List[Dict[str, Any]] = []

    def generate(self, slot: QuestionSlot, evidence_pack, excluded_concepts: Set[str] = None) -> GeneratedQuestion:
        if excluded_concepts is None:
            excluded_concepts = set()

        # Read retry budget from runtime profile
        try:
            from runtime import get_active_profile
            prof = self.profile or get_active_profile()
            MAX_ATTEMPTS = prof.max_slot_attempts
            slot_budget_sec = getattr(prof, "slot_budget_sec", 120)
        except Exception:
            MAX_ATTEMPTS = 4
            slot_budget_sec = 120

        attempt = 1
        extra_hints = ""
        failure_history: List[str] = []
        start_time = time.monotonic()
        
        while attempt <= MAX_ATTEMPTS:
            # Slot budget check
            if time.monotonic() - start_time > slot_budget_sec:
                raise SlotBudgetExceeded(
                    slot_id=slot.slot_id,
                    budget_sec=slot_budget_sec,
                    attempts_made=attempt - 1
                )

            # Generate candidate attempt slot with iterated seed
            attempt_slot = slot.make_attempt_slot(attempt - 1) if hasattr(slot, "make_attempt_slot") else slot

            # 1. Format prompt using template and recovery hint
            recovery_hint = _build_recovery_hint(failure_history)
            combined_hints = (extra_hints + "\n" + recovery_hint).strip()
            prompt = self._format_prompt(attempt_slot, evidence_pack, combined_hints)
            
            try:
                # 2. Call LLM
                raw_json = self._call_llm(prompt)
                
                # 3. Parse JSON
                data = self._parse_json(raw_json)
                output = QuestionOutput(**data)

                # Populate MathBlock source fields automatically from evidence chunk metadata
                from core.generation.output_schema import MathSource
                chunk_id = attempt_slot.evidence_ids[0] if attempt_slot.evidence_ids else "unknown_chunk"
                page_num = getattr(evidence_pack, "page", None)
                if page_num is None:
                    if hasattr(evidence_pack, "page_number"):
                        page_num = getattr(evidence_pack, "page_number")
                    elif isinstance(evidence_pack, dict):
                        page_num = evidence_pack.get("page") or evidence_pack.get("page_number")
                
                for block in output.math_blocks:
                    if not block.source:
                        block.source = MathSource(chunk_id=chunk_id, page=page_num)

            except Exception as e:
                failure_code_str = "SCHEMA_FAILURE"
                failure_history.append(failure_code_str)
                _check_oscillation(failure_history, attempt_slot.slot_id)

                failure = GenerationFailure(
                    slot_id=attempt_slot.slot_id,
                    code=GenerationFailureCode.SCHEMA_FAILURE,
                    category="SCHEMA",
                    message=f"JSON or schema parsing failed: {str(e)}",
                    retryable=(attempt < MAX_ATTEMPTS),
                    attempt=attempt
                )
                self.session_log.append(failure.__dict__)
                LOG.warning(f"[ORCHESTRATOR] Slot {attempt_slot.slot_id} Attempt {attempt} failed: {failure.message}")
                
                if attempt == MAX_ATTEMPTS:
                    raise SlotRegenerationExhausted(attempt_slot.slot_id, MAX_ATTEMPTS, failure_history)
                
                extra_hints = self._compile_extra_hints(failure)
                attempt += 1
                continue

            # 4. Create GeneratedQuestion candidate
            candidate = GeneratedQuestion(output, attempt_slot)

            # 5. Run Linter & Two-Layer Bloom check
            contract = attempt_slot.to_contract()
            if hasattr(evidence_pack, "combined_text"):
                evidence_text = getattr(evidence_pack, "combined_text")
            elif hasattr(evidence_pack, "text"):
                evidence_text = getattr(evidence_pack, "text")
            elif isinstance(evidence_pack, dict):
                evidence_text = evidence_pack.get("text") or evidence_pack.get("combined_text") or str(evidence_pack)
            elif isinstance(evidence_pack, str):
                evidence_text = evidence_pack
            else:
                evidence_text = str(evidence_pack)
            report = run_linter(output, candidate, attempt_slot, contract, evidence_text=evidence_text)

            # Two-layer Bloom validation integration
            try:
                from core.validation.bloom_validator import check_bloom_two_layer
                bloom_res = check_bloom_two_layer(output.instruction, attempt_slot)
                if not bloom_res.passed and report.passed:
                    # Upgrade bloom failure if linter passed
                    report.add_result(bloom_res.to_check_result())
            except Exception as be:
                LOG.debug(f"[BLOOM VALIDATOR] Check skipped: {be}")

            if getattr(report, "status", None) == "QUARANTINE":
                raise SlotQuarantined(attempt_slot.slot_id, report)
            
            if report.passed:
                candidate.status = "VALIDATED"
                LOG.info(f"[ORCHESTRATOR] Slot {attempt_slot.slot_id} passed validation on attempt {attempt}")
                self.session_log.append({
                    "slot_id": attempt_slot.slot_id,
                    "attempt": attempt,
                    "status": "PASS"
                })
                return candidate
            
            # FAIL! Reconcile failure
            failed_check = report.get_failure()
            failure_code = self._map_check_to_failure_code(failed_check.code)
            failure_history.append(failed_check.code)
            _check_oscillation(failure_history, attempt_slot.slot_id)

            failure = GenerationFailure(
                slot_id=attempt_slot.slot_id,
                code=failure_code,
                category="LINTER",
                message=failed_check.message,
                retryable=(attempt < MAX_ATTEMPTS and failed_check.action != RetryAction.CRITICAL),
                attempt=attempt
            )
            
            self.session_log.append(failure.__dict__)
            LOG.warning(f"[ORCHESTRATOR] Slot {attempt_slot.slot_id} Attempt {attempt} failed linter check: {failed_check.code} - {failed_check.message}")
            
            if attempt == MAX_ATTEMPTS or not failure.retryable:
                raise SlotRegenerationExhausted(attempt_slot.slot_id, attempt, failure_history)
            
            extra_hints = self._compile_extra_hints(failure)
            
            if failed_check.action == RetryAction.REBUILD_EVIDENCE or failure.code == GenerationFailureCode.EVIDENCE_FAILURE:
                evidence_pack = self._reload_evidence(attempt_slot, excluded_concepts)
                
            attempt += 1
            
        raise SlotRegenerationExhausted(slot.slot_id, MAX_ATTEMPTS, failure_history)

    def _format_prompt(self, slot: QuestionSlot, evidence_pack, extra_hints: str) -> str:
        evidence_text = getattr(evidence_pack, "combined_text", "") if hasattr(evidence_pack, "combined_text") else str(evidence_pack)
        math_artifacts = getattr(evidence_pack, "math_artifacts", "") if hasattr(evidence_pack, "math_artifacts") else "none"
        
        math_policy = "REQUIRED" if slot.math_required else "FORBIDDEN"
        visual_policy = "REQUIRED" if slot.visual_required else "FORBIDDEN"
        
        allowed_sec = ", ".join(slot.task_signature.allowed_secondary_operations) if slot.task_signature.allowed_secondary_operations else "none"
        
        from core.contracts.demand_profile import DemandProfile
        profile = DemandProfile.from_contract(slot.to_contract())
        min_dims = profile.min_dimensions
        
        # Determine a compliant second verb for the few-shot example
        from core.validation.linter import VERB_OPERATION_MAP
        sec_verb = None
        if slot.task_signature.allowed_secondary_operations:
            for verb, op in VERB_OPERATION_MAP.items():
                if op in slot.task_signature.allowed_secondary_operations:
                    sec_verb = verb.capitalize()
                    break
        if not sec_verb:
            for verb, op in VERB_OPERATION_MAP.items():
                if op == slot.bloom_operation and verb.lower() != slot.bloom_verb.lower():
                    sec_verb = verb.capitalize()
                    break
        if not sec_verb:
            sec_verb = "List" if slot.bloom_verb.lower() != "list" else "Define"

        if min_dims > 1:
            example_text = f"{slot.bloom_verb} how Binary Search Trees operate and {sec_verb.lower()} how they maintain sorted values."
        else:
            example_text = f"{slot.bloom_verb} how Binary Search Trees operate."

        prompt = f"""Generate ONE examination sub-question matching this contract:
Topic: {slot.topic}
Bloom Verb: {slot.bloom_verb} (primary operation: {slot.bloom_operation})
Marks: {slot.marks}
Difficulty: {slot.difficulty}
Question Type: {slot.question_type}
Allowed Operations: {slot.bloom_operation} and {allowed_sec}
Math Policy: {math_policy}
Visual Policy: {visual_policy}
Min Clauses: {min_dims} (requires at least {min_dims} distinct parts split by 'and', 'or', 'as well as', or commas)

EVIDENCE:
{evidence_text}

MATH ARTIFACTS:
{math_artifacts}

EXAMPLE OF A VALID OUTPUT FORMAT:
If the Topic was "Binary Search Trees", Bloom Verb was "{slot.bloom_verb}", and Min Clauses was {min_dims}, a valid output would be:
{{
  "instruction": "{example_text}",
  "question_text": "{example_text}",
  "math_blocks": [],
  "diagram_request": null
}}

RULES:
1. Both "instruction" and "question_text" fields MUST begin with the word "{slot.bloom_verb}" (case-insensitive).
2. Write a single compound sentence containing at least {min_dims} clauses connected by conjunctions (such as 'and', 'or', 'as well as') or commas.
3. Place comparison or justification words (like 'compare', 'justify', 'evaluate') LATER in the sentence, NEVER at the start.
4. Use ONLY these action verbs: {slot.bloom_verb} or verbs from {slot.bloom_operation} / {allowed_sec} operations. Do NOT use other verbs.
5. Do NOT include any question numbers, parts like "(a)", "3(a)", or words like "OR".
6. Do NOT refer to source notes, provided materials, or documents.
7. Math policy: {math_policy} (declare exactly {1 if slot.math_required else 0} MathBlocks in math_blocks).
8. Return ONLY valid JSON matching this schema:
{{
  "instruction": "{slot.bloom_verb} [clause 1] and [clause 2]...",
  "question_text": "{slot.bloom_verb} [clause 1] and [clause 2]...",
  "math_blocks": [],
  "diagram_request": null
}}"""

        if profile.requires_comparison:
            prompt += f"\n9. COMPARISON REQUIRED: The question text and instruction MUST contain comparison keywords such as 'compare', 'contrast', 'distinguish', or 'differentiate' placed in the middle or end of the sentence."
        if profile.requires_justification:
            prompt += f"\n10. JUSTIFICATION REQUIRED: The question text and instruction MUST contain justification keywords such as 'justify', 'critique', 'reconcile', or 'evaluate' placed in the middle or end of the sentence."

        prompt += f"\n11. QUESTION TYPE TARGET: This question must be formatted as {slot.question_type}. "
        if slot.question_type == "NUMERICAL":
            prompt += "You MUST write a calculation, problem-solving, or numerical analysis question using the formulas or numerical data present in the evidence. Do NOT invent external formulas."
        elif slot.question_type == "APPLICATION":
            prompt += "Focus on practical engineering application scenarios derived from the evidence."
        else:
            prompt += "Focus on core theory, conceptual understanding, definitions, or descriptive explanations."

        if extra_hints:
            prompt += f"\n\nADDITIONAL RECOVERY INSTRUCTIONS:\n{extra_hints}"

        return prompt

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client:
            if hasattr(self.llm_client, "call"):
                from core.generation.robust_llm_caller import LLMRequest
                req = LLMRequest(model="qwen2.5:7b", prompt=prompt)
                resp = self.llm_client.call(req)
                if resp.success and resp.text:
                    return resp.text
                elif resp.timed_out:
                    raise TimeoutError("LLM call timed out")
                else:
                    raise RuntimeError(f"LLM call failed: {resp.error}")
            elif hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt)
        
        from v0_1.llm import get_best_llm
        caller = get_best_llm()
        res = caller.call(prompt, max_tokens=1024)
        if not res:
            raise RuntimeError("LLM call returned empty response or timed out.")
        return res

    def _parse_json(self, raw: str) -> dict:
        cleaned = raw.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        return json.loads(cleaned)

    def _map_check_to_failure_code(self, check_code: str) -> GenerationFailureCode:
        try:
            return GenerationFailureCode(check_code)
        except ValueError:
            mapping = {
                "BLOOM_VERB_NOT_AT_START": GenerationFailureCode.BLOOM_MISMATCH,
                "BLOOM_VERB_WRONG": GenerationFailureCode.BLOOM_MISMATCH,
                "BLOOM_SEMANTIC_MISMATCH": GenerationFailureCode.BLOOM_MISMATCH,
                "DISALLOWED_SECONDARY_TASK": GenerationFailureCode.BLOOM_MISMATCH,
                "COMPARISON_NOT_DECLARED": GenerationFailureCode.DEMAND_FAILURE,
                "JUSTIFICATION_NOT_DECLARED": GenerationFailureCode.DEMAND_FAILURE,
                "CALCULATION_VERB_MISSING": GenerationFailureCode.DEMAND_FAILURE,
                "INSUFFICIENT_DECLARED_DIMENSIONS": GenerationFailureCode.DEMAND_FAILURE,
                "MATH_PLACEHOLDER_UNRESOLVED": GenerationFailureCode.MATH_FAILURE,
                "MATH_BLOCK_UNREFERENCED": GenerationFailureCode.MATH_FAILURE,
                "MATH_CORRUPTED": GenerationFailureCode.MATH_FAILURE,
                "MATH_RENDER_ERROR": GenerationFailureCode.MATH_FAILURE,
                "MATH_RENDER_EMPTY": GenerationFailureCode.MATH_FAILURE,
                "MATH_RENDER_FAILURE": GenerationFailureCode.MATH_FAILURE,
                "MATH_POLICY_VIOLATION": GenerationFailureCode.MATH_FAILURE,
                "VISUAL_POLICY_VIOLATION": GenerationFailureCode.MATH_FAILURE,
                "MULTI_SLOT_CONTAMINATION": GenerationFailureCode.MULTI_SLOT_CONTAMINATION,
                "ANSWERABILITY_FAILURE": GenerationFailureCode.ANSWERABILITY_FAILURE,
                "CONTRACT_FAILURE": GenerationFailureCode.CONTRACT_FAILURE,
                "EVIDENCE_FAILURE": GenerationFailureCode.EVIDENCE_FAILURE,
            }
            return mapping.get(check_code, GenerationFailureCode.SCHEMA_FAILURE)

    def _get_recovery_hints(self, code: GenerationFailureCode) -> str:
        hints = "\n    ADDITIONAL RECOVERY INSTRUCTIONS:\n    ────────────────────────────────\n"
        if code == GenerationFailureCode.META_LANGUAGE:
            hints += "    - DANGER: Do NOT make references to source notes, provided materials, or uploaded files.\n"
        elif code == GenerationFailureCode.ANSWER_LEAK:
            hints += "    - DANGER: Do NOT include solutions, correct options, model answers, or phrases like 'therefore', 'result is' in your output.\n"
        elif code == GenerationFailureCode.BLOOM_MISMATCH:
            hints += "    - DANGER: Ensure the instruction clause starts EXACTLY with the requested Bloom verb, and contains only the primary/allowed secondary tasks.\n"
        elif code == GenerationFailureCode.DEMAND_FAILURE:
            hints += "    - DANGER: Ensure the question text has adequate complexity. Split the tasks clearly to meet all required dimensions.\n"
        elif code == GenerationFailureCode.MATH_FAILURE:
            hints += "    - DANGER: Fix unbalanced braces, invalid fractions, or unresolved placeholders. Keep math canonical.\n"
        elif code == GenerationFailureCode.MULTI_SLOT_CONTAMINATION:
            hints += "    - DANGER: Do NOT write structural labels like (a), (b), Q1, or OR in your question text. Only generate the raw question body.\n"
        elif code == GenerationFailureCode.ANSWERABILITY_FAILURE:
            hints += "    - DANGER: Provide a clear instruction with an action verb, and write a question whose complexity is fully matching the marks.\n"
        elif code == GenerationFailureCode.SCHEMA_FAILURE:
            hints += "    - DANGER: You must output ONLY valid JSON matching the schema. No markdown formatting outside of code blocks.\n"
        else:
            hints += "    - Ensure strict alignment with the requested policies and evidence.\n"
        return hints

    def _compile_extra_hints(self, failure: GenerationFailure) -> str:
        history_lines = []
        for entry in self.session_log:
            if entry.get("slot_id") == failure.slot_id and entry.get("status") != "PASS":
                code_val = entry.get("code")
                msg_val = entry.get("message", "")
                attempt_val = entry.get("attempt", 1)
                short_msg = msg_val[:120] + "..." if len(msg_val) > 120 else msg_val
                code_str = code_val.value if hasattr(code_val, "value") else str(code_val)
                history_lines.append(f"  - Attempt {attempt_val} failed: {code_str} ({short_msg})")
        
        history_text = "\n".join(history_lines)
        base_hint = self._get_recovery_hints(failure.code)
        
        return (
            f"\nPREVIOUS ATTEMPTS HISTORY FOR THIS SLOT:\n"
            f"{history_text}\n"
            f"{base_hint}"
        )

    def _reload_evidence(self, slot: QuestionSlot, excluded_concepts: Set[str]):
        if self.artifact:
            try:
                from core.evidence.pack_builder import EvidencePackBuilder
                return EvidencePackBuilder.build(
                    slot=slot,
                    artifact=self.artifact,
                    rng=self.rng,
                    excluded_concepts=excluded_concepts
                )
            except Exception:
                pass
        return self.artifact
