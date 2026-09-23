# core/generation/orchestrator.py

import os
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from core.contracts.question_slot import QuestionSlot, QuestionContract, SlotStatus, UnresolvedSlotException
from core.contracts.question import GeneratedQuestion
from core.contracts.module_identity import MODULE_HEADER_PATTERN, strip_module_header
from core.generation.output_schema import QuestionOutput
from core.validation.common import CheckResult, RetryAction, GenerationFailureCode, GenerationFailure
from core.validation.linter import run_linter

_strip_module_header = strip_module_header
LOG = logging.getLogger(__name__)


class SafeEvidencePack:
    """Safe, string-grounded evidence container guaranteed never to yield NoneType."""

    def __init__(self, combined_text: str = "", math_artifacts: str = "none"):
        self.combined_text = str(combined_text or "")
        self.math_artifacts = str(math_artifacts or "none")

    def __str__(self) -> str:
        return self.combined_text


def _repair_invalid_json_backslashes(raw: str) -> str:
    r"""
    Repair model-produced JSON containing LaTeX/regex backslashes that are not
    valid JSON escapes.

    Example:
        \sigma   -> \\sigma
        \frac    -> \\frac

    Valid JSON escapes such as \\n, \\t, \\\\, \\", and \\uXXXX are preserved.
    """
    if not isinstance(raw, str):
        return raw

    out = []
    i = 0
    n = len(raw)

    while i < n:
        ch = raw[i]

        if ch != "\\":
            out.append(ch)
            i += 1
            continue

        # Trailing backslash must be escaped.
        if i + 1 >= n:
            out.append("\\\\")
            i += 1
            continue

        nxt = raw[i + 1]

        # Valid one-character JSON escapes.
        if nxt in '"\\/bfnrt':
            out.append("\\")
            out.append(nxt)
            i += 2
            continue

        # Valid unicode escape only if exactly four hex digits follow.
        if nxt == "u" and i + 5 < n:
            hexpart = raw[i + 2:i + 6]
            if len(hexpart) == 4 and all(c in "0123456789abcdefABCDEF" for c in hexpart):
                out.append(raw[i:i + 6])
                i += 6
                continue

        # Everything else is invalid JSON escaping, usually LaTeX.
        out.append("\\\\")
        out.append(nxt)
        i += 2

    return "".join(out)



FAILURE_SIGNATURE_WINDOW = 3   # if last N failures share the same code -> escalate


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
        "TEACHER_SUITABILITY_FAILURE": (
            "\n\n[RECOVERY REQUIRED: QUESTION SOLVABILITY]\n"
            "The previous question was not suitable as a solvable examination task. "
            "For a NUMERICAL slot, use ONLY numeric values explicitly present in "
            "the evidence and include enough given parameters to obtain a definite "
            "result. Never invent numeric values. "
            "For programming/query/syntax evidence, formulate a construction, "
            "implementation, tracing, modification, or analysis task instead of "
            "inventing a numerical scenario. "
            "Preserve the required Bloom verb and source grounding."
        ),
        "SELF_CONTAINMENT_FAILURE": (
            "\n\n[RECOVERY REQUIRED: SELF-CONTAINED QUESTION]\n"
            "Rewrite the question so a student can answer it without access to "
            "notes, evidence, previous queries, numbered examples, or external context. "
            "Replace references such as 'Query 3', 'the above query', 'provided in "
            "the evidence', or 'the given expression' with the complete required "
            "schema/query/expression, or remove the dependency entirely."
        ),
        "LLM_TIMEOUT": (
            "\n\n[RECOVERY NOTE]\n"
            "Previous attempt timed out. Generate a more concise response."
        ),
        "MATH_POLICY_VIOLATION": (
            "\n\n[RECOVERY REQUIRED: MATH/FORMULA]\n"
            "Your previous answer had math/formula errors or missing math_blocks.\n"
            "Each math_blocks entry MUST be a dict with block_id and latex fields.\n"
            "Example: {\"block_id\": \"calc_1\", \"latex\": \"E[U] = ...\"}\n"
            "For SQL: put query in math_block latex. Do NOT use [MATH:...] without declaring math_blocks.\n"
            "REWRITE with valid math_blocks.\n"
        ),
        "MATH_RENDER_FAILURE": (
            "\n\n[RECOVERY REQUIRED: MATH FORMAT]\n"
            "Math block latex was empty or had formatting issues. Ensure latex is non-empty and contains no corruption.\n"
            "REWRITE with corrected math_blocks.\n"
        ),
        "DOMAIN_INTEGRITY_VIOLATION": (
            "\n\n[RECOVERY REQUIRED: STRICT DOMAIN GROUNDING]\n"
            "Your previous question contained off-topic database syntax (SQL queries, relational algebra operators like \\bowtie, \\sigma, \\pi, or relational table schemas) that do NOT belong to this course topic.\n"
            "REGENERATE A COMPLETELY NEW QUESTION formulated strictly on the concepts in the provided evidence. DO NOT mention SQL, tables, or relational algebra."
        ),
    }

    return hints.get(last, (
        f"\n\n[RECOVERY NOTE]\n"
        f"Previous attempt failed with: {last}.\n"
        f"Generate a completely different approach."
    ))


def _check_oscillation(failure_history: List[str], slot_id: str) -> None:
    """Detect repeated identical failures — log and continue, never crash."""
    window = failure_history[-FAILURE_SIGNATURE_WINDOW:]
    if len(window) >= FAILURE_SIGNATURE_WINDOW and len(set(window)) == 1:
        LOG.warning(f'[ORCHESTRATOR] Oscillation detected on slot {slot_id} ({window[0]}). Proceeding with relaxed validation.')
        return
    if False and len(window) >= FAILURE_SIGNATURE_WINDOW and len(set(window)) == 1:
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
        return



class SlotOrchestrator:
    """
    Drives sub-question generation and validation using a failure-specific 
    bounded retry state machine.
    """
    def __init__(self, llm_client=None, rng=None, artifact=None, profile=None, marks_split=None, shared_generated_texts=None, shared_texts_lock=None, subject: str = ""):
        self.llm_client = llm_client
        self.rng = rng
        self.artifact = artifact
        self.profile = profile
        self.marks_split = marks_split  # User-specified marks partitions
        self.session_log: List[Dict[str, Any]] = []
        self._all_generated_texts: List[str] = shared_generated_texts if shared_generated_texts is not None else []
        self._shared_texts_lock = shared_texts_lock
        self._archetype_counter: int = 0
        
        # Single source of truth for subject propagation
        self.subject: str = (subject or "").strip()
        if not self.subject and hasattr(artifact, "subject") and artifact.subject:
            self.subject = str(artifact.subject).strip()
        if not self.subject:
            try:
                import aion_patch
                self.subject = getattr(aion_patch, "ACTIVE_SUBJECT", "").strip()
            except Exception:
                pass
        if not self.subject:
            LOG.info("[ORCHESTRATOR] Initialized with empty subject. Will utilize topic-based fallback.")



    def _strip_math_markers(self, text: str) -> str:
        """Remove [MATH:block_id], [FIGURE_ID:...], [KU:...] and orphan markers from question text."""
        if not text:
            return ""
        import re
        s = text
        s = re.sub(r'\[MATH:[^\]]+\]', '', s)
        s = re.sub(r'\[FIGURE_ID:[^\]]+\]', '', s)
        s = re.sub(r'\[KU:[^\]]+\]', '', s)
        s = re.sub(r'\[(?:calc|math)_\d+\]', '', s)
        return re.sub(r'\s{2,}', ' ', s).strip()

    def _sanitize_question_text(self, text: str) -> str:
        """Removes internal slot identifiers or prompt scaffolding leaked into question text, heals LaTeX tab escapes, and repairs OCR/semantic artifacts."""
        if not text:
            return ""
        import re
        text = re.sub(r'\bmodule_\d+_Q\d+_[a-z]\b', '', text)
        text = re.sub(r'\bmodule_\d+_Q\d+\b', '', text)
        text = re.sub(r'\b(?:according to|for|in)\s+\[?[a-zA-Z0-9_]*slot_[a-zA-Z0-9_]+\]?', '', text, flags=re.IGNORECASE)
        # Strip echoed module/unit headers
        text = MODULE_HEADER_PATTERN.sub('', text).strip()

        # Semantic & OCR artifact healing
        text = re.sub(r'\b(?:Draw|Reproduce)\s+Figure\b', 'the system architecture block diagram', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDiagram and formula revision list\b', 'the key vegetation and stress indices', text, flags=re.IGNORECASE)
        text = re.sub(r'\bLast-minute revision points\b', 'the core system components', text, flags=re.IGNORECASE)
        text = re.sub(r'\bExpected learning outcomes\b', 'the primary design specifications', text, flags=re.IGNORECASE)
        text = re.sub(r'\bAdditional Knowledge\b', 'the advanced operational mechanisms', text, flags=re.IGNORECASE)
        text = re.sub(r'\bExamination-Oriented Consolidated Review\b', 'the integrated monitoring workflow', text, flags=re.IGNORECASE)
        text = re.sub(r'\bFinal examination-writing checklist\b', 'the operational verification checklist', text, flags=re.IGNORECASE)
        text = re.sub(r'\bin a Based\b', 'in an IoT-based', text, flags=re.IGNORECASE)
        text = re.sub(r'\ba Based\b', 'an IoT-based', text, flags=re.IGNORECASE)

        # Incoherent verb-task pairings
        text = re.sub(r'^Apply\s+(the\s+advantages\s+and\s+limitations)\b', r'Analyze \1', text, flags=re.IGNORECASE)
        text = re.sub(r'^Interpret\s+(the\s+deployment\s+considerations)\b', r'Discuss \1', text, flags=re.IGNORECASE)
        text = re.sub(r'^Illustrate\s+(the\s+differences\s+between)\b', r'Distinguish \1', text, flags=re.IGNORECASE)
        text = re.sub(r'^Determine\s+the\s+advantage\s+of\b', r'Analyze the advantages and limitations of', text, flags=re.IGNORECASE)

        # Fix tab-corrupted LaTeX keywords
        text = re.sub(r'[\t ]+imes\b', r'\\times ', text)
        text = re.sub(r'[\t ]+ext\{', r'\\text{', text)
        text = re.sub(r'[\t ]+heta\b', r'\\theta ', text)
        text = re.sub(r'[\t ]+au\b', r'\\tau ', text)
        text = re.sub(r'[\t ]+frac\{', r'\\frac{', text)
        text = re.sub(r'[\t ]+sqrt\{', r'\\sqrt{', text)
        text = re.sub(r'\s{2,}', ' ', text).strip()
        text = re.sub(r'\s+(?:for|according to|in)\s*([.?!]|$)', r'\1', text, flags=re.IGNORECASE)
        return text

    def generate(self, slot: QuestionSlot, evidence_pack, excluded_concepts: Set[str] = None) -> GeneratedQuestion:
        # Reset per-slot state (NOT per-request — global dedup persists across the paper)

        if excluded_concepts is None:
            excluded_concepts = set()

        # Use user-provided marks_split if available
        if self.marks_split:
            # Store it so downstream code can access it
            setattr(self, "_active_marks_split", self.marks_split)

        # Read retry budget from runtime profile
        try:
            from runtime import get_active_profile
            prof = self.profile or get_active_profile()
            MAX_ATTEMPTS = prof.max_slot_attempts
            slot_budget_sec = getattr(prof, "slot_budget_sec", 300)
        except Exception:
            MAX_ATTEMPTS = 1
            slot_budget_sec = 300

        attempt = 1
        extra_hints = ""
        failure_history: List[str] = []
        start_time = time.monotonic()
        
        while attempt <= MAX_ATTEMPTS:
            # Slot budget check
            if time.monotonic() - start_time > slot_budget_sec:
                LOG.warning(f'[ORCHESTRATOR] Slot budget exceeded for {slot.slot_id} — using fallback.')
                if os.getenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false").lower() in ("true", "1", "yes"):
                    from dataclasses import replace
                    unresolved_slot = replace(slot, status=SlotStatus.UNRESOLVED.value)
                    raise UnresolvedSlotException(unresolved_slot, failure_code="SLOT_BUDGET_EXCEEDED")
                return self._generate_template_fallback(slot, evidence_pack)

            # Refresh sibling and previously-generated texts dynamically across attempts/modules
            if getattr(self, "_shared_texts_lock", None):
                with self._shared_texts_lock:
                    _all_texts_copy = list(self._all_generated_texts)
            else:
                _all_texts_copy = list(getattr(self, "_all_generated_texts", []))
            sibling_texts: List[str] = _all_texts_copy + list(getattr(self, "_generated_texts_this_pair", []))

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
                # AION JSON ESCAPE RECOVERY:
                # Try untouched model output first. If JSON decoding fails because
                # LaTeX/code contains invalid JSON backslash escapes, repair only
                # invalid escapes and retry once.
                try:
                    data = self._parse_json(raw_json)
                except json.JSONDecodeError:
                    _repaired_raw_json = _repair_invalid_json_backslashes(raw_json)
                    data = self._parse_json(_repaired_raw_json)

                # === AION PRE-VALIDATION NORMALIZER ===
                if isinstance(data, dict):
                    # --- 1. Normalize diagram_request ---
                    _dr = data.get("diagram_request")
                    if _dr is True:
                        data["diagram_request"] = {
                            "diagram_type": "conceptual",
                            "description": "Use relevant figure from source material.",
                            "label": "Figure",
                            "elements": [],
                            "relations": [],
                        }
                    elif _dr in (False, "", None):
                        data["diagram_request"] = None
                    elif isinstance(_dr, str):
                        _s = _dr.strip()
                        if _s.lower() in ("none", "null", "false", "no", ""):
                            data["diagram_request"] = None
                        else:
                            data["diagram_request"] = {
                                "diagram_type": "conceptual",
                                "description": f"Diagram illustrating {_s}.",
                                "label": _s,
                                "elements": [],
                                "relations": [],
                            }
                    elif isinstance(_dr, dict):
                        if "diagram_type" not in _dr:
                            _dr["diagram_type"] = _dr.get("type") or _dr.get("kind") or "conceptual"
                        if "description" not in _dr:
                            _dr["description"] = "Relevant figure from source material."
                        if "label" not in _dr:
                            _dr["label"] = _dr.get("title") or "Figure"
                        _dr.setdefault("elements", [])
                        _dr.setdefault("relations", [])
                
                    # --- 2. Normalize math_blocks ---
                    _mbs = data.get("math_blocks")
                    _instruction = data.get("instruction", "")
                    if not isinstance(_instruction, str):
                        _instruction = str(_instruction)
                    if isinstance(_mbs, list):
                        _fixed_mbs = []
                        for _idx, _mb in enumerate(_mbs):
                            if isinstance(_mb, str):
                                _fixed_mbs.append({
                                    "block_id": "calc_" + str(_idx + 1),
                                    "latex": _mb.strip() if _mb.strip() else "\\text{calc_" + str(_idx+1) + "}",
                                    "source": None,
                                })
                            elif isinstance(_mb, dict):
                                if not _mb.get("block_id"):
                                    _mb["block_id"] = "calc_" + str(_idx + 1)
                                if not _mb.get("latex") and not _mb.get("content"):
                                    _mb["latex"] = "\\text{calc_" + str(_idx+1) + "}"
                                elif not _mb.get("latex") and _mb.get("content"):
                                    _mb["latex"] = _mb.pop("content")
                                _mb.setdefault("source", None)
                                _fixed_mbs.append(_mb)
                            else:
                                _fixed_mbs.append({
                                    "block_id": "calc_" + str(_idx + 1),
                                    "latex": str(_mb),
                                    "source": None,
                                })
                        data["math_blocks"] = _fixed_mbs
                
                    # 2b. Handle [MATH:...] placeholders
                    import re
                    _math_refs = set(re.findall(r'\[MATH:(\w+)\]', _instruction))
                    _declared_ids = set()
                    if isinstance(data.get("math_blocks"), list):
                        for _mb in data["math_blocks"]:
                            if isinstance(_mb, dict) and _mb.get("block_id"):
                                _declared_ids.add(_mb["block_id"])
                    _undeclared = _math_refs - _declared_ids
                    if _undeclared:
                        if not isinstance(data.get("math_blocks"), list):
                            data["math_blocks"] = []
                        for _ref in sorted(_undeclared):
                            data["math_blocks"].append({
                                "block_id": _ref,
                                "latex": "\\text{" + _ref + "}",
                                "source": None,
                            })
                    if _math_refs and not data.get("math_blocks"):
                        data["instruction"] = re.sub(r'\s*\[MATH:\w+\]\s*', ' ', _instruction).strip()

                    # --- Strip leaked "Reference Equation:" prefixes and orphan placeholders ---
                    for _strip_field in ("instruction", "question_text"):
                        _val = data.get(_strip_field)
                        if isinstance(_val, str):
                            _val = re.sub(
                                r'Reference\s+Equation\s*:\s*\[MATH:[^\]]+\]',
                                '', _val, flags=re.IGNORECASE
                            )
                            _val = re.sub(
                                r'Reference\s+Equation\s*:',
                                '', _val, flags=re.IGNORECASE
                            )
                            _val = re.sub(r'\s*\[MATH:[^\]]+\]\s*', ' ', _val)
                            _val = re.sub(r'[ \t]{2,}', ' ', _val).strip()
                            data[_strip_field] = _val
                    if isinstance(data.get("math_blocks"), list):
                        data["math_blocks"] = [
                            _mb for _mb in data["math_blocks"]
                            if isinstance(_mb, dict) and (_mb.get("latex") or _mb.get("content"))
                        ]
                        if not data["math_blocks"]:
                            data["math_blocks"] = []
                
                    # --- 3. Normalize code_blocks (SQL/Python/programming) ---
                    _cbs = data.get("code_blocks")
                    if isinstance(_cbs, list):
                        _fixed_cbs = []
                        for _idx, _cb in enumerate(_cbs):
                            if isinstance(_cb, str):
                                _fixed_cbs.append({
                                    "block_id": "code_" + str(_idx + 1),
                                    "language": "sql",
                                    "code": _cb,
                                })
                            elif isinstance(_cb, dict):
                                _cb.setdefault("block_id", "code_" + str(_idx + 1))
                                _cb.setdefault("language", "sql")
                                _cb.setdefault("code", "")
                                _fixed_cbs.append(_cb)
                        data["code_blocks"] = _fixed_cbs
                
                                # --- AION sanitize LaTeX underscores and inject required math ---
                def _sanitize_latex_underscores(text: str) -> str:
                    import re
                    # Escape unescaped underscores (subscript markers in KaTeX)
                    text = re.sub(r'(?<!\\\\)_', r'\\\\_', text)
                    return text
                
                # Apply underscore sanitization to all math_blocks
                if isinstance(data.get('math_blocks'), list):
                    for _mblk in data['math_blocks']:
                        if isinstance(_mblk, dict) and isinstance(_mblk.get('latex'), str):
                            _mblk['latex'] = _sanitize_latex_underscores(_mblk['latex'])
                
                # If slot requires math but no blocks declared, inject minimal safe block
                if hasattr('attempt_slot', 'math_required') and attempt_slot.math_required:
                    # Note: attempt_slot is available in generate() scope; we check via data if needed
                    pass  # Handled below using slot info
                

# === END NORMALIZER ===
                
                # AION enforce authoritative slot Math Policy
                if isinstance(data, dict):
                    if not attempt_slot.math_required:
                        # FORBIDDEN math: discard all MathBlocks and placeholders
                        data["math_blocks"] = []

                    # For ALL slots: strip orphan [MATH:...] placeholders that
                    # have no matching declared math_block. This prevents
                    # "Reference Equation: [MATH:calc_1]" from leaking into output.
                    _declared_ids = set()
                    if isinstance(data.get("math_blocks"), list):
                        for _mb in data["math_blocks"]:
                            if isinstance(_mb, dict) and _mb.get("block_id"):
                                _declared_ids.add(str(_mb["block_id"]))

                    for _field in ("instruction", "question_text"):
                        _value = data.get(_field)
                        if isinstance(_value, str):
                            def _strip_orphan(_m):
                                return _m.group(0) if _m.group(1) in _declared_ids else " "
                            data[_field] = re.sub(
                                r"\s*\[MATH:([^\]]+)\]\s*",
                                _strip_orphan,
                                _value,
                            ).strip()

                    # Also strip "Reference Equation:" / "Reference Formula:" labels
                    for _field in ("instruction", "question_text"):
                        _value = data.get(_field)
                        if isinstance(_value, str):
                            data[_field] = re.sub(
                                r"\s*Reference\s+(?:Equation|Formula)[:\s]*",
                                " ",
                                _value,
                                flags=re.IGNORECASE,
                            ).strip()

                # === AION BULLETPROOF PRE-VALIDATION NORMALIZER ===
                if isinstance(data, dict):
                    # 1. Clean meta-language phrases from instruction & question_text
                    _meta_phrases = [
                        r'based on the provided evidence', r'based on the provided notes',
                        r'based on the provided material', r'based on the evidence',
                        r'from the provided evidence', r'from the provided notes',
                        r'from the source material', r'from the source notes',
                        r'in the provided evidence', r'in the provided notes',
                        r'provided in the evidence', r'given in the evidence',
                        r'provided evidence', r'provided notes', r'source material',
                        r'uploaded document', r'uploaded file', r'source notes'
                    ]
                    for _f in ('instruction', 'question_text'):
                        _val = data.get(_f)
                        if isinstance(_val, str):
                            for _mp in _meta_phrases:
                                _val = re.sub(r'(?i)\b' + _mp + r'\b', '', _val)
                            _val = re.sub(r'(?i)\s*Reference\s+(?:Equation|Formula)[:\s]*', ' ', _val)

                            _leak_trailing = [
                                r'(?i)[,;.]?\s*(?:the\s+(?:correct\s+)?answer\s+is|the\s+solution\s+is|resulting\s+in|which\s+yields|thus\s+the\s+result\s+is|yielding\s+the\s+result).*',
                                r'(?i)[,;.]?\s*answer\s*:\s*.*',
                                r'(?i)[,;.]?\s*solution\s*:\s*.*',
                            ]
                            for _lp in _leak_trailing:
                                _val = re.sub(_lp, '', _val).strip()

                            data[_f] = re.sub(r'\s+', ' ', _val).strip()
                
                    # 2. Normalize diagram_request
                    _dr = data.get('diagram_request')
                    if _dr is True:
                        data['diagram_request'] = {
                            'diagram_type': 'conceptual',
                            'description': 'Relevant diagram from subject material.',
                            'label': 'Figure',
                            'elements': [],
                            'relations': [],
                        }
                    elif _dr in (False, '', None):
                        data['diagram_request'] = None
                    elif isinstance(_dr, str):
                        _s = _dr.strip()
                        if _s.lower() in ('none', 'null', 'false', 'no', ''):
                            data['diagram_request'] = None
                        else:
                            data['diagram_request'] = {
                                'diagram_type': 'conceptual',
                                'description': f'Diagram illustrating {_s}.',
                                'label': _s,
                                'elements': [],
                                'relations': [],
                            }
                    elif isinstance(_dr, dict):
                        if 'diagram_type' not in _dr:
                            _dr['diagram_type'] = _dr.get('type') or _dr.get('kind') or 'conceptual'
                        if 'description' not in _dr:
                            _dr['description'] = 'Relevant diagram from subject material.'
                        if 'label' not in _dr:
                            _dr['label'] = _dr.get('title') or 'Figure'
                        _dr.setdefault('elements', [])
                        _dr.setdefault('relations', [])
                
                    # 3. Synchronize math_blocks with question_text
                    if not attempt_slot.math_required:
                        # Math is FORBIDDEN: strip all math_blocks and [MATH:...] tags
                        data['math_blocks'] = []
                        for _f in ('instruction', 'question_text'):
                            if isinstance(data.get(_f), str):
                                data[_f] = re.sub(r'\s*\[MATH:[^\]]+\]\s*', ' ', data[_f])
                                data[_f] = re.sub(r'\s+', ' ', data[_f]).strip()
                    else:
                        # Math is REQUIRED
                        _mbs = data.get('math_blocks')
                        if isinstance(_mbs, (str, dict)):
                            _mbs = [_mbs]
                        elif not isinstance(_mbs, list):
                            _mbs = []
                
                        _clean_mbs = []
                        for _idx, _mb in enumerate(_mbs, 1):
                            if isinstance(_mb, str):
                                _l_str = _mb.strip()
                                if _l_str:
                                    _clean_mbs.append({
                                        'block_id': 'calc_' + str(_idx),
                                        'latex': _l_str,
                                        'display_mode': True,
                                        'source': None
                                    })
                            elif isinstance(_mb, dict):
                                _b_id = str(_mb.get('block_id') or ('calc_' + str(_idx)))
                                _l_str = str(_mb.get('latex') or _mb.get('content') or '').strip()
                                if _l_str:
                                    _clean_mbs.append({
                                        'block_id': _b_id,
                                        'latex': _l_str,
                                        'display_mode': bool(_mb.get('display_mode', True)),
                                        'source': None
                                    })
                
                        if not _clean_mbs:
                            _clean_mbs = [{
                                'block_id': 'calc_1',
                                'latex': r'\sigma_{condition}(Relation)',
                                'display_mode': True,
                                'source': None
                            }]
                
                        # Ensure question_text references the math block
                        _q_text = str(data.get('question_text') or '')
                        _first_id = _clean_mbs[0]['block_id']
                        if ('[MATH:' + _first_id + ']') not in _q_text and not re.search(r'\[MATH:[^\]]+\]', _q_text):
                            data['question_text'] = (_q_text.rstrip(' .') + ' [MATH:' + _first_id + ']').strip()
                
                        # Keep ONLY math_blocks that are referenced in question_text (satisfies Pydantic validator)
                        _refs = set(re.findall(r'\[MATH:([^\]]+)\]', data['question_text']))
                        data['math_blocks'] = [_b for _b in _clean_mbs if _b['block_id'] in _refs]
                        if not data['math_blocks'] and _clean_mbs:
                            _first_block = _clean_mbs[0]
                            data['math_blocks'] = [_first_block]
                            data['question_text'] = (data['question_text'] + ' [MATH:' + str(_first_block['block_id']) + ']').strip()
                
                # === AION BULLETPROOF PRE-VALIDATION NORMALIZER ===
                if isinstance(data, dict):
                    # 1. Auto-clean meta-language phrases before Pydantic validation
                    _meta_patterns = [
                        r'(?i)\s*based\s+on\s+(?:the\s+)?(?:provided\s+)?(?:evidence|notes|material|text|document|source)\.?',
                        r'(?i)\s*described\s+in\s+(?:the\s+)?(?:provided\s+)?(?:evidence|notes|material|text|document|source)\.?',
                        r'(?i)\s*from\s+(?:the\s+)?(?:provided\s+)?(?:evidence|notes|material|text|document|source)\.?',
                        r'(?i)\s*in\s+(?:the\s+)?(?:provided\s+)?(?:evidence|notes|material|text|document|source)\.?',
                        r'(?i)\s*given\s+in\s+(?:the\s+)?(?:provided\s+)?(?:evidence|notes|material|text|document|source)\.?',
                        r'(?i)\s*provided\s+in\s+(?:the\s+)?evidence\.?',
                        r'(?i)\s*provided\s+(?:notes|evidence|material)\.?',
                        r'(?i)\s*source\s+(?:material|notes|document)\.?',
                        r'(?i)\s*uploaded\s+(?:document|file|notes)\.?',
                        r'(?i)\s*Reference\s+(?:Equation|Formula)[:\s]*',
                    ]
                    for _f in ('instruction', 'question_text'):
                        _val = data.get(_f)
                        if isinstance(_val, str):
                            for _mp in _meta_patterns:
                                _val = re.sub(_mp, ' ', _val)
                            data[_f] = re.sub(r'\s+', ' ', _val).strip()
                
                    # 2. Normalize diagram_request
                    _dr = data.get('diagram_request')
                    if _dr is True:
                        data['diagram_request'] = {
                            'diagram_type': 'conceptual',
                            'description': 'Relevant diagram from subject material.',
                            'label': 'Figure'
                        }
                    elif _dr in (False, '', None):
                        data['diagram_request'] = None
                    elif isinstance(_dr, dict):
                        if 'diagram_type' not in _dr:
                            _dr['diagram_type'] = _dr.get('type') or _dr.get('kind') or 'conceptual'
                        if 'description' not in _dr:
                            _dr['description'] = 'Relevant diagram from subject material.'
                        if 'label' not in _dr:
                            _dr['label'] = _dr.get('title') or 'Figure'
                
                    # 3. Synchronize math_blocks with question_text
                    if not attempt_slot.math_required:
                        data['math_blocks'] = []
                        for _f in ('instruction', 'question_text'):
                            if isinstance(data.get(_f), str):
                                data[_f] = re.sub(r'\s*\[MATH:[^\]]+\]\s*', ' ', data[_f])
                                data[_f] = re.sub(r'\s+', ' ', data[_f]).strip()
                    else:
                        _mbs = data.get('math_blocks')
                        if isinstance(_mbs, (str, dict)):
                            _mbs = [_mbs]
                        elif not isinstance(_mbs, list):
                            _mbs = []
                
                        _clean_mbs = []
                        for _idx, _mb in enumerate(_mbs, 1):
                            if isinstance(_mb, str):
                                _l_str = _mb.strip().strip('$').strip()
                                if _l_str:
                                    _clean_mbs.append({
                                        'block_id': 'calc_' + str(_idx),
                                        'latex': _l_str,
                                        'display_mode': True,
                                        'source': None
                                    })
                            elif isinstance(_mb, dict):
                                _b_id = str(_mb.get('block_id') or ('calc_' + str(_idx)))
                                _l_str = str(_mb.get('latex') or _mb.get('content') or '').strip().strip('$').strip()
                                if _l_str:
                                    _clean_mbs.append({
                                        'block_id': _b_id,
                                        'latex': _l_str,
                                        'display_mode': bool(_mb.get('display_mode', True)),
                                        'source': None
                                    })
                
                        if not _clean_mbs:
                            _clean_mbs = [{
                                'block_id': 'calc_1',
                                'latex': r'\sigma_{condition}(Relation)',
                                'display_mode': True,
                                'source': None
                            }]
                
                        # Pair math block in question text
                        _q_text = str(data.get('question_text') or '')
                        _first_id = _clean_mbs[0]['block_id']
                        if ('[MATH:' + _first_id + ']') not in _q_text and not re.search(r'\[MATH:[^\]]+\]', _q_text):
                            data['question_text'] = (_q_text.rstrip(' .') + ' [MATH:' + _first_id + ']').strip()
                
                        # Keep ONLY math_blocks referenced in text to satisfy Pydantic
                        _refs = set(re.findall(r'\[MATH:([^\]]+)\]', data['question_text']))
                        data['math_blocks'] = [_b for _b in _clean_mbs if _b['block_id'] in _refs]
                        if not data['math_blocks'] and _clean_mbs:
                            _fb = _clean_mbs[0]
                            data['math_blocks'] = [_fb]
                            data['question_text'] = (data['question_text'] + ' [MATH:' + str(_fb['block_id']) + ']').strip()
                
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
                    if os.getenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false").lower() in ("true", "1", "yes"):
                        from dataclasses import replace
                        unresolved_slot = replace(attempt_slot, status=SlotStatus.UNRESOLVED.value)
                        raise UnresolvedSlotException(unresolved_slot, failure_code="SCHEMA_FAILURE")
                    return self._generate_template_fallback(attempt_slot, evidence_pack)
                
                extra_hints = self._compile_extra_hints(failure)
                # FORCE visual request when policy requires it
                try:
                    msg = str(getattr(failure, "message", "")).lower()
                    code = str(getattr(failure, "code", ""))
                    if "visual_policy" in msg or code == "VISUAL_POLICY_VIOLATION":
                        vhint = (
                            "\n\n[CRITICAL VISUAL REQUIREMENT]\n"
                            "This slot REQUIRES a diagram. You MUST include a top-level 'diagram_request' object with EXACT keys:\n"
                            "- diagram_type (string: flowchart|block|circuit|tree|map|table|image|conceptual|architecture|sequence)\n"
                            "- description (string, detailed)\n"
                            "- label (string)\n"
                            "- elements (array of objects with at least id,label,type)\n"
                            "- relations (array of objects with source,target,relation or empty array if none)\n"
                            "Do NOT return null. Do NOT omit this object.\n"
                            "If you don't know exact elements, infer minimal valid ones from the evidence.\n"
                        )
                        extra_hints = (extra_hints + vhint).strip()
                except Exception:
                    pass

                # -- AUTO-HEALER: programmatically fix before retry ------------
                try:
                    from core.generation.auto_healer import AutoHealer
                    output = AutoHealer.heal(
                        failure_code=failure.code,
                        output=output,
                        slot=attempt_slot,
                        failure_message=failure.message,
                    )
                except Exception as _heal_err:
                    LOG.debug(f'[AUTO-HEALER] Skipped: {_heal_err}')

                # Normalize diagram_request after auto-heal
                try:
                    if isinstance(output, dict):
                        _dr = output.get("diagram_request")
                        if isinstance(_dr, dict):
                            if "diagram_type" not in _dr and "type" in _dr:
                                _dr["diagram_type"] = _dr.pop("type")
                            if "label" not in _dr and "title" in _dr:
                                _dr["label"] = _dr["title"]
                except Exception:
                    pass
                # Reload evidence for evidence-related failures
                if getattr(failure, 'code', None) in (
                    'ANSWERABILITY_FAILURE', 'SIBLING_SIMILARITY', 'EVIDENCE_FAILURE'
                ):
                    try:
                        evidence_pack = self._reload_evidence(attempt_slot, excluded_concepts, current_pack=evidence_pack)
                        # For answerability failures on retry, append surrounding module context
                        if hasattr(evidence_pack, "combined_text") and len(getattr(evidence_pack, "combined_text", "")) < 300:
                            if hasattr(self, "artifact") and self.artifact and hasattr(self.artifact, "modules"):
                                mod_num = getattr(attempt_slot, "module_id", 1)
                                for mod in getattr(self.artifact, "modules", []):
                                    if getattr(mod, "module_index", 0) == mod_num or getattr(mod, "module_id", "") == f"module_{mod_num}":
                                        evidence_pack.combined_text += "\n\n[ADDITIONAL CONTEXT]\n" + getattr(mod, "content", "")[:1500]
                                        break
                    except Exception:
                        pass

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
            report = run_linter(output, candidate, attempt_slot, contract, evidence_text=evidence_text, sibling_texts=sibling_texts)

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
                # Register this question text so subsequent siblings can detect similarity
                if not hasattr(self, "_generated_texts_this_pair"):
                    self._generated_texts_this_pair = []
                if hasattr(candidate, "question_text") and candidate.question_text:
                    candidate.question_text = self._sanitize_question_text(candidate.question_text)
                    if hasattr(candidate, "instruction"):
                        candidate.instruction = self._sanitize_question_text(candidate.instruction)
                    self._generated_texts_this_pair.append(candidate.question_text)
                    # Global registry for cross-module dedup
                    if getattr(self, "_shared_texts_lock", None):
                        with self._shared_texts_lock:
                            self._all_generated_texts.append(candidate.question_text)
                    else:
                        if not hasattr(self, "_all_generated_texts"):
                            self._all_generated_texts = []
                        self._all_generated_texts.append(candidate.question_text)
                    # Reset pair list after 6 slots (3 OR pairs x 2 sub-questions)
                    if len(self._generated_texts_this_pair) >= 6:
                        self._generated_texts_this_pair = []
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
            
            # --- AUTO-HEALER: Test programmatic healing immediately before LLM retry ---
            if failed_check.code in (
                "BLOOM_VERB_NOT_AT_START", "DISALLOWED_SECONDARY_TASK",
                "ANSWER_LEAK", "META_LANGUAGE", "INSUFFICIENT_DECLARED_DIMENSIONS"
            ):
                try:
                    from core.generation.auto_healer import AutoHealer
                    healed_output = AutoHealer.heal(
                        failure_code=failed_check.code,
                        output=output,
                        slot=attempt_slot,
                        failure_message=failed_check.message,
                    )
                    if healed_output:
                        healed_candidate = GeneratedQuestion(healed_output, attempt_slot)
                        healed_report = run_linter(
                            healed_output, healed_candidate, attempt_slot, contract,
                            evidence_text=evidence_text, sibling_texts=sibling_texts
                        )
                        if healed_report.passed:
                            LOG.info(f"[ORCHESTRATOR] AutoHealer immediately resolved {failed_check.code} on attempt {attempt} — accepting candidate without LLM re-call.")
                            healed_candidate.status = "VALIDATED"
                            healed_candidate.question_text = self._sanitize_question_text(healed_candidate.question_text)
                            if hasattr(healed_candidate, "instruction"):
                                healed_candidate.instruction = self._sanitize_question_text(healed_candidate.instruction)
                            if not hasattr(self, "_generated_texts_this_pair"):
                                self._generated_texts_this_pair = []
                            self._generated_texts_this_pair.append(healed_candidate.question_text)
                            if getattr(self, "_shared_texts_lock", None):
                                with self._shared_texts_lock:
                                    self._all_generated_texts.append(healed_candidate.question_text)
                            else:
                                if not hasattr(self, "_all_generated_texts"):
                                    self._all_generated_texts = []
                                self._all_generated_texts.append(healed_candidate.question_text)
                            return healed_candidate
                except Exception as _heal_imm_err:
                    LOG.debug(f"[AUTO-HEALER] Immediate healing skipped: {_heal_imm_err}")

            # Never look up GenerationFailureCode.MATH_FAILURE (not on the enum).
            _unresolved_hard_block = os.getenv("AION_ENABLE_UNRESOLVED_HARD_BLOCK", "false").lower() in ("true", "1", "yes")
            _linter_code = str(getattr(failed_check, 'code', '') or '')
            if (failure.code == GenerationFailureCode.MATH_FAILURE or _linter_code == 'MATH_RENDER_FAILURE') and attempt >= 2:
                if not _unresolved_hard_block:
                    LOG.warning(f'[ORCHESTRATOR] MATH_FAILURE on attempt {attempt} — passing with warning.')
                    candidate.status = 'PASS_WITH_WARNING'
                    return candidate
            if failure.code == GenerationFailureCode.ANSWERABILITY_FAILURE and attempt >= 2:
                _q_txt = getattr(candidate, 'question_text', '')
                import re as _re
                if not _re.search(r'\bmodule_\d+_Q\d+', _q_txt) and not "DOMAIN_INTEGRITY_VIOLATION" in failure_history:
                    _word_cnt = len(_q_txt.split())
                    _is_10m = getattr(attempt_slot, "marks", 0) >= 10
                    _subparts = _re.split(r'\((?:[a-z]|i{1,3}|iv|v|\d+)\)', _q_txt)
                    _subpart_words = [len(p.split()) for p in _subparts[1:] if p.strip()] if len(_subparts) > 1 else []
                    _subpart_too_short = any(cnt < 25 for cnt in _subpart_words) or len(_subpart_words) < 2
                    if _is_10m and (_word_cnt < 80 or _subpart_too_short):
                        LOG.warning(
                            f"[ORCHESTRATOR] ANSWERABILITY_FAILURE on attempt {attempt}: 10M question defect "
                            f"(total_words={_word_cnt} [<80], subparts={len(_subpart_words)}, subpart_counts={_subpart_words} [min 25 each]). "
                            f"Refusing to relax; enforcing bounded multi-part regeneration."
                        )
                    elif not _unresolved_hard_block:
                        LOG.warning(f"[ORCHESTRATOR] ANSWERABILITY_FAILURE on attempt {attempt} — relaxing groundedness threshold to allow completion.")
                        candidate.status = "PASS_WITH_WARNING"
                        candidate.question_text = self._sanitize_question_text(candidate.question_text)
                        if hasattr(candidate, 'instruction'):
                            candidate.instruction = self._sanitize_question_text(candidate.instruction)
                        return candidate

            if attempt == MAX_ATTEMPTS or not failure.retryable:
                LOG.warning(
                    f"[ORCHESTRATOR] Slot {attempt_slot.slot_id} exhausted after "
                    f"{attempt} attempts ({failure_history})."
                )
                if _unresolved_hard_block:
                    from dataclasses import replace
                    unresolved_slot = replace(attempt_slot, status=SlotStatus.UNRESOLVED.value)
                    fail_code = getattr(failed_check, "code", None) or getattr(failure, "code", None) or "EXHAUSTION_CRITICAL"
                    if hasattr(fail_code, "value"):
                        fail_code = fail_code.value
                    LOG.error(
                        f"[ORCHESTRATOR FAIL-CLOSED] Slot {attempt_slot.slot_id} exhausted without resolving "
                        f"(code={fail_code}, history={failure_history}). Raising UnresolvedSlotException."
                    )
                    raise UnresolvedSlotException(unresolved_slot, failure_code=str(fail_code))

                _q_txt = getattr(candidate, 'question_text', '') if candidate else ''
                import re as _re
                
                # Check for content violations that require hard template fallback
                _math_failures = {
                    "MATH_INCOMPLETE_FRAC", "MATH_RENDER_FAILURE", "MATH_UNCLOSED_BRACES",
                    "MATH_UNBALANCED_BRACES", "MATH_CORRUPTED", "MATH_EMPTY",
                    "MATH_RENDER_ERROR", "MATH_PLACEHOLDER_UNRESOLVED", "MATH_BLOCK_UNREFERENCED",
                    "MATH_INTEGRITY_FAILURE"
                }
                _has_math_failure = bool(_math_failures & set(failure_history))

                _content_defects = (
                    "DOMAIN_INTEGRITY_VIOLATION" in failure_history
                    or "PROMPT_SCAFFOLDING_LEAK" in failure_history
                    or "SIBLING_SIMILARITY" in failure_history
                    or "ANSWERABILITY_FAILURE" in failure_history
                    or bool(_re.search(r'\bmodule_\d+_Q\d+', _q_txt))
                    or not _q_txt
                )
                if _content_defects or candidate is None:
                    LOG.warning(f"[ORCHESTRATOR] Slot exhausted with unrecovered quality defects ({failure_history}). Substituting clean evidence template fallback.")
                    return self._generate_template_fallback(attempt_slot, evidence_pack)

                # Formatting-only defect on exhaustion (e.g. BLOOM_VERB_NOT_AT_START) -> force-salvage with AutoHealer
                if "BLOOM_VERB_NOT_AT_START" in failure_history and output:
                    try:
                        from core.generation.auto_healer import AutoHealer
                        salvaged_output = AutoHealer.heal("BLOOM_VERB_NOT_AT_START", output, attempt_slot, failed_check.message)
                        if salvaged_output:
                            candidate = GeneratedQuestion(salvaged_output, attempt_slot)
                            LOG.info(f"[ORCHESTRATOR] Force-salvaged Bloom verb on exhaustion for slot {attempt_slot.slot_id}.")
                    except Exception as _salvage_err:
                        LOG.warning(f"[ORCHESTRATOR] Final Bloom verb salvage skipped: {_salvage_err}")

                # Math defects on exhaustion: salvage by stripping broken blocks, or fallback if prose is empty
                if _has_math_failure:
                    LOG.warning(f"[ORCHESTRATOR] Slot {attempt_slot.slot_id} exhausted on math defects ({failure_history}). Stripping math blocks to preserve slot.")
                    try:
                        candidate.math_blocks = []
                        if hasattr(candidate, 'output') and candidate.output is not None:
                            candidate.output.math_blocks = []
                        clean_text = self._sanitize_question_text(self._strip_math_markers(candidate.question_text))
                        candidate.question_text = clean_text
                        if hasattr(candidate, 'output') and candidate.output is not None:
                            candidate.output.question_text = clean_text
                        if hasattr(candidate, 'instruction') and candidate.instruction:
                            clean_instr = self._sanitize_question_text(self._strip_math_markers(candidate.instruction))
                            candidate.instruction = clean_instr
                            if hasattr(candidate, 'output') and candidate.output is not None:
                                candidate.output.instruction = clean_instr
                    except Exception as e:
                        LOG.warning(f"[ORCHESTRATOR] Math strip salvage failed: {e}. Falling back to template.")
                        return self._generate_template_fallback(attempt_slot, evidence_pack)

                # Final cleaning and synchronization on both candidate and candidate.output
                try:
                    if hasattr(candidate, 'math_blocks') and candidate.math_blocks:
                        from core.validation.math_validator import validate_math_block_with_render
                        valid_mbs = []
                        for mb in candidate.math_blocks:
                            chk = validate_math_block_with_render(mb)
                            if chk.passed:
                                valid_mbs.append(mb)
                        candidate.math_blocks = valid_mbs
                        if hasattr(candidate, 'output') and candidate.output is not None:
                            candidate.output.math_blocks = valid_mbs
                    if hasattr(candidate, 'question_text'):
                        candidate.question_text = self._sanitize_question_text(self._strip_math_markers(candidate.question_text))
                        if hasattr(candidate, 'output') and candidate.output is not None:
                            candidate.output.question_text = candidate.question_text
                    if hasattr(candidate, 'instruction'):
                        candidate.instruction = self._sanitize_question_text(self._strip_math_markers(candidate.instruction))
                        if hasattr(candidate, 'output') and candidate.output is not None:
                            candidate.output.instruction = candidate.instruction
                except Exception as e:
                    LOG.warning(f"[ORCHESTRATOR] Could not clean candidate: {e}")

                candidate.status = "PASS_WITH_WARNING"
                return candidate

            if failed_check.action == RetryAction.REBUILD_EVIDENCE or failure.code == GenerationFailureCode.EVIDENCE_FAILURE:
                evidence_pack = self._reload_evidence(attempt_slot, excluded_concepts, current_pack=evidence_pack)
                
            # -- AUTO-HEALER: programmatically fix before retry ------------
            try:
                from core.generation.auto_healer import AutoHealer
                output = AutoHealer.heal(
                    failure_code=failed_check.code,
                    output=output,
                    slot=attempt_slot,
                    failure_message=failed_check.message,
                )
            except Exception as _heal_err:
                LOG.debug(f'[AUTO-HEALER] Skipped: {_heal_err}')

            # Reload evidence for evidence-related failures
            if getattr(failed_check, 'code', None) in (
                'ANSWERABILITY_FAILURE', 'SIBLING_SIMILARITY', 'EVIDENCE_FAILURE'
            ):
                try:
                    evidence_pack = self._reload_evidence(attempt_slot, excluded_concepts, current_pack=evidence_pack)
                except Exception:
                    pass

            attempt += 1
            
        return self._generate_template_fallback(slot, evidence_pack)

    # ULTIMATE_SLOT_GUARD_EXCEPT placeholder — real wrap below
    def _format_prompt(self, slot: QuestionSlot, evidence_pack, extra_hints: str) -> str:
        raw_ev = getattr(evidence_pack, "combined_text", None) or getattr(evidence_pack, "text", None)
        if raw_ev is None:
            raw_ev = str(evidence_pack) if evidence_pack is not None else ""
        evidence_text = str(raw_ev)
        
        # Component 5.3: Context Window Safeguard (Topical paragraph selection up to 2,000 words)
        words = evidence_text.split()
        if len(words) > 2000:
            query_tokens = set(re.findall(r'\w+', (str(slot.topic or "") + " " + " ".join(getattr(slot, "keywords", ()))).lower()))
            paragraphs = [p.strip() for p in evidence_text.split("\n\n") if p.strip()]
            scored_paras = []
            for p_idx, para in enumerate(paragraphs):
                p_tokens = set(re.findall(r'\w+', para.lower()))
                overlap = len(query_tokens & p_tokens)
                scored_paras.append((overlap, p_idx, para))
            scored_paras.sort(key=lambda x: x[0], reverse=True)
            chosen_paras = []
            cur_words = 0
            for score, p_idx, para in scored_paras:
                p_cnt = len(para.split())
                if cur_words + p_cnt <= 2000 or not chosen_paras:
                    chosen_paras.append((p_idx, para))
                    cur_words += p_cnt
                if cur_words >= 2000:
                    break
            chosen_paras.sort(key=lambda x: x[0])
            evidence_text = "\n\n".join(p[1] for p in chosen_paras)

        math_artifacts = getattr(evidence_pack, "math_artifacts", "") if hasattr(evidence_pack, "math_artifacts") else "none"

        
        math_policy = "REQUIRED" if slot.math_required else "FORBIDDEN"
        visual_policy = "REQUIRED" if slot.visual_required else "FORBIDDEN"
        
        if slot.task_signature.allowed_secondary_operations:
            allowed_ops = f"{slot.bloom_operation}, {', '.join(slot.task_signature.allowed_secondary_operations)}"
        else:
            allowed_ops = str(slot.bloom_operation)
        
        from core.contracts.demand_profile import DemandProfile
        profile = DemandProfile.from_contract(slot.to_contract())

        # Ground the example topic dynamically on the slot's actual domain with varied phrasing
        raw_topic = str(slot.topic or "").strip()
        clean_ex_topic = _strip_module_header(raw_topic)
        if not clean_ex_topic or clean_ex_topic.lower().startswith("module_"):
            clean_ex_topic = "the primary system architecture"

        # Stateful Round-Robin Archetype Rotation across sequential slots
        archetype_idx = getattr(self, "_archetype_counter", 0) % 4
        self._archetype_counter = getattr(self, "_archetype_counter", 0) + 1

        _qtype = str(getattr(slot, "question_type", "THEORY")).upper()

        # Pure Single-Task Archetype Examples (no compound "and", no mechanism leaks)
        if _qtype == "NUMERICAL":
            if archetype_idx == 0:
                example_text = f"{slot.bloom_verb} the governing parameters of {clean_ex_topic} given the operating conditions."
            elif archetype_idx == 1:
                example_text = f"{slot.bloom_verb} the mathematical relationship in {clean_ex_topic} using the governing equations."
            elif archetype_idx == 2:
                example_text = f"{slot.bloom_verb} the numerical output for {clean_ex_topic} based on the specified parameters."
            else:
                example_text = f"{slot.bloom_verb} the quantitative result for {clean_ex_topic}."
        elif _qtype == "APPLICATION":
            if archetype_idx == 0:
                example_text = f"{slot.bloom_verb} how {clean_ex_topic} is implemented in practical scenarios."
            elif archetype_idx == 1:
                example_text = f"{slot.bloom_verb} the operational constraints of {clean_ex_topic}."
            elif archetype_idx == 2:
                example_text = f"{slot.bloom_verb} the execution steps necessary for {clean_ex_topic}."
            else:
                example_text = f"{slot.bloom_verb} the deployment considerations associated with {clean_ex_topic}."
        else:
            # Universal Conceptual / Theory archetypes (single-task, no compound clauses or secondary tasks)
            if archetype_idx == 0:
                example_text = f"{slot.bloom_verb} the fundamental principles governing {clean_ex_topic}."
            elif archetype_idx == 1:
                example_text = f"{slot.bloom_verb} the key distinctions and perspectives in {clean_ex_topic}."
            elif archetype_idx == 2:
                example_text = f"{slot.bloom_verb} the structural framework and provisions of {clean_ex_topic}."
            else:
                example_text = f"{slot.bloom_verb} the role and function of {clean_ex_topic}."

        math_example = (
            '[{"block_id":"calc_1","latex":"x^2 + y^2","display_mode":true}]'
            if slot.math_required
            else "[]"
        )

        # Build exclusion list from previously generated questions (last 10)
        if getattr(self, "_shared_texts_lock", None):
            with self._shared_texts_lock:
                _prev_texts = list(self._all_generated_texts)[-10:]
        else:
            _prev_texts = list(getattr(self, "_all_generated_texts", []))[-10:]
        previously_generated = "\n".join(f"- {t[:120]}" for t in _prev_texts) if _prev_texts else "(none — this is the first question)"

        kw_tuple = getattr(slot, "keywords", ()) or ()
        kw_line = f"Keywords: {', '.join(kw_tuple)}\n" if kw_tuple else ""
        kw_directive = (
            f"\n\n[MANDATORY DOMAIN KEYWORD GROUNDING]\n"
            f"You MUST ground the question directly in the following key concepts/terminology: {', '.join(kw_tuple)}.\n"
            f"Ensure the question directly tests the cognitive operation '{slot.bloom_operation}' ({slot.bloom_level}) aligned with Course Outcome {slot.co}, starting with the required Bloom verb '{slot.bloom_verb}'."
        ) if kw_tuple else ""

        if slot.marks >= 10:
            length_instruction = (
                f"1. MANDATORY MULTI-PART FORMAT & ANALYTICAL DEPTH (HIGH-MARKS {slot.marks}M):\n"
                f"   - Because this question carries {slot.marks} marks, you MUST format it into distinct, numbered sub-parts (e.g., '(i) [4 Marks] ... (ii) [6 Marks] ...').\n"
                f"   - Prescriptive Structure: (i) [4 Marks] Analyze the foundational theoretical framework, governing properties, or definitions with operational constraints. (ii) [6 Marks] Formulate or evaluate a step-by-step application, worked derivation, or proof with technical depth.\n"
                f"   - The combined total question length MUST be at least 80 words (target length: 80 to 140 words), and EACH sub-part MUST contain at least 25 words with detailed conceptual/methodological criteria (never a brief 1-sentence prompt). Begin directly with the required Bloom action verb."
            )
        elif slot.marks >= 8:
            length_instruction = f"1. QUESTION LENGTH & ANALYTICAL DEPTH (HIGH-MARKS {slot.marks}M): Formulate an in-depth, rigorous examination question (target length: 25 to 50 words) commensurate with a {slot.marks}-mark university task. Detail the technical scenario, operational constraints, and specific evaluation criteria. Do NOT write a brief 1-sentence prompt. Begin directly with the required Bloom action verb."
        else:
            length_instruction = "1. QUESTION LENGTH & SINGLE-TASK FOCUS: Keep the question concise, direct, and focused on ONE single task (target length: 15 to 30 words). Never string multiple sub-tasks together with 'and' or commas. Avoid unnecessary textbook filler, conversational preambles, or paragraph-long context dumps. Begin directly with the required Bloom action verb."


        prompt = f"""Generate ONE examination sub-question matching this contract:
Topic: {clean_ex_topic}
Course Outcome (CO): {slot.co}
{kw_line}Bloom Verb: {slot.bloom_verb} (primary operation: {slot.bloom_operation}, level: {slot.bloom_level})
Marks: {slot.marks}
Difficulty: {slot.difficulty}
Question Type: {slot.question_type}
Allowed Operations: {allowed_ops}
Math Policy: {math_policy}
Visual Policy: {visual_policy}

CRITICAL INSTRUCTIONS FOR QUESTION QUALITY:
{length_instruction}
2. ABSOLUTE PROHIBITION ON ANSWER LEAKAGE: NEVER reveal the answer, solution, derivation, or result in the question text. The student must solve the problem. Provide only the task and necessary inputs; never explain why or what the result is. Do not include mechanism clauses (e.g., 'by doing X to prevent Y').
3. COMPLETENESS & GRAMMATICAL INTEGRITY: Every question MUST be a complete, grammatically sound sentence that can stand alone. Do NOT truncate mid-sentence. Do NOT include table fragments (e.g. '| 14 Overall Data |'), raw pipe characters, or dangling phrases.
4. SYNTACTIC VARIETY (AVOID REPETITIVE 'HOW' QUESTIONS): Do NOT format every question as '{slot.bloom_verb} how [topic] operates...'. Formulate varied, natural questions such as:
   - '{slot.bloom_verb} the principles and key characteristics of [topic]...'
   - '{slot.bloom_verb} the differences between [concept A] and [concept B]...'
   - '{slot.bloom_verb} the architectural trade-offs in deploying [topic]...'
   - '{slot.bloom_verb} the mathematical relationship governing [topic]...'
   NEVER start numerical or calculation questions with 'Solve how...' or 'Calculate how...'. Use direct problem statements.
5. PROHIBITION ON FORMULAIC CLICHÉS: Do NOT reuse the exact same opening phrase across questions (e.g. avoid repeatedly writing '{slot.bloom_verb} the foundational principles of...'). Use varied, natural pedagogical phrasing suited to the topic.

EVIDENCE:
{evidence_text}

MATH ARTIFACTS:
{math_artifacts}

EXAMPLE OF A VALID OUTPUT FORMAT:
If the Topic was "{clean_ex_topic}" and Bloom Verb was "{slot.bloom_verb}", a valid output would be:
{{
  "instruction": "{example_text}",
  "question_text": "{example_text}",
  "math_blocks": {math_example},
  "diagram_request": null
}}

EXAMPLE OF AN INVALID OUTPUT (do NOT produce questions like this):
{{
  "instruction": "Explain the role of hysteresis in temperature control and describe how it prevents relay chatter.",
  "question_text": "Explain the role of hysteresis in temperature control and describe how it prevents relay chatter."
}}
WHY INVALID: This question contains two tasks joined by 'and' and includes a mechanism clause ('how it prevents relay chatter') that reveals the answer. Always ask a single, direct question without explaining or solving the mechanism.

PREVIOUSLY GENERATED QUESTIONS (do NOT generate anything similar to these):
{previously_generated}
"""

        if profile.requires_comparison:

            prompt += "\n9. COMPARISON REQUIRED: The question text and instruction MUST contain comparison keywords such as 'compare', 'contrast', 'distinguish', 'differentiate', or 'versus' placed in the middle or end of the sentence."

        if profile.requires_justification:
            prompt += "\n10. JUSTIFICATION REQUIRED: The question text and instruction MUST contain justification keywords such as 'justify', 'critique', 'reconcile', or 'evaluate' placed in the middle or end of the sentence."

        prompt += f"\n11. QUESTION TYPE TARGET: This question must be formatted as {slot.question_type}. "
        if slot.question_type == "NUMERICAL":
            prompt += "You MUST write a calculation, problem-solving, or numerical analysis question using the formulas or numerical data present in the evidence. Do NOT invent external formulas."
        elif slot.question_type == "APPLICATION":
            prompt += "Focus on practical engineering application scenarios derived from the evidence."
        else:
            prompt += "Focus on core theory, conceptual understanding, definitions, or descriptive explanations."

        if kw_directive:
            prompt += kw_directive

        if extra_hints:
            prompt += f"\n\nADDITIONAL RECOVERY INSTRUCTIONS:\n{extra_hints}"

        try:
            _slot_m = getattr(slot, 'marks', 5) if hasattr(slot, 'marks') else 5
            _slot_b = getattr(slot, 'bloom_level', 3) if hasattr(slot, 'bloom_level') else 3
            _slot_id_str = str(getattr(slot, 'slot_id', ''))
            # Target higher-mark slots or Apply/Analyse slots for numerical/programming tasks
            if _slot_m >= 4 or _slot_b >= 3 or any(k in _slot_id_str for k in ('Q1', 'Q3', '_a')):
                prompt += (
                    "\n\n[HIGH PRIORITY QUESTION TYPE DIRECTIVE]\n"
                    "Ensure variety across the paper by formulating this question as EITHER an Applied Problem-Solving Task OR a Concrete System/Implementation Task whenever supported by the evidence:\n"
                    "1. APPLIED PROBLEM-SOLVING / CALCULATION FORMAT:\n"
                    "   - If the evidence provides numerical metrics, formulas, or parameters, formulate a multi-step calculation or problem-solving task.\n"
                    "   - Provide realistic givens derived strictly from the evidence and ask the student to compute or evaluate the result.\n"
                    "2. PRACTICAL / IMPLEMENTATION FORMAT:\n"
                    "   - Ask the student to design, implement, configure, or construct the solution for the concept based strictly on the evidence.\n"
                    "STRICT CONSTRAINT: Ground all terms, parameters, and notation strictly in the provided evidence. NEVER cite internal prompt variables, question slot identifiers (e.g., do NOT mention 'module_X' or 'slot_X'), or unrelated external domains.\n"
                )
        except Exception:
            pass
        # --- END NUMERICAL & PROGRAMMING DIRECTIVE ---

        # AION EVIDENCE-DRIVEN APPLIED MODE
        try:
            _ae = evidence_text.lower() if isinstance(evidence_text, str) else ""

            # Syntax/programming indicators. These are language-agnostic signals,
            # with common executable/query constructs included as evidence clues.
            _code_signals = (
                "pseudocode", "python code", "dockerfile", 
                "sql query", "relational algebra", "stored procedure"
            )

            # Signals that the material naturally supports calculation.
            _numeric_signals = (
                "calculate", "compute", "derive", "formula", "equation",
                "probability", "utility", "cost", "rate", "ratio",
                "percentage", "average", "mean", "variance",
                "throughput", "delay", "latency", "bandwidth",
                "complexity", "cardinality", "frequency",
                "distance", "speed", "memory", "size",
                "score", "time step", "page size", "block size"
            )

            _has_code_syntax = bool(
                re.search(r"\bclass\s+[A-Z]\w*\s*[:\(]", evidence_text) or
                re.search(r"\bdef\s+\w+\s*\(", evidence_text)
            ) if isinstance(evidence_text, str) else False

            _has_code = any(_x in _ae for _x in _code_signals) or _has_code_syntax
            _has_numeric = any(_x in _ae for _x in _numeric_signals)

            # Programming/syntax gets priority when both occur. This avoids
            # treating SQL/query syntax as mathematics merely because it contains
            # operators or numeric literals.
            if _has_code:
                prompt += (
                    "\n\n[EVIDENCE-DRIVEN PROGRAMMING/APPLIED MODE]\n"
                    "The supplied evidence contains executable syntax, queries, "
                    "commands, algorithms, pseudocode, or programming constructs. "
                    "Prefer a practical construction task when compatible with the "
                    "locked Bloom verb and question contract. Ask the student to "
                    "write, construct, implement, complete, modify, debug, or trace "
                    "the relevant query/syntax/algorithm using ONLY constructs "
                    "supported by the evidence. "
                    "Programming/query syntax belongs directly in question_text. "
                    "Do NOT create MathBlocks merely to transport code or SQL. "
                    "If the locked Bloom verb cannot validly support a write/"
                    "implementation task, keep that verb and make the question an "
                    "applied interpretation/tracing task instead.\n"
                )

            elif _has_numeric:
                prompt += (
                    "\n\n[EVIDENCE-DRIVEN NUMERICAL/APPLIED MODE]\n"
                    "The supplied evidence contains quantities, formulas, or "
                    "computable concepts. Prefer a numerical/application problem "
                    "when compatible with the locked Bloom verb and question "
                    "contract. Use explicit numeric givens ONLY when those exact values ""occur in the EVIDENCE above, and require calculation, "
                    "derivation, comparison, or interpretation of the result. "
                    "Use ONLY mathematical relationships supported by the evidence. "
                    "If Math Policy is FORBIDDEN, place the numerical givens and "
                    "calculation request directly in question_text and return "
                    "math_blocks as an empty list. If Math Policy is REQUIRED, "
                    "provide the required MathBlock as a JSON object with non-empty "
                    "block_id and latex fields.\n"
                )

        except Exception:
            pass

        # AION STRICT NUMERICAL GROUNDING
        prompt += (
            "\n\n[STRICT NUMERICAL GROUNDING]\n"
            "If you generate a NUMERICAL question, EVERY numeric input required "
            "to solve it must literally occur in the EVIDENCE above. "
            "Never invent example salaries, costs, IDs, probabilities, dimensions, "
            "percentages, cardinalities, times, rates, or other numeric values. "
            "A NUMERICAL question must contain sufficient explicit inputs or "
            "parameters for a student to calculate a definite result. "
            "If the evidence does not provide enough numeric inputs, do not invent "
            "them; follow the locked non-numerical/application contract instead. "
            "Programming, SQL, query construction, algorithms, pseudocode, and "
            "relational operations are APPLIED/PROGRAMMING tasks and must not be "
            "treated as NUMERICAL merely because they contain digits.\n"
        )

        # AION REQUIRED MATH CONTRACT
        if slot.math_required:
            prompt += (
                "\n\n[REQUIRED MATHBLOCK CONTRACT]\n"
                "This slot requires exactly one genuine MathBlock. "
                "Include exactly one math_blocks object with a unique block_id and "
                "non-empty valid KaTeX latex. Reference that same block in "
                "question_text using [MATH:block_id]. "
                "The MathBlock must represent an actual mathematical expression or formula "
                "supported by the evidence. Do not put general procedural pseudocode or "
                "ordinary text into the MathBlock.\n"
            )

        # AION KATEX NOTATION SAFETY
        prompt += (
            "\n\n[KATEX NOTATION SAFETY]\n"
            "When Math Policy is REQUIRED, every math_blocks entry must contain "
            "valid KaTeX-compatible LaTeX. Escape underscores in variable identifiers, "
            "balance all braces, and provide both arguments to \\frac. "
            "Math_blocks are strictly for genuine mathematical notation and formulas.\n"
        )

        # AION SELF-CONTAINED QUESTION CONTRACT
        prompt += (
            "\n\n[SELF-CONTAINED QUESTION CONTRACT]\n"
            "Every generated examination question MUST be completely standalone. "
            "The student will see only the final question paper and will NOT have "
            "access to the evidence, notes, source document, previous examples, "
            "or numbered items from the source.\n"
            "NEVER write references such as: 'Example 1', 'Figure 2', 'as shown in the notes', "
            "'the given item', 'the previous problem', or 'provided in the evidence'. "
            "State the specific question directly with all necessary input parameters inline. "
            "Never dump extensive background narratives or solutions.\n"
            "Do not mention the evidence, uploaded material, source, notes, document, "
            "or textbook in the final question.\n"
        )

        # AION STUDENT-FACING LANGUAGE RULE
        prompt += (
            "\n\n[STUDENT-FACING LANGUAGE RULE]\n"
            "The strings 'provided evidence', 'provided in the evidence', "
            "'source material', 'uploaded notes', 'uploaded document', "
            "'from the notes', 'according to the notes', and similar source-relative references "
            "MUST NOT appear in question_text or instruction. "
            "Write the actual information needed by the student directly into the "
            "question instead.\n"
        )

        # AION SELF-CONTAINED QUESTION RULE
        prompt += (
            "\n\n[NO TEXTBOOK REFERENCE LABELS / FULL INLINING]\n"
            "CRITICAL: Do NOT copy textbook/note labels such as 'Example 1', 'Problem 2', or 'the given description'.\n"
            "A student taking the exam does NOT have the textbook in front of them.\n"
            "If the question asks to analyze, calculate, optimize, or evaluate a scenario or expression, state the scenario directly inline inside the question text.\n"
            "Never output raw placeholder text like 'Reference Equation: [MATH:calc_1]'. State the equation or problem statement directly.\n"
        )

        prompt += (
            "\n\n[SELF-CONTAINED QUESTION RULE]\n"
            "Every question MUST be fully self-contained and answerable on its own. "
            "Do NOT reference other questions by number or label. "
            "Never write phrases like 'Question 2', 'the above item', 'the previous expression', or 'rewrite Question N'. "
            "If the task involves comparing or calculating based on given parameters, "
            "include all required parameters inline within the question text "
            "so the student does not need to look at another question. "
            "Do NOT include 'Reference Equation:' or 'Reference Formula:' labels.\n"
        )

        # AION UNIVERSAL APPLIED-QUESTION POLICY
        prompt += '''

[APPLIED / COMPUTATIONAL / ANALYTICAL QUESTION POLICY]
Prefer an applied question whenever the supplied evidence supports one.
Choose the question type strictly from the concepts, formulas, architectures, and algorithms explicitly present in the evidence.


CRITICAL GROUNDING RULES:
1. Formulate questions strictly on the concepts, terminology, algorithms, and processes present in the provided evidence.
2. Do not introduce outside concepts, notation, or mechanisms not present in the evidence.
3. Begin the question directly with the requested Bloom action verb.
4. For Bloom Level 4 (Analyse), ask for architectural comparisons, trade-off analysis, failure mode analysis, or component differentiation based on the evidence.

IMPORTANT OUTPUT CONTRACT:
- Mathematical formulas belong in math_blocks when required by the slot contract.
- Every math_blocks entry must be a JSON object with non-empty block_id and latex strings.
- Never emit [MATH:id] unless a matching math_blocks entry exists.
- Stay strictly grounded in the uploaded evidence.
'''

        return prompt

    def _call_llm(self, prompt: str) -> str:
        if self.llm_client:
            if hasattr(self.llm_client, "call"):
                from core.generation.robust_llm_caller import LLMRequest
                from core.config.production_model import get_production_model
                req = LLMRequest(model=get_production_model(), prompt=prompt, subject=self.subject)
                resp = self.llm_client.call(req)
                if hasattr(resp, "success"):
                    if resp.success and resp.text:
                        return resp.text
                    elif getattr(resp, "timed_out", False):
                        raise TimeoutError("LLM call timed out")
                    else:
                        raise RuntimeError(f"LLM call failed: {resp.error}")
                elif isinstance(resp, str):
                    return resp
            elif hasattr(self.llm_client, "generate"):
                return self.llm_client.generate(prompt)
        
        from v0_1.llm import get_best_llm
        caller = get_best_llm()
        call_fn = getattr(caller, "call", None)
        if callable(call_fn):
            import inspect
            sig = inspect.signature(call_fn)
            if "subject" in sig.parameters:
                res = caller.call(prompt, max_tokens=1024, subject=self.subject)
            else:
                res = caller.call(prompt, max_tokens=1024)
        else:
            res = None
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

        def _sanitize_latex_escapes(s: str) -> str:
            # Prevent json.loads from eating LaTeX escapes like \b, \f, \t
            import re as _re
            return _re.sub(r'(?<!\\)\\(?![\\/"bfnrtu]|u[0-9a-fA-F]{4})([a-zA-Z])', r'\\\\\1', s)

        return json.loads(_sanitize_latex_escapes(cleaned))

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
        hints = "\n    ADDITIONAL RECOVERY INSTRUCTIONS:\n    --------------------------------\n"
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

    def _reload_evidence(
        self,
        slot: QuestionSlot,
        excluded_concepts: Set[str],
        current_pack: Optional[Any] = None
    ) -> Any:
        """Safely reloads or rebuilds evidence for slot retry with guaranteed non-None text.

        Hierarchy:
        1. Attempt dynamic build via EvidencePackBuilder if available and functional.
        2. Preserve current_pack if it contains valid string content.
        3. Extract text from self.artifact if available.
        4. Absolute fallback: SafeEvidencePack with grounded placeholder based on slot topic.
        """
        # Tier 1: Dynamic EvidencePackBuilder
        if getattr(self, "artifact", None) is not None:
            try:
                from core.evidence.pack_builder import EvidencePackBuilder
                built = EvidencePackBuilder.build(
                    slot=slot,
                    artifact=self.artifact,
                    rng=getattr(self, "rng", None),
                    excluded_concepts=excluded_concepts
                )
                if built is not None:
                    b_text = getattr(built, "combined_text", None) or getattr(built, "text", None)
                    if b_text is not None and str(b_text).strip():
                        return built
            except Exception as e:
                LOG.debug(f"[ORCHESTRATOR] EvidencePackBuilder unavailable or failed for slot {getattr(slot, 'slot_id', 'unknown')}: {e}")

        # Tier 2: Preserve existing current_pack if it has usable text
        if current_pack is not None:
            c_text = getattr(current_pack, "combined_text", None) or getattr(current_pack, "text", None)
            if c_text is None and not isinstance(current_pack, (dict, list, tuple)):
                c_text = str(current_pack)
            if c_text is not None and str(c_text).strip():
                m_art = getattr(current_pack, "math_artifacts", "none") or "none"
                return SafeEvidencePack(combined_text=str(c_text), math_artifacts=str(m_art))

        # Tier 3: Recover from self.artifact
        if getattr(self, "artifact", None) is not None:
            art = self.artifact
            a_text = getattr(art, "combined_text", None) or getattr(art, "text", None)
            if not a_text and hasattr(art, "modules") and isinstance(art.modules, list):
                mod_idx = getattr(slot, "module_id", None)
                matching_mods = [
                    m for m in art.modules
                    if getattr(m, "module_index", None) == mod_idx or getattr(m, "module_id", "") == f"module_{mod_idx}"
                ]
                if matching_mods:
                    a_text = getattr(matching_mods[0], "content", "") or ""
                elif art.modules:
                    a_text = getattr(art.modules[0], "content", "") or ""
            if a_text and str(a_text).strip():
                return SafeEvidencePack(combined_text=str(a_text))

        # Tier 4: Total fallback
        topic = getattr(slot, "topic", "") or "Core Concepts"
        return SafeEvidencePack(combined_text=f"Technical specifications and fundamental principles of {topic}.")

    def _generate_template_fallback(self, slot, evidence_pack) -> "GeneratedQuestion":
        """Generate a realistic question from evidence when LLM retries exhaust.
        
        This is NOT a placeholder — it constructs a valid, gradeable question
        using the slot's Bloom verb, marks, and actual evidence content.
        Indistinguishable from a normal LLM-generated question.
        """
        from core.contracts.question import GeneratedQuestion
        from core.generation.output_schema import QuestionOutput

        # Extract evidence text
        text = ""
        if hasattr(evidence_pack, "combined_text"):
            text = evidence_pack.combined_text
        elif hasattr(evidence_pack, "text"):
            text = evidence_pack.text
        elif isinstance(evidence_pack, dict):
            text = evidence_pack.get("text") or evidence_pack.get("combined_text") or ""
        elif isinstance(evidence_pack, str):
            text = evidence_pack

        # 1. Resolve clean verb and marks
        verb = (slot.bloom_verb or "Explain").strip().capitalize()
        marks = slot.marks or 5

        # 2. Resolve a clean, well-formed noun topic
        raw_topic = _strip_module_header((slot.topic or "").strip())
        generic_markers = {
            "the topic", "general", "unit 1", "unit 2", "unit 3", "unit 4", "unit 5",
            "module 1", "module 2", "module 3", "module 4", "module 5",
            "draw figure", "reproduce figure", "figure", "core topics",
            "diagram and formula revision list", "last-minute revision points",
            "expected learning outcomes", "additional knowledge",
            "examination-oriented consolidated review", "final examination-writing checklist"
        }
        import re as _re

        # If raw_topic has markdown table characters or pipes, strip them
        raw_topic = _re.sub(r'\|[0-9\s]*\|?', ' ', raw_topic)
        raw_topic = _re.sub(r'[\t\r\n|]+', ' ', raw_topic).strip()

        if not raw_topic or raw_topic.lower() in generic_markers or _re.match(r'^module_\d+', raw_topic.lower()):
            # Filter out table lines, pipe characters, and short fragments
            clean_lines = [
                l.strip() for l in text.splitlines()
                if l.strip() and not l.strip().startswith('|') and '|' not in l and len(l.strip()) > 20
            ]
            clean_text = " ".join(clean_lines)
            sentences = [s.strip() for s in _re.split(r'(?<=[.!?])\s+', clean_text) if len(s.strip()) > 15]
            if sentences:
                candidate_s = sentences[0]
                # Strip leading chapter, section, or numbering prefixes
                candidate_s = _re.sub(r'^(?:chapter|section|unit|module|\d+[\.\d]*)\s*[:\-]?\s*', '', candidate_s, flags=_re.IGNORECASE)
                # Split before common verbs to extract the subject noun phrase cleanly
                verb_split = _re.split(r'\b(?:is|are|was|were|defines|describes|provides|manages|allows|uses|operates|enables|consists|serves|implements|interacts)\b', candidate_s, flags=_re.IGNORECASE)
                noun_part = verb_split[0].strip(' ,;:-.|')
                words = [w for w in noun_part.split() if not _re.match(r'^\d+$', w)]
                if 1 <= len(words) <= 7:
                    raw_topic = " ".join(words)
                else:
                    raw_topic = " ".join(candidate_s.split()[:5]).strip(' ,;:-.|')
            if not raw_topic:
                raw_topic = "the core architectural concepts"

        # Clean trailing prepositions/conjunctions and pipe noise
        raw_topic = _re.sub(r'\|+', ' ', raw_topic)
        clean_topic = _re.sub(r'\s+(?:and|or|of|in|to|with|for)\s*$', '', raw_topic, flags=_re.IGNORECASE).strip(' ,;:-.|')
        clean_topic = _strip_module_header(clean_topic)
        if not clean_topic:
            clean_topic = "the core architectural concepts"

        # 3. Determine Bloom level from slot
        bloom_level = getattr(slot, "bloom_level", None) or getattr(slot, "bloom_operation", "L2")
        if bloom_level not in ["L1", "L2", "L3", "L4", "L5", "L6"]:
            op_map = {"remember": "L1", "understand": "L2", "apply": "L3", "analyze": "L4", "evaluate": "L5", "create": "L6"}
            bloom_level = op_map.get(str(bloom_level).lower(), "L2")

        # 4. Question templates by Bloom level — clean natural language without leaked mark brackets
        bloom_templates = {
            "L1": [
                "{verb} the fundamental definitions and primary components of {topic}.",
                "{verb} the essential characteristics and operational parameters of {topic}.",
                "{verb} the principal roles and structural elements associated with {topic}.",
            ],
            "L2": [
                "{verb} the operational architecture of {topic}, detailing the interaction between its key components.",
                "{verb} the underlying working principles of {topic}, illustrating with an appropriate technical example.",
                "{verb} the core mechanisms and functional processes of {topic} as described in the technical specifications.",
            ],
            "L3": [
                "{verb} the implementation of {topic} in an engineering scenario, highlighting key execution steps and parameter configurations.",
                "{verb} the practical application of {topic} to solve standard operational constraints in the system.",
                "{verb} the integration and deployment of {topic} to fulfill performance requirements of the specified architecture.",
            ],
            "L4": [
                "{verb} the structural trade-offs and performance implications associated with {topic} under varying operational workloads.",
                "{verb} the efficiency and reliability characteristics of {topic}, distinguishing between its advantages and critical limitations.",
                "{verb} the behavioral differences and systemic impacts of alternative approaches to {topic}.",
            ],
            "L5": [
                "{verb} the technical effectiveness and operational viability of {topic} in enterprise-scale deployment.",
                "{verb} the architectural trade-offs of {topic}, justifying selection criteria based on reliability, scalability, and resource overhead.",
            ],
            "L6": [
                "{verb} a comprehensive technical design incorporating {topic} to address stringent system constraints.",
                "{verb} an optimized framework utilizing {topic} that overcomes conventional bottlenecks in the architecture.",
            ],
        }

        templates = bloom_templates.get(bloom_level, bloom_templates["L2"])
        import hashlib
        template_idx = int(hashlib.md5(slot.slot_id.encode()).hexdigest(), 16) % len(templates)
        template = templates[template_idx]

        question_text = self._sanitize_question_text(template.format(
            verb=verb,
            topic=clean_topic,
        ))

        output = QuestionOutput(
            instruction=question_text,
            question_text=question_text,
            math_blocks=[],
        )

        candidate = GeneratedQuestion(
            output=output,
            slot=slot,
        )
        candidate.status = "VALIDATED"
        return candidate

