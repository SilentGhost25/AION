"""
Writing Agent — question generation via local LLM or VLM.

Inputs (read from AgentContext):
    - context.plans : List[QuestionPlan]

Output (AgentResult.payload):
    - WritingAgentOutput(questions=[...], failures=[...])

The agent processes each plan independently. One slot's failure does
not abort the batch. The orchestrator inspects `failures` afterward
and decides whether to escalate.

Does NOT:
    - Call the API (that is the Refinement Agent's / orchestrator's job)
    - Repair invalid questions (that is the Refinement Agent)

Enforces at generation time:
    - Subpart Bloom verb alignment and cognitive depth
    - Multi-part sibling independence
    - Raw text artifact and template fragment rejection
    - Multi-level evidence provenance aggregation
"""

from __future__ import annotations

import json
import re
import time
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

# UNTESTED — run quickstart.py on server first
from core.evaluation.irt_calibrator import IRTCalibrator, DifficultyEstimate
from core.validation.bloom_validator import (
    BLOOM_VERB_LEVEL_MAP,
    classify_verb,
    get_all_canonical_verbs,
    get_canonical_verbs,
    is_exact_slot_compliant,
)
from .base import Agent, AgentContext, AgentResult
from .generated_question import (
    GeneratedQuestion,
    SlotFailure,
    SubPart,
    WritingAgentOutput,
)
from .local_evaluators import (
    _first_meaningful_word,
    ALL_BLOOM_VERBS,
    evaluate_cognitive_demand,
    SemanticDemandOutcome,
    validate_no_source_tags,
    validate_singular_directive,
    validate_single_topic,
)
from .planning_agent import PlanStatus
from .prompts import (
    BLOOM_VERBS,
    build_archetype_subpart_prompt,
    build_difficulty_feedback,
    build_prompt,
    build_subpart_prompt,
    get_figure_field,
    IMAGE_REFERENCE_PHRASES,
)


# -----------------------------------------------------------------------------
# Caller protocols
# -----------------------------------------------------------------------------


class LLMCaller(Protocol):
    """Minimal interface for a text LLM caller."""
    def call(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Any:
        ...


class VLMCaller(Protocol):
    """Minimal interface for a vision-capable caller."""
    def call(
        self,
        prompt: str,
        image_path: str,
        schema: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Any:
        ...


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------


MAX_ATTEMPTS = 3
DEFAULT_SEED = 42

# Phrases that signal a question references a figure are imported from .prompts
# (IMAGE_REFERENCE_PHRASES is re-exported here for backward compatibility)

# Required keys in the LLM's JSON output for monolithic generation.
REQUIRED_OUTPUT_KEYS = ("question_text", "solution", "marking_scheme")

# Required keys in the LLM's JSON output for structured SubPart generation.
REQUIRED_SUBPART_KEYS = ("text", "solution", "marking_scheme")


@dataclass
class GenerationBudget:
    """Bounded budget for whole-paper LLM generation and IRT retries."""
    max_total_llm_calls: int = 40          # hard cap for whole paper
    max_irt_retries_per_slot: int = 2
    max_total_irt_retries: int = 6         # global cap for whole paper
    used_calls: int = 0
    used_irt: int = 0

    def can_retry_irt(self) -> bool:
        return (
            self.used_irt < self.max_total_irt_retries
            and self.used_calls < self.max_total_llm_calls
        )

    def charge(self, irt: bool = False) -> None:
        self.used_calls += 1
        if irt:
            self.used_irt += 1


# -----------------------------------------------------------------------------
# Writing Agent
# -----------------------------------------------------------------------------


class WritingAgent(Agent):
    name = "writing"

    def __init__(
        self,
        llm_caller: LLMCaller,
        vlm_caller: Optional[VLMCaller] = None,
        seed: int = DEFAULT_SEED,
        max_attempts: int = MAX_ATTEMPTS,
        clock: Optional[Callable[[], float]] = None,
        irt_calibrator: Optional[IRTCalibrator] = None,
        difficulty_bands: Optional[Dict[str, Tuple[float, float]]] = None,
        enable_irt: bool = True,
    ) -> None:
        if llm_caller is None:
            raise ValueError("llm_caller is required")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        self._llm = llm_caller
        self._vlm = vlm_caller
        self._seed = seed
        self._max_attempts = max_attempts
        self._clock = clock or time.monotonic

        # Closed-loop IRT difficulty controller
        env_irt = os.getenv("AION_ENABLE_IRT", "true").lower().strip() not in ("0", "false", "no")
        self._enable_irt = enable_irt and env_irt
        self._current_bloom = "L2"
        if self._enable_irt:
            import random
            self._irt = irt_calibrator or IRTCalibrator(
                student_sim_fn=self._default_student_sim,
                samples_per_level=3,
                rng=random.Random(self._seed),
            )
        else:
            self._irt = None

        self._difficulty_bands = difficulty_bands or {
            "easy": (-3.0, -0.5),
            "medium": (-0.5, 0.8),
            "hard": (0.8, 3.0),
        }

    def _default_student_sim(self, question_text: str, theta: float) -> float:
        """
        Simulate a student at ability theta answering the question.
        Uses a heuristic: harder questions (longer, more clauses,
        higher Bloom level verbs) reduce P(correct).
        """
        import math
        word_count = len(question_text.split())
        bloom_penalty = {
            "L1": 0.0, "L2": 0.1, "L3": 0.3, "L4": 0.5, "L5": 0.7, "L6": 0.9,
        }.get(getattr(self, "_current_bloom", "L2"), 0.2)
        feature_difficulty = (min(word_count, 40) / 100.0) + bloom_penalty
        z = theta - feature_difficulty
        if z >= 0:
            ez = math.exp(-z)
            return 1.0 / (1.0 + ez)
        else:
            ez = math.exp(z)
            return ez / (1.0 + ez)

    def _band_for_plan(self, plan) -> str:
        """Derive target difficulty band from plan."""
        band = getattr(plan, "difficulty_band", None)
        if band:
            return band
        if plan.bloom in ("L1", "L2") or plan.total_marks <= 4:
            return "easy"
        if plan.bloom in ("L3",) or plan.total_marks <= 6:
            return "medium"
        return "hard"

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> AgentResult:
        plans = list(getattr(context, "plans", []) or [])
        if not plans:
            return AgentResult(
                success=True,
                payload=WritingAgentOutput(questions=[], failures=[]),
                telemetry={"questions_generated": 0},
            )

        start = self._clock()
        questions: List[GeneratedQuestion] = []
        failures: List[SlotFailure] = []
        sources: Dict[str, int] = {"local_llm": 0, "local_vlm": 0}
        total_attempts = 0
        budget = GenerationBudget()

        for plan in plans:
            try:
                gq, attempts = self._generate_one(plan, budget=budget)
                questions.append(gq)
                sources[gq.generation_source] = sources.get(gq.generation_source, 0) + 1
                total_attempts += attempts
            except GenerationFailed as e:
                failures.append(SlotFailure(
                    slot_id=plan.slot_id,
                    failure_code=e.code,
                    failure_detail=e.detail,
                ))
            except Exception as e:
                failures.append(SlotFailure(
                    slot_id=plan.slot_id,
                    failure_code="UNEXPECTED_ERROR",
                    failure_detail=str(e),
                ))

        elapsed = self._clock() - start
        self.log(
            "done",
            questions=len(questions),
            failures=len(failures),
            elapsed=f"{elapsed:.2f}s",
        )

        return AgentResult(
            success=True,
            payload=WritingAgentOutput(questions=questions, failures=failures),
            telemetry={
                "questions_generated": len(questions),
                "failures": len(failures),
                "sources": sources,
                "total_attempts": total_attempts,
                "elapsed_seconds": round(elapsed, 3),
            },
        )

    # -------------------------------------------------------------- Internals

    def _generate_one(
        self,
        plan,
        budget: Optional[GenerationBudget] = None,
    ) -> tuple:
        """
        Generate one question with closed-loop IRT difficulty feedback.
        Returns (GeneratedQuestion, attempts).
        """
        if getattr(plan, "status", None) == PlanStatus.UNRESOLVED:
            return self._generate_one_raw(plan)

        # For PLANNED structured questions, generate once without whole-question IRT retries.
        # Calibrate with IRT if enabled, and record calibration result in telemetry.
        if getattr(plan, "status", None) == PlanStatus.PLANNED:
            gq, attempts = self._generate_one_raw(plan)
            if self._enable_irt and self._irt is not None:
                self._current_bloom = plan.bloom
                target_band = self._band_for_plan(plan)
                try:
                    estimate = self._irt.calibrate(gq.question_text)
                    lo, hi = self._difficulty_bands.get(target_band, (-3.0, 3.0))
                    gq.telemetry["irt_difficulty"] = estimate.difficulty
                    gq.telemetry["irt_converged"] = estimate.converged
                    gq.telemetry["irt_attempts"] = 1
                    if lo <= estimate.difficulty <= hi:
                        gq.telemetry["irt_band"] = target_band
                    else:
                        gq.telemetry["irt_band_miss"] = target_band
                except Exception as e:
                    print(f"[IRT] calibration exception for {plan.slot_id}: {e}", flush=True)
                    gq.telemetry["irt_calibration_error"] = str(e)
            return gq, attempts

        if not self._enable_irt or self._irt is None:
            return self._generate_one_raw(plan)

        if budget is None:
            budget = GenerationBudget()

        max_difficulty_retries = 2
        target_band = self._band_for_plan(plan)
        difficulty_feedback = None
        total_inner_attempts = 0

        for difficulty_attempt in range(max_difficulty_retries + 1):
            budget.charge(irt=(difficulty_attempt > 0))
            gq, attempts = self._generate_one_raw(
                plan,
                difficulty_feedback=difficulty_feedback,
            )
            total_inner_attempts += attempts

            # Calibrate difficulty using IRT
            self._current_bloom = plan.bloom
            try:
                estimate = self._irt.calibrate(gq.question_text)
            except Exception as e:
                print(f"[IRT] calibration exception for {plan.slot_id}: {e}", flush=True)
                gq.telemetry["irt_calibration_error"] = str(e)
                return gq, total_inner_attempts

            lo, hi = self._difficulty_bands.get(target_band, (-3.0, 3.0))
            if lo <= estimate.difficulty <= hi:
                gq.telemetry["irt_difficulty"] = estimate.difficulty
                gq.telemetry["irt_band"] = target_band
                gq.telemetry["irt_converged"] = estimate.converged
                gq.telemetry["irt_attempts"] = difficulty_attempt + 1
                return gq, total_inner_attempts

            # Out of band — check if budget allows retry
            if not budget.can_retry_irt():
                print(
                    f"[IRT] {plan.slot_id}: global IRT retry budget exhausted "
                    f"({budget.used_irt}/{budget.max_total_irt_retries}); accepting out-of-band question",
                    flush=True,
                )
                break

            # Out of band — adjust feedback and retry
            if difficulty_attempt < max_difficulty_retries:
                difficulty_feedback = build_difficulty_feedback(estimate, target_band)
                print(
                    f"[IRT] {plan.slot_id}: measured b={estimate.difficulty:.2f} "
                    f"target={target_band} [{lo}, {hi}] — regenerating with feedback",
                    flush=True,
                )

        # Exhausted difficulty retries — accept with warning telemetry
        gq.telemetry["irt_difficulty"] = estimate.difficulty
        gq.telemetry["irt_band_miss"] = target_band
        gq.telemetry["irt_converged"] = estimate.converged
        gq.telemetry["irt_attempts"] = max_difficulty_retries + 1
        return gq, total_inner_attempts

    def _generate_one_raw(
        self,
        plan,
        difficulty_feedback: Optional[str] = None,
    ) -> tuple:
        """
        Validate plan and dispatch to structured subpart generation or legacy generation.
        """
        plan_status = getattr(plan, "status", None)

        if plan_status == PlanStatus.UNRESOLVED:
            f_code = getattr(plan, "failure_code", None)
            code_str = f_code.value if hasattr(f_code, "value") else (str(f_code) if f_code else "UNRESOLVED_PLAN")
            f_detail = getattr(plan, "failure_detail", None) or f"QuestionPlan {plan.slot_id} is unresolved"
            raise GenerationFailed(code=code_str, detail=f_detail)

        if plan_status == PlanStatus.PLANNED:
            # Must fail closed if sub_part_plans missing or invalid
            sub_part_plans = getattr(plan, "sub_part_plans", None)
            if not sub_part_plans or not plan.is_currently_valid():
                is_valid, f_code, f_detail = plan.validation_result()
                code_str = f_code.value if hasattr(f_code, "value") else (str(f_code) if f_code else "INVALID_PLAN")
                detail_str = f_detail or f"QuestionPlan {plan.slot_id} marked PLANNED but sub_part_plans are invalid or missing"
                raise GenerationFailed(code=code_str, detail=detail_str)
            return self._generate_structured_subparts(plan, difficulty_feedback=difficulty_feedback)

        if plan_status == PlanStatus.LEGACY_UNVALIDATED:
            return self._generate_legacy_raw(plan, difficulty_feedback=difficulty_feedback)

        raise GenerationFailed(
            code="INVALID_PLAN_STATUS",
            detail=f"Unknown plan status {plan_status!r} for slot {getattr(plan, 'slot_id', 'unknown')}",
        )

    def _validate_subpart_raw_artifacts(self, text_raw: str) -> Optional[str]:
        """Validate raw model response text for forbidden bullet markers, template placeholders, and truncation."""
        forbidden_bullets = ("•", "▪", "→", "◆", "■")
        for b in forbidden_bullets:
            if b in text_raw:
                return f"subpart text contains forbidden formatting artifact: bullet marker {b!r}"

        # Unparsed template placeholders like {TOPIC}, [TOPIC], {CONCEPT}, etc.
        placeholder_match = re.search(r"\{[A-Z_]{2,}\}|\[[A-Z_]{2,}\]", text_raw)
        if placeholder_match:
            return f"subpart text contains unparsed template placeholder: {placeholder_match.group(0)!r}"

        # Mid-clause ending / trailing punctuation
        stripped = text_raw.strip()
        if stripped.endswith((",", ";", "-")):
            return f"subpart text ends mid-clause with trailing punctuation: {stripped[-1]!r}"

        if len(stripped.split()) < 2:
            return "subpart text is too short or incomplete"

        return None

    def _validate_subpart_bloom(self, text_raw: str, bloom: str, marks: int, label: str = "a") -> Optional[str]:
        """
        Validate opening Bloom verb and cognitive demand alignment for an individual subpart.
        Enforces:
          Layer 1 & 3: Exact canonical slot compliance (no adjacency tolerance).
          Layer 2: Semantic cognitive demand (PASS / FAIL / INDETERMINATE with fail-closed policy).
        """
        if not bloom:
            bloom = "L2"
        bloom = bloom.upper()

        first = _first_meaningful_word(text_raw)
        if not first:
            return "subpart text is empty or contains only stopwords"

        # Layer 1 & 3: Exact Slot Compliance via Canonical Classifier
        compliant, detected_lvl, slot_reason = is_exact_slot_compliant(first, bloom)
        if not compliant:
            return slot_reason

        # Singular Directive Enforcement (no compound/multi-directive tasks)
        directive_err = validate_singular_directive(text_raw)
        if directive_err:
            return directive_err

        # Source Tag Leakage Enforcement (no Figure 5.1 / Chapter N / Section N / Module N tags)
        tag_err = validate_no_source_tags(text_raw)
        if tag_err:
            return tag_err

        # Layer 2: Semantic Cognitive Demand Classification
        outcome, demand_reason = evaluate_cognitive_demand(text_raw, bloom)
        if outcome == SemanticDemandOutcome.FAIL:
            return f"Cognitive demand failure for {bloom}: {demand_reason}"
        if outcome == SemanticDemandOutcome.INDETERMINATE:
            return (
                f"Cognitive demand is indeterminate for {bloom}: {demand_reason} "
                f"Please frame the subpart with explicit scenario context, criteria, or constraints."
            )

        return None

    def _validate_subpart_independence(self, text_raw: str, label: str, prior_subparts: List[SubPart]) -> Optional[str]:
        """
        Validate multi-part sibling independence: subsequent subparts must not chain to
        earlier subparts or begin with referring pronouns.
        """
        if not prior_subparts:
            return None

        chain_patterns = [
            (r"\bwith\s+reference\s+to\s+(?:part\s+\(?[a-zA-Z]\)?|\([a-zA-Z]\))", "explicit chaining to previous part"),
            (r"\bin\s+(?:the\s+)?context\s+of\s+(?:part\s+\(?[a-zA-Z]\)?|\([a-zA-Z]\))", "contextual dependency on previous part"),
            (r"\busing\s+(?:your|the)\s+answer\s+(?:from|to)\s+(?:part\s+\(?[a-zA-Z]\)?|\([a-zA-Z]\)|above)", "dependency on previous part's answer"),
            (r"\bextend(?:ing)?\s+(?:the\s+)?(?:above|part\s*\(?[a-zA-Z]\)?)", "extends previous part instead of testing distinct topic"),
            (r"\bcontinuing\s+from\s+(?:part\s*\(?[a-zA-Z]\)?|above)", "continuity chaining from previous part"),
            (r"\bcompare\s+(?:this\s+)?to\s+(?:your|the)\s+answer\b", "references previous answer"),
            (r"\bhow\s+does\s+(?:this|that)\s+compare\b", "relies on comparison with previous part"),
            (r"\bsubsequent\s+to\s+(?:the\s+)?above\b", "sequential dependency on previous part"),
            (r"\bbased\s+on\s+the\s+result\s+of\b", "result dependency on previous part"),
        ]

        for pattern, desc in chain_patterns:
            if re.search(pattern, text_raw, re.IGNORECASE):
                return f"Sibling chaining detected in subpart ({label}): {desc}"

        clean_leading = re.sub(r"^\s*\(?[a-zA-Z0-9ivxIVX]+\)?[.:]\s*", "", text_raw).strip()
        first_token = (clean_leading.split()[0].rstrip(".,:;!?").lower()) if clean_leading.split() else ""
        first_meaningful = _first_meaningful_word(clean_leading)
        referring_pronouns = {"it", "this", "that", "these", "those", "they", "them", "such"}
        if first_token in referring_pronouns or first_meaningful in referring_pronouns:
            return f"Subpart ({label}) starts with referring pronoun {first_token or first_meaningful!r} rather than an independent concept or verb"

        return None

    def _generate_structured_subparts(
        self,
        plan,
        difficulty_feedback: Optional[str] = None,
    ) -> Tuple[GeneratedQuestion, int]:
        """
        Generate structured SubPart objects for each SubPartPlan independently.
        Fails closed on any subpart exhaustion without partial return or mark redistribution.
        """
        # Upfront check for multiple visuals across subparts in plan
        visual_subparts = [sp for sp in plan.sub_part_plans if sp.visual_required]
        if len(visual_subparts) > 1:
            fig_keys = set()
            for sp in visual_subparts:
                if sp.figure is not None:
                    f_key = get_figure_field(sp.figure, "image_path") or get_figure_field(sp.figure, "id")
                    if f_key:
                        fig_keys.add(f_key)
                else:
                    fig_keys.add(f"missing_fig_{sp.label}")
            if len(fig_keys) > 1:
                raise GenerationFailed(
                    code="UNSUPPORTED_MULTIPLE_VISUALS",
                    detail=(
                        f"Slot {plan.slot_id} has {len(visual_subparts)} subparts requiring distinct figures "
                        f"({sorted(list(fig_keys))}), exceeding single-visual contract of GeneratedQuestion"
                    ),
                )

        subparts: List[SubPart] = []
        subpart_attempts: Dict[str, int] = {}
        subpart_diagram_requests: Dict[str, Any] = {}
        subpart_bloom_verbs: Dict[str, str] = {}
        vlm_used = False

        for sp_plan in plan.sub_part_plans:
            use_vlm = False
            if sp_plan.visual_required:
                fig_path = get_figure_field(sp_plan.figure, "image_path") if sp_plan.figure is not None else None
                if sp_plan.figure is None or not fig_path:
                    raise GenerationFailed(
                        code="MISSING_FIGURE_ASSET",
                        detail=f"Subpart ({sp_plan.label}) requires a visual but no figure asset with image_path was provided",
                    )
                if self._vlm is None:
                    raise GenerationFailed(
                        code="MISSING_VLM_CAPABILITY",
                        detail=f"Subpart ({sp_plan.label}) has visual_required=True but no VLM caller is available",
                    )
                use_vlm = True
                vlm_used = True

            last_error = ""
            subpart_succeeded = False

            for attempt in range(1, self._max_attempts + 1):
                strict_json = attempt > 1
                sp_spec = getattr(sp_plan, "specification", None)
                if sp_spec is not None:
                    prompt = build_archetype_subpart_prompt(
                        plan,
                        sp_plan,
                        sp_spec,
                        strict_json=strict_json,
                        difficulty_feedback=difficulty_feedback,
                        validation_feedback=last_error if attempt > 1 else None,
                    )
                else:
                    prompt = build_subpart_prompt(
                        plan,
                        sp_plan,
                        strict_json=strict_json,
                        difficulty_feedback=difficulty_feedback,
                        validation_feedback=last_error if attempt > 1 else None,
                    )

                try:
                    if use_vlm:
                        raw = self._vlm.call(
                            prompt,
                            image_path=get_figure_field(sp_plan.figure, "image_path", ""),
                            schema=None,
                            seed=self._seed,
                        )
                    else:
                        raw = self._llm.call(
                            prompt,
                            schema=None,
                            image_path=None,
                            seed=self._seed,
                        )
                except Exception as e:
                    is_to = isinstance(e, TimeoutError) or "timeout" in str(e).lower() or "timed out" in str(e).lower()
                    last_error = f"caller raised: {'timeout: ' if is_to else ''}{e}"
                    print(f"[WRITING-WARN] slot={plan.slot_id} subpart={sp_plan.label} attempt={attempt} caller raised: {e}", flush=True)
                    continue

                parsed = self._parse_response(raw)
                if parsed is None:
                    last_error = "response was not valid JSON"
                    continue

                # Check required keys for subpart: text, solution, marking_scheme
                missing = [k for k in REQUIRED_SUBPART_KEYS if k not in parsed]
                if missing:
                    last_error = f"missing required keys: {missing}"
                    continue

                text_raw = parsed.get("text")
                if not isinstance(text_raw, str) or not text_raw.strip():
                    last_error = "subpart 'text' must be a non-empty string"
                    continue

                sol_raw = parsed.get("solution")
                if not isinstance(sol_raw, str) or not sol_raw.strip():
                    last_error = "subpart 'solution' must be a non-empty string"
                    continue

                scheme_raw = parsed.get("marking_scheme")
                if not isinstance(scheme_raw, list) or len(scheme_raw) < 1 or len(scheme_raw) > 4:
                    last_error = f"marking_scheme must be list of 1 to 4 items, got {len(scheme_raw) if isinstance(scheme_raw, list) else type(scheme_raw).__name__}"
                    continue

                # Validate marking scheme entries and sum
                scheme_sum = 0
                valid_entries = True
                for i_c, entry in enumerate(scheme_raw):
                    if not isinstance(entry, dict):
                        last_error = f"marking_scheme[{i_c}] must be dict"
                        valid_entries = False
                        break
                    crit_text = entry.get("criterion")
                    if not isinstance(crit_text, str) or not crit_text.strip():
                        last_error = f"marking_scheme[{i_c}] missing valid criterion"
                        valid_entries = False
                        break
                    crit_marks = entry.get("marks")
                    if type(crit_marks) is not int or isinstance(crit_marks, bool) or crit_marks <= 0:
                        last_error = f"marking_scheme[{i_c}] marks must be strictly positive int"
                        valid_entries = False
                        break
                    scheme_sum += crit_marks

                if not valid_entries:
                    continue

                if scheme_sum != sp_plan.marks:
                    last_error = f"marking_scheme criteria sum ({scheme_sum}) != subpart marks ({sp_plan.marks})"
                    continue

                # --- TRACK A / STAGE G2 VALIDATION ---
                # 1. Raw text formatting artifacts & template fragments (MUST run BEFORE cleaning)
                raw_err = self._validate_subpart_raw_artifacts(text_raw)
                if raw_err:
                    last_error = raw_err
                    continue

                # 2. Sibling independence (must not chain to previous subparts)
                indep_err = self._validate_subpart_independence(text_raw, sp_plan.label, subparts)
                if indep_err:
                    last_error = indep_err
                    continue

                # 3. Bloom verb, singular directive, and cognitive depth alignment
                bloom_err = self._validate_subpart_bloom(text_raw, getattr(sp_plan, "bloom", "L2"), sp_plan.marks, label=sp_plan.label)
                if bloom_err:
                    last_error = bloom_err
                    continue

                # 4. Single-topic focus (must not bundle sibling topics)
                sibling_topics = [
                    getattr(s, "topic", "")
                    for s in plan.sub_part_plans
                    if getattr(s, "label", "") != sp_plan.label and getattr(s, "topic", None)
                ]
                topic_err = validate_single_topic(text_raw, getattr(sp_plan, "topic", None), sibling_topics)
                if topic_err:
                    last_error = topic_err
                    continue

                # Clean subpart text (strip preamble and any accidental leading label)
                clean_t = self._clean_text(text_raw)
                # Strip leading subpart label like "(a)", "a.", "a)" if model included it
                clean_t = re.sub(r"^\s*\(?[a-zA-Z0-9ivxIVX]+\)?[.:]\s*", "", clean_t)
                clean_t = clean_t.strip()
                if not clean_t:
                    last_error = "subpart text was empty after cleaning"
                    continue

                clean_sol = sol_raw.strip()

                # Enforce figure reference in question text for visual-required subparts
                if sp_plan.visual_required and not self._references_figure(clean_t):
                    last_error = f"visual-required subpart ({sp_plan.label}) produced text without a figure reference"
                    continue

                # Authoritative label assignment: sp_plan.label governs, ignoring any LLM-supplied label
                try:
                    ev_ids = []
                    for b in (getattr(sp_plan, "evidence_blocks", None) or []):
                        b_id = getattr(b, "id", None) if not isinstance(b, dict) else b.get("id")
                        if b_id:
                            ev_ids.append(str(b_id))

                    sp = SubPart(
                        label=sp_plan.label,
                        marks=sp_plan.marks,
                        text=clean_t,
                        topic=sp_plan.topic or "",
                        solution=clean_sol,
                        marking_scheme=scheme_raw,
                        bloom=getattr(sp_plan, "bloom", "") or "",
                        co=getattr(sp_plan, "co", "") or "",
                        evidence_block_ids=tuple(ev_ids),
                        concept_footprint=getattr(sp_plan, "concept_footprint", None),
                        expected_response=getattr(sp_plan, "expected_response", None),
                        visual_task=getattr(sp_plan, "visual_task", None),
                    )
                except Exception as exc:
                    last_error = f"SubPart construction failed: {exc}"
                    continue

                # Capture diagram request and bloom verb if emitted
                if parsed.get("diagram_request"):
                    subpart_diagram_requests[sp_plan.label] = parsed.get("diagram_request")
                if parsed.get("bloom_verb_used"):
                    subpart_bloom_verbs[sp_plan.label] = parsed.get("bloom_verb_used")
                else:
                    first_v = _first_meaningful_word(clean_t)
                    if first_v:
                        subpart_bloom_verbs[sp_plan.label] = first_v

                sp_spec = getattr(sp_plan, "specification", None)
                if sp_spec is not None:
                    sp.specification = sp_spec

                if sp_plan.figure is not None:
                    sp.figure_id = getattr(sp_plan.figure, "id", None) if not isinstance(sp_plan.figure, dict) else sp_plan.figure.get("id")
                    sp.image_path = getattr(sp_plan.figure, "image_path", None) if not isinstance(sp_plan.figure, dict) else sp_plan.figure.get("image_path")
                    sp.figure_caption = getattr(sp_plan.figure, "caption", None) if not isinstance(sp_plan.figure, dict) else sp_plan.figure.get("caption")
                    sp.image_sha256 = getattr(sp_plan.figure, "image_sha256", None) if not isinstance(sp_plan.figure, dict) else sp_plan.figure.get("image_sha256")

                subparts.append(sp)
                subpart_attempts[sp_plan.label] = attempt
                subpart_succeeded = True
                break

            if not subpart_succeeded:
                failure_code = "TIMEOUT" if ("timeout" in last_error.lower() or "timed out" in last_error.lower()) else "SUBPART_GENERATION_FAILED"
                raise GenerationFailed(
                    code=failure_code,
                    detail=f"Subpart ({sp_plan.label}) failed after {self._max_attempts} attempts: {last_error}",
                )

        # All subparts succeeded. Now assemble parent GeneratedQuestion.
        # Visual mapping:
        references_image = False
        image_path = None
        figure_caption = None
        primary_diagram_request = None

        for sp_plan, sp in zip(plan.sub_part_plans, subparts):
            has_fig = bool(get_figure_field(sp_plan.figure, "image_path")) if sp_plan.figure is not None else False
            text_ref = self._references_figure(sp.text)
            if sp_plan.visual_required and has_fig and text_ref:
                references_image = True
            if has_fig and image_path is None:
                image_path = get_figure_field(sp_plan.figure, "image_path")
                figure_caption = get_figure_field(sp_plan.figure, "caption")

        if not image_path and plan.figure:
            image_path = get_figure_field(plan.figure, "image_path")
            figure_caption = get_figure_field(plan.figure, "caption")
            if plan.visual_required and any(self._references_figure(sp.text) for sp in subparts):
                references_image = True

        figure_id = None
        image_sha256 = None
        visual_sps = [sp for sp in subparts if getattr(sp, "figure_id", None)]
        if visual_sps:
            figure_id = visual_sps[0].figure_id
            image_sha256 = visual_sps[0].image_sha256
        elif plan.figure is not None:
            figure_id = getattr(plan.figure, "id", None) if not isinstance(plan.figure, dict) else plan.figure.get("id")
            image_sha256 = getattr(plan.figure, "image_sha256", None) if not isinstance(plan.figure, dict) else plan.figure.get("image_sha256")

        if len(subpart_diagram_requests) > 1:
            def _normalize_diagram_request(d: Any) -> Optional[tuple]:
                if not isinstance(d, dict):
                    return None

                def _freeze_val(val: Any) -> Any:
                    if isinstance(val, dict):
                        return tuple(sorted((str(k).strip().lower(), _freeze_val(v)) for k, v in val.items()))
                    elif isinstance(val, (list, tuple)):
                        return tuple(_freeze_val(x) for x in val)
                    elif isinstance(val, str):
                        return val.strip().lower()
                    return val

                norm = {}
                for k, v in d.items():
                    norm[str(k).strip().lower()] = _freeze_val(v)
                if not any(norm.values()):
                    return None
                return tuple(sorted(norm.items()))

            distinct_requests = set()
            for d in subpart_diagram_requests.values():
                norm = _normalize_diagram_request(d)
                if norm is not None:
                    distinct_requests.add(norm)

            if len(distinct_requests) > 1:
                raise GenerationFailed(
                    code="UNSUPPORTED_MULTIPLE_VISUALS",
                    detail=(
                        f"Slot {plan.slot_id} produced conflicting diagram requests across subparts: "
                        f"{list(subpart_diagram_requests.values())}"
                    ),
                )

        if subpart_diagram_requests:
            primary_diagram_request = (
                subpart_diagram_requests.get("a")
                or list(subpart_diagram_requests.values())[0]
            )

        source = "local_vlm" if vlm_used else "local_llm"
        parent_attempts = max(subpart_attempts.values()) if subpart_attempts else 1

        # Build evidence summary
        evidence_summary = self._build_evidence_summary(plan)

        gq = GeneratedQuestion(
            slot_id=plan.slot_id,
            module_id=plan.module_id,
            global_q_idx=plan.global_q_idx,
            question_text="",   # Populated by sync_materialized_projections()
            solution="",        # Populated by sync_materialized_projections()
            marking_scheme=[],  # Populated by sync_materialized_projections()
            marks=plan.total_marks,
            partition=list(plan.partition),
            bloom=plan.bloom,
            co=plan.co,
            topic=plan.topic,
            evidence_summary=evidence_summary,
            references_image=references_image,
            image_path=image_path,
            figure_caption=figure_caption,
            diagram_request=primary_diagram_request,
            generation_source=source,
            attempts=parent_attempts,
            telemetry={
                "subpart_attempts": subpart_attempts,
                "bloom_verb_used": subpart_bloom_verbs.get("a", ""),
                "subpart_bloom_verbs": subpart_bloom_verbs,
                "subpart_diagram_requests": subpart_diagram_requests,
            },
            sub_parts=subparts,
            figure_id=figure_id,
            image_sha256=image_sha256,
        )

        # Synchronize projections
        gq.sync_materialized_projections()

        valid_proj, err_proj = gq.verify_projection_consistency()
        if not valid_proj:
            raise GenerationFailed(
                code="PROJECTION_INCONSISTENCY",
                detail=f"Synthesized question failed projection consistency check: {err_proj}",
            )

        valid_struct, err_struct = gq.validate_structure(allow_legacy=False, check_projections=True)
        if not valid_struct:
            raise GenerationFailed(
                code="INVALID_STRUCTURE",
                detail=f"Synthesized question failed structural validation: {err_struct}",
            )

        return gq, parent_attempts

    def _generate_legacy_raw(
        self,
        plan,
        difficulty_feedback: Optional[str] = None,
    ) -> tuple:
        """
        Monolithic question generation attempt loop for LEGACY_UNVALIDATED plans.
        Returns (GeneratedQuestion, attempts).
        """
        # Route to VLM if visual required and VLM available
        use_vlm = (
            plan.visual_required
            and plan.figure is not None
            and self._vlm is not None
        )
        source = "local_vlm" if use_vlm else "local_llm"

        last_error = ""
        for attempt in range(1, self._max_attempts + 1):
            strict_json = attempt > 1
            prompt = build_prompt(
                plan,
                strict_json=strict_json,
                difficulty_feedback=difficulty_feedback,
            )

            try:
                if use_vlm:
                    raw = self._vlm.call(
                        prompt,
                        image_path=get_figure_field(plan.figure, "image_path", ""),
                        schema=None,
                        seed=self._seed,
                    )
                else:
                    raw = self._llm.call(
                        prompt,
                        schema=None,
                        image_path=None,
                        seed=self._seed,
                    )
            except Exception as e:
                last_error = f"caller raised: {e}"
                print(f"[WRITING-WARN] slot={plan.slot_id} attempt={attempt} caller raised: {e}", flush=True)
                continue

            parsed = self._parse_response(raw)
            if parsed is None:
                last_error = "response was not valid JSON"
                continue

            missing = [k for k in REQUIRED_OUTPUT_KEYS if k not in parsed]
            if missing:
                last_error = f"missing required keys: {missing}"
                continue

            # Check figure reference if visual required
            # Distinguish asset availability from reference wording:
            # references_image is True ONLY when question references an actually available figure asset.
            has_figure = bool(get_figure_field(plan.figure, "image_path", None)) if plan.figure else False
            text_references = self._references_figure(parsed["question_text"])
            references_image = bool(text_references and has_figure)
            if plan.visual_required and not text_references:
                last_error = "visual-required slot produced a question with no figure reference"
                # We accept it anyway — the Evaluation Agent will flag it.
                # (Design choice: don't waste a retry on phrasing.)

            return self._build_question(
                plan=plan,
                parsed=parsed,
                source=source,
                attempts=attempt,
                references_image=references_image,
            ), attempt

        failure_code = "TIMEOUT" if "timeout" in last_error.lower() else "GENERATION_FAILED"
        raise GenerationFailed(
            code=failure_code,
            detail=f"failed after {self._max_attempts} attempts: {last_error}",
        )

    def _parse_response(self, raw: Any) -> Optional[dict]:
        """
        Accept dict, JSON string, or string with code fences.
        Returns a dict or None.
        """
        if raw is None:
            return None

        if isinstance(raw, dict):
            return raw

        if not isinstance(raw, str):
            return None

        text = raw.strip()

        # Strip markdown code fences
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        # Extract first {...} block if there's surrounding prose
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)

        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _references_figure(self, question_text: str) -> bool:
        if not question_text:
            return False
        lower = question_text.lower()
        return any(phrase in lower for phrase in IMAGE_REFERENCE_PHRASES)

    def _clean_text(self, text: str) -> str:
        """Strip forbidden preambles, bullet markers, and trailing stray markers."""
        q_text = str(text or "").strip()
        _preamble_regex = re.compile(
            r"^(?:Based\s+on\s+(?:the\s+)?evidence\s*(?:provided)?\s*[,.:;]?\s*|"
            r"Based\s+on\s+the\s+provided\s+evidence\s*[,.:;]?\s*|"
            r"Apply\s+the\s+following\s*[:.,;]?\s*|"
            r"Refer\s+to\s+the\s+(?:text|notes|book)\s*[,.:;]?\s*|"
            r"Using\s+the\s+provided\s+(?:evidence|text)\s*[,.:;]?\s*)",
            re.IGNORECASE,
        )
        m = _preamble_regex.match(q_text)
        if m:
            stripped = q_text[m.end():].strip()
            if stripped:
                q_text = stripped[0].upper() + stripped[1:]

        # Strip bullet markers and template artifacts
        q_text = re.sub(r"[▪•→◆■]+", " ", q_text)
        # Strip trailing/inline stray subpart markers like "...their functions (i)." or "...processes. (ii)"
        q_text = re.sub(r"\s*\([ivxIVX\d]+\)\s*(?=[.?!]|$)", "", q_text)
        q_text = re.sub(r"\.\s*\([ivxIVX\d]+\)\s*$", ".", q_text)
        q_text = re.sub(r"\s+", " ", q_text).strip()
        return q_text

    def _build_evidence_summary(self, plan) -> str:
        """Build evidence summary from plan's and subparts' evidence blocks or fallback."""
        all_blocks = []
        seen_keys = set()

        for sp in getattr(plan, "sub_part_plans", None) or []:
            for block in getattr(sp, "evidence_blocks", None) or []:
                b_id = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
                b_text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
                k = b_id or (b_text[:80] if b_text else None)
                if k and k not in seen_keys:
                    seen_keys.add(k)
                    all_blocks.append(block)

        for block in getattr(plan, "evidence_blocks", None) or []:
            b_id = getattr(block, "id", None) if not isinstance(block, dict) else block.get("id")
            b_text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
            k = b_id or (b_text[:80] if b_text else None)
            if k and k not in seen_keys:
                seen_keys.add(k)
                all_blocks.append(block)

        evidence_texts = []
        for block in all_blocks[:5]:
            text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
            if text:
                evidence_texts.append(text[:500])
        evidence_summary = " ".join(evidence_texts)[:2500]
        if not evidence_summary or len(evidence_summary.strip()) < 80:
            concepts = ", ".join(getattr(plan, "concept_neighborhood", []) or [])
            evidence_summary = (
                f"Module topic: {plan.topic}. "
                f"Related curriculum concepts: {concepts if concepts else plan.topic}. "
                f"Evidence context for question {plan.slot_id}."
            )
        return evidence_summary

    def _build_question(
        self,
        plan,
        parsed: dict,
        source: str,
        attempts: int,
        references_image: bool,
    ) -> GeneratedQuestion:
        figure = plan.figure
        image_path = get_figure_field(figure, "image_path", None)
        caption = get_figure_field(figure, "caption", None)
        fig_id = getattr(figure, "id", None) if not isinstance(figure, dict) else figure.get("id")
        fig_hash = getattr(figure, "image_sha256", None) if not isinstance(figure, dict) else figure.get("image_sha256")

        q_text = self._clean_text(parsed.get("question_text", ""))

        scheme = list(parsed.get("marking_scheme", []) or [])
        if scheme and all(isinstance(e, dict) for e in scheme):
            current_sum = sum(e.get("marks", 0) for e in scheme if isinstance(e.get("marks"), (int, float)))
            if current_sum > 0 and current_sum != plan.total_marks:
                scale = plan.total_marks / current_sum
                new_scheme = []
                for i, entry in enumerate(scheme):
                    if i < len(scheme) - 1:
                        new_marks = max(1, round(entry.get("marks", 0) * scale))
                    else:
                        new_marks = plan.total_marks - sum(e["marks"] for e in new_scheme)
                    new_scheme.append(dict(entry, marks=new_marks))
                scheme = new_scheme

        evidence_summary = self._build_evidence_summary(plan)

        return GeneratedQuestion(
            slot_id=plan.slot_id,
            module_id=plan.module_id,
            global_q_idx=plan.global_q_idx,
            question_text=q_text,
            solution=str(parsed.get("solution", "")).strip(),
            marking_scheme=scheme,
            marks=plan.total_marks,
            partition=list(plan.partition),
            bloom=plan.bloom,
            co=plan.co,
            topic=plan.topic,
            evidence_summary=evidence_summary,
            references_image=references_image,
            image_path=image_path,
            figure_caption=caption,
            diagram_request=parsed.get("diagram_request"),
            generation_source=source,
            attempts=attempts,
            telemetry={
                "bloom_verb_used": parsed.get("bloom_verb_used", ""),
                "prompt_length_chars": None,  # placeholder for future
            },
            figure_id=fig_id,
            image_sha256=fig_hash,
        )


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class GenerationFailed(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
