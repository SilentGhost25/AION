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
    - Enforce Bloom verb-at-start (that is the Evaluation Agent)
"""

from __future__ import annotations

import json
import re
import time
import os
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

# UNTESTED — run quickstart.py on server first
from core.evaluation.irt_calibrator import IRTCalibrator, DifficultyEstimate
from .base import Agent, AgentContext, AgentResult
from .generated_question import (
    GeneratedQuestion,
    SlotFailure,
    WritingAgentOutput,
)
from .prompts import build_difficulty_feedback, build_prompt


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

# Phrases that signal a question references a figure.
IMAGE_REFERENCE_PHRASES = (
    "given figure", "given diagram", "given schematic",
    "shown in the figure", "shown in the diagram",
    "refer to the figure", "refer to the diagram",
    "with the aid of the given",
    "using the provided",
    "using the given",
)

# Required keys in the LLM's JSON output.
REQUIRED_OUTPUT_KEYS = ("question_text", "solution", "marking_scheme")


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
        feature_difficulty = (word_count / 100.0) + bloom_penalty
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

        for plan in plans:
            try:
                gq, attempts = self._generate_one(plan)
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

    def _generate_one(self, plan) -> tuple:
        """
        Generate one question with closed-loop IRT difficulty feedback.
        Returns (GeneratedQuestion, attempts).
        """
        if not self._enable_irt or self._irt is None:
            return self._generate_one_raw(plan)

        max_difficulty_retries = 2
        target_band = self._band_for_plan(plan)
        difficulty_feedback = None
        total_inner_attempts = 0

        for difficulty_attempt in range(max_difficulty_retries + 1):
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
        Raw question generation attempt loop. Returns (GeneratedQuestion, attempts).
        Raises GenerationFailed on final failure.
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
                        image_path=getattr(plan.figure, "image_path", ""),
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
            references_image = self._references_figure(parsed["question_text"])
            if plan.visual_required and not references_image:
                last_error = "visual-required slot produced a question with no figure reference"
                # We accept it anyway — the Evaluation Agent will flag it.
                # (Design choice: don't waste a retry on phrasing.)
                references_image = False

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

    def _build_question(
        self,
        plan,
        parsed: dict,
        source: str,
        attempts: int,
        references_image: bool,
    ) -> GeneratedQuestion:
        figure = plan.figure
        image_path = getattr(figure, "image_path", None) if figure else None
        caption = getattr(figure, "caption", None) if figure else None

        q_text = str(parsed.get("question_text", "")).strip()
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

        # Build evidence summary from the plan's evidence blocks
        evidence_texts = []
        for block in (getattr(plan, "evidence_blocks", None) or [])[:5]:
            text = getattr(block, "text", "")
            if text:
                evidence_texts.append(text[:500])
        evidence_summary = " ".join(evidence_texts)[:2500]

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
        )


# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------


class GenerationFailed(Exception):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
