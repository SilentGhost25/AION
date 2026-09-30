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
from typing import Any, Callable, Dict, List, Optional, Protocol

from .base import Agent, AgentContext, AgentResult
from .generated_question import (
    GeneratedQuestion,
    SlotFailure,
    WritingAgentOutput,
)
from .prompts import build_prompt


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
        Generate one question. Returns (GeneratedQuestion, attempts).
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
            prompt = build_prompt(plan, strict_json=strict_json)

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

        return GeneratedQuestion(
            slot_id=plan.slot_id,
            module_id=plan.module_id,
            global_q_idx=plan.global_q_idx,
            question_text=str(parsed.get("question_text", "")).strip(),
            solution=str(parsed.get("solution", "")).strip(),
            marking_scheme=list(parsed.get("marking_scheme", []) or []),
            marks=plan.total_marks,
            partition=list(plan.partition),
            bloom=plan.bloom,
            co=plan.co,
            topic=plan.topic,
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
