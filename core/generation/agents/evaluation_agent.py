"""
Evaluation Agent — the first agent that talks to the API.

Pipeline per run:
    1. Run all local deterministic evaluators on every question.
    2. Collect questions that passed local checks.
    3. Batch-call the API for semantic audit.
    4. Merge local + API verdicts into a single AuditReport.
    5. On API unavailability, degrade to unverified verdicts.

The agent never raises on expected failures. All error handling
produces a well-formed AuditReport.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, List, Optional, Protocol

from .base import Agent, AgentContext, AgentResult
from .evaluation_contracts import (
    ALL_REASON_CODES,
    UNVERIFIED,
    AuditReport,
    SlotVerdict,
)
from .evaluation_prompts import (
    MAX_QUESTIONS_PER_BATCH,
    build_batch_prompt,
)
from .local_evaluators import run_local_evaluators


# -----------------------------------------------------------------------------
# API caller protocol
# -----------------------------------------------------------------------------


class APIUnavailable(Exception):
    """Raised by the API caller when no provider can serve the request."""


class APICaller(Protocol):
    """Injected API caller. Uses router + circuit breaker internally."""
    def call(self, prompt: str, schema: Optional[dict] = None) -> Any:
        ...
    @property
    def last_provider_name(self) -> Optional[str]:
        ...


# -----------------------------------------------------------------------------
# Agent
# -----------------------------------------------------------------------------


class EvaluationAgent(Agent):
    name = "evaluation"

    def __init__(
        self,
        api_caller: Optional[APICaller] = None,
        batch_size: int = MAX_QUESTIONS_PER_BATCH,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._api = api_caller
        self._batch_size = batch_size
        self._clock = clock or time.monotonic

    # ------------------------------------------------------------------ API

    def run(self, context: AgentContext) -> AgentResult:
        questions = list(getattr(context, "questions", []) or [])
        start = self._clock()

        if not questions:
            report = AuditReport(elapsed_seconds=0.0)
            return AgentResult(success=True, payload=report)

        # Phase 1: local evaluations
        verdicts: List[SlotVerdict] = []
        candidates = []
        for q in questions:
            local_failure = run_local_evaluators(q)
            if local_failure is not None:
                verdicts.append(local_failure)
            else:
                candidates.append(q)

        # Phase 2: API evaluations (only for locally-passed questions)
        api_used = False
        api_provider: Optional[str] = None
        api_calls = 0
        degraded = False

        if candidates and self._api is not None:
            api_verdicts, api_meta = self._evaluate_via_api(candidates)
            verdicts.extend(api_verdicts)
            api_used = api_meta["used"]
            api_provider = api_meta["provider"]
            api_calls = api_meta["calls"]
            degraded = api_meta["degraded"]
        elif candidates:
            # No API caller — mark all candidates as unverified
            for q in candidates:
                verdicts.append(SlotVerdict(
                    slot_id=q.slot_id,
                    verdict="unverified",
                    source="degraded",
                    reason_code=UNVERIFIED,
                    detail="no API caller configured",
                ))
            degraded = True

        elapsed = self._clock() - start

        # If any verdict ended up unverified, mark degraded mode
        if any(v.verdict == "unverified" for v in verdicts):
            degraded = True

        # Stable ordering by original question order
        order = {q.slot_id: i for i, q in enumerate(questions)}
        verdicts.sort(key=lambda v: order.get(v.slot_id, 999999))

        report = AuditReport(
            verdicts=verdicts,
            degraded_mode=degraded,
            api_used=api_used,
            api_provider=api_provider,
            api_calls=api_calls,
            elapsed_seconds=round(elapsed, 3),
            telemetry={
                "questions_evaluated": len(questions),
                "local_failures": len([v for v in verdicts if v.source == "local"]),
                "api_evaluated": len([v for v in verdicts if v.source == "api"]),
                "unverified": len([v for v in verdicts if v.verdict == "unverified"]),
            },
        )

        self.log(
            "done",
            evaluated=len(verdicts),
            failures=len(report.failed_slots()),
            degraded=degraded,
            api_calls=api_calls,
        )
        return AgentResult(success=True, payload=report, telemetry=report.telemetry)

    # -------------------------------------------------------------- API batching

    def _evaluate_via_api(self, questions: List[object]) -> tuple:
        """
        Batch-evaluate questions. Returns (verdicts, meta).

        meta = {"used": bool, "provider": Optional[str], "calls": int, "degraded": bool}
        """
        all_verdicts: List[SlotVerdict] = []
        provider_name: Optional[str] = None
        degraded = False

        batches = [
            questions[i : i + self._batch_size]
            for i in range(0, len(questions), self._batch_size)
        ]

        for batch in batches:
            try:
                prompt = build_batch_prompt(batch)
                raw = self._api.call(prompt, schema=None)
                provider_name = getattr(self._api, "last_provider_name", None) or provider_name
            except APIUnavailable as e:
                self.log("api_unavailable", detail=str(e), batch_size=len(batch))
                for q in batch:
                    all_verdicts.append(SlotVerdict(
                        slot_id=q.slot_id,
                        verdict="unverified",
                        source="degraded",
                        reason_code=UNVERIFIED,
                        detail="all providers unavailable",
                    ))
                degraded = True
                continue
            except Exception as e:
                self.log("api_error", detail=str(e), batch_size=len(batch))
                for q in batch:
                    all_verdicts.append(SlotVerdict(
                        slot_id=q.slot_id,
                        verdict="unverified",
                        source="degraded",
                        reason_code=UNVERIFIED,
                        detail=f"api error: {type(e).__name__}",
                    ))
                degraded = True
                continue

            parsed = self._parse_api_response(raw)
            if parsed is None:
                for q in batch:
                    all_verdicts.append(SlotVerdict(
                        slot_id=q.slot_id,
                        verdict="unverified",
                        source="degraded",
                        reason_code=UNVERIFIED,
                        detail="malformed api response",
                    ))
                degraded = True
                continue

            batch_verdicts = self._verdicts_from_response(parsed, batch)
            if any(v.verdict == "unverified" for v in batch_verdicts):
                degraded = True
            all_verdicts.extend(batch_verdicts)

        return all_verdicts, {
            "used": len(all_verdicts) > 0,
            "provider": provider_name,
            "calls": len(batches),
            "degraded": degraded,
        }

    def _parse_api_response(self, raw: Any) -> Optional[dict]:
        if raw is None:
            return None
        if isinstance(raw, dict):
            return raw
        if not isinstance(raw, str):
            return None

        text = raw.strip()
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        if not text.startswith("{"):
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                text = m.group(0)
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None

    def _verdicts_from_response(
        self, parsed: dict, batch: List[object]
    ) -> List[SlotVerdict]:
        """
        Convert the API's parsed JSON into SlotVerdicts. Fills in
        unverified verdicts for any slot the API omitted.
        """
        api_by_slot = {}
        for entry in parsed.get("verdicts", []):
            if not isinstance(entry, dict):
                continue
            slot_id = entry.get("slot_id")
            if slot_id:
                api_by_slot[slot_id] = entry

        result: List[SlotVerdict] = []
        for q in batch:
            entry = api_by_slot.get(q.slot_id)
            if entry is None:
                result.append(SlotVerdict(
                    slot_id=q.slot_id,
                    verdict="unverified",
                    source="degraded",
                    reason_code=UNVERIFIED,
                    detail="slot missing from api response",
                ))
                continue
            result.append(self._entry_to_verdict(q.slot_id, entry))
        return result

    def _entry_to_verdict(self, slot_id: str, entry: dict) -> SlotVerdict:
        verdict_str = str(entry.get("verdict", "pass")).lower()
        if verdict_str not in ("pass", "fail"):
            verdict_str = "unverified"

        reason_codes = entry.get("reason_codes") or []
        if isinstance(reason_codes, str):
            reason_codes = [reason_codes]
        # Pick the first known code
        known_code = None
        for rc in reason_codes:
            if rc in ALL_REASON_CODES:
                known_code = rc
                break

        if verdict_str == "fail" and not known_code:
            known_code = "UNGROUNDED"  # safe default if API returns unknown code

        if verdict_str == "pass":
            return SlotVerdict(
                slot_id=slot_id,
                verdict="pass",
                source="api",
                detail=str(entry.get("detail", ""))[:300],
            )
        if verdict_str == "unverified":
            return SlotVerdict(
                slot_id=slot_id,
                verdict="unverified",
                source="degraded",
                reason_code=UNVERIFIED,
                detail=str(entry.get("detail", ""))[:300],
            )
        return SlotVerdict(
            slot_id=slot_id,
            verdict="fail",
            source="api",
            reason_code=known_code,
            detail=str(entry.get("detail", ""))[:300],
            suggested_fix=entry.get("suggested_fix"),
        )
