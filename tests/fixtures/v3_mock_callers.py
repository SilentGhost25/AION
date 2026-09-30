"""
Mock callers for v3 pipeline testing.

Provides controlled LLM, VLM, and API implementations that produce
valid, deterministic responses. Tests can override specific behaviors
to exercise failure paths.

Design principles:
    - Deterministic: same prompt + same seed → same response.
    - Faithful to the real contract: returns dicts, not strings.
    - No network, no model loading, no GPU.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


# -----------------------------------------------------------------------------
# Deterministic response builder
# -----------------------------------------------------------------------------


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]


def _first_verb(prompt: str) -> str:
    """Extract a plausible Bloom verb from the prompt's allowed list."""
    m = re.search(r"approved \w+ verb: ([^\.]+)", prompt)
    if not m:
        return "explain"
    verbs = [v.strip().strip("'") for v in m.group(1).split(",")]
    return verbs[0] if verbs else "explain"


def _marks_from_prompt(prompt: str) -> int:
    m = re.search(r"Marks: (\d+)", prompt)
    return int(m.group(1)) if m else 10


def _partition_from_prompt(prompt: str) -> List[int]:
    m = re.search(r"Marks: \d+\s*\(([\d\s\+]+)\)", prompt)
    if m:
        try:
            return [int(x.strip()) for x in m.group(1).split("+")]
        except Exception:
            pass
    return [_marks_from_prompt(prompt)]


def _make_valid_response(prompt: str, seed: int = 42) -> dict:
    """Produce a syntactically valid question response."""
    verb = _first_verb(prompt)
    marks = _marks_from_prompt(prompt)
    partition = _partition_from_prompt(prompt)
    is_multi = len(partition) > 1
    h = _hash_prompt(prompt)

    if is_multi:
        sub_parts = []
        labels = ["(i)", "(ii)", "(iii)", "(iv)"]
        for idx, p_marks in enumerate(partition):
            lbl = labels[idx] if idx < len(labels) else f"({idx+1})"
            sub_parts.append(f"{lbl} Explain key aspect {idx+1} ({p_marks} marks)")
        question_text = (
            f"{verb.capitalize()} the following aspects related to module {h}: "
            + " ".join(sub_parts)
            + "."
        )
        marking_scheme = [
            {"criterion": f"Sub-part {idx+1} analysis", "marks": p_marks}
            for idx, p_marks in enumerate(partition)
        ]
    else:
        question_text = (
            f"{verb.capitalize()} the core concepts related to module content "
            f"{h}. Provide a self-contained answer demonstrating understanding "
            f"of the topic and its application."
        )
        marking_scheme = [
            {"criterion": "Correct definition", "marks": max(1, marks // 3)},
            {"criterion": "Mechanism explained", "marks": max(1, marks // 3)},
            {"criterion": "Application", "marks": marks - 2 * (marks // 3)},
        ]

    return {
        "question_text": question_text,
        "solution": (
            f"A complete model answer covering the key points: "
            f"(1) definition and context, (2) mechanism or derivation, "
            f"(3) application example. Reference specific technical details "
            f"from module {h}."
        ),
        "marking_scheme": marking_scheme,
        "diagram_request": None,
        "references_image": False,
        "bloom_verb_used": verb,
    }


# -----------------------------------------------------------------------------
# Mock LLM caller
# -----------------------------------------------------------------------------


class MockLLMCaller:
    """
    Deterministic LLM mock.

    Behaviors:
        normal  — returns a valid question response
        fail    — raises RuntimeError
        bad_json — returns a non-JSON string
        circular — returns a question marked CIRCULAR (for testing eval)

    The `mode` parameter selects behavior for all calls.
    The `fail_on_nth` parameter can raise on specific call numbers.
    """

    def __init__(
        self,
        mode: str = "normal",
        fail_on_nth: Optional[List[int]] = None,
        seed: int = 42,
    ) -> None:
        self._mode = mode
        self._fail_on = set(fail_on_nth or [])
        self._seed = seed
        self._call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def call(
        self,
        prompt: str,
        schema: Optional[dict] = None,
        image_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Any:
        self._call_count += 1
        self.calls.append({
            "prompt": prompt,
            "schema": schema,
            "image_path": image_path,
            "seed": seed,
            "call_number": self._call_count,
        })

        if self._call_count in self._fail_on:
            raise RuntimeError(f"MockLLM: forced failure at call {self._call_count}")

        if self._mode == "fail":
            raise RuntimeError("MockLLM: mode=fail")
        if self._mode == "bad_json":
            return "this is not json at all"
        if self._mode == "circular":
            return {
                "question_text": (
                    "Recall the definition of Docker isolation, which uses "
                    "namespaces to provide process isolation."
                ),
                "solution": "Docker uses namespaces.",
                "marking_scheme": [{"criterion": "x", "marks": 10}],
                "diagram_request": None,
                "references_image": False,
            }

        return _make_valid_response(prompt, seed or self._seed)


# -----------------------------------------------------------------------------
# Mock VLM caller
# -----------------------------------------------------------------------------


class MockVLMCaller:
    """Deterministic VLM mock. Returns figure-aware responses."""

    def __init__(self, mode: str = "normal") -> None:
        self._mode = mode
        self._call_count = 0
        self.calls: List[Dict[str, Any]] = []

    def call(
        self,
        prompt: str,
        image_path: str,
        schema: Optional[dict] = None,
        seed: Optional[int] = None,
    ) -> Any:
        self._call_count += 1
        self.calls.append({
            "prompt": prompt,
            "image_path": image_path,
            "call_number": self._call_count,
        })

        if self._mode == "fail":
            raise RuntimeError("MockVLM: mode=fail")

        marks = _marks_from_prompt(prompt)
        verb = _first_verb(prompt)
        partition = _partition_from_prompt(prompt)
        is_multi = len(partition) > 1

        if is_multi:
            question_text = (
                f"{verb.capitalize()} the depicted system with the aid of the given figure: "
                f"(i) Analyze its primary architecture ({partition[0]} marks). "
                f"(ii) Detail the functional operational flow ({partition[1]} marks)."
            )
            marking_scheme = [
                {"criterion": "Architecture analysis", "marks": partition[0]},
                {"criterion": "Operational flow", "marks": partition[1]},
            ]
        else:
            question_text = (
                f"{verb.capitalize()} the structure and significance of the depicted system "
                f"with the aid of the given figure, analyzing its key components and relationships."
            )
            marking_scheme = [
                {"criterion": "Figure interpretation", "marks": marks // 2},
                {"criterion": "Analysis of components", "marks": marks - marks // 2},
            ]

        return {
            "question_text": question_text,
            "solution": (
                "The figure illustrates the system architecture. Key components "
                "include the input stage, processing layer, and output interface."
            ),
            "marking_scheme": marking_scheme,
            "diagram_request": {
                "diagram_type": "schematic",
                "description": "System architecture as shown",
            },
            "references_image": True,
            "bloom_verb_used": verb,
        }


# -----------------------------------------------------------------------------
# Mock API caller
# -----------------------------------------------------------------------------


class MockAPICaller:
    """
    Deterministic API mock for the Evaluation Agent.

    Behaviors:
        all_pass       — every question passes
        all_fail       — every question fails with UNGROUNDED
        per_slot       — dict {slot_id: verdict_dict} for mixed results
        unavailable    — raises APIUnavailable
    """

    def __init__(
        self,
        mode: str = "all_pass",
        per_slot: Optional[Dict[str, dict]] = None,
    ) -> None:
        self._mode = mode
        self._per_slot = per_slot or {}
        self.calls: List[Dict[str, Any]] = []
        self._last_provider = "mock_provider"

    @property
    def last_provider_name(self) -> Optional[str]:
        return self._last_provider

    def call(self, prompt: str, schema: Optional[dict] = None) -> Any:
        from core.generation.agents.evaluation_agent import APIUnavailable

        self.calls.append({"prompt": prompt, "schema": schema})

        if self._mode == "unavailable":
            raise APIUnavailable("MockAPI: mode=unavailable")

        # Extract slot_ids from the prompt
        slot_ids = re.findall(r"^slot_id: (\S+)$", prompt, re.MULTILINE)

        if self._mode == "all_pass":
            verdicts = [
                {"slot_id": sid, "verdict": "pass",
                 "reason_codes": [], "detail": "", "suggested_fix": None}
                for sid in slot_ids
            ]
        elif self._mode == "all_fail":
            verdicts = [
                {"slot_id": sid, "verdict": "fail",
                 "reason_codes": ["UNGROUNDED"],
                 "detail": "not present in evidence",
                 "suggested_fix": "Rephrase to align with evidence"}
                for sid in slot_ids
            ]
        elif self._mode == "per_slot":
            verdicts = []
            for sid in slot_ids:
                entry = self._per_slot.get(sid, {"verdict": "pass", "reason_codes": []})
                verdicts.append({
                    "slot_id": sid,
                    "verdict": entry.get("verdict", "pass"),
                    "reason_codes": entry.get("reason_codes", []),
                    "detail": entry.get("detail", ""),
                    "suggested_fix": entry.get("suggested_fix"),
                })
        else:
            verdicts = []

        return {"verdicts": verdicts}


# -----------------------------------------------------------------------------
# Mock SymPy verifier
# -----------------------------------------------------------------------------


class MockSympyVerifier:
    def __init__(self, result: bool = True) -> None:
        self._result = result
        self.calls: List[Any] = []

    def verify(self, question_text: str, solution_text: str) -> bool:
        self.calls.append((question_text, solution_text))
        return self._result
