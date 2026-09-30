"""
Unit tests for the Evaluation Agent.

Uses a SequenceAPI mock that returns pre-configured responses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from core.generation.agents import (
    AgentContext,
    APIUnavailable,
    AuditReport,
    EvaluationAgent,
    GeneratedQuestion,
    SlotVerdict,
    BLOOM_MISMATCH,
    MARKING_SCHEME_MISMATCH,
    MISSING_FIGURE_REFERENCE,
    MULTI_PART_STRUCTURE_MISMATCH,
    QUESTION_TOO_SHORT,
    SOLUTION_MISSING,
    FACTUAL_ERROR,
    UNGROUNDED,
    UNVERIFIED,
)


# -----------------------------------------------------------------------------
# Mocks
# -----------------------------------------------------------------------------


class SequenceAPI:
    """Returns pre-configured responses in order."""
    def __init__(self, responses: List[Any], provider_name: str = "mock_provider"):
        self._responses = list(responses)
        self._provider_name = provider_name
        self.calls: List[Dict[str, Any]] = []

    @property
    def last_provider_name(self) -> Optional[str]:
        return self._provider_name if self.calls else None

    def call(self, prompt: str, schema: Optional[dict] = None) -> Any:
        self.calls.append({"prompt": prompt, "schema": schema})
        if not self._responses:
            raise RuntimeError("SequenceAPI exhausted")
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class AlwaysUnavailableAPI:
    @property
    def last_provider_name(self) -> Optional[str]:
        return None

    def call(self, prompt, schema=None):
        raise APIUnavailable("no providers")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


def make_question(
    slot_id="module_1_Q1",
    question_text="Explain how Docker manages container isolation.",
    solution="Docker uses namespaces and cgroups to provide isolation.",
    marking_scheme=None,
    marks=10,
    partition=None,
    bloom="L2",
    co="CO1",
    image_path=None,
    references_image=False,
):
    return GeneratedQuestion(
        slot_id=slot_id,
        module_id="module_1",
        global_q_idx=1,
        question_text=question_text,
        solution=solution,
        marking_scheme=marking_scheme or [{"criterion": "x", "marks": marks}],
        marks=marks,
        partition=partition or [marks],
        bloom=bloom,
        co=co,
        image_path=image_path,
        references_image=references_image,
    )


def api_pass_response(slot_ids):
    return {"verdicts": [
        {"slot_id": sid, "verdict": "pass", "reason_codes": [],
         "detail": "", "suggested_fix": None}
        for sid in slot_ids
    ]}


def api_fail_response(slot_id, reason_code, detail="issue"):
    return {"verdicts": [
        {"slot_id": slot_id, "verdict": "fail", "reason_codes": [reason_code],
         "detail": detail, "suggested_fix": "fix it"}
    ]}


# -----------------------------------------------------------------------------
# Local evaluators via agent
# -----------------------------------------------------------------------------


def test_all_local_evaluators_pass_then_api_passes():
    api = SequenceAPI([api_pass_response(["Q1"])])
    agent = EvaluationAgent(api_caller=api)
    ctx = AgentContext(questions=[
        make_question(slot_id="Q1", bloom="L2",
                      question_text="Explain how Docker manages container isolation.")
    ])
    result = agent.run(ctx)
    assert result.success is True
    report = result.payload
    assert isinstance(report, AuditReport)
    assert report.api_used is True
    assert report.all_passed


def test_local_failure_short_circuits_api():
    api = SequenceAPI([])  # should not be called
    agent = EvaluationAgent(api_caller=api)
    # bloom mismatch: L2 but starts with "calculate"
    ctx = AgentContext(questions=[
        make_question(slot_id="Q1", bloom="L2",
                      question_text="Calculate the orbital velocity.")
    ])
    result = agent.run(ctx)
    report = result.payload
    assert len(report.failed_slots()) == 1
    assert report.failed_slots()[0].reason_code == BLOOM_MISMATCH
    assert len(api.calls) == 0


def test_local_failure_marking_scheme():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    q = make_question(slot_id="Q1", marks=10,
                      marking_scheme=[{"criterion": "x", "marks": 4}])
    result = agent.run(AgentContext(questions=[q]))
    report = result.payload
    assert report.failed_slots()[0].reason_code == MARKING_SCHEME_MISMATCH


def test_local_failure_figure_consistency():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    q = make_question(slot_id="Q1", image_path="/tmp/fig.png",
                      references_image=False)
    result = agent.run(AgentContext(questions=[q]))
    assert result.payload.failed_slots()[0].reason_code == MISSING_FIGURE_REFERENCE


def test_local_failure_question_too_short():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    q = make_question(slot_id="Q1", question_text="Explain.")
    result = agent.run(AgentContext(questions=[q]))
    assert result.payload.failed_slots()[0].reason_code == QUESTION_TOO_SHORT


def test_local_failure_solution_missing():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    q = make_question(slot_id="Q1", solution="")
    result = agent.run(AgentContext(questions=[q]))
    assert result.payload.failed_slots()[0].reason_code == SOLUTION_MISSING


def test_local_failure_multi_part_mismatch():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    q = make_question(
        slot_id="Q1", marks=10, partition=[6, 4],
        question_text="Explain Docker isolation without sub-parts.",
        marking_scheme=[
            {"criterion": "a", "marks": 6},
            {"criterion": "b", "marks": 4},
        ],
    )
    result = agent.run(AgentContext(questions=[q]))
    assert result.payload.failed_slots()[0].reason_code == MULTI_PART_STRUCTURE_MISMATCH


# -----------------------------------------------------------------------------
# API failures
# -----------------------------------------------------------------------------


def test_api_reports_failure():
    api = SequenceAPI([api_fail_response("Q1", FACTUAL_ERROR, "incorrect physics")])
    agent = EvaluationAgent(api_caller=api)
    ctx = AgentContext(questions=[make_question(slot_id="Q1")])
    report = agent.run(ctx).payload
    assert len(report.failed_slots()) == 1
    assert report.failed_slots()[0].reason_code == FACTUAL_ERROR
    assert report.failed_slots()[0].source == "api"


def test_api_ungrounded_failure():
    api = SequenceAPI([api_fail_response("Q1", UNGROUNDED)])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[make_question(slot_id="Q1")])).payload
    assert report.failed_slots()[0].reason_code == UNGROUNDED


# -----------------------------------------------------------------------------
# Degraded mode
# -----------------------------------------------------------------------------


def test_no_api_caller_marks_unverified():
    agent = EvaluationAgent(api_caller=None)
    ctx = AgentContext(questions=[make_question(slot_id="Q1")])
    report = agent.run(ctx).payload
    assert report.degraded_mode is True
    assert len(report.unverified_slots()) == 1
    assert report.unverified_slots()[0].reason_code == UNVERIFIED


def test_api_unavailable_degrades_gracefully():
    agent = EvaluationAgent(api_caller=AlwaysUnavailableAPI())
    ctx = AgentContext(questions=[make_question(slot_id="Q1")])
    result = agent.run(ctx)
    assert result.success is True
    report = result.payload
    assert report.degraded_mode is True
    assert len(report.unverified_slots()) == 1


def test_api_exception_degrades_gracefully():
    api = SequenceAPI([RuntimeError("network error")])
    agent = EvaluationAgent(api_caller=api)
    result = agent.run(AgentContext(questions=[make_question(slot_id="Q1")]))
    assert result.success is True
    assert result.payload.degraded_mode is True


def test_malformed_api_response_degrades():
    api = SequenceAPI(["not json at all"])
    agent = EvaluationAgent(api_caller=api)
    result = agent.run(AgentContext(questions=[make_question(slot_id="Q1")]))
    assert result.payload.degraded_mode is True


def test_api_omitting_slot_marks_unverified():
    api = SequenceAPI([{"verdicts": []}])  # API returns nothing
    agent = EvaluationAgent(api_caller=api)
    result = agent.run(AgentContext(questions=[make_question(slot_id="Q1")]))
    assert result.payload.degraded_mode is True
    assert result.payload.unverified_slots()[0].reason_code == UNVERIFIED


# -----------------------------------------------------------------------------
# Batching
# -----------------------------------------------------------------------------


def test_batching_splits_into_multiple_calls():
    # 25 questions with batch_size=10 → 3 batches
    questions = [make_question(slot_id=f"Q{i}") for i in range(25)]
    # Two full batches of pass responses
    api = SequenceAPI([
        api_pass_response([f"Q{i}" for i in range(10)]),
        api_pass_response([f"Q{i}" for i in range(10, 20)]),
        api_pass_response([f"Q{i}" for i in range(20, 25)]),
    ])
    agent = EvaluationAgent(api_caller=api, batch_size=10)
    report = agent.run(AgentContext(questions=questions)).payload
    assert len(api.calls) == 3
    assert report.api_calls == 3
    assert len(report.passed_slots()) == 25


def test_verdicts_ordered_by_question_order():
    # Mixed failure order; verdict list must match question order
    q1 = make_question(slot_id="Q1")
    q2 = make_question(slot_id="Q2", bloom="L2",
                       question_text="Calculate the orbital velocity.")
    q3 = make_question(slot_id="Q3")
    api = SequenceAPI([api_pass_response(["Q1", "Q3"])])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[q1, q2, q3])).payload
    assert [v.slot_id for v in report.verdicts] == ["Q1", "Q2", "Q3"]


# -----------------------------------------------------------------------------
# Empty input
# -----------------------------------------------------------------------------


def test_empty_questions_returns_empty_report():
    api = SequenceAPI([])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[])).payload
    assert report.verdicts == []
    assert report.api_used is False


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_invalid_batch_size_rejected():
    with pytest.raises(ValueError):
        EvaluationAgent(api_caller=SequenceAPI([]), batch_size=0)


# -----------------------------------------------------------------------------
# API JSON parsing variants
# -----------------------------------------------------------------------------


def test_api_response_with_code_fences_parsed():
    import json
    fenced = "```json\n" + json.dumps(api_pass_response(["Q1"])) + "\n```"
    api = SequenceAPI([fenced])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[make_question(slot_id="Q1")])).payload
    assert report.all_passed


def test_api_response_with_surrounding_prose_parsed():
    import json
    noisy = "Here are the verdicts:\n" + json.dumps(api_pass_response(["Q1"])) + "\nDone."
    api = SequenceAPI([noisy])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[make_question(slot_id="Q1")])).payload
    assert report.all_passed


def test_api_unknown_reason_code_defaults():
    """Unknown codes are mapped to a safe default."""
    api = SequenceAPI([{
        "verdicts": [{
            "slot_id": "Q1", "verdict": "fail",
            "reason_codes": ["SOME_NEW_CODE"],
            "detail": "unknown issue", "suggested_fix": None,
        }]
    }])
    agent = EvaluationAgent(api_caller=api)
    report = agent.run(AgentContext(questions=[make_question(slot_id="Q1")])).payload
    assert len(report.failed_slots()) == 1
    assert report.failed_slots()[0].reason_code == UNGROUNDED


# -----------------------------------------------------------------------------
# SlotVerdict / AuditReport contracts
# -----------------------------------------------------------------------------


def test_slot_verdict_requires_reason_for_fail():
    with pytest.raises(ValueError):
        SlotVerdict(slot_id="Q1", verdict="fail", source="local")  # missing reason


def test_slot_verdict_rejects_unknown_verdict():
    with pytest.raises(ValueError):
        SlotVerdict(slot_id="Q1", verdict="maybe", source="local")


def test_audit_report_helpers():
    v1 = SlotVerdict(slot_id="Q1", verdict="pass", source="api")
    v2 = SlotVerdict(slot_id="Q2", verdict="fail", source="local",
                     reason_code=BLOOM_MISMATCH)
    v3 = SlotVerdict(slot_id="Q3", verdict="unverified", source="degraded",
                     reason_code=UNVERIFIED)
    r = AuditReport(verdicts=[v1, v2, v3])
    assert [v.slot_id for v in r.failed_slots()] == ["Q2"]
    assert [v.slot_id for v in r.passed_slots()] == ["Q1"]
    assert [v.slot_id for v in r.unverified_slots()] == ["Q3"]
    assert r.verdict_for("Q2") is v2
    assert r.all_passed is False
