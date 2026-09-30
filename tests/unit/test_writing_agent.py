"""
Unit tests for the Writing Agent.

Uses a SequenceLLM mock that returns pre-configured responses.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from core.generation.agents import (
    AgentContext,
    GeneratedQuestion,
    QuestionPlan,
    WritingAgent,
    WritingAgentOutput,
)


# -----------------------------------------------------------------------------
# Mocks
# -----------------------------------------------------------------------------


class SequenceLLM:
    """Returns pre-configured responses in order. Records calls."""
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt, schema=None, image_path=None, seed=None):
        self.calls.append({
            "prompt": prompt,
            "schema": schema,
            "image_path": image_path,
            "seed": seed,
        })
        if not self._responses:
            raise RuntimeError("SequenceLLM exhausted")
        return self._responses.pop(0)


class SequenceVLM:
    def __init__(self, responses: List[Any]):
        self._responses = list(responses)
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt, image_path, schema=None, seed=None):
        self.calls.append({
            "prompt": prompt,
            "image_path": image_path,
            "schema": schema,
            "seed": seed,
        })
        if not self._responses:
            raise RuntimeError("SequenceVLM exhausted")
        return self._responses.pop(0)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@dataclass
class StubFigure:
    id: str
    image_path: str
    caption: str = "Keplerian orbit diagram"
    page: int = 5


def make_plan(
    slot_id="module_1_Q1",
    module_id="module_1",
    module_idx=1,
    global_q_idx=1,
    partition=None,
    bloom="L2",
    co="CO1",
    topic="Docker",
    visual_required=False,
    figure=None,
    neighborhood=None,
    evidence=None,
):
    return QuestionPlan(
        slot_id=slot_id,
        module_id=module_id,
        module_idx=module_idx,
        slot_idx=0,
        global_q_idx=global_q_idx,
        partition=partition or [6, 4],
        total_marks=sum(partition or [6, 4]),
        bloom=bloom,
        co=co,
        topic=topic,
        concept_neighborhood=neighborhood or [],
        evidence_blocks=evidence or [],
        visual_required=visual_required,
        figure=figure,
    )


def valid_response():
    return {
        "question_text": "Explain how Docker manages container isolation.",
        "solution": "Docker uses namespaces and cgroups...",
        "marking_scheme": [
            {"criterion": "Mentions namespaces", "marks": 3},
            {"criterion": "Mentions cgroups", "marks": 3},
        ],
        "diagram_request": None,
        "references_image": False,
        "bloom_verb_used": "explain",
    }


def valid_response_with_image():
    return {
        "question_text": "With the aid of the given diagram, explain orbital mechanics.",
        "solution": "The figure shows Keplerian elements...",
        "marking_scheme": [{"criterion": "Correct interpretation", "marks": 10}],
        "diagram_request": {"diagram_type": "schematic", "description": "Orbit"},
        "references_image": True,
        "bloom_verb_used": "explain",
    }


# -----------------------------------------------------------------------------
# Happy path
# -----------------------------------------------------------------------------


def test_single_plan_success():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    ctx = AgentContext(plans=[make_plan()])
    result = agent.run(ctx)
    assert result.success is True
    assert isinstance(result.payload, WritingAgentOutput)
    assert len(result.payload.questions) == 1
    assert result.payload.all_succeeded
    q = result.payload.questions[0]
    assert q.slot_id == "module_1_Q1"
    assert q.generation_source == "local_llm"
    assert q.attempts == 1
    assert q.first_word == "explain"


def test_multiple_plans_in_order():
    llm = SequenceLLM([
        valid_response(),
        valid_response(),
        valid_response(),
    ])
    agent = WritingAgent(llm_caller=llm)
    plans = [
        make_plan(slot_id="module_1_Q1", global_q_idx=1),
        make_plan(slot_id="module_1_Q2", global_q_idx=2),
        make_plan(slot_id="module_1_Q3", global_q_idx=3),
    ]
    ctx = AgentContext(plans=plans)
    result = agent.run(ctx)
    assert len(result.payload.questions) == 3
    assert [q.slot_id for q in result.payload.questions] == [
        "module_1_Q1", "module_1_Q2", "module_1_Q3"
    ]


def test_empty_plans_returns_empty_output():
    llm = SequenceLLM([])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[]))
    assert result.success is True
    assert result.payload.questions == []
    assert result.payload.failures == []


# -----------------------------------------------------------------------------
# Response parsing
# -----------------------------------------------------------------------------


def test_json_string_response_parsed():
    import json
    llm = SequenceLLM([json.dumps(valid_response())])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert result.payload.questions[0].question_text.startswith("Explain")


def test_json_with_code_fences_parsed():
    import json
    fenced = f"```json\n{json.dumps(valid_response())}\n```"
    llm = SequenceLLM([fenced])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert len(result.payload.questions) == 1


def test_json_with_surrounding_prose_parsed():
    import json
    noisy = f"Here is the question:\n{json.dumps(valid_response())}\nHope this helps!"
    llm = SequenceLLM([noisy])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert len(result.payload.questions) == 1


# -----------------------------------------------------------------------------
# Retry on invalid output
# -----------------------------------------------------------------------------


def test_retry_on_invalid_json():
    llm = SequenceLLM([
        "not json at all",
        valid_response(),
    ])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert result.payload.all_succeeded
    q = result.payload.questions[0]
    assert q.attempts == 2
    # Second call should have strict_json directive
    assert "previous response" in llm.calls[1]["prompt"]


def test_retry_on_missing_keys():
    llm = SequenceLLM([
        {"question_text": "..."},  # missing solution + marking_scheme
        valid_response(),
    ])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert result.payload.all_succeeded
    assert result.payload.questions[0].attempts == 2


def test_failure_after_max_attempts():
    llm = SequenceLLM(["bad", "bad", "bad"])
    agent = WritingAgent(llm_caller=llm)
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert len(result.payload.questions) == 0
    assert len(result.payload.failures) == 1
    assert result.payload.failures[0].failure_code == "GENERATION_FAILED"


def test_caller_exception_retries():
    class FlakyLLM:
        def __init__(self):
            self.n = 0
        def call(self, *args, **kwargs):
            self.n += 1
            if self.n < 3:
                raise RuntimeError("temporary")
            return valid_response()

    agent = WritingAgent(llm_caller=FlakyLLM())
    result = agent.run(AgentContext(plans=[make_plan()]))
    assert result.payload.all_succeeded
    assert result.payload.questions[0].attempts == 3


# -----------------------------------------------------------------------------
# VLM routing
# -----------------------------------------------------------------------------


def test_vlm_routing_when_visual_required():
    llm = SequenceLLM([])  # should not be called
    vlm = SequenceVLM([valid_response_with_image()])
    agent = WritingAgent(llm_caller=llm, vlm_caller=vlm)
    plan = make_plan(
        bloom="L4",
        partition=[10],
        visual_required=True,
        figure=StubFigure(id="fig1", image_path="/tmp/fig1.png"),
    )
    result = agent.run(AgentContext(plans=[plan]))
    assert result.payload.all_succeeded
    assert result.payload.questions[0].generation_source == "local_vlm"
    assert len(vlm.calls) == 1
    assert vlm.calls[0]["image_path"] == "/tmp/fig1.png"


def test_llm_when_visual_required_but_no_vlm():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm, vlm_caller=None)
    plan = make_plan(
        bloom="L4",
        partition=[10],
        visual_required=True,
        figure=StubFigure(id="fig1", image_path="/tmp/fig1.png"),
    )
    result = agent.run(AgentContext(plans=[plan]))
    assert result.payload.questions[0].generation_source == "local_llm"


def test_llm_when_not_visual_required():
    llm = SequenceLLM([valid_response()])
    vlm = SequenceVLM([])
    agent = WritingAgent(llm_caller=llm, vlm_caller=vlm)
    plan = make_plan(visual_required=False)
    result = agent.run(AgentContext(plans=[plan]))
    assert result.payload.questions[0].generation_source == "local_llm"
    assert len(vlm.calls) == 0


# -----------------------------------------------------------------------------
# Image reference detection
# -----------------------------------------------------------------------------


def test_references_image_detected():
    vlm = SequenceVLM([valid_response_with_image()])
    agent = WritingAgent(llm_caller=SequenceLLM([]), vlm_caller=vlm)
    plan = make_plan(
        partition=[10],
        visual_required=True,
        figure=StubFigure(id="fig1", image_path="/tmp/fig1.png"),
    )
    result = agent.run(AgentContext(plans=[plan]))
    assert result.payload.questions[0].references_image is True


def test_references_image_absent_without_visual():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(visual_required=False)
    result = agent.run(AgentContext(plans=[plan]))
    assert result.payload.questions[0].references_image is False


# -----------------------------------------------------------------------------
# Prompt construction
# -----------------------------------------------------------------------------


def test_prompt_contains_slot_contract():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan()
    agent.run(AgentContext(plans=[plan]))
    prompt = llm.calls[0]["prompt"]
    assert "[SLOT CONTRACT]" in prompt
    assert "Marks: 10" in prompt
    assert "Bloom level: L2" in prompt
    assert "Course Outcome: CO1" in prompt


def test_prompt_contains_topic():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(topic="Docker networking")
    agent.run(AgentContext(plans=[plan]))
    assert "Docker networking" in llm.calls[0]["prompt"]


def test_prompt_includes_bloom_verbs():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(bloom="L3")
    agent.run(AgentContext(plans=[plan]))
    prompt = llm.calls[0]["prompt"]
    assert "apply" in prompt or "calculate" in prompt or "solve" in prompt


def test_prompt_includes_multi_part_directive():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(partition=[6, 4])
    agent.run(AgentContext(plans=[plan]))
    assert "part 1: 6 marks" in llm.calls[0]["prompt"]


def test_prompt_excludes_multi_part_directive_for_single():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(partition=[10])
    agent.run(AgentContext(plans=[plan]))
    assert "part 1" not in llm.calls[0]["prompt"]


def test_prompt_includes_visual_directive_when_required():
    vlm = SequenceVLM([valid_response_with_image()])
    agent = WritingAgent(llm_caller=SequenceLLM([]), vlm_caller=vlm)
    plan = make_plan(
        partition=[10],
        visual_required=True,
        figure=StubFigure(id="fig1", image_path="/tmp/fig1.png",
                          caption="Keplerian orbit"),
    )
    agent.run(AgentContext(plans=[plan]))
    prompt = vlm.calls[0]["prompt"]
    assert "[VISUAL DIRECTIVE" in prompt
    assert "Keplerian orbit" in prompt


def test_prompt_excludes_visual_directive_when_not_required():
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(visual_required=False)
    agent.run(AgentContext(plans=[plan]))
    assert "[VISUAL DIRECTIVE" not in llm.calls[0]["prompt"]


def test_kg_neighborhood_in_prompt():
    @dataclass
    class FakeTriple:
        subject: str
        predicate: str
        object: str
    llm = SequenceLLM([valid_response()])
    agent = WritingAgent(llm_caller=llm)
    plan = make_plan(neighborhood=[
        FakeTriple("Docker", "is_a", "platform"),
        FakeTriple("Docker", "requires", "runtime"),
    ])
    agent.run(AgentContext(plans=[plan]))
    prompt = llm.calls[0]["prompt"]
    assert "KNOWLEDGE GRAPH" in prompt
    assert "Docker --is_a--> platform" in prompt


# -----------------------------------------------------------------------------
# Determinism
# -----------------------------------------------------------------------------


def test_same_inputs_same_output():
    def run():
        llm = SequenceLLM([valid_response()])
        agent = WritingAgent(llm_caller=llm)
        return agent.run(AgentContext(plans=[make_plan()])).payload

    r1 = run()
    r2 = run()
    assert r1.questions[0].question_text == r2.questions[0].question_text
    assert r1.questions[0].slot_id == r2.questions[0].slot_id


# -----------------------------------------------------------------------------
# Mixed batch
# -----------------------------------------------------------------------------


def test_partial_failure_in_batch():
    llm = SequenceLLM([
        valid_response(),
        "bad json", "bad json", "bad json",
        valid_response(),
    ])
    agent = WritingAgent(llm_caller=llm)
    plans = [
        make_plan(slot_id="Q1", global_q_idx=1),
        make_plan(slot_id="Q2", global_q_idx=2),
        make_plan(slot_id="Q3", global_q_idx=3),
    ]
    result = agent.run(AgentContext(plans=plans))
    assert len(result.payload.questions) == 2
    assert len(result.payload.failures) == 1
    assert result.payload.failures[0].slot_id == "Q2"


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_missing_llm_caller_rejected():
    with pytest.raises(ValueError):
        WritingAgent(llm_caller=None)


def test_invalid_max_attempts_rejected():
    with pytest.raises(ValueError):
        WritingAgent(llm_caller=SequenceLLM([]), max_attempts=0)


# -----------------------------------------------------------------------------
# GeneratedQuestion properties
# -----------------------------------------------------------------------------


def test_first_word_strips_punctuation():
    gq = GeneratedQuestion(
        slot_id="Q1", module_id="m1", global_q_idx=1,
        question_text="Explain, in detail, how Docker works.",
        solution="...", marking_scheme=[],
    )
    assert gq.first_word == "explain"


def test_is_multi_part():
    gq_multi = GeneratedQuestion(
        slot_id="Q1", module_id="m1", global_q_idx=1,
        question_text="...", solution="...", marking_scheme=[],
        partition=[6, 4],
    )
    assert gq_multi.is_multi_part is True
    gq_single = GeneratedQuestion(
        slot_id="Q2", module_id="m1", global_q_idx=2,
        question_text="...", solution="...", marking_scheme=[],
        partition=[10],
    )
    assert gq_single.is_multi_part is False
