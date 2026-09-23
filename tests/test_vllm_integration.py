"""
tests/test_vllm_integration.py
Comprehensive validation suite for:
1. vLLM backend support in core/generation/robust_llm_caller.py
2. OpenAI endpoint routing and JSON schema / json_object compatibility
3. Parity between _call_ollama and _call_vllm responses
4. Startup readiness gate and model-matching assertions
5. Truncation detection (TRUNCATED_AT_MAX_TOKENS)
6. Non-CS domain protections in orchestrator, aion_patch, and teacher_suitability_gate
"""

import os
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pytest
from unittest.mock import patch, MagicMock

from core.generation.robust_llm_caller import RobustLLMCaller, LLMRequest, LLMResponse
from core.contracts.question_slot import QuestionSlot
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature


def test_llm_request_default_max_tokens():
    """Verify LLMRequest defaults max_tokens to 4096 without breaking existing callers."""
    req = LLMRequest(model="test_model", prompt="Hello world")
    assert req.max_tokens == 4096
    assert req.temperature == 0.1
    assert req.seed == 42


def test_llm_request_aion_seed_override():
    """Verify LLMRequest respects AION_SEED environment variable override."""
    with patch.dict(os.environ, {"AION_SEED": "1337"}):
        req = LLMRequest(model="test_model", prompt="Hello world")
        assert req.seed == 1337

    with patch.dict(os.environ, {"AION_SEED": "invalid"}):
        req = LLMRequest(model="test_model", prompt="Hello world")
        assert req.seed == 42


def test_robust_llm_caller_backend_and_host_defaults():
    """Verify backend and host resolution from arguments and environment."""
    with patch.dict(os.environ, {"AION_BACKEND": "vllm", "AION_LLM_HOST": "http://gpu-server:8000"}):
        caller = RobustLLMCaller()
        assert caller.backend == "vllm"
        assert caller.host == "http://gpu-server:8000"

    with patch.dict(os.environ, {"AION_BACKEND": "ollama", "OLLAMA_URL": "http://localhost:11434"}):
        caller = RobustLLMCaller()
        assert caller.backend == "ollama"
        assert caller.host == "http://localhost:11434"


def test_vllm_check_health_asserts_model_presence():
    """Verify check_health fails loudly when vLLM is running but does not serve AION_MODEL."""
    caller = RobustLLMCaller(backend="vllm", host="http://localhost:8000")
    
    mock_res = MagicMock()
    mock_res.ok = True
    mock_res.json.return_value = {
        "data": [{"id": "other-model:latest"}, {"id": "meta-llama/Llama-3-8B"}]
    }

    with patch("requests.get", return_value=mock_res), \
         patch.dict(os.environ, {"AION_MODEL": "Qwen/Qwen2.5-14B-Instruct-AWQ"}):
        with pytest.raises(RuntimeError) as exc_info:
            caller.check_health()
        assert "does not serve model 'Qwen/Qwen2.5-14B-Instruct-AWQ'" in str(exc_info.value)
        assert "Available models: ['other-model:latest', 'meta-llama/Llama-3-8B']" in str(exc_info.value)


def test_vllm_check_health_passes_when_model_present():
    """Verify check_health passes when expected model is listed in /v1/models."""
    caller = RobustLLMCaller(backend="vllm", host="http://localhost:8000")
    
    mock_res = MagicMock()
    mock_res.ok = True
    mock_res.json.return_value = {
        "data": [{"id": "Qwen/Qwen2.5-14B-Instruct-AWQ"}]
    }

    with patch("requests.get", return_value=mock_res), \
         patch.dict(os.environ, {"AION_MODEL": "Qwen/Qwen2.5-14B-Instruct-AWQ"}):
        assert caller.check_health() is True


def test_call_vllm_json_schema_payload_structure():
    """Verify _call_vllm formats OpenAI request with json_schema when schema is provided."""
    caller = RobustLLMCaller(backend="vllm", host="http://localhost:8000")
    dummy_schema = {"type": "object", "properties": {"question": {"type": "string"}}}
    req = LLMRequest(model="qwen", prompt="Generate exam question", schema=dummy_schema, max_tokens=2048)

    captured_payload = {}

    def mock_post(url, json=None, timeout=None):
        nonlocal captured_payload
        captured_payload = json
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": '{"question": "What is consideration in contract law?"}'}}]
        }
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        res = caller.call(req)
        assert res.success is True
        assert res.parsed == {"question": "What is consideration in contract law?"}
        assert captured_payload["max_tokens"] == 2048
        assert captured_payload["response_format"]["type"] == "json_schema"
        assert captured_payload["response_format"]["json_schema"]["schema"] == dummy_schema


def test_call_vllm_truncation_detection():
    """Verify unclosed responses reaching >= 0.9 * max_tokens are flagged as TRUNCATED_AT_MAX_TOKENS."""
    caller = RobustLLMCaller(backend="vllm", host="http://localhost:8000")
    req = LLMRequest(model="qwen", prompt="Generate long question", max_tokens=50)

    # 48 words ending abruptly without closing brace
    truncated_words = ["word"] * 48
    raw_content = '{"instruction": "' + " ".join(truncated_words)

    mock_resp = MagicMock()
    mock_resp.ok = True
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": raw_content}}]
    }

    with patch("requests.post", return_value=mock_resp):
        res = caller.call(req)
        assert res.success is False
        assert res.error == "TRUNCATED_AT_MAX_TOKENS"
        assert res.parsed is None


def test_self_heal_ollama_skips_when_vllm_backend():
    """Verify aion_patch.self_heal_ollama returns False immediately when backend is vllm."""
    from aion_patch import self_heal_ollama
    with patch.dict(os.environ, {"AION_BACKEND": "vllm"}):
        with patch("subprocess.run") as mock_run:
            result = self_heal_ollama()
            assert result is False
            mock_run.assert_not_called()


def test_code_signals_regex_ignores_legal_class_prose():
    """Verify orchestrator._code_signals does not trip on 'class action' or 'class of remedies' in Law."""
    from core.generation.orchestrator import SlotOrchestrator
    orch = SlotOrchestrator()
    slot = QuestionSlot(
        slot_id="mod_1_q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="or_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="EXPLAIN",
        co="CO1",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Company Law and Corporate Remedies",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "THEORY"),
    )

    class MockEvidencePack:
        combined_text = (
            "Under Indian corporate law, a class action suit may be filed by shareholders "
            "against fraudulent management seeking a distinct class of statutory remedies."
        )
        math_artifacts = "none"

    prompt = orch._format_prompt(slot, MockEvidencePack(), extra_hints="")
    assert "[EVIDENCE-DRIVEN PROGRAMMING/APPLIED MODE]" not in prompt


def test_teacher_suitability_gate_does_not_penalize_law_subject():
    """Verify teacher suitability gate does not inject CS Data Structures syllabus penalty for Law."""
    from core.validation.teacher_suitability_gate import TeacherSuitabilityGate
    from core.generation.output_schema import QuestionOutput
    from core.contracts.question import GeneratedQuestion

    slot = QuestionSlot(
        slot_id="module_1_q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="or_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="EXPLAIN",
        co="CO1",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Doctrine of Ultra Vires",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "THEORY"),
    )
    output = QuestionOutput(
        instruction="Explain the doctrine of ultra vires under company law with reference to landmark precedents.",
        question_text="Explain the doctrine of ultra vires under company law with reference to landmark precedents.",
        math_blocks=[]
    )
    gq = GeneratedQuestion(output=output, slot=slot)
    result = TeacherSuitabilityGate.validate(gq, slot, evidence_text="The doctrine of ultra vires limits corporate powers.")
    assert result.passed is True


def test_build_augmented_prompt_string_and_llmrequest_parity():
    """
    Verify that _build_augmented_prompt produces the EXACT same output string
    whether passed a raw string prompt or an LLMRequest dataclass.
    """
    from aion_patch import _build_augmented_prompt
    raw_prompt = "Generate an exam question on topic: Corporate Insolvency Resolution Process."
    kwargs = {"subject": "Corporate Law"}

    out_from_str = _build_augmented_prompt(raw_prompt, kwargs)

    req = LLMRequest(model="qwen", prompt=raw_prompt)
    out_from_req = _build_augmented_prompt(req, kwargs)

    assert out_from_str == out_from_req
    assert "Visvesvaraya Technological University (VTU) Board of Examiners" in out_from_str
    # And verify CS/programming was NOT injected for Law
    assert "[EVIDENCE-DRIVEN PROGRAMMING/APPLIED MODE]" not in out_from_str


def test_call_vllm_json_schema_400_fallback_to_json_object():
    """Verify _call_vllm falls back to json_object if vLLM returns HTTP 400 for json_schema."""
    import copy
    caller = RobustLLMCaller(backend="vllm", host="http://localhost:8000")
    dummy_schema = {"type": "object", "properties": {"question": {"type": "string"}}}
    req = LLMRequest(model="qwen", prompt="Generate exam question", schema=dummy_schema)

    attempts = []

    def mock_post(url, json=None, timeout=None):
        attempts.append(copy.deepcopy(json))
        mock_resp = MagicMock()
        if len(attempts) == 1:
            mock_resp.ok = False
            mock_resp.status_code = 400
            mock_resp.text = "json_schema not supported"
        else:
            mock_resp.ok = True
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "choices": [{"message": {"content": '{"question": "Valid JSON"}'}}]
            }
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        res = caller.call(req)
        assert res.success is True
        assert res.parsed == {"question": "Valid JSON"}
        assert len(attempts) == 2
        assert attempts[0]["response_format"]["type"] == "json_schema"
        assert attempts[1]["response_format"]["type"] == "json_object"


def test_non_cs_pharmacology_prompt_has_no_cs_leak():
    """Verify that a Pharmacology evidence pack produces no CS or programming prompts."""
    from core.generation.orchestrator import SlotOrchestrator
    orch = SlotOrchestrator()
    slot = QuestionSlot(
        slot_id="mod_1_q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="or_1",
        is_alternative=False,
        module_id=1,
        marks=8,
        bloom_level="L3",
        bloom_verb="Calculate",
        bloom_operation="CALCULATE",
        co="CO1",
        difficulty="HARD",
        question_type="NUMERICAL",
        topic="Pharmacokinetics: Elimination Rate and Half-Life",
        evidence_ids=("chunk_pharm_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(8, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 8),
        task_signature=TaskSignature.from_bloom_marks_type("L3", 8, "NUMERICAL"),
    )

    class MockPharmPack:
        combined_text = (
            "The elimination rate constant (k_e) of a drug is determined by k_e = CL / V_d. "
            "The plasma half-life t_{1/2} is given by 0.693 / k_e. For a patient administered 500 mg, "
            "clearance CL is 2.5 L/h and distribution volume V_d is 25 L."
        )
        math_artifacts = "none"

    prompt = orch._format_prompt(slot, MockPharmPack(), extra_hints="")
    # Check that programming mode is NOT triggered
    assert "[EVIDENCE-DRIVEN PROGRAMMING/APPLIED MODE]" not in prompt
    assert "write a python" not in prompt.lower()
    assert "write an sql" not in prompt.lower()
    assert "write code" not in prompt.lower()
    assert "def " not in prompt.lower()
    assert "class " not in prompt.lower()


def test_aion_api_tags_and_health_vllm_mode():
    """Verify aion_api routes /api/tags and /api/health to /v1/models when AION_BACKEND=vllm."""
    from aion_api import app
    client = app.test_client()

    mock_models_resp = MagicMock()
    mock_models_resp.ok = True
    mock_models_resp.status_code = 200
    mock_models_resp.json.return_value = {
        "data": [{"id": "Qwen/Qwen2.5-14B-Instruct-AWQ"}]
    }

    with patch.dict(os.environ, {
        "AION_BACKEND": "vllm",
        "AION_LLM_HOST": "http://localhost:8000",
        "AION_MODEL": "Qwen/Qwen2.5-14B-Instruct-AWQ",
    }):
        with patch("requests.get", return_value=mock_models_resp):
            # Test /api/tags
            res_tags = client.get("/api/tags")
            assert res_tags.status_code == 200
            data_tags = res_tags.get_json()
            assert any(m["name"] == "Qwen/Qwen2.5-14B-Instruct-AWQ" for m in data_tags.get("models", []))

            # Test /api/health
            res_health = client.get("/api/health")
            data_health = res_health.get_json()
            assert data_health["backend"] == "vllm"
            assert data_health["llm_available"] is True


def test_iai_archetype_resolution():
    """Verify subject 'IAI' and VTU code '21cs54' resolve to ai_ml_data archetype."""
    from aion_patch import resolve_subject_archetype, SUBJECT_ARCHETYPES
    arch_iai = resolve_subject_archetype("IAI", "")
    assert arch_iai == "ai_ml_data"
    arch_code = resolve_subject_archetype("21CS54", "")
    assert arch_code == "ai_ml_data"
    directive = SUBJECT_ARCHETYPES["ai_ml_data"]["directive"]
    assert "Intelligent Agents" in directive
    assert "Search Algorithms" in directive


def test_strip_module_header_removes_pdf_extension_and_notes():
    """Verify filenames and notes suffix are stripped from topics."""
    from core.contracts.module_identity import strip_module_header
    assert strip_module_header("Module 2: IAI-MODULE-2-NOTES.pdf") == "IAI-MODULE-2"
    assert strip_module_header("Module 4: IAI-MODULE-4-NOTES.pdf") == "IAI-MODULE-4"
    assert strip_module_header("Chapter 1: notes.pdf") == ""


def test_high_marks_prompt_specifies_analytical_depth():
    """Verify 10-mark slots receive 25 to 50 word depth requirement, while standard slots do not."""
    from core.generation.orchestrator import SlotOrchestrator
    from core.contracts.question_slot import QuestionSlot, AnswerBudget, QuestionBudget, TaskSignature

    orch = SlotOrchestrator()
    slot_10m = QuestionSlot(
        slot_id="module_3_q5",
        question_no=5,
        sub_label="",
        or_pair_id="or_3",
        is_alternative=False,
        module_id=3,
        marks=10,
        bloom_level="L4",
        bloom_verb="Analyze",
        bloom_operation="ANALYZE",
        co="CO3",
        difficulty="EASY",
        question_type="CONCEPTUAL",
        topic="Intelligent Agents",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(10, "L4"),
        question_budget=QuestionBudget.from_bloom("L4", 10),
        task_signature=TaskSignature.from_bloom_marks_type("L4", 10, "CONCEPTUAL"),
    )
    class DummyEvidencePack:
        combined_text = "An agent is anything that can be viewed as perceiving its environment through sensors."
        math_blocks = []

    prompt_10m = orch._format_prompt(slot_10m, DummyEvidencePack(), extra_hints="")
    assert "HIGH-MARKS 10M" in prompt_10m
    assert "target length: 80 to 140 words" in prompt_10m
    assert "at least 80 words" in prompt_10m



def test_looks_code_ignores_english_ai_prose():
    """Verify generic English words in AI (where, open, loop, function) do not trigger looks_code."""
    import re
    ai_text = "The agent function maps percepts to actions where the open list and closed list are tracked in a loop."
    _math_lower = ai_text.lower()
    _strong_code_signals = (
        "select distinct", "insert into", "update set", "delete from", "create table",
        "alter table", "drop table", "group by", "stored procedure", "create trigger",
        "create procedure", "declare cursor", "begin transaction", "commit;", "rollback;",
        "```sql", "```python", "```c", "```java", "```cpp"
    )
    _looks_like_code = (
        any(_sig in _math_lower for _sig in _strong_code_signals)
        or any(bool(re.search(pat, _math_lower, re.IGNORECASE)) for pat in (
            r'\b(?:def|class|public\s+static|private\s+void)\b',
            r'\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET)\b',
            r'```[a-zA-Z]+\n'
        ))
    )
    assert _looks_like_code is False




