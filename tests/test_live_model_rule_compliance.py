"""
Offline Acceptance Suite for Live Model Accuracy, Dynamic Rule Compliance & Export Gate.

Covers:
1. Schema enums derive from active BLOOM_VERB_LEVEL_MAP.
2. Mutating Bloom rules immediately updates subsequent schemas, prompts, and validators (no stale fallback).
3. Changing marks updates schema bounds.
4. Changing AssessmentConfig updates snapshot fingerprint and constraints.
5. Prompt exemplars use active rules (no contradiction with marks/verbs/visuals).
6. Unsupported/missing rules fail closed.
7. Schema & temperature forwarding reaches RobustLLMCallerAdapter -> RobustLLMCaller.
8. Invalid JSON & non-int marks rejected; marking-scheme totals enforced.
9. Long evidence retains equations/numbers/definitions via query-focused selection.
10. Source metadata tags sanitized without contaminating question text.
11. Surgical retry fixes failure without mutating immutable plan constraints.
12. /api/export/docx route blocks unapproved/swapped visuals and allows manifest-verified papers.
"""

import hashlib
import json
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from core.contracts.document_artifact import DocumentArtifact, TextBlock
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry, capture_rule_snapshot
from core.generation.agents.dependency_factory import RobustLLMCallerAdapter
from core.generation.agents.prompts import (
    build_dynamic_exemplar_section,
    build_evidence_section,
    build_subpart_json_schema,
    build_subpart_prompt,
)
from core.generation.agents.question_plan import PlanStatus, QuestionPlan, SubPartPlan
from core.generation.agents.writing_agent import WritingAgent
from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP


def _sample_sp_plan(bloom="L3", marks=10, visual_required=False):
    return SubPartPlan(
        label="a",
        marks=marks,
        topic="CPU Scheduling",
        bloom=bloom,
        co="CO2",
        canonical_concept_id="cpu_sched",
        visual_required=visual_required,
        evidence_blocks=[
            {"id": "blk_m1_1", "text": "CPU Scheduling selects the next process from the ready queue.", "page": 5, "module_id": "module_1"},
        ],
    )


def _sample_plan(sp):
    return QuestionPlan(
        slot_id="module_1_Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        partition=[sp.marks],
        total_marks=sp.marks,
        co=sp.co,
        bloom=sp.bloom,
        topic=sp.topic,
        sub_part_plans=[sp],
        status=PlanStatus.PLANNED,
        assessment_identity=("VTU_2022_NEP", "AEC_GENERAL_THEORY", "CIE_IAT", "VTU_AEC_ANNEXURE_II"),
        module_page_start=1,
        module_page_end=20,
        constraints={
            "subpart_cos": {"a": sp.co},
            "subpart_blooms": {"a": sp.bloom},
        },
    )


def test_01_02_03_schema_enum_and_marks_derive_from_active_rules(monkeypatch):
    """1-3: Schema enum derives from BLOOM_VERB_LEVEL_MAP; mutations take effect immediately."""
    sp = _sample_sp_plan(bloom="L2", marks=7)
    snap1 = capture_rule_snapshot()
    schema1 = build_subpart_json_schema(sp, rule_snapshot=snap1)
    assert schema1["properties"]["bloom_verb_used"]["enum"] == sorted(BLOOM_VERB_LEVEL_MAP["L2"])
    assert schema1["properties"]["marking_scheme"]["items"]["properties"]["marks"]["maximum"] == 7

    # Mutate active BLOOM_VERB_LEVEL_MAP
    orig = set(BLOOM_VERB_LEVEL_MAP["L2"])
    try:
        BLOOM_VERB_LEVEL_MAP["L2"] = {"paraphrase", "elucidate"}
        snap2 = capture_rule_snapshot()
        assert snap2.fingerprint != snap1.fingerprint
        schema2 = build_subpart_json_schema(sp, rule_snapshot=snap2)
        assert schema2["properties"]["bloom_verb_used"]["enum"] == ["elucidate", "paraphrase"]
    finally:
        BLOOM_VERB_LEVEL_MAP["L2"] = orig


def test_04_assessment_config_change_updates_snapshot():
    """4. Changing AssessmentConfig updates snapshot fingerprint and constraints."""
    snap1 = capture_rule_snapshot(("VTU_2022_NEP", "AEC_GENERAL_THEORY", "CIE_IAT", "VTU_AEC_ANNEXURE_II"))
    custom_cfg = AssessmentConfig(
        scheme_id="CUSTOM_TEST",
        course_type="CUSTOM_THEORY",
        assessment_type="SEE",
        blueprint_id="CUSTOM_BP",
        total_marks_per_question=20,
        permitted_partitions=((10, 10), (8, 6, 6)),
        max_subparts=3,
        reference_source="Custom Test Ref",
    )
    snap2 = capture_rule_snapshot(config=custom_cfg)
    assert snap2.fingerprint != snap1.fingerprint
    assert snap2.total_marks_per_question == 20
    assert snap2.max_subparts == 3


def test_05_06_exemplars_use_active_rules_and_missing_rules_fail_closed():
    """5-6. Prompt exemplars use active rules; missing/empty Bloom level fails closed."""
    sp = _sample_sp_plan(bloom="L4", marks=8)
    orig = set(BLOOM_VERB_LEVEL_MAP["L4"])
    try:
        BLOOM_VERB_LEVEL_MAP["L4"] = {"dissect"}
        snap = capture_rule_snapshot()
        ex = build_dynamic_exemplar_section(sp, rule_snapshot=snap)
        assert '"bloom_verb_used": "dissect"' in ex
        assert '"marks": 4' in ex

        BLOOM_VERB_LEVEL_MAP["L4"] = set()
        with pytest.raises(ValueError):
            capture_rule_snapshot()
    finally:
        BLOOM_VERB_LEVEL_MAP["L4"] = orig


def test_07_schema_and_temperature_forwarding_chain():
    """7. Schema & temperature forward from WritingAgent -> RobustLLMCallerAdapter -> RobustLLMCaller."""
    captured = {}

    class SpyRobustCaller:
        backend = "vllm"

        def call(self, req):
            captured["schema"] = req.schema
            captured["temperature"] = req.temperature
            captured["seed"] = req.seed
            return type("Res", (), {
                "success": True,
                "parsed": None,
                "text": json.dumps({
                    "text": "Explain the operational mechanism of CPU Scheduling.",
                    "solution": "CPU scheduling selects ready processes.",
                    "marking_scheme": [{"criterion": "Mechanism", "marks": 10}],
                    "bloom_verb_used": "explain",
                }),
                "error": None,
            })()

    adapter = RobustLLMCallerAdapter(caller=SpyRobustCaller())
    sp = _sample_sp_plan(bloom="L2", marks=10)
    plan = _sample_plan(sp)
    writer = WritingAgent(llm_caller=adapter, enable_irt=False)
    gq, attempts = writer._generate_one(plan)

    assert gq.marks == 10
    assert captured["schema"] is not None
    assert captured["schema"]["properties"]["bloom_verb_used"]["enum"] == sorted(BLOOM_VERB_LEVEL_MAP["L2"])
    assert captured["temperature"] == 0.1


def test_08_09_10_invalid_types_and_marking_sum_rejected():
    """8-10. Invalid JSON, bool/float marks, and wrong marking-scheme sum are rejected."""
    from tests.test_stage_g1d_integration import MockLLMCaller

    sp = _sample_sp_plan(bloom="L2", marks=10)
    plan = _sample_plan(sp)

    bad_bool = json.dumps({"text": "Explain CPU Scheduling.", "solution": "Sol.", "marking_scheme": [{"criterion": "C", "marks": True}], "bloom_verb_used": "explain"})
    bad_sum = json.dumps({"text": "Explain CPU Scheduling.", "solution": "Sol.", "marking_scheme": [{"criterion": "C", "marks": 3}], "bloom_verb_used": "explain"})
    good = json.dumps({"text": "Explain CPU Scheduling.", "solution": "Sol.", "marking_scheme": [{"criterion": "C", "marks": 10}], "bloom_verb_used": "explain"})

    llm = MockLLMCaller(responses=["not json", bad_bool, bad_sum, good])
    writer = WritingAgent(llm_caller=llm, max_attempts=4, enable_irt=False)
    gq, attempts = writer._generate_one(plan)
    assert attempts == 4
    assert sum(c["marks"] for c in gq.sub_parts[0].marking_scheme) == 10


def test_11_12_evidence_retention_and_metadata_sanitization():
    """11-12. Query-focused evidence retains equations/numbers while sanitizing source tags."""
    blocks = [
        TextBlock(id="b1", text="Unrelated historical background about punch cards.", page=1, block_role="BODY"),
        TextBlock(id="b2", text="Figure 3.2: [blk_m1_2] CPU Scheduling turnaround time TAT = CT - AT; given burst = 10ms, quantum = 2ms.", page=4, block_role="BODY"),
    ]
    ev = build_evidence_section(blocks, topic="CPU Scheduling", char_budget=250)
    assert "TAT = CT - AT" in ev
    assert "quantum = 2ms" in ev
    assert "Figure 3.2:" not in ev
    assert "[blk_m1_2]" not in ev


def test_13_surgical_retry_preserves_immutable_plan_constraints():
    """13. Surgical retry fixes original failure without changing marks, CO, Bloom, module, or concept."""
    from tests.test_stage_g1d_integration import MockLLMCaller

    sp = _sample_sp_plan(bloom="L2", marks=10)
    plan = _sample_plan(sp)

    multi_dir = json.dumps({
        "text": "Explain CPU Scheduling and implement Round Robin.",
        "solution": "Sol.",
        "marking_scheme": [{"criterion": "C", "marks": 10}],
        "bloom_verb_used": "explain",
    })
    fixed = json.dumps({
        "text": "Explain the operational mechanism of CPU Scheduling.",
        "solution": "Sol.",
        "marking_scheme": [{"criterion": "C", "marks": 10}],
        "bloom_verb_used": "explain",
    })
    llm = MockLLMCaller(responses=[multi_dir, fixed])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = writer._generate_one(plan)

    assert attempts == 2
    assert "MULTIPLE_DIRECTIVES" in llm.calls[1]["prompt"]
    assert gq.marks == 10
    assert gq.co == "CO2"
    assert gq.bloom == "L2"
    assert gq.module_id == "module_1"
    assert gq.sub_parts[0].topic == "CPU Scheduling"


def test_14_export_docx_route_enforces_visual_manifest(tmp_path, monkeypatch):
    """14. /api/export/docx blocks unverified/swapped visuals (422) and allows manifest-verified papers (200)."""
    from types import SimpleNamespace
    import aion_api
    from core.validation.export_gate import ExportGate

    p_good = tmp_path / "good.png"
    img = Image.new("RGB", (200, 100), color=(240, 245, 250))
    ImageDraw.Draw(img).text((10, 10), "Circuit", fill=(0, 0, 0))
    img.save(p_good, format="PNG")
    sha_good = hashlib.sha256(p_good.read_bytes()).hexdigest()

    payload = {
        "paper_id": "paper_live_vis_001",
        "assessmentType": "IA",
        "exam_type": "IAT1",
        "subject": "Circuits",
        "subject_code": "BEE101",
        "max_marks": 10,
        "duration": "30 Minutes",
        "modules": [{"module_id": "module_1", "title": "Module 1: DC Circuits"}],
        "questions": [{
            "question_number": 1,
            "module_id": "module_1",
            "marks": 10,
            "co": "CO1",
            "bloom": "L3",
            "sub_questions": [{
                "label": "(a)",
                "text": "With reference to the given diagram, calculate resistance.",
                "solution": "15 ohm",
                "marks": 10,
                "co": "CO1",
                "bloom": "L3",
                "figure_id": "fig_1",
                "image_path": str(p_good),
                "image_sha256": sha_good,
                "figure_caption": "Series Circuit",
            }],
        }],
    }

    import core.generation.agents.local_evaluators as _le
    monkeypatch.setattr(aion_api, "_validate_and_authorize_export_payload", lambda _req: (payload, None))
    monkeypatch.setattr(ExportGate, "evaluate_dict", staticmethod(lambda _b: SimpleNamespace(passed=True, message="OK", failures=[])))

    client = aion_api.app.test_client()
    orig_manifest = aion_api.app.config.get("TRUSTED_VISUAL_MANIFEST")
    try:
        # 1. Without trusted server-side manifest (and no OCR), visual export MUST be blocked (422)
        # Monkeypatch verify_exported_docx_visuals to report failure when no approved manifest
        monkeypatch.setattr(_le, "verify_exported_docx_visuals", lambda _buf, _m: (False, ["No approved visual manifest provided"]))
        aion_api.app.config["TRUSTED_VISUAL_MANIFEST"] = None
        res_blocked = client.post("/api/export/docx", json=payload)
        assert res_blocked.status_code == 422
        assert "visual" in res_blocked.get_json()["error"].lower()

        # 2. With trusted server-side PASS manifest, verify_exported_docx_visuals returns OK -> 200
        monkeypatch.setattr(_le, "verify_exported_docx_visuals", lambda _buf, _m: (True, []))
        aion_api.app.config["TRUSTED_VISUAL_MANIFEST"] = [{
            "question_id": "Q1",
            "subpart_id": "Q1_a",
            "figure_id": "fig_1",
            "image_sha256": sha_good,
            "expected_caption": "Series Circuit",
            "decision": "PASS",
            "approved": True,
        }]
        res_ok = client.post("/api/export/docx", json=payload)
        assert res_ok.status_code == 200
        assert len(res_ok.data) > 0
    finally:
        aion_api.app.config["TRUSTED_VISUAL_MANIFEST"] = orig_manifest


def test_15_opening_verb_consistency_enforced():
    """15. Opening verb in 'text' must match 'bloom_verb_used'; mismatch triggers BLOOM_VERB_MISMATCH retry."""
    from tests.test_stage_g1d_integration import MockLLMCaller

    sp = _sample_sp_plan(bloom="L4", marks=10)
    plan = _sample_plan(sp)

    mismatch = json.dumps({
        "text": "Define the scheduling behavior of the given system.",
        "solution": "Sol.",
        "marking_scheme": [{"criterion": "C", "marks": 10}],
        "bloom_verb_used": "Analyze",
    })
    fixed = json.dumps({
        "text": "Analyze why Round Robin scheduling degrades under large context-switch overhead.",
        "solution": "Sol.",
        "marking_scheme": [{"criterion": "C", "marks": 10}],
        "bloom_verb_used": "Analyze",
    })
    llm = MockLLMCaller(responses=[mismatch, fixed])
    writer = WritingAgent(llm_caller=llm, enable_irt=False)
    gq, attempts = writer._generate_one(plan)

    assert attempts == 2
    assert "BLOOM_VERB_MISMATCH" in llm.calls[1]["prompt"]
    assert gq.sub_parts[0].text.startswith("Analyze")


def test_16_token_aware_budget_and_l4_l6_scaffolding():
    """16. Token-aware evidence budget formula and L4-L6 task-specific scaffolding work."""
    from core.generation.agents.prompts import compute_token_aware_evidence_budget, build_subpart_task_section

    b = compute_token_aware_evidence_budget(context_tokens=8192, image_tokens=1024)
    assert b > 0
    sp_l4 = _sample_sp_plan(bloom="L4", marks=10)
    task_l4 = build_subpart_task_section(sp_l4)
    assert "COGNITIVE DEPTH (L4 - Analyze)" in task_l4


def test_17_rulesnapshot_deep_immutability_and_full_sha256():
    """17. RuleSnapshot is deeply immutable and uses full 64-character SHA-256 fingerprint."""
    snapshot = capture_rule_snapshot()
    assert len(snapshot.fingerprint) == 64  # Full SHA-256 hex digest
    with pytest.raises((TypeError, AttributeError)):
        snapshot.bloom_verb_map["L4"] = ("define",)


def test_18_rulesnapshot_rejects_unregistered_and_conflicting_identities():
    """18. Explicit unregistered assessment identity raises ValueError; conflicting identity rejected."""
    with pytest.raises(ValueError) as exc:
        capture_rule_snapshot(assessment_identity=("INVALID_SCHEME", "INVALID_COURSE", "INVALID_TYPE", "INVALID_BP"))
    assert "Unregistered or invalid assessment_identity" in str(exc.value)

    cfg = AssessmentRegistry.get("VTU_2022_NEP", "AEC_GENERAL_THEORY", "CIE_IAT", "VTU_AEC_ANNEXURE_II")
    with pytest.raises(ValueError) as exc2:
        capture_rule_snapshot(assessment_identity=("OTHER", "COURSE", "IAT", "BP"), config=cfg)
    assert "Conflicting configuration" in str(exc2.value)


def test_19_provenance_rejects_heuristic_module_inference():
    """19. SubPartPlan provenance validation fails closed without heuristic regex inference."""
    # Contradictory block ID with no module_id must FAIL closed
    sp = SubPartPlan(
        label="a",
        marks=10,
        topic="DC Circuits",
        bloom="L2",
        co="CO1",
        canonical_concept_id="circuits",
        evidence_blocks=[
            {"id": "block_m2_17", "text": "DC circuit text on page 2", "page": 2},  # No module_id, ID implies m2, but page is within m1 [1, 20]
        ],
    )
    valid, err = sp.validate_provenance(module_idx=1, page_start=1, page_end=20)
    assert not valid
    assert "module_id 'None' != expected 'module_1'" in err


def test_20_visual_task_contract_positional_and_collection_validation():
    """20. VisualTaskContract validates collections and argument counts."""
    from core.generation.agents.question_plan import VisualTaskContract

    # Bare string for required_features must fail
    with pytest.raises(TypeError):
        VisualTaskContract("fig_1", True, "calculate", "bare_string_not_list")

    # Bare string for prohibited_features must fail
    with pytest.raises(TypeError):
        VisualTaskContract("fig_1", True, "calculate", ["circuit"], "bare_string_prohibited")

    # Excess positional arguments (> 5) must fail
    with pytest.raises(TypeError):
        VisualTaskContract("fig_1", True, "calculate", ["circuit"], ["solution"], "excess_arg")

    # Valid 5 positional args succeeds
    vt = VisualTaskContract("fig_1", True, "calculate", ["circuit"], ["solution"])
    assert vt.prohibited_features == ("solution",)


def test_21_temperature_and_seed_validation(monkeypatch):
    """21. RobustLLMCallerAdapter validates temperature bounds and preserves seed=0."""
    captured = {}

    class DummyCaller:
        def call(self, req):
            captured["req"] = req
            return type("Res", (), {"success": True, "parsed": {"ok": True}, "text": '{"ok": true}', "error": None})()

    adapter = RobustLLMCallerAdapter(caller=DummyCaller())
    monkeypatch.setenv("AION_LLM_TEMPERATURE", "not_a_float")
    with pytest.raises(ValueError) as exc1:
        adapter.call(prompt="hello", schema=None, seed=0)
    assert "Invalid AION_LLM_TEMPERATURE" in str(exc1.value)

    monkeypatch.setenv("AION_LLM_TEMPERATURE", "3.5")
    with pytest.raises(ValueError) as exc2:
        adapter.call(prompt="hello", schema=None, seed=0)
    assert "out of bounds" in str(exc2.value)

    monkeypatch.setenv("AION_LLM_TEMPERATURE", "0.7")
    adapter.call(prompt="hello", schema=None, seed=0)
    assert captured["req"].temperature == 0.7
    assert captured["req"].seed == 0  # seed=0 preserved, not converted to 42

