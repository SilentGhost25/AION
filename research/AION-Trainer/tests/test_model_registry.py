import pytest
from pathlib import Path

from server.model_registry import ModelRegistry


def test_register_candidate_adds_first_version_0_1(model_registry):
    scores = {"overall_score": 0.92, "grammar": 0.95}
    record = model_registry.register_candidate(
        subject="BAI401",
        weights_path="checkpoints/aion_model_latest.pt",
        dataset_version="8",
        knowledge_version="2.1",
        benchmark_scores=scores,
        job_id="JOB-1234",
    )
    assert record.version == "0.1"
    assert record.subject == "BAI401"
    assert record.dataset_version == "8"
    assert record.knowledge_version == "2.1"
    assert record.is_production is False
    assert record.benchmark_scores == scores


def test_register_candidate_increments_version_float_friendly(model_registry):
    scores = {"overall_score": 0.92}
    model_registry.register_candidate(
        subject="BAI401",
        weights_path="ckpt1",
        dataset_version="8",
        knowledge_version="2.1",
        benchmark_scores=scores,
        job_id="JOB-1",
    )
    record2 = model_registry.register_candidate(
        subject="BAI401",
        weights_path="ckpt2",
        dataset_version="9",
        knowledge_version="2.2",
        benchmark_scores=scores,
        job_id="JOB-2",
    )
    assert record2.version == "0.2"


def test_get_production_returns_none_when_empty(model_registry):
    assert model_registry.get_production("BAI401") is None


def test_promote_to_production_marks_flag_and_disables_others(model_registry):
    scores = {"overall_score": 0.92}
    r1 = model_registry.register_candidate("BAI401", "ckpt1", "8", "2.1", scores, "JOB-1")
    r2 = model_registry.register_candidate("BAI401", "ckpt2", "8", "2.1", scores, "JOB-2")

    # Promote r1
    assert model_registry.promote_to_production("BAI401", r1.version) is True
    prod = model_registry.get_production("BAI401")
    assert prod.version == r1.version

    # Promote r2
    assert model_registry.promote_to_production("BAI401", r2.version) is True
    prod2 = model_registry.get_production("BAI401")
    assert prod2.version == r2.version

    # Verify r1 is no longer production
    assert model_registry.get_candidate("BAI401", r1.version).is_production is False


def test_promote_to_production_returns_false_for_unknown_version(model_registry):
    assert model_registry.promote_to_production("BAI401", "9.9") is False


def test_list_candidates_filters_by_subject(model_registry):
    model_registry.register_candidate("BAI401", "ckpt1", "8", "2.1", {}, "JOB-1")
    model_registry.register_candidate("BCS402", "ckpt2", "8", "2.1", {}, "JOB-2")

    candidates = model_registry.list_candidates("BAI401")
    assert len(candidates) == 1
    assert candidates[0].subject == "BAI401"


def test_compare_to_production_gives_green_signal_when_no_production(model_registry):
    gate = model_registry.compare_to_production("BAI401", {"overall_score": 0.95})
    assert gate["can_promote"] is True
    assert gate["comparisons"] == {}


def test_compare_to_production_rejects_degraded_overall_score(model_registry):
    r = model_registry.register_candidate("BAI401", "ckpt1", "8", "2.1", {"overall_score": 0.90}, "JOB-1")
    model_registry.promote_to_production("BAI401", r.version)

    # Candidate has lower overall score
    gate1 = model_registry.compare_to_production("BAI401", {"overall_score": 0.88})
    assert gate1["can_promote"] is False

    # Candidate has higher overall score
    gate2 = model_registry.compare_to_production("BAI401", {"overall_score": 0.92})
    assert gate2["can_promote"] is True
