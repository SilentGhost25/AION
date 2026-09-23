# tests/unit/test_phase_b_benchmark_and_calibration.py

import json
from pathlib import Path
import pytest
from scripts.calibrate_quality_thresholds import calibrate_thresholds

FIXTURES_DIR = Path("tests/fixtures")


def test_block_roles_corpus_invariants():
    """Verify block_roles.jsonl satisfies >=150 total records and >=15 per role across 10 roles."""
    roles_file = FIXTURES_DIR / "block_roles.jsonl"
    assert roles_file.exists(), "block_roles.jsonl fixture missing"

    expected_roles = {
        "BODY", "HEADING", "CAPTION", "LIST_ITEM", "METADATA",
        "ADMIN", "BIBLIOGRAPHY", "TOC", "EXERCISE", "UNKNOWN"
    }

    counts = {r: 0 for r in expected_roles}
    total_records = 0

    with open(roles_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            total_records += 1
            role = rec.get("role")
            assert role in expected_roles, f"Unexpected role {role} found"
            assert "text" in rec and len(rec["text"].strip()) > 0
            counts[role] += 1

    assert total_records >= 150, f"Expected >= 150 records, got {total_records}"
    for role, count in counts.items():
        assert count >= 15, f"Role {role} has {count} samples (< 15 required)"


def test_paper_quality_corpus_split_and_schema():
    """Verify paper_quality_corpus.jsonl satisfies >=80 records and 70/30 train/holdout split."""
    corpus_file = FIXTURES_DIR / "paper_quality_corpus.jsonl"
    assert corpus_file.exists(), "paper_quality_corpus.jsonl fixture missing"

    total = 0
    train_count = 0
    holdout_count = 0
    labels = set()

    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            total += 1
            assert "text" in item
            assert "label" in item
            assert item["label"] in ("good", "bad", "borderline")
            labels.add(item["label"])

            split = item.get("split")
            assert split in ("train", "holdout"), f"Invalid split: {split}"
            if split == "train":
                train_count += 1
            else:
                holdout_count += 1

    assert total >= 80, f"Expected >= 80 samples, got {total}"
    assert labels == {"good", "bad", "borderline"}
    # Verify split is roughly 70/30 (+/- 5%)
    train_ratio = train_count / total
    assert 0.65 <= train_ratio <= 0.75, f"Train ratio {train_ratio:.2f} not in expected 70% range"


def test_thresholds_json_schema():
    """Verify thresholds.json has all required parameters including delta margin and confidence threshold."""
    thresh_file = FIXTURES_DIR / "thresholds.json"
    assert thresh_file.exists(), "thresholds.json missing"

    with open(thresh_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("parameters", {})
    required_keys = [
        "faithfulness_threshold",
        "qa_threshold",
        "delta_margin",
        "unknown_confidence_threshold",
        "borderline_acceptance_rate",
        "classifier_starvation_threshold",
        "starvation_min_chars",
        "drift_alert_unknown_ratio",
        "unresolved_soft_alert_ratio",
        "unresolved_rollback_ratio",
        "max_added_latency_p50_sec"
    ]

    for k in required_keys:
        assert k in params, f"Key {k} missing from thresholds.json parameters"

    assert params["delta_margin"] == 0.15
    assert params["unknown_confidence_threshold"] == 0.70
    assert params["borderline_acceptance_rate"] == 0.20


def test_calibration_script_execution(tmp_path):
    """Verify calibrate_quality_thresholds executes and writes valid output."""
    temp_out = tmp_path / "test_thresholds.json"
    result = calibrate_thresholds(
        corpus_path=str(FIXTURES_DIR / "paper_quality_corpus.jsonl"),
        output_path=str(temp_out)
    )
    assert temp_out.exists()
    assert result["parameters"]["delta_margin"] == 0.15
