# tests/unit/test_block_classifier.py

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from core.extraction.block_classifier import ContentAwareBlockClassifier, BlockRole
from core.generation.topic_validator import MultiDomainTopicValidator

FIXTURES_DIR = Path("tests/fixtures")


def test_classifier_trained_and_holdout_accuracy():
    """
    D.1: Holdout Verification
    - 0% false acceptance on 'bad' samples
    - <= 20% acceptance on 'borderline' samples (with review routing)
    - >= 95% true acceptance on 'good' technical body topics
    """
    corpus_file = FIXTURES_DIR / "paper_quality_corpus.jsonl"
    assert corpus_file.exists()

    holdout_samples = []
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("split") == "holdout":
                    holdout_samples.append(item)

    assert len(holdout_samples) >= 20, f"Expected at least 20 holdout samples, got {len(holdout_samples)}"

    validator = MultiDomainTopicValidator()

    bad_accepted = 0
    bad_total = 0
    borderline_accepted = 0
    borderline_total = 0
    good_accepted = 0
    good_total = 0

    file_domain_map = {
        "cloud_computing.pdf": "CLOUD_COMPUTING",
        "smart_irrigation.pdf": "IOT_AGRICULTURE",
        "iot_sensing.pdf": "IOT_AGRICULTURE",
        "computer_networks.pdf": "COMPUTER_NETWORKS",
        "dbms_textbook.pdf": "DATABASE_SYSTEMS",
        "os_textbook.pdf": "OPERATING_SYSTEMS",
        "distributed_systems.pdf": "CLOUD_COMPUTING",
        "machine_learning.pdf": "MACHINE_LEARNING",
        "network_security.pdf": "COMPUTER_NETWORKS",
    }

    with patch.dict(os.environ, {"AION_ENABLE_TOPIC_VALIDATOR": "true"}):
        for s in holdout_samples:
            label = s["label"]
            text = s["text"]
            domain = file_domain_map.get(s.get("file", ""), "IOT_AGRICULTURE")
            res = validator.validate_topic(text, current_domain=domain)

            if label == "bad":
                bad_total += 1
                if res.is_valid:
                    bad_accepted += 1
            elif label == "borderline":
                borderline_total += 1
                if res.is_valid:
                    borderline_accepted += 1
                    # When borderline is accepted, it must require human review
                    assert res.requires_human_review is True
            elif label == "good":
                good_total += 1
                if res.is_valid:
                    good_accepted += 1

    # D.1 Metrics verification
    assert bad_total > 0
    false_acceptance_rate = bad_accepted / bad_total
    assert false_acceptance_rate == 0.0, f"0% false acceptance required on bad samples; got {false_acceptance_rate:.1%}"

    assert borderline_total > 0
    borderline_rate = borderline_accepted / borderline_total
    assert borderline_rate <= 0.20, f"<= 20% acceptance required on borderline; got {borderline_rate:.1%}"

    assert good_total > 0
    true_acceptance_rate = good_accepted / good_total
    assert true_acceptance_rate >= 0.95, f">= 95% true acceptance required on good samples; got {true_acceptance_rate:.1%}"


def test_classifier_concrete_starvation_handling():
    """
    R6: Starvation Fallback
    If < 5 BODY blocks in a module, fallback to {BODY, LIST_ITEM} with > 80 chars.
    """
    from core.extraction.contracts import TextBlock

    short_body = TextBlock(text="Short text.", block_id="b1")
    long_list = TextBlock(text="- " + "Detailed specification bullet item " * 3, block_id="b2")  # > 80 chars list item
    admin_block = TextBlock(text="USN: [        ]", block_id="b3")

    classified = ContentAwareBlockClassifier.classify_blocks([short_body, long_list, admin_block])
    handled = ContentAwareBlockClassifier.handle_starvation(classified, module_id=2)

    provisional = [b for b in handled if b.provisional_topic_eligible]
    assert len(provisional) == 1
    assert provisional[0].block_id == "b2"


def test_classifier_drift_telemetry_trigger():
    """
    R8: Drift Telemetry
    If UNKNOWN blocks exceed 25%, triggers data drift alert and disables topic validator.
    """
    from core.extraction.contracts import TextBlock

    # 4 blocks, 2 UNKNOWN noise blocks = 50% UNKNOWN > 25% threshold
    blocks = [
        TextBlock(text="Valid cloud computing body description with sufficient words.", block_id="b1"),
        TextBlock(text="Another descriptive body paragraph explaining virtualization layers.", block_id="b2"),
        TextBlock(text="  0x00 0xFF ???", block_id="b3"),
        TextBlock(text="----------------------------------", block_id="b4"),
    ]

    classified = ContentAwareBlockClassifier.classify_blocks(blocks)
    drift_result = ContentAwareBlockClassifier.check_data_drift(classified)

    assert drift_result["drift_detected"] is True
    assert drift_result["unknown_ratio"] >= 0.25
    assert drift_result["disable_topic_validator"] is True
