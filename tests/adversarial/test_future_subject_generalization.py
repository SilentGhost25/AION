# tests/adversarial/test_future_subject_generalization.py

import json
import os
from pathlib import Path
from unittest.mock import patch
import pytest

from core.extraction.block_classifier import ContentAwareBlockClassifier, BlockRole
from core.generation.topic_validator import MultiDomainTopicValidator

CN_FIXTURE_PATH = Path("tests/fixtures/future_paper_cn/cn_blocks.jsonl")


def test_future_subject_cn_classification_and_topic_validity():
    """
    D.3: Future-Paper Generalization Test
    Tests the block classifier and topic validity gate on unseen third subject
    (Computer Networks), labeled blindly to verify zero false acceptance on bad
    blocks and full recognition of core networking concepts.
    """
    assert CN_FIXTURE_PATH.exists(), "CN fixture missing"

    blocks = []
    with open(CN_FIXTURE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                blocks.append(json.loads(line))

    # 1. Test Block Role Classification
    classified = ContentAwareBlockClassifier.classify_blocks(blocks)
    for orig, clf in zip(blocks, classified):
        expected_role = orig["role"]
        # Bad metadata/exercise/admin must not be classified as BODY
        if expected_role in ("EXERCISE", "METADATA", "ADMIN", "BIBLIOGRAPHY"):
            assert clf.role != BlockRole.BODY, f"Block {orig['id']} wrongly classified as BODY"

    # 2. Test Multi-Domain Topic Validator with domain COMPUTER_NETWORKS
    validator = MultiDomainTopicValidator()
    with patch.dict(os.environ, {"AION_ENABLE_TOPIC_VALIDATOR": "true"}):
        for block in blocks:
            text = block["text"]
            label = block["label"]
            res = validator.validate_topic(text, current_domain="COMPUTER_NETWORKS")

            if label == "bad":
                assert res.is_valid is False, f"Bad CN block '{text}' was improperly accepted"
            elif label == "good":
                assert res.is_valid is True, f"Good CN concept '{text}' was rejected: {res.reason}"
                assert res.assigned_domain == "COMPUTER_NETWORKS"
