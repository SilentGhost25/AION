# tests/adversarial/test_topic_mutation_resistance.py

import os
from unittest.mock import patch
import pytest

from core.generation.topic_validator import MultiDomainTopicValidator

MUTATION_SEEDS = [
    # Pre-labeled adversarial mutations
    "Draw Figure 1.2: Architecture overview",
    "draw figure 99: Soil moisture sensor layout",
    "Reproduce Figure 4: Greenhouse ventilation curve",
    "REPRODUCE FIGURE: Irrigation pipe hydraulic network",
    "Diagram and formula revision list for end-sem",
    "Last-minute revision points for semester exams",
    "Revision checklist: Module 1 to 5",
    "Author: Dr. Arvind Sharma and Prof. K. R. Ramanathan",
    "ISBN: 978-93-5316-890-1",
    "Table of Contents: Chapter 1 Introduction",
    "USN: 1DS20CS001",
    "Visvesvaraya Technological University Belagavi regulations",
    "Answer any FIVE full questions choosing ONE from each module",
    "Review Questions: 1. Define cloud computing",
    "Laboratory Exercise 4: Connect sensor to breadboard",
    "All rights reserved. Copyright 2024 McGraw-Hill",
    "https://aws.amazon.com/architecture/reference",
    "Scheme of Evaluation: 6M theory + 4M sketch",
    "Internal Assessment Test I - Question Paper",
    "Signature of Course Instructor and HOD",
]


def test_adversarial_prelabeled_mutation_catch_rate():
    """
    D.2: Scripted Adversarial Mutation Resistance
    Assert 100% catch rate on pre-labeled adversarial mutations.
    """
    validator = MultiDomainTopicValidator()

    with patch.dict(os.environ, {"AION_ENABLE_TOPIC_VALIDATOR": "true"}):
        caught = 0
        for mutation in MUTATION_SEEDS:
            res = validator.validate_topic(mutation, current_domain="IOT_AGRICULTURE")
            if not res.is_valid:
                caught += 1

        catch_rate = caught / len(MUTATION_SEEDS)
        assert catch_rate == 1.0, f"Expected 100% catch rate on pre-labeled mutations, got {catch_rate:.1%}"


def test_adversarial_dynamic_perturbations():
    """
    D.2: Perturbations testing (typos, case variations, whitespace, symbols)
    Target: >= 98% catch rate.
    """
    validator = MultiDomainTopicValidator()

    dynamic_variants = []
    for seed in MUTATION_SEEDS:
        dynamic_variants.append(seed.upper())
        dynamic_variants.append(seed.lower())
        dynamic_variants.append(f"   {seed}   ")
        dynamic_variants.append(f"*** {seed} ***")
        dynamic_variants.append(f"[ {seed} ]")

    with patch.dict(os.environ, {"AION_ENABLE_TOPIC_VALIDATOR": "true"}):
        caught = 0
        for variant in dynamic_variants:
            res = validator.validate_topic(variant, current_domain="IOT_AGRICULTURE")
            if not res.is_valid:
                caught += 1

        catch_rate = caught / len(dynamic_variants)
        assert catch_rate >= 0.98, f"Expected >= 98% catch rate on perturbed variants, got {catch_rate:.1%}"
