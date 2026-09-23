# tests/performance/test_latency_budget.py

import time
import pytest
from core.extraction.block_classifier import ContentAwareBlockClassifier
from core.generation.topic_validator import MultiDomainTopicValidator
from core.validation.verb_task_linter import VerbTaskCoherenceLinter


def test_architectural_layers_latency_budget():
    """
    R9: Latency budget verification
    Asserts that the added computational overhead of the block classifier,
    topic validator, and verb-task linter remains < 1.5s per paper (P50) and < 3.0s (P99).
    Across 20 slots x 10 candidate iterations = 200 checks.
    """
    sample_text = "Cloud computing provides elastic on-demand compute resources across distributed datacenters."
    validator = MultiDomainTopicValidator()

    latencies = []
    # Benchmark 200 evaluations (equivalent to full paper candidate evaluation)
    for _ in range(200):
        t0 = time.perf_counter()
        _ = ContentAwareBlockClassifier.classify_block_single(sample_text)
        _ = validator.validate_topic(sample_text, current_domain="CLOUD_COMPUTING")
        _ = VerbTaskCoherenceLinter.lint_instruction("Explain the elastic cloud computing architecture.")
        latencies.append(time.perf_counter() - t0)

    total_latency = sum(latencies)
    sorted_latencies = sorted(latencies)
    p50 = sorted_latencies[len(sorted_latencies) // 2]
    p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]

    # Assert total latency across 200 checks is well under 1.5 seconds
    assert total_latency < 1.5, f"Total latency {total_latency:.3f}s exceeds 1.5s budget"
    assert p50 < 0.01, f"P50 latency per check {p50*1000:.2f}ms too high"
    assert p99 < 0.03, f"P99 latency per check {p99*1000:.2f}ms too high"
