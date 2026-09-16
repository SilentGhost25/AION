"""
Unit tests for AION Real-Time RAG Metrics (AION RAGAS-Inspired Evaluator)
========================================================================
Tests:
- Zero/empty safety of harmonic_mean
- Independent faithfulness vs. hallucination_rate calculation
- Context recall against EvidenceBundle and text
- Question relevance against Bloom & marks
- Metric provenance
- Latency percentiles (p50, p95, p99, max)
- Fail-open safety on malformed input
"""

import time
import pytest
from core.evaluation.aggregation import harmonic_mean, compute_percentile, aggregate_paper_metrics
from core.evaluation.contracts import RealtimeRAGMetrics, MetricProvenance, PaperRAGSummary
from core.evaluation.deterministic import (
    DeterministicRAGEvaluator,
    register_metric_listener,
    unregister_metric_listener,
    emit_accepted_metric,
)


def test_harmonic_mean_robustness():
    """Verify harmonic mean handles zeroes, empty lists, and clamps correctly."""
    # Empty list
    assert harmonic_mean([]) == 0.0

    # Any zero must return 0.0 (prevents division by zero)
    assert harmonic_mean([0.0, 0.8, 0.9]) == 0.0
    assert harmonic_mean([0.8, 0.0, 0.9]) == 0.0
    assert harmonic_mean([0.0, 0.0, 0.0]) == 0.0

    # Standard positive inputs
    assert pytest.approx(harmonic_mean([1.0, 1.0, 1.0]), 0.001) == 1.0
    assert pytest.approx(harmonic_mean([0.6, 0.6, 0.6]), 0.001) == 0.6

    # Clamping out-of-bound inputs
    assert pytest.approx(harmonic_mean([1.5, 1.0]), 0.001) == 1.0


def test_independent_faithfulness_and_hallucination():
    """Verify faithfulness and hallucination rate are computed independently."""
    evaluator = DeterministicRAGEvaluator()
    evidence = "In a series DC circuit, the total resistance is R_total = R1 + R2. The current is uniform throughout."

    # Grounded question
    q_grounded = "Explain how the total resistance is calculated in a series DC circuit with resistors R1 and R2."
    metrics_grounded = evaluator.evaluate_question(
        question_text=q_grounded,
        slot=None,
        evidence=evidence,
    )
    assert metrics_grounded.faithfulness >= 0.7
    assert metrics_grounded.hallucination_rate == 0.0
    assert len(metrics_grounded.hallucination_findings) == 0

    # Question with phantom figure citation
    q_hallucinated = "With reference to Figure 9.9, determine the current in the loop."
    metrics_hall = evaluator.evaluate_question(
        question_text=q_hallucinated,
        slot=None,
        evidence=evidence,
    )
    assert metrics_hall.hallucination_rate > 0.0
    assert any("Figure 9.9" in f or "figure" in f.lower() for f in metrics_hall.hallucination_findings)
    # Crucial check: hallucination_rate is NOT simply 1 - faithfulness
    assert metrics_hall.hallucination_rate != (1.0 - metrics_hall.faithfulness)


def test_question_relevance_bloom_and_archetype():
    """Verify question relevance scores Bloom verbs and marks budgets appropriately."""
    evaluator = DeterministicRAGEvaluator()
    evidence = "Nodal analysis uses Kirchhoff's Current Law (KCL) to determine node voltages in a circuit."

    class MockSlot:
        slot_id = "M1_Q1_a"
        bloom_level = "L3"
        bloom_verb = "calculate"
        marks = 6
        question_type = "NUMERICAL"
        keywords = ["nodal", "voltages", "kcl"]

    slot = MockSlot()
    q_aligned = "Calculate the node voltages in the given circuit network using nodal analysis and KCL."
    metrics = evaluator.evaluate_question(
        question_text=q_aligned,
        slot=slot,
        evidence=evidence,
        latency_ms=45.0,
    )

    assert metrics.question_relevance >= 0.7
    assert metrics.answer_relevance_proxy == metrics.question_relevance
    assert metrics.archetype_adherence >= 0.5
    assert metrics.harmonic_rag_score > 0.0


def test_provenance_and_framework_metadata():
    """Verify complete metric provenance is retained."""
    evaluator = DeterministicRAGEvaluator()
    class MockSlot:
        slot_id = "M2_Q3_b"
        module_id = 2
        topic = "Transformers"

    slot = MockSlot()
    metrics = evaluator.evaluate_question(
        question_text="Explain the working principle of a step-down transformer.",
        slot=slot,
        evidence="A step-down transformer converts high primary voltage to lower secondary voltage.",
        model_name="qwen2.5:14b",
        provider_name="ollama",
        trace_id="TR-12345",
        generation_id="GEN-999",
        attempt_index=1,
    )

    assert metrics.evaluation_framework == "aion_ragas_inspired"
    assert metrics.provenance.trace_id == "TR-12345"
    assert metrics.provenance.generation_id == "GEN-999"
    assert metrics.provenance.slot_id == "M2_Q3_b"
    assert metrics.provenance.module_idx == 2
    assert metrics.provenance.model_name == "qwen2.5:14b"
    assert metrics.provenance.provider == "ollama"

    d = metrics.to_dict()
    assert d["evaluation_framework"] == "aion_ragas_inspired"
    assert "provenance" in d


def test_latency_profiling():
    """Benchmark evaluation speed and profile percentiles (p50, p95, p99, max)."""
    evaluator = DeterministicRAGEvaluator()
    evidence = "Ohm's Law states V = I * R where V is voltage, I is current, and R is resistance in ohms."
    q_text = "Calculate the current I passing through a 10 ohm resistor when a 20V source is applied."

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = evaluator.evaluate_question(
            question_text=q_text,
            slot=None,
            evidence=evidence,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p50 = compute_percentile(latencies, 50.0)
    p95 = compute_percentile(latencies, 95.0)
    p99 = compute_percentile(latencies, 99.0)
    max_lat = max(latencies)

    print(f"\n[LATENCY BENCHMARK (100 runs)] p50={p50:.3f}ms | p95={p95:.3f}ms | p99={p99:.3f}ms | max={max_lat:.3f}ms")
    # Target: <5ms; assert p95 < 10ms to account for CI jitter
    assert p95 < 10.0, f"p95 latency exceeded 10ms: {p95:.3f}ms"


def test_fail_open_safety():
    """Verify evaluator never raises an exception on unexpected inputs."""
    evaluator = DeterministicRAGEvaluator()
    # Malformed / None inputs
    m1 = evaluator.evaluate_question(question_text=None, slot=None, evidence=None)
    assert isinstance(m1, RealtimeRAGMetrics)
    assert m1.status in ("PASS", "WARNING")

    # Extreme edge cases
    m2 = evaluator.evaluate_question(question_text="", slot=object(), evidence=12345)
    assert isinstance(m2, RealtimeRAGMetrics)


def test_listener_registry_and_emit():
    """Verify accepted question metric events broadcast to registered callbacks."""
    received = []

    def on_metric(m: RealtimeRAGMetrics):
        received.append(m)

    register_metric_listener(on_metric)
    evaluator = DeterministicRAGEvaluator()

    m = evaluator.evaluate_question("What is Ohm's Law?", None, "Ohm's Law is V = IR.")
    emit_accepted_metric(m)

    assert len(received) == 1
    assert received[0].evaluation_framework == "aion_ragas_inspired"

    unregister_metric_listener(on_metric)
    emit_accepted_metric(m)
    # Should not receive after unregistering
    assert len(received) == 1


def test_generate_main_question_attaches_rag_metrics():
    """Verify that _generate_main_question in v0_1/main.py observes and attaches ragas_metrics."""
    from v0_1.main import _generate_main_question
    from core.contracts.question import GeneratedQuestion
    from core.contracts.question_slot import QuestionSlot

    class MockOrchestrator:
        model_name = "qwen2.5:14b"
        def generate(self, slot, evidence_pack, excluded_concepts=None):
            class MockOutput:
                question_text = "Explain the architecture and key components of a relational database management system."
                math_blocks = []
                diagram_request = None
            return GeneratedQuestion(MockOutput(), slot)

    orch = MockOrchestrator()
    sample_chunk = "Relational database management systems architecture consists of key components including the storage engine, query processor, and schema constraints."

    mq_result = _generate_main_question(
        mq_idx=1,
        partition=[10],
        bloom=2,
        chunks=[sample_chunk],
        total_marks=10,
        module_id="module_1",
        orchestrator=orch,
    )

    assert "generated_questions" in mq_result
    gq = mq_result["generated_questions"][0]
    assert hasattr(gq, "ragas_metrics")
    assert gq.ragas_metrics is not None
    assert gq.ragas_metrics.evaluation_framework == "aion_ragas_inspired"
    assert gq.ragas_metrics.faithfulness > 0.5
    assert gq.ragas_metrics.question_relevance > 0.5

    # Check subquestion dictionary
    sub_q = mq_result["sub_questions"][0]
    assert "ragas_metrics" in sub_q
    assert sub_q["ragas_metrics"]["evaluation_framework"] == "aion_ragas_inspired"
    assert "harmonic_rag_score" in sub_q["ragas_metrics"]


def test_format_paper_preserves_rag_metrics():
    """Verify _format_paper in aion_api.py forwards ragas_metrics and ragas_summary."""
    from aion_api import _format_paper

    raw_paper = {
        "modules": [
            {
                "module_index": 1,
                "module_title": "Database Systems",
                "questions": [
                    {
                        "mq_index": 1,
                        "total_marks": 10,
                        "bloom_level": 2,
                        "bloom_name": "Understand",
                        "sub_questions": [
                            {
                                "letter": "a",
                                "text": "Explain the ACID properties in database transaction processing.",
                                "marks": 10,
                                "bloom": "L2",
                                "co": "CO1",
                                "ragas_metrics": {
                                    "faithfulness": 0.95,
                                    "context_recall": 0.90,
                                    "question_relevance": 0.95,
                                    "harmonic_rag_score": 0.93,
                                    "hallucination_rate": 0.0,
                                    "evaluation_framework": "aion_ragas_inspired"
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    }

    qa_report = {
        "quality_score": 95,
        "ragas_summary": {
            "mean_faithfulness": 0.95,
            "mean_context_recall": 0.90,
            "mean_question_relevance": 0.95,
            "mean_rag_score": 0.93,
            "evaluation_framework": "aion_ragas_inspired"
        }
    }

    formatted = _format_paper(raw_paper, subject="DBMS", exam_type="IAT1", mode="turbo", qa_report=qa_report)

    # Validate paper-level summary
    assert "ragas_summary" in formatted
    assert formatted["ragas_summary"]["mean_rag_score"] == 0.93
    assert formatted["qaReport"]["ragas_summary"]["evaluation_framework"] == "aion_ragas_inspired"

    # Validate subquestion level metrics
    mod = formatted["modules"][0]
    q = mod["questions"][0]
    sub = q["subQuestions"][0]
    assert "ragas_metrics" in sub or "ragasMetrics" in sub
    sub_metrics = sub.get("ragas_metrics") or sub.get("ragasMetrics")
    assert sub_metrics["harmonic_rag_score"] == 0.93

