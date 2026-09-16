"""
AION RAG Metric Aggregation & Statistics
========================================
Robust aggregation mathematical helpers:
- Guaranteed zero-division safe harmonic mean
- Latency percentiles (p50, p95, p99, max)
- Paper-level summary aggregation
"""

from __future__ import annotations

import math
from typing import List, Sequence

from .contracts import RealtimeRAGMetrics, PaperRAGSummary


def harmonic_mean(values: Sequence[float]) -> float:
    """
    Robust harmonic mean calculation for bounded [0.0, 1.0] metric scores.

    Guarantees:
    - Values are clamped to [0.0, 1.0].
    - Returns 0.0 if the input list is empty.
    - If ANY value is 0.0, returns 0.0 (preventing division by zero).
    - Returns exact float: len(values) / sum(1.0 / v for v in values).
    """
    if not values:
        return 0.0

    clamped = [max(0.0, min(1.0, float(v))) for v in values]

    if any(v == 0.0 for v in clamped):
        return 0.0

    inv_sum = sum(1.0 / v for v in clamped)
    if inv_sum <= 0.0:
        return 0.0

    return float(len(clamped)) / inv_sum


def compute_percentile(data: Sequence[float], percentile: float) -> float:
    """
    Calculates the given percentile (0.0 to 100.0) from a sequence of floats.
    Uses linear interpolation between nearest ranks.
    """
    if not data:
        return 0.0
    sorted_data = sorted(float(x) for x in data)
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]

    k = (n - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return d0 + d1


def aggregate_paper_metrics(metrics_list: List[RealtimeRAGMetrics]) -> PaperRAGSummary:
    """
    Aggregates per-question RealtimeRAGMetrics into an authoritative PaperRAGSummary.
    Safe against empty lists or partial generation.
    """
    if not metrics_list:
        return PaperRAGSummary(questions_evaluated=0)

    n = len(metrics_list)
    faithfulness_vals = [m.faithfulness for m in metrics_list]
    recall_vals = [m.context_recall for m in metrics_list]
    relevance_vals = [m.question_relevance for m in metrics_list]
    rag_scores = [m.harmonic_rag_score for m in metrics_list]
    hallucination_vals = [m.hallucination_rate for m in metrics_list]
    equation_vals = [m.equation_fidelity for m in metrics_list]
    latencies = [m.latency_ms for m in metrics_list]
    tps_vals = [m.tokens_per_sec for m in metrics_list if m.tokens_per_sec > 0]

    summary = PaperRAGSummary(
        mean_faithfulness=sum(faithfulness_vals) / n,
        mean_context_recall=sum(recall_vals) / n,
        mean_question_relevance=sum(relevance_vals) / n,
        mean_rag_score=sum(rag_scores) / n,
        mean_hallucination_rate=sum(hallucination_vals) / n,
        mean_equation_fidelity=sum(equation_vals) / n,
        mean_latency_ms=sum(latencies) / n,
        p50_latency_ms=compute_percentile(latencies, 50.0),
        p95_latency_ms=compute_percentile(latencies, 95.0),
        p99_latency_ms=compute_percentile(latencies, 99.0),
        max_latency_ms=max(latencies) if latencies else 0.0,
        total_context_tokens=sum(m.context_tokens for m in metrics_list),
        total_generation_tokens=sum(m.generation_tokens for m in metrics_list),
        avg_tokens_per_sec=(sum(tps_vals) / len(tps_vals)) if tps_vals else 0.0,
        questions_evaluated=n,
    )
    return summary
