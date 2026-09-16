"""
AION RAG Evaluation Package
===========================
Exports contracts, evaluator implementation, listener registration, and aggregation.
"""

from .contracts import MetricProvenance, RealtimeRAGMetrics, PaperRAGSummary
from .aggregation import harmonic_mean, compute_percentile, aggregate_paper_metrics
from .deterministic import (
    DeterministicRAGEvaluator,
    register_metric_listener,
    unregister_metric_listener,
    emit_accepted_metric,
)

__all__ = [
    "MetricProvenance",
    "RealtimeRAGMetrics",
    "PaperRAGSummary",
    "harmonic_mean",
    "compute_percentile",
    "aggregate_paper_metrics",
    "DeterministicRAGEvaluator",
    "register_metric_listener",
    "unregister_metric_listener",
    "emit_accepted_metric",
]
