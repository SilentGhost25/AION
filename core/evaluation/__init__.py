"""
AION RAG Evaluation & Psychometric Calibration Package.
======================================================
Exports contracts, evaluator implementation, listener registration,
aggregation, and Rasch 1PL IRT difficulty calibration.
"""

from .contracts import MetricProvenance, RealtimeRAGMetrics, PaperRAGSummary
from .aggregation import harmonic_mean, compute_percentile, aggregate_paper_metrics
from .deterministic import (
    DeterministicRAGEvaluator,
    register_metric_listener,
    unregister_metric_listener,
    emit_accepted_metric,
)
from .irt_calibrator import (
    IRTCalibrator,
    DifficultyEstimate,
    StudentSimFn,
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
    "IRTCalibrator",
    "DifficultyEstimate",
    "StudentSimFn",
]
