# runtime/benchmark/benchmark_pipeline.py
"""End-to-end pipeline benchmark for LAPTOP_FAST profile.

Measures total generation time across all phases and validates the
acceptance predicate.

Usage:
    python runtime/benchmark/benchmark_pipeline.py \
        --dataset "C:\\Users\\Tarun J\\Downloads\\New Dataset\\IAI" \
        --profile LAPTOP_FAST
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from runtime.profiles.base import TimeoutBudget
from runtime.watchdog import PipelineWatchdog, Phase


@dataclass
class PhaseResult:
    """Timing result for a single pipeline phase."""
    name: str
    elapsed_seconds: float
    budget_seconds: float
    exceeded: bool


@dataclass
class PipelineBenchmarkResult:
    """Complete benchmark result for the acceptance predicate."""

    # Timing
    total_time: float = 0.0
    target_time: float = 540.0
    phases: List[PhaseResult] = field(default_factory=list)

    # Generation metrics
    questions_generated: int = 0
    valid_questions: int = 0
    all_slots_generated: bool = False
    first_pass_validation_rate: float = 0.0

    # Integrity checks
    contract_integrity: float = 0.0
    evidence_integrity: float = 0.0
    co_integrity: float = 0.0
    bloom_integrity: float = 0.0
    marks_integrity: float = 0.0
    module_integrity: float = 0.0
    math_integrity: float = 0.0

    # Failure counts
    answer_leakage: int = 0
    unicode_corruption: int = 0
    pdf_internal_leakage: int = 0
    fused_subquestions: int = 0
    evidence_failures: int = 0
    bloom_failures: int = 0
    co_failures: int = 0
    marks_failures: int = 0
    math_failures: int = 0
    or_duplicates: int = 0
    module_leakage: int = 0

    # Export
    export_gate: str = "NOT_RUN"

    # Backend info
    model: str = ""
    backend: str = ""
    dataset: str = ""
    modules: int = 0

    @property
    def passed(self) -> bool:
        """Evaluate the complete acceptance predicate."""
        return (
            self.total_time <= 600.0
            and self.total_time <= self.target_time
            and self.all_slots_generated
            and self.contract_integrity == 100
            and self.evidence_integrity == 100
            and self.co_integrity == 100
            and self.bloom_integrity == 100
            and self.marks_integrity == 100
            and self.module_integrity == 100
            and self.math_integrity == 100
            and self.answer_leakage == 0
            and self.unicode_corruption == 0
            and self.pdf_internal_leakage == 0
            and self.fused_subquestions == 0
            and self.export_gate == "PASS"
        )


def print_report(result: PipelineBenchmarkResult) -> None:
    """Print the formatted benchmark report."""
    print()
    print("=" * 60)
    print("AION LAPTOP FAST FINAL BENCHMARK")
    print("=" * 60)
    print()
    print(f"  Dataset:          {result.dataset}")
    print(f"  Modules:          {result.modules}")
    print(f"  Model:            {result.model}")
    print(f"  Backend:          {result.backend}")
    print()
    print("-" * 60)

    for phase in result.phases:
        status = "  " if not phase.exceeded else " !"
        mins = int(phase.elapsed_seconds // 60)
        secs = int(phase.elapsed_seconds % 60)
        print(f"  {phase.name:<20} {mins:02d}:{secs:02d}{status}")

    print("-" * 60)
    total_mins = int(result.total_time // 60)
    total_secs = int(result.total_time % 60)
    print(f"  {'TOTAL':<20} {total_mins:02d}:{total_secs:02d}")
    print("-" * 60)
    print()
    print(f"  Questions generated  {result.questions_generated}")
    print(f"  Valid questions      {result.valid_questions}")
    print(f"  Evidence failures    {result.evidence_failures}")
    print(f"  Bloom failures       {result.bloom_failures}")
    print(f"  CO failures          {result.co_failures}")
    print(f"  Marks failures       {result.marks_failures}")
    print(f"  Math failures        {result.math_failures}")
    print(f"  Unicode failures     {result.unicode_corruption}")
    print(f"  Answer leakage       {result.answer_leakage}")
    print(f"  OR duplicates        {result.or_duplicates}")
    print(f"  Module leakage       {result.module_leakage}")
    print()
    print("-" * 60)
    print(f"  TIME TARGET          ≤ 10:00")
    print(f"  RESULT               {'PASS' if result.passed else 'FAIL'}")
    print("=" * 60)


def run_pipeline_benchmark(
    dataset_dir: str,
    profile_name: str = "LAPTOP_FAST",
) -> PipelineBenchmarkResult:
    """Run the full pipeline benchmark.

    NOTE: This is currently a framework stub.  The actual generation
    pipeline integration (step 10 in the implementation plan) will wire
    this to the real slot-contract generation flow.
    """
    result = PipelineBenchmarkResult(
        dataset=dataset_dir,
        target_time=540.0,
    )

    # Load benchmark profile
    profile_path = Path(".aion_cache") / "runtime_profile.json"
    if profile_path.exists():
        with open(profile_path, "r", encoding="utf-8") as f:
            prof = json.load(f)
        result.model = prof.get("model", "unknown")
        result.backend = prof.get("backend", "unknown")

    # Scan dataset
    try:
        from runtime.laptop_dataset import scan_dataset
        modules = scan_dataset(dataset_dir)
        result.modules = len(modules)
    except Exception as exc:
        print(f"  ERROR: Dataset scan failed — {exc}")
        return result

    # The actual generation pipeline will be wired here after step 10.
    # For now, this reports the framework structure.
    print(f"\n  Pipeline benchmark framework ready.")
    print(f"  Dataset: {dataset_dir}")
    print(f"  Modules: {result.modules}")
    print(f"  Model: {result.model}")
    print(f"  Backend: {result.backend}")
    print(f"\n  ⚠ Full pipeline integration pending (step 10).")

    return result


def main():
    parser = argparse.ArgumentParser(description="AION Pipeline Benchmark")
    parser.add_argument(
        "--dataset",
        default=r"C:\Users\Tarun J\Downloads\New Dataset\IAI",
        help="Path to the IAI dataset directory",
    )
    parser.add_argument(
        "--profile",
        default="LAPTOP_FAST",
        help="Runtime profile to use",
    )
    args = parser.parse_args()

    result = run_pipeline_benchmark(args.dataset, args.profile)
    print_report(result)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
