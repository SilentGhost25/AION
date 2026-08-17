"""
AION Core — Pipeline Integrity Contracts
==========================================
Machine-enforced pre-generation and post-generation gates.

These dataclasses are the single source of truth for pipeline readiness
and paper integrity state. They block — they do not merely warn.

Importable by:
  - v0_1/main.py   (CLI pipeline)
  - core/api/*     (server pipeline)
  - runtime/*      (laptop profile)
  - tests/*        (integration tests)

Never put PipelineReadiness inline in main.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PipelineReadiness:
    """
    Pre-generation gate state.

    Built after ingestion is complete, before the ThreadPoolExecutor starts.
    If not ready, the pipeline MUST raise — never proceed with generation.
    """
    # Dataset state
    dataset_files_expected : int
    dataset_files_found    : int
    modules_expected       : List[int]    # e.g. [1, 2, 3, 4, 5]
    modules_found          : List[int]    # actual parsed module numbers
    ingestion_errors       : List[dict]   # {"file": ..., "error": ...}

    # Model state
    model_name             : str
    model_capability       : str          # "chat" | "generate" | "none"

    # Plan state
    required_slot_ids      : List[str]    # from SlotOrchestrator plan
    export_gate_ready      : bool

    @property
    def modules_missing(self) -> List[int]:
        return sorted(set(self.modules_expected) - set(self.modules_found))

    @property
    def modules_unexpected(self) -> List[int]:
        return sorted(set(self.modules_found) - set(self.modules_expected))

    @property
    def ready(self) -> bool:
        return (
            len(self.modules_missing) == 0
            and len(self.modules_unexpected) == 0
            and len(self.ingestion_errors) == 0
            and self.model_capability != "none"
            and len(self.required_slot_ids) > 0
            and self.export_gate_ready
        )

    def print_gate(self) -> None:
        status = "READY" if self.ready else "BLOCKED"
        sep = "=" * 60
        print(sep)
        print("AION PRE-GENERATION CONTRACT")
        print(sep)
        print(f"  Dataset files        : {self.dataset_files_found}/{self.dataset_files_expected}")
        print(f"  Expected modules     : {sorted(self.modules_expected)}")
        print(f"  Found modules        : {sorted(self.modules_found)}")
        print(f"  Missing modules      : {self.modules_missing or 'none'}")
        print(f"  Unexpected modules   : {self.modules_unexpected or 'none'}")
        print(f"  Ingestion errors     : {len(self.ingestion_errors)}")
        for e in self.ingestion_errors:
            print(f"    ✗ {e.get('file', '?')}: {e.get('error', '?')}")
        print(f"  Model                : {self.model_name}")
        print(f"  Model API            : {self.model_capability}")
        print(f"  Required slots       : {len(self.required_slot_ids)}")
        print(f"  ExportGate           : {'ENABLED' if self.export_gate_ready else 'DISABLED'}")
        print(f"  STATUS               : {status}")
        print(sep)

    def raise_if_blocked(self) -> None:
        """Call after print_gate(). Raises RuntimeError if not ready."""
        if not self.ready:
            raise RuntimeError(
                f"[PRE-GENERATION GATE] Pipeline BLOCKED — see contract above.\n"
                f"  Missing modules  : {self.modules_missing}\n"
                f"  Ingestion errors : {len(self.ingestion_errors)}\n"
                f"  Model capability : {self.model_capability}"
            )


@dataclass
class GenerationIntegrity:
    """
    Post-generation gate state.

    Built after all slots have been generated and ExportGate has run.
    Only returned to the caller if paper_ready is True.
    """
    # Slot completeness
    expected_slots       : int
    generated_slots      : int
    missing_slot_ids     : List[str] = field(default_factory=list)
    extra_slot_ids       : List[str] = field(default_factory=list)

    # Module completeness
    modules_expected     : int = 0
    modules_generated    : int = 0

    # Gate results
    export_gate_pass     : bool = False
    marks_integrity_pass : bool = False
    co_integrity_pass    : bool = False
    bloom_integrity_pass : bool = False
    provenance_pass      : bool = False

    # Performance (optional — populated by LAPTOP_FAST profile)
    generation_time_s    : float = 0.0
    extraction_time_s    : float = 0.0
    peak_ram_gb          : float = 0.0
    retries_total        : int   = 0

    @property
    def paper_ready(self) -> bool:
        return (
            len(self.missing_slot_ids) == 0
            and len(self.extra_slot_ids) == 0
            and self.modules_generated == self.modules_expected
            and self.export_gate_pass
        )

    def print_gate(self) -> None:
        status = "PAPER_READY" if self.paper_ready else "PAPER_BLOCKED"
        sep = "=" * 60
        print(sep)
        print("AION GENERATION INTEGRITY")
        print(sep)
        print(f"  Expected slots       : {self.expected_slots}")
        print(f"  Generated slots      : {self.generated_slots}")
        print(f"  Missing slots        : {len(self.missing_slot_ids)}")
        if self.missing_slot_ids:
            for sid in self.missing_slot_ids:
                print(f"    ✗ {sid}")
        print(f"  Extra slots          : {len(self.extra_slot_ids)}")
        print(f"  Modules              : {self.modules_generated}/{self.modules_expected}")
        print(f"  ExportGate           : {'PASS' if self.export_gate_pass else 'FAIL'}")
        print(f"  Marks integrity      : {'PASS' if self.marks_integrity_pass else 'FAIL'}")
        print(f"  CO integrity         : {'PASS' if self.co_integrity_pass else 'FAIL'}")
        print(f"  Bloom integrity      : {'PASS' if self.bloom_integrity_pass else 'FAIL'}")
        print(f"  Provenance           : {'PASS' if self.provenance_pass else 'FAIL'}")
        if self.generation_time_s > 0:
            print(f"  Generation time      : {self.generation_time_s:.1f}s")
            print(f"  Extraction time      : {self.extraction_time_s:.1f}s")
            print(f"  Peak RAM             : {self.peak_ram_gb:.2f} GB")
            print(f"  Total retries        : {self.retries_total}")
        print(f"  STATUS               : {status}")
        print(sep)

    def raise_if_blocked(self) -> None:
        """Call after print_gate(). Raises RuntimeError if not paper_ready."""
        if not self.paper_ready:
            raise RuntimeError(
                f"[POST-GENERATION GATE] Paper BLOCKED — integrity check failed.\n"
                f"  Missing slots    : {self.missing_slot_ids}\n"
                f"  ExportGate       : {'PASS' if self.export_gate_pass else 'FAIL'}"
            )
