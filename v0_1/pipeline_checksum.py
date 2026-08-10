"""
AION Pipeline Checksum Tracker
==============================
Computes sha256 checksums across every stage boundary for instant root-cause tracing:
request_hash -> extraction_hash -> retrieval_hash -> question_plan_hash -> vko_hash -> final_question_hash.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class PipelineChecksumTracker:
    """Tracks sha256 checksums across pipeline stages."""
    request_hash: str = ""
    extraction_hash: str = ""
    retrieval_hash: str = ""
    question_plan_hash: str = ""
    vko_hash: str = ""
    final_question_hash: str = ""
    trace_events: list[Dict[str, Any]] = field(default_factory=list)

    def compute_request_hash(self, payload: Dict[str, Any]) -> str:
        self.request_hash = self._hash_payload(payload)
        self._log("request", self.request_hash)
        return self.request_hash

    def compute_extraction_hash(self, raw_text: str) -> str:
        self.extraction_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()[:16]
        self._log("extraction", self.extraction_hash)
        return self.extraction_hash

    def compute_retrieval_hash(self, chunks: list[str]) -> str:
        combined = "\n".join(chunks)
        self.retrieval_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]
        self._log("retrieval", self.retrieval_hash)
        return self.retrieval_hash

    def compute_vko_hash(self, vko_data: Dict[str, Any]) -> str:
        self.vko_hash = self._hash_payload(vko_data)
        self._log("vko", self.vko_hash)
        return self.vko_hash

    def compute_final_question_hash(self, question_text: str) -> str:
        self.final_question_hash = hashlib.sha256(question_text.encode("utf-8")).hexdigest()[:16]
        self._log("final_question", self.final_question_hash)
        return self.final_question_hash

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_hash": self.request_hash,
            "extraction_hash": self.extraction_hash,
            "retrieval_hash": self.retrieval_hash,
            "question_plan_hash": self.question_plan_hash,
            "vko_hash": self.vko_hash,
            "final_question_hash": self.final_question_hash,
            "trace_events": self.trace_events,
        }

    def _log(self, stage: str, hash_val: str) -> None:
        event = {"stage": stage, "hash": hash_val}
        self.trace_events.append(event)
        print(f"[CHECKSUM] {stage.upper()}_HASH: {hash_val}")

    @staticmethod
    def _hash_payload(payload: Dict[str, Any]) -> str:
        try:
            raw = json.dumps(payload, sort_keys=True, default=str)
        except Exception:
            raw = str(payload)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
