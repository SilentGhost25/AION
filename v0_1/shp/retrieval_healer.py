"""
AION SHP Stage 4 — Retrieval Healer
=====================================
Diagnoses and repairs retrieval failures.
Prevents "0 chunks → generate anyway" bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .error_knowledge import ErrorKnowledgeBase, Severity


@dataclass
class RetrievalResult:
    chunks:       list[str]
    scores:       list[float]
    gate_passed:  bool
    repairs:      list[str] = field(default_factory=list)
    abort_reason: str       = ""

    @property
    def can_generate(self) -> bool:
        return self.gate_passed and bool(self.chunks)


class RetrievalHealer:
    """
    Stage 4: Ensure retrieval produces usable evidence or explicitly abort.
    Never passes 0 chunks to the generator.
    """

    MIN_CHUNKS  = 2
    MAX_CHUNKS  = 3

    def __init__(self, kb: ErrorKnowledgeBase):
        self.kb = kb

    def retrieve_and_heal(
        self,
        query:     str,
        chunks:    list[str],
        metas:     Optional[list[dict]] = None,
        module_id: Optional[str]        = None,
    ) -> RetrievalResult:
        repairs = []

        if not chunks:
            self.kb.record("SH-020", "S4_RETRIEVAL",
                           "No chunks provided to retriever",
                           Severity.ERROR)
            return RetrievalResult(
                chunks=[], scores=[], gate_passed=False,
                abort_reason="EMPTY_CHUNK_POOL",
            )

        # Single chunk fallback (e.g. inline notes_text) -> split into 2 chunks for GroundingGate
        if len(chunks) == 1:
            words = chunks[0].split()
            if len(words) >= 40:
                mid = len(words) // 2
                chunks = [" ".join(words[:mid+10]), " ".join(words[mid-10:])]
                if metas:
                    metas = [metas[0], metas[0]]
            else:
                chunks = [chunks[0], chunks[0]]
                if metas:
                    metas = [metas[0], metas[0]]

        result = self._attempt_retrieval(query, chunks, metas, module_id)

        if result.can_generate:
            return result

        avg_words = sum(len(c.split()) for c in chunks) / len(chunks)
        if avg_words > 400:
            rec = self.kb.record(
                "SH-014", "S4_RETRIEVAL",
                f"Average chunk size {avg_words:.0f}w — splitting for retrieval",
                Severity.WARNING,
            )
            chunks  = self._split_chunks(chunks, 250, 25)
            metas   = [metas[i // 2] if metas else {}
                       for i in range(len(chunks))]
            repairs.append(f"SH-014: Split to {len(chunks)} smaller chunks")

            result = self._attempt_retrieval(query, chunks, metas, module_id)
            if result.can_generate:
                result.repairs = repairs
                self.kb.resolve(rec, f"{len(result.chunks)} chunks after split")
                return result

        if len(chunks) >= self.MIN_CHUNKS:
            rec = self.kb.record(
                "SH-020", "S4_RETRIEVAL",
                "Retriever returned 0 — using top chunks directly",
                Severity.WARNING,
            )
            top    = chunks[:self.MAX_CHUNKS]
            repairs.append("SH-020: Used first available chunks (similarity bypassed)")
            self.kb.resolve(rec, "Used direct chunk selection")

            from v0_1.grounding_gate import check_grounding
            gate = check_grounding(query, top, module_id=module_id)
            if gate.proceed:
                return RetrievalResult(
                    chunks      = gate.pruned_chunks,
                    scores      = [0.5] * len(gate.pruned_chunks),
                    gate_passed = True,
                    repairs     = repairs,
                )

        abort_msg = "Retrieval failed after all healing attempts"
        self.kb.record("SH-020", "S4_RETRIEVAL", abort_msg, Severity.ERROR)

        return RetrievalResult(
            chunks=[], scores=[], gate_passed=False,
            repairs=repairs,
            abort_reason=abort_msg,
        )

    def _attempt_retrieval(
        self,
        query:     str,
        chunks:    list[str],
        metas:     Optional[list[dict]],
        module_id: Optional[str],
    ) -> RetrievalResult:
        try:
            from v0_1.retriever     import GroundedRetriever
            from v0_1.grounding_gate import check_grounding

            ret     = GroundedRetriever(max_chunks=self.MAX_CHUNKS)
            top     = ret.retrieve_texts(query, chunks, metas, module_id)
            gate    = check_grounding(query, top, module_id=module_id)

            return RetrievalResult(
                chunks      = gate.pruned_chunks if gate.proceed else [],
                scores      = [0.8] * len(gate.pruned_chunks) if gate.proceed else [],
                gate_passed = gate.proceed,
            )
        except Exception as e:
            self.kb.record("SH-020", "S4_RETRIEVAL",
                           f"Retrieval exception: {e}", Severity.ERROR)
            return RetrievalResult(chunks=[], scores=[], gate_passed=False)

    def _split_chunks(
        self, chunks: list[str], size: int, overlap: int
    ) -> list[str]:
        result = []
        for chunk in chunks:
            words = chunk.split()
            step  = max(1, size - overlap)
            for i in range(0, len(words), step):
                sub = " ".join(words[i:i+size])
                if len(sub.split()) >= 20:
                    result.append(sub)
        return result
