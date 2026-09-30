"""
GeneratedQuestion — the output contract of the Writing Agent.

Every downstream agent consumes this structure. The Evaluation Agent
audits it, the Refinement Agent repairs it, the Checking Agent gates it,
the DOCX exporter renders it.

No field is optional except `diagram_request` and `image_path`. The
Writing Agent must produce a question with a solution and marking scheme,
or the slot is a failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GeneratedQuestion:
    # Identity
    slot_id: str
    module_id: str
    global_q_idx: int

    # Content
    question_text: str
    solution: str
    marking_scheme: List[Dict[str, Any]]

    # Pedagogical metadata (echoed from the plan for downstream use)
    marks: int = 0
    partition: List[int] = field(default_factory=list)
    bloom: str = "L2"
    co: str = "CO1"
    topic: str = ""

    # Visual contract
    references_image: bool = False
    image_path: Optional[str] = None
    figure_caption: Optional[str] = None
    diagram_request: Optional[Dict[str, Any]] = None

    # Generation provenance
    generation_source: str = "local_llm"     # local_llm | local_vlm | api
    attempts: int = 1

    # Telemetry (populated by agents)
    telemetry: Dict[str, Any] = field(default_factory=dict)

    @property
    def first_word(self) -> str:
        """First token of the question text (lowercased, stripped of punctuation)."""
        if not self.question_text:
            return ""
        token = self.question_text.strip().split()[0] if self.question_text.strip() else ""
        return token.rstrip(".,:;!?").lower()

    @property
    def is_multi_part(self) -> bool:
        return len(self.partition) > 1


@dataclass
class SlotFailure:
    """A slot that failed during the Writing Agent's run."""
    slot_id: str
    failure_code: str
    failure_detail: str


@dataclass
class WritingAgentOutput:
    """The payload of the Writing Agent's successful AgentResult."""
    questions: List[GeneratedQuestion]
    failures: List[SlotFailure] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        return len(self.failures) == 0
