"""
Paper specs and fake artifacts for v3 testing.

Provides pre-built DocumentArtifact stubs and paper specs matching
the real exam types (IAT1, IAT2, ELECTIVE_3MOD). No PDF parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# -----------------------------------------------------------------------------
# DocumentArtifact stubs
# -----------------------------------------------------------------------------


@dataclass
class StubTextBlock:
    id: str
    text: str
    page: int
    block_role: str = "BODY"
    source_pdf_sha256: str = "stub"


@dataclass
class StubFigure:
    id: str
    page: int
    module_id: str
    provenance_score: float = 0.85
    caption: str = "Diagram from module"
    image_path: str = "/tmp/stub_figure.png"


@dataclass
class StubArtifact:
    text_blocks: List[StubTextBlock] = field(default_factory=list)
    figures: List[StubFigure] = field(default_factory=list)
    tables: List = field(default_factory=list)
    equations: List = field(default_factory=list)
    title: str = ""


# -----------------------------------------------------------------------------
# Fake artifact builder
# -----------------------------------------------------------------------------


def make_fake_artifact(
    module_count: int,
    pages_per_module: int = 10,
    figures_per_module: int = 2,
    subject_prefix: str = "Subject",
) -> StubArtifact:
    """
    Build a fake artifact with N modules, each having M pages and K figures.
    Text content is realistic-looking academic prose so the KG builder
    can extract triples from it.
    """
    blocks: List[StubTextBlock] = []
    figures: List[StubFigure] = []

    for module_idx in range(1, module_count + 1):
        module_start = (module_idx - 1) * pages_per_module + 1

        for page_offset in range(pages_per_module):
            page = module_start + page_offset
            block_id = f"m{module_idx}_p{page}_b1"
            text = (
                f"{subject_prefix} Module {module_idx} discusses fundamental "
                f"concepts on page {page}. The primary topic consists of "
                f"architecture and design principles. Virtualization requires "
                f"a hypervisor and enables isolation. The system includes "
                f"compute, storage, and networking components. Modern systems "
                f"is used for scalable deployment of applications."
            )
            blocks.append(StubTextBlock(
                id=block_id, text=text, page=page, block_role="BODY",
            ))

        for fig_num in range(figures_per_module):
            fig_id = f"m{module_idx}_fig_{fig_num + 1}"
            figures.append(StubFigure(
                id=fig_id,
                page=module_start + fig_num * 3,
                module_id=f"module_{module_idx}",
                caption=f"Figure {fig_num + 1}: {subject_prefix} Module {module_idx} schematic",
            ))

    return StubArtifact(
        text_blocks=blocks,
        figures=figures,
        title=f"{subject_prefix} Course Materials",
    )


# -----------------------------------------------------------------------------
# Paper specs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class V3PaperSpec:
    """Minimal PaperSpec matching the real interface."""
    exam_type: str
    module_count: int
    questions_per_module: int
    marks_per_question: int
    co_count: int = 5

    @property
    def total_questions(self) -> int:
        return self.module_count * self.questions_per_module

    def allocate_partitions(self, marks_split: List[List[int]]) -> Dict:
        """Mirror the real allocate_partitions API."""
        if len(marks_split) != self.total_questions:
            raise ValueError(
                f"marks_split has {len(marks_split)} entries, "
                f"expected {self.total_questions}"
            )
        assignments = {}
        cumulative = 0
        for m in range(self.module_count):
            for q in range(self.questions_per_module):
                partition = marks_split[cumulative]
                if sum(partition) != self.marks_per_question:
                    raise ValueError(
                        f"partition {partition} sums to {sum(partition)}, "
                        f"expected {self.marks_per_question}"
                    )
                assignments[(m, q)] = partition
                cumulative += 1
        return assignments


# -----------------------------------------------------------------------------
# Standard specs
# -----------------------------------------------------------------------------


IAT1_SPEC = V3PaperSpec(
    exam_type="IAT1",
    module_count=5,
    questions_per_module=2,
    marks_per_question=10,
    co_count=5,
)

IAT2_SPEC = V3PaperSpec(
    exam_type="IAT2",
    module_count=5,
    questions_per_module=2,
    marks_per_question=10,
    co_count=5,
)

ELECTIVE_3MOD_SPEC = V3PaperSpec(
    exam_type="ELECTIVE_3MOD",
    module_count=3,
    questions_per_module=3,
    marks_per_question=10,
    co_count=3,
)


from core.contracts.paper_spec import STANDARD_MARKS_SPLITS

IAT1_MARKS_SPLIT = STANDARD_MARKS_SPLITS["IAT1"]
IAT2_MARKS_SPLIT = STANDARD_MARKS_SPLITS["IAT2"]
ELECTIVE_3MOD_MARKS_SPLIT = STANDARD_MARKS_SPLITS["ELECTIVE_3MOD"]


# -----------------------------------------------------------------------------
# Test subject catalog
# -----------------------------------------------------------------------------


@dataclass
class TestSubject:
    __test__ = False
    name: str
    spec: V3PaperSpec
    marks_split: List[List[int]]
    artifact: StubArtifact


def build_test_catalog() -> List[TestSubject]:
    """Return the standard four test subjects."""
    return [
        TestSubject(
            name="Satellite Communication",
            spec=IAT1_SPEC,
            marks_split=IAT1_MARKS_SPLIT,
            artifact=make_fake_artifact(5, subject_prefix="Satellite Communication"),
        ),
        TestSubject(
            name="Cloud Computing",
            spec=IAT2_SPEC,
            marks_split=IAT2_MARKS_SPLIT,
            artifact=make_fake_artifact(5, subject_prefix="Cloud Computing"),
        ),
        TestSubject(
            name="VLSI Design",
            spec=ELECTIVE_3MOD_SPEC,
            marks_split=ELECTIVE_3MOD_MARKS_SPLIT,
            artifact=make_fake_artifact(3, subject_prefix="VLSI Design"),
        ),
        TestSubject(
            name="Unseen Subject",
            spec=IAT1_SPEC,
            marks_split=IAT1_MARKS_SPLIT,
            artifact=make_fake_artifact(5, subject_prefix="Unseen"),
        ),
    ]
