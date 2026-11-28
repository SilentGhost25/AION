"""
Tests for Module Segmenter Resolution & Propagation
Validates:
1. merge_artifacts generates sequential, non-overlapping ModuleSegments with defensive page offsetting.
2. Single-PDF auto-segmentation in pipeline_bridge when module_segments is absent.
3. Fail-closed invariant for empty synthetic/mock artifacts without segments.
"""

import pytest
from core.contracts.document_artifact import (
    DocumentArtifact,
    TextBlock,
    FigureArtifact,
    TableArtifact,
    EquationArtifact,
)
from core.extraction.artifact_cache import merge_artifacts
from core.extraction.module_segmenter import ModuleSegmenter, ModuleSegment, SegmentationStrategy
from core.generation.agents.pipeline_bridge import run_v3_pipeline, _build_kg_for_artifact
from core.generation.paper_spec_resolver import PaperSpecResolver


def test_merge_artifacts_produces_non_overlapping_segments():
    art1 = DocumentArtifact(
        source_pdf_sha256="sha1",
        source_pdf_path="p1.pdf",
        text_blocks=[
            TextBlock(id="b1", text="Intro to Module 1", page=1),
            TextBlock(id="b2", text="Details of Module 1", page=3),
        ],
        title="Module 1 Notes",
    )
    art2 = DocumentArtifact(
        source_pdf_sha256="sha2",
        source_pdf_path="p2.pdf",
        text_blocks=[
            TextBlock(id="b3", text="Intro to Module 2", page=1),
            TextBlock(id="b4", text="Details of Module 2", page=2),
        ],
        title="Module 2 Notes",
    )

    merged = merge_artifacts([art1, art2], subject="Sample Subject")

    assert hasattr(merged, "module_segments")
    segments = merged.module_segments
    assert len(segments) == 2
    assert segments[0].module_id == 1
    assert segments[0].page_start == 1
    assert segments[0].page_end == 3

    assert segments[1].module_id == 2
    assert segments[1].page_start == 4
    assert segments[1].page_end == 5

    # Check that text blocks in art2 were offset
    art2_blocks = [b for b in merged.text_blocks if b.id.startswith("m2_")]
    assert len(art2_blocks) == 2
    assert art2_blocks[0].page == 4
    assert art2_blocks[1].page == 5


def test_auto_segmentation_single_pdf_artifact():
    # 10-page artifact with no explicit module segments
    blocks = [
        TextBlock(id=f"b_{p}", text=f"Page {p} content on advanced algorithms", page=p)
        for p in range(1, 11)
    ]
    artifact = DocumentArtifact(
        source_pdf_sha256="sha_single",
        source_pdf_path="single.pdf",
        text_blocks=blocks,
        title="Full Syllabus",
    )
    assert not getattr(artifact, "module_segments", None)

    segments = ModuleSegmenter.segment(artifact)
    assert len(segments) == 5
    assert segments[0].module_id == 1
    assert segments[0].page_start == 1
    assert segments[0].page_end == 2
    assert segments[-1].module_id == 5
    assert segments[-1].page_end == 10


def test_build_kg_with_auto_segmented_artifact():
    blocks = [
        TextBlock(id=f"b_{p}", text=f"Page {p} B-Tree indexing and SQL queries", page=p)
        for p in range(1, 11)
    ]
    artifact = DocumentArtifact(
        source_pdf_sha256="sha_kg",
        source_pdf_path="kg.pdf",
        text_blocks=blocks,
        title="Database Systems",
    )
    spec = PaperSpecResolver.resolve("IAT1")
    # Before resolution, building without segments on real artifact fails closed if None passed
    with pytest.raises(ValueError, match="SEGMENTS_ABSENT"):
        _build_kg_for_artifact(artifact, spec, module_segments=None)

    # But once auto-segmented or passed segments, it succeeds
    artifact.module_segments = ModuleSegmenter.segment(artifact)
    kg = _build_kg_for_artifact(artifact, spec, module_segments=None)
    assert kg is not None
    assert getattr(kg, "module_segments_source", None) == "artifact_module_segments"
