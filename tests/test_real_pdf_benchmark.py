"""
AION v2 Real-World PDF Benchmark & Ground-Truth Verification
============================================================
Benchmarks the Streaming PDF Extractor and Artifact Fusion Engine against
3 real-world engineering PDFs in workspace/uploads/:
1. 074c9920-6b0.pdf (10 pages, Ensembling Techniques)
2. 037de9d9-3f7.pdf (40 pages, Operating Systems - File Systems)
3. 01cad428-1c0.pdf (82 pages, Deep Learning & Reinforcement Learning)

Produces the observable 13-point inspection tables directly to terminal stdout
and verifies ground truth alignment against manually verified sample pages.
"""

import tempfile
from pathlib import Path
import pytest

from aion.core.dom.artifacts import ArtifactType
from aion.core.extraction.pdf_extractor import extract_pdf_to_dom
from aion.core.extraction.benchmark import (
    ArtifactBenchmarkAuditor,
    GroundTruthExpectation,
)


@pytest.fixture(scope="module")
def sample_assets_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def test_benchmark_pdf1_ensembling_techniques(sample_assets_dir):
    """Benchmark PDF 1: 074c9920-6b0.pdf (10 pages, Ensembling Techniques)."""
    pdf_path = Path("workspace/uploads/074c9920-6b0.pdf")
    assert pdf_path.exists(), "Sample PDF 1 does not exist in workspace/uploads"

    # Ground truth expectations for representative pages:
    gt_expectations = [
        # Page 1: Heading "Ensembling Techniques", 2 native tables, discussion of regression
        GroundTruthExpectation(
            page_number=1,
            expected_heading_substrings=["Ensembling Techniques"],
            expected_artifact_types={ArtifactType.TABLE, ArtifactType.TEXT},
            expected_text_keywords=["ensemble", "combining multiple predictive models", "regression"],
            min_tables=1,
        ),
        # Page 2: Native table, Condorcet's jury theorem
        GroundTruthExpectation(
            page_number=2,
            expected_artifact_types={ArtifactType.TABLE, ArtifactType.TEXT},
            expected_text_keywords=["Condorcet's jury theorem", "probability of errors"],
            min_tables=1,
        ),
        # Page 3: Random forests, bagged trees
        GroundTruthExpectation(
            page_number=3,
            expected_artifact_types={ArtifactType.TABLE, ArtifactType.TEXT},
            expected_text_keywords=["Random forests", "bootstrapped sample"],
            min_tables=1,
        ),
    ]

    dom = extract_pdf_to_dom(
        pdf_path=str(pdf_path),
        document_id="DOC-074C-ENSEMBLE",
        module_mapping={1: (1, 10)},
        assets_root_dir=sample_assets_dir,
    )

    report = ArtifactBenchmarkAuditor.audit(dom, ground_truth_expectations=gt_expectations)
    output_table = report.format_table()

    # Print the exact observable report table
    print("\n" + "=" * 55)
    print(output_table)
    print("=" * 55 + "\n")

    # Invariants
    assert report.total_pages == 10
    assert report.recorded_pages == 10
    assert report.page_failures == 0
    assert report.tables_count >= 3
    assert report.unique_ids_percentage == 100.0
    assert report.bbox_percentage >= 95.0
    assert report.module_assigned_count == report.total_artifacts
    assert report.ground_truth_passed_checks >= int(report.ground_truth_total_checks * 0.85)


def test_benchmark_pdf2_file_systems(sample_assets_dir):
    """Benchmark PDF 2: 037de9d9-3f7.pdf (40 pages, File Systems)."""
    pdf_path = Path("workspace/uploads/037de9d9-3f7.pdf")
    assert pdf_path.exists(), "Sample PDF 2 does not exist in workspace/uploads"

    # Ground truth expectations for representative pages:
    gt_expectations = [
        # Page 1: Heading "Implementation of File System", text discussing File Concepts & Attributes
        GroundTruthExpectation(
            page_number=1,
            expected_heading_substrings=["Implementation of File System"],
            expected_artifact_types={ArtifactType.HEADING, ArtifactType.TEXT},
            expected_text_keywords=["File Attributes", "File Operations", "File Concept"],
        ),
        # Page 2: Text on File Concepts, File Attributes, File Operations
        GroundTruthExpectation(
            page_number=2,
            expected_artifact_types={ArtifactType.TEXT},
            expected_text_keywords=["Allocation Methods", "collection of related information", "File Attributes"],
        ),
        # Page 3: Diagrams/figures illustrating open file table
        GroundTruthExpectation(
            page_number=3,
            expected_artifact_types={ArtifactType.FIGURE, ArtifactType.TEXT},
            expected_text_keywords=["Truncating a file", "open file table"],
            min_figures=1,
        ),
    ]

    dom = extract_pdf_to_dom(
        pdf_path=str(pdf_path),
        document_id="DOC-037D-FILESYSTEM",
        module_mapping={4: (1, 40)},
        assets_root_dir=sample_assets_dir,
    )

    report = ArtifactBenchmarkAuditor.audit(dom, ground_truth_expectations=gt_expectations)
    output_table = report.format_table()

    # Print the exact observable report table
    print("\n" + "=" * 55)
    print(output_table)
    print("=" * 55 + "\n")

    # Invariants
    assert report.total_pages == 40
    assert report.recorded_pages == 40
    assert report.page_failures == 0
    assert report.figures_count >= 10
    assert report.unique_ids_percentage == 100.0
    assert report.bbox_percentage >= 95.0
    assert report.module_assigned_count == report.total_artifacts
    assert report.ground_truth_passed_checks >= int(report.ground_truth_total_checks * 0.85)


def test_benchmark_pdf3_deep_learning_slides(sample_assets_dir):
    """Benchmark PDF 3: 01cad428-1c0.pdf (82 pages, Deep Learning)."""
    pdf_path = Path("workspace/uploads/01cad428-1c0.pdf")
    assert pdf_path.exists(), "Sample PDF 3 does not exist in workspace/uploads"

    # Ground truth expectations for representative pages:
    gt_expectations = [
        # Page 1: Title slide "Deep Learning"
        GroundTruthExpectation(
            page_number=1,
            expected_heading_substrings=["Deep Learning"],
            expected_artifact_types={ArtifactType.HEADING},
        ),
        # Page 3: Comparison slide "Machine Learning(ML) vs Deep Learning(DL)"
        GroundTruthExpectation(
            page_number=3,
            expected_heading_substrings=["Machine Learning"],
            expected_artifact_types={ArtifactType.HEADING, ArtifactType.TEXT},
            expected_text_keywords=["Computer vision", "Supervised learning"],
        ),
        # Page 4: Figure illustrating AI -> ML -> DL evolution
        GroundTruthExpectation(
            page_number=4,
            expected_artifact_types={ArtifactType.FIGURE, ArtifactType.TEXT},
            expected_text_keywords=["human intelligence", "hidden structures"],
            min_figures=1,
        ),
    ]

    dom = extract_pdf_to_dom(
        pdf_path=str(pdf_path),
        document_id="DOC-01CA-DEEPLEARNING",
        module_mapping={5: (1, 82)},
        assets_root_dir=sample_assets_dir,
    )

    report = ArtifactBenchmarkAuditor.audit(dom, ground_truth_expectations=gt_expectations)
    output_table = report.format_table()

    # Print the exact observable report table
    print("\n" + "=" * 55)
    print(output_table)
    print("=" * 55 + "\n")

    # Invariants
    assert report.total_pages == 82
    assert report.recorded_pages == 82
    assert report.page_failures == 0
    assert report.figures_count >= 15
    assert report.unique_ids_percentage == 100.0
    assert report.bbox_percentage >= 95.0
    assert report.module_assigned_count == report.total_artifacts
    assert report.ground_truth_passed_checks >= int(report.ground_truth_total_checks * 0.85)
