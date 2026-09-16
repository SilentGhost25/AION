"""
AION v2 Observable 13-Point Artifact Inventory Benchmark
========================================================
Audits an extracted DocumentDOM against the 13 required architectural checks.
Formats the exact observable inspection table requested for Phase 2.
Also compares extracted DOM artifacts against manually verified ground truth samples
to distinguish inventory presence from extraction correctness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from aion.core.dom.artifacts import (
    ArtifactType,
    BaseArtifact,
    EquationArtifact,
    FigureArtifact,
    HeadingArtifact,
    TableArtifact,
    TextArtifact,
)
from aion.core.dom.document_dom import (
    DocumentDOM,
    PageState,
)
from aion.core.dom.relationships import RelationshipType


@dataclass
class GroundTruthExpectation:
    """
    Manually verified expectations for a representative page in a sample PDF.
    """
    page_number: int
    expected_heading_substrings: List[str] = field(default_factory=list)
    expected_artifact_types: Set[ArtifactType] = field(default_factory=set)
    expected_text_keywords: List[str] = field(default_factory=list)
    min_tables: int = 0
    min_figures: int = 0
    min_equations: int = 0


@dataclass
class BenchmarkReport:
    """
    Structured 13-point audit report for a single document.
    """
    doc_title: str
    total_pages: int
    recorded_pages: int
    text_count: int
    h1_count: int
    h2_count: int
    h3_count: int
    eq_verified: int
    eq_candidate: int
    eq_failed: int
    figures_count: int
    linked_captions_count: int
    tables_count: int
    academic_semantics_count: int
    module_assigned_count: int
    section_assigned_count: int
    total_artifacts: int
    bbox_percentage: float
    unique_ids_percentage: float
    cross_refs_count: int
    page_failures: int
    partial_pages: int
    ocr_required_pages: int

    # Ground truth alignment
    ground_truth_total_checks: int = 0
    ground_truth_passed_checks: int = 0
    ground_truth_failures: List[str] = field(default_factory=list)

    def format_table(self) -> str:
        """
        Produce the exact observable ASCII / Markdown audit table requested.
        """
        pg_status = "PASS" if self.recorded_pages == self.total_pages and self.page_failures == 0 else "FAIL"
        txt_status = "PASS" if self.text_count > 0 else "REVIEW"
        h_status = "PASS" if (self.h1_count + self.h2_count + self.h3_count) > 0 else "REVIEW"
        eq_status = "PASS" if self.eq_verified > 0 else ("INFO" if self.eq_candidate > 0 else "N/A")
        fig_status = "PASS" if self.figures_count > 0 else "INFO"
        cap_status = "PASS" if self.figures_count == 0 or (self.linked_captions_count > 0) else "REVIEW"
        tbl_status = "PASS" if self.tables_count > 0 else "INFO"
        sem_status = "PASS" if self.academic_semantics_count > 0 else "INFO"
        mod_status = "PASS" if self.module_assigned_count == self.total_artifacts else "FAIL"
        sec_status = "PASS" if self.section_assigned_count >= int(self.total_artifacts * 0.70) else "REVIEW"
        bbox_status = "PASS" if self.bbox_percentage >= 95.0 else "REVIEW"
        id_status = "PASS" if self.unique_ids_percentage == 100.0 else "FAIL"
        xref_status = "PASS" if self.cross_refs_count > 0 else "INFO"

        cap_str = f"{self.linked_captions_count}/{self.figures_count}" if self.figures_count else "0/0"
        mod_str = f"{self.module_assigned_count}/{self.total_artifacts}"
        sec_str = f"{self.section_assigned_count}/{self.total_artifacts}"

        lines = [
            f"DOCUMENT: {self.doc_title}",
            f"PAGES:    {self.total_pages}",
            "",
            f"{'METRIC':<25} {'COUNT':<15} {'STATUS'}",
            "-" * 50,
            f"{'Pages':<25} {f'{self.recorded_pages}/{self.total_pages}':<15} {pg_status}",
            f"{'Text artifacts':<25} {self.text_count:<15} {txt_status}",
            f"{'H1 (Module/Major)':<25} {self.h1_count:<15} {h_status}",
            f"{'H2 (Section)':<25} {self.h2_count:<15} {h_status}",
            f"{'H3 (Subsection)':<25} {self.h3_count:<15} {h_status}",
            f"{'Equations (Verified)':<25} {self.eq_verified:<15} {eq_status}",
            f"{'Equations (Candidate)':<25} {self.eq_candidate:<15} INFO",
            f"{'Equations (Failed)':<25} {self.eq_failed:<15} {'WARN' if self.eq_failed > 0 else 'PASS'}",
            f"{'Figures':<25} {self.figures_count:<15} {fig_status}",
            f"{'Linked captions':<25} {cap_str:<15} {cap_status}",
            f"{'Tables':<25} {self.tables_count:<15} {tbl_status}",
            f"{'Academic semantics':<25} {self.academic_semantics_count:<15} {sem_status}",
            f"{'Module assignment':<25} {mod_str:<15} {mod_status}",
            f"{'Section assignment':<25} {sec_str:<15} {sec_status}",
            f"{'Bounding boxes':<25} {f'{self.bbox_percentage:.1f}%':<15} {bbox_status}",
            f"{'Unique artifact IDs':<25} {f'{self.unique_ids_percentage:.1f}%':<15} {id_status}",
            f"{'Cross references':<25} {self.cross_refs_count:<15} {xref_status}",
            "-" * 50,
            f"PAGE FAILURES:        {self.page_failures}",
            f"PARTIAL PAGES:        {self.partial_pages}",
            f"OCR REQUIRED:         {self.ocr_required_pages}",
        ]

        if self.ground_truth_total_checks > 0:
            pct = (self.ground_truth_passed_checks / self.ground_truth_total_checks) * 100.0
            lines.extend([
                "",
                f"GROUND TRUTH ALIGNMENT: {self.ground_truth_passed_checks}/{self.ground_truth_total_checks} ({pct:.1f}%)",
            ])
            for f in self.ground_truth_failures:
                lines.append(f"  [DISCREPANCY] {f}")

        return "\n".join(lines)


class ArtifactBenchmarkAuditor:
    """
    Executes the 13-point audit on any DocumentDOM.
    """

    @classmethod
    def audit(
        cls,
        dom: DocumentDOM,
        ground_truth_expectations: Optional[List[GroundTruthExpectation]] = None,
    ) -> BenchmarkReport:
        reg = dom.registry
        total_arts = reg.count()

        # 1. Page accounting
        total_pages = dom.total_pages
        audit_records = dom.page_audit
        rec_pages = len(audit_records)
        pg_fail = sum(1 for r in audit_records.values() if r.state == PageState.FAILED)
        pg_part = sum(1 for r in audit_records.values() if r.state == PageState.PARTIAL)
        pg_ocr = sum(1 for r in audit_records.values() if r.state == PageState.OCR_REQUIRED)

        # 2. Text count
        texts = reg.get_by_type(ArtifactType.TEXT)

        # 3. Headings by level
        headings = reg.get_by_type(ArtifactType.HEADING)
        h1 = sum(1 for h in headings if isinstance(h, HeadingArtifact) and h.level == 1)
        h2 = sum(1 for h in headings if isinstance(h, HeadingArtifact) and h.level == 2)
        h3 = sum(1 for h in headings if isinstance(h, HeadingArtifact) and h.level == 3)

        # 4. Equations by verification status
        equations = reg.get_by_type(ArtifactType.EQUATION)
        eq_ver = sum(1 for e in equations if isinstance(e, EquationArtifact) and e.verification_status == "VERIFIED")
        eq_cand = sum(1 for e in equations if isinstance(e, EquationArtifact) and e.verification_status == "CANDIDATE")
        eq_fail = sum(1 for e in equations if isinstance(e, EquationArtifact) and e.verification_status == "FAILED")

        # 5. Figures & Captions
        figures = reg.get_by_type(ArtifactType.FIGURE)
        fig_cnt = len(figures)
        linked_caps = sum(1 for f in figures if isinstance(f, FigureArtifact) and bool(f.caption.strip()))

        # 6. Tables
        tables = reg.get_by_type(ArtifactType.TABLE)
        tbl_cnt = len(tables)

        # 7. Academic semantics (Definitions, Examples, Procedures, Algorithms)
        definitions = reg.get_by_type(ArtifactType.DEFINITION)
        examples = reg.get_by_type(ArtifactType.EXAMPLE)
        procedures = reg.get_by_type(ArtifactType.PROCEDURE)
        algorithms = reg.get_by_type(ArtifactType.ALGORITHM)
        academic_sem = len(definitions) + len(examples) + len(procedures) + len(algorithms)

        # 8 & 9. Module & Section assignments
        all_artifacts = [reg.get(aid) for aid in reg._artifacts]
        valid_artifacts = [a for a in all_artifacts if a is not None]
        mod_assigned = sum(1 for a in valid_artifacts if a.module_index > 0)
        sec_assigned = sum(1 for a in valid_artifacts if bool(a.section_id))

        # 10. Bounding box coverage
        bbox_count = sum(1 for a in valid_artifacts if a.source_bbox is not None)
        bbox_pct = (bbox_count / max(1, len(valid_artifacts))) * 100.0

        # 11. Unique artifact IDs
        all_ids = [a.artifact_id for a in valid_artifacts]
        unique_ids = set(all_ids)
        unique_pct = (len(unique_ids) / max(1, len(all_ids))) * 100.0

        # 12. Cross references
        cross_refs = 0
        for edge_list in dom.relationships._outgoing.values():
            cross_refs += len(edge_list)

        # 13. Ground truth evaluation
        gt_total = 0
        gt_passed = 0
        gt_failures: List[str] = []

        if ground_truth_expectations:
            for exp in ground_truth_expectations:
                page_artifacts = reg.get_by_page(exp.page_number)
                page_types = {a.artifact_type for a in page_artifacts}

                # Check expected headings
                for h_sub in exp.expected_heading_substrings:
                    gt_total += 1
                    found_h = any(
                        h_sub.lower() in (a.title.lower() if isinstance(a, HeadingArtifact) else "")
                        for a in page_artifacts
                    )
                    if found_h:
                        gt_passed += 1
                    else:
                        gt_failures.append(f"Page {exp.page_number}: Expected heading '{h_sub}' was not detected.")

                # Check expected artifact types
                for exp_type in exp.expected_artifact_types:
                    gt_total += 1
                    if exp_type in page_types:
                        gt_passed += 1
                    else:
                        gt_failures.append(f"Page {exp.page_number}: Expected artifact type {exp_type.value} was not found.")

                # Check expected text keywords
                for kw in exp.expected_text_keywords:
                    gt_total += 1
                    found_kw = any(
                        kw.lower() in (a.normalized_text.lower() if isinstance(a, TextArtifact) else "")
                        for a in page_artifacts
                    )
                    if found_kw:
                        gt_passed += 1
                    else:
                        gt_failures.append(f"Page {exp.page_number}: Expected keyword '{kw}' was not found in page text blocks.")

                # Check min tables / figures / equations
                if exp.min_tables > 0:
                    gt_total += 1
                    act_tbl = sum(1 for a in page_artifacts if a.artifact_type == ArtifactType.TABLE)
                    if act_tbl >= exp.min_tables:
                        gt_passed += 1
                    else:
                        gt_failures.append(f"Page {exp.page_number}: Expected >= {exp.min_tables} tables, found {act_tbl}.")

                if exp.min_figures > 0:
                    gt_total += 1
                    act_fig = sum(1 for a in page_artifacts if a.artifact_type == ArtifactType.FIGURE)
                    if act_fig >= exp.min_figures:
                        gt_passed += 1
                    else:
                        gt_failures.append(f"Page {exp.page_number}: Expected >= {exp.min_figures} figures, found {act_fig}.")

        return BenchmarkReport(
            doc_title=dom.title,
            total_pages=total_pages,
            recorded_pages=rec_pages,
            text_count=len(texts),
            h1_count=h1,
            h2_count=h2,
            h3_count=h3,
            eq_verified=eq_ver,
            eq_candidate=eq_cand,
            eq_failed=eq_fail,
            figures_count=fig_cnt,
            linked_captions_count=linked_caps,
            tables_count=tbl_cnt,
            academic_semantics_count=academic_sem,
            module_assigned_count=mod_assigned,
            section_assigned_count=sec_assigned,
            total_artifacts=total_arts,
            bbox_percentage=bbox_pct,
            unique_ids_percentage=unique_pct,
            cross_refs_count=cross_refs,
            page_failures=pg_fail,
            partial_pages=pg_part,
            ocr_required_pages=pg_ocr,
            ground_truth_total_checks=gt_total,
            ground_truth_passed_checks=gt_passed,
            ground_truth_failures=gt_failures,
        )
