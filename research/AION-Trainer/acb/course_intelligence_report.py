# AION-Trainer/acb/course_intelligence_report.py
"""
Course Intelligence Report (CIR) Generator.

Compiles content coverage metrics, confidence analysis, and syllabus mapping
into a premium, actionable markdown report and a corresponding JSON artifact.
Provides a clear feedback loop to the university faculty on material readiness
before exam paper generation.
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from acb.completeness_analyzer import CompletenessProfile, ModuleCoverage
from acb.confidence_engine import ConceptReasoning

logger = logging.getLogger("aion.acb.report")


class CourseIntelligenceReport:
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else None

    def generate(
        self,
        profile: CompletenessProfile,
        reasonings: List[ConceptReasoning],
        subject_name: str = "",
        semester: int = 0,
    ) -> Tuple[str, Dict[str, Any]]:
        # Map reasoning list for quick lookup
        reasoning_map = {r.concept_name: r for r in reasonings}

        md = []
        md.append(f"# Course Intelligence Report (CIR): {profile.subject_code}")
        md.append(f"Generated at: `{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}`")
        md.append("")
        md.append("## 1. Executive Summary")
        md.append("| Metric | Value | Status |")
        md.append("| :--- | :--- | :--- |")
        md.append(f"| **Subject Name** | {subject_name or 'N/A'} | - |")
        md.append(f"| **Subject Code** | {profile.subject_code} | - |")
        md.append(f"| **Semester** | Semester {semester} | - |")
        
        status_color = "[Ready to Refine]"
        if profile.overall_completeness >= 0.85:
            status_color = "[Ready for Generation]"
        elif profile.overall_completeness >= 0.70:
            status_color = "[Verification Required]"
            
        md.append(f"| **Overall Syllabus Coverage** | **{profile.overall_completeness * 100:.1f}%** | {status_color} |")
        md.append(f"| **Syllabus Topics (Covered / Total)** | {profile.covered_syllabus_topics} / {profile.total_syllabus_topics} | - |")
        md.append(f"| **Concepts Stubs (Incomplete)** | {profile.stubs_count} | Needs Content Ingestion |")
        md.append(f"| **Needs Verification** | {profile.needs_verification_count} | Requires Faculty Approval |")
        md.append("")

        md.append("## 2. Module-wise Coverage Analysis")
        md.append("| Module | Title | Topics | Covered | Missing | Coverage % | Max Bloom | Average Confidence |")
        md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
        
        for m in profile.modules:
            avg_conf = f"{m.average_confidence * 100:.1f}%"
            cov_ratio = f"{m.coverage_ratio * 100:.1f}%"
            md.append(f"| {m.module_number} | {m.title} | {m.total_topics} | {len(m.covered_topics)} | {len(m.missing_topics)} | **{cov_ratio}** | {m.highest_bloom_level} | {avg_conf} |")
        md.append("")

        md.append("## 3. Coverage Gaps & Missing Syllabus Topics")
        gaps_found = False
        for m in profile.modules:
            if m.missing_topics:
                gaps_found = True
                md.append(f"### Module {m.module_number}: {m.title}")
                md.append("The following topics defined in the syllabus were **not found** in any ingested textbook or notes:")
                for topic in m.missing_topics:
                    md.append(f"* [MISSING] {topic}")
                md.append("")
        if not gaps_found:
            md.append("[OK] **Perfect Coverage!** No missing topics found across any syllabus module.")
            md.append("")

        md.append("## 4. Concepts Requiring Faculty Verification")
        unverified_list = [r for r in reasonings if r.needs_verification]
        if unverified_list:
            md.append("The following concepts have been flagged due to low confidence scores or structural module link conflicts:")
            for r in unverified_list[:20]:  # Limit to top 20 for readability
                md.append(f"### [WARN] {r.concept_name}")
                md.append(f"* **Confidence Score**: `{r.confidence * 100:.1f}%` (Threshold: `85.0%`)")
                md.append(f"* **Recommended Module**: Module `{r.recommended_module}` (Is Primary Link: `{r.is_primary}`)")
                if r.conflicts:
                    md.append("* **Conflicts Detected**:")
                    for conflict in r.conflicts:
                        md.append(f"  * [CONFLICT] {conflict}")
                md.append("* **Evidence Chains**:")
                for ev in r.evidence:
                    sign = "[PASS]" if ev.weight > 0 else "[FAIL]"
                    md.append(f"  * {sign} {ev.description} (Weight score: `{ev.weight:+.2f}`)")
                md.append("")
        else:
            md.append("[PASS] No low-confidence concepts or link conflicts found. All concepts fully verified.")
            md.append("")

        md.append("## 5. Actionable Recommendations")
        recommendations = []
        if profile.overall_completeness < 0.85:
            recommendations.append("[Syllabus Coverage] **Ingest more reference materials**: The overall syllabus coverage is below the 85% generation readiness gate. Upload additional textbooks, notes, or previous exam papers.")
        
        for m in profile.modules:
            if m.coverage_ratio < 0.70:
                recommendations.append(f"[Content Gap] **Module {m.module_number} Content Gap**: Ingest additional notes or textbook chapters specifically addressing: {', '.join(m.missing_topics[:3])}.")
            if m.highest_bloom_level in ["L1", "L2"] and m.total_topics > 5:
                recommendations.append(f"[Bloom Level] **Bloom Progression Warning**: Module {m.module_number} is capped at **{m.highest_bloom_level}**. To enable higher-order question generation (L3+ Apply/Analyze), provide textbooks or question banks detailing practical problems.")

        if profile.needs_verification_count > 0:
            recommendations.append(f"[Verification] **Verify Concept database**: Use the AION desktop client to verify the {profile.needs_verification_count} concepts flagged under Section 4.")

        if not recommendations:
            recommendations.append("[Optimized] **System is fully optimized**: All readiness indicators are green. You can proceed with exam paper candidate generation.")

        for i, rec in enumerate(recommendations, 1):
            md.append(f"{i}. {rec}")

        report_markdown = "\n".join(md)

        # JSON Metadata bundle
        report_json = {
            "subject_code": profile.subject_code,
            "subject_name": subject_name,
            "overall_completeness": profile.overall_completeness,
            "total_syllabus_topics": profile.total_syllabus_topics,
            "covered_syllabus_topics": profile.covered_syllabus_topics,
            "stubs_count": profile.stubs_count,
            "needs_verification_count": profile.needs_verification_count,
            "modules": [
                {
                    "module_number": m.module_number,
                    "title": m.title,
                    "coverage_ratio": m.coverage_ratio,
                    "average_confidence": m.average_confidence,
                    "highest_bloom_level": m.highest_bloom_level,
                    "missing_topics": m.missing_topics,
                }
                for m in profile.modules
            ],
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "course_intelligence_report.md").write_text(report_markdown, encoding="utf-8")
            (self.output_dir / "course_intelligence_report.json").write_text(json.dumps(report_json, indent=2, default=str), encoding="utf-8")

        return report_markdown, report_json


from typing import Tuple  # noqa: used in type annotations
