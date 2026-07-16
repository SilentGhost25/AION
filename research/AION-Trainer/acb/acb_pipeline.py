# AION-Trainer/acb/acb_pipeline.py
"""
ACB Pipeline — orchestrates the entire course building process.

Runs syllabus parsing, quality registering, candidate discovery, merging,
confidence reasoning, importance scoring, completeness analysis, and
Course Intelligence Report generation.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from acb.concept import ConceptStore
from acb.source_registry import SourceRegistry, SourceType
from acb.syllabus_parser import SyllabusParser, ParsedSyllabus
from acb.concept_discoverer import ConceptDiscoverer, ConceptCandidate
from acb.concept_merger import ConceptMerger
from acb.confidence_engine import ConfidenceEngine
from acb.importance_scorer import ImportanceScorer
from acb.completeness_analyzer import CompletenessAnalyzer
from acb.course_intelligence_report import CourseIntelligenceReport

logger = logging.getLogger("aion.acb.pipeline")


class ACBPipeline:
    def __init__(
        self,
        subject_code: str,
        academic_root: str,
        db_dir: Optional[str] = None,
        department: str = "AIML",
        semester: int = 4,
    ):
        self.subject_code = subject_code
        self.academic_root = Path(academic_root)
        self.department = department
        self.semester = semester
        
        self.subject_dir = self.academic_root / department / f"semester_{semester}" / subject_code
        if not self.subject_dir.exists():
            # Fallback direct path matching
            self.subject_dir = self.academic_root / subject_code

        self.db_dir = Path(db_dir) if db_dir else self.subject_dir
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self.store_path = self.db_dir / "concepts.json"
        self.registry_path = self.db_dir / "sources.json"

        self.concept_store = ConceptStore(str(self.store_path))
        self.source_registry = SourceRegistry(str(self.registry_path))
        
        # Load existing data if available
        if self.store_path.exists():
            self.concept_store.load()
        if self.registry_path.exists():
            self.source_registry.load()

    def run(self) -> Dict[str, Any]:
        logger.info(f"[ACB Pipeline] Starting run for {self.subject_code}")
        self.subject_dir.mkdir(parents=True, exist_ok=True)

        # 1. Syllabus Parsing
        syllabus = self._parse_syllabus()
        if not syllabus:
            logger.error("[ACB Pipeline] Syllabus not found or failed to parse. Aborting.")
            return {"status": "failed", "error": "syllabus_missing"}

        # 2. Ingest and Register Sources
        source_files = self._scan_sources()
        logger.info(f"[ACB Pipeline] Registered {len(source_files)} source files in registry")

        # 3. Concept Discovery & Merging
        discoverer = ConceptDiscoverer()
        merger = ConceptMerger(self.concept_store, self.source_registry)
        
        candidates: List[ConceptCandidate] = []
        for file_path, source_type in source_files:
            profile = self.source_registry.create_and_register(str(file_path), source_type, self.subject_code)
            
            # Extract text and split into blocks
            text = self._extract_text(file_path)
            if not text.strip():
                continue
                
            if source_type in (SourceType.TEXTBOOK, SourceType.NOTES, SourceType.QUESTION_BANK):
                blocks = self._split_into_blocks(text)
                cands = discoverer.discover_from_blocks(blocks, profile.source_id, source_type)
                candidates.extend(cands)
            elif source_type == SourceType.PREVIOUS_PAPER:
                # Use PYQ Parser
                from server.pyq_extractor import PYQParser
                pyq_parser = PYQParser()
                records = pyq_parser.parse_text(text)
                cands = discoverer.discover_from_pyq_records(records, profile.source_id)
                candidates.extend(cands)

        # Merge candidates into store
        merge_stats = merger.merge_candidates(candidates)

        # 4. Confidence & Importance Scoring
        confidence_engine = ConfidenceEngine(syllabus)
        importance_scorer = ImportanceScorer()

        all_concepts = self.concept_store.concepts_for_subject(self.subject_code)
        if not all_concepts:
            all_concepts = self.concept_store.all_concepts()

        importance_scorer.compute_all(all_concepts, syllabus)
        reasonings = confidence_engine.compute_all(all_concepts)

        # 5. Completeness Analysis
        analyzer = CompletenessAnalyzer(self.concept_store)
        profile = analyzer.analyze(syllabus)

        # 6. Report Generation
        reporter = CourseIntelligenceReport(str(self.db_dir))
        md_report, json_report = reporter.generate(
            profile, reasonings, subject_name=syllabus.subject_name, semester=self.semester
        )

        # Save Stores
        self.concept_store.save()
        self.source_registry.save()

        logger.info(f"[ACB Pipeline] Completed successfully. Coverage: {profile.overall_completeness*100:.1f}%")
        return {
            "status": "success",
            "overall_completeness": profile.overall_completeness,
            "concepts_count": self.concept_store.size(),
            "sources_count": len(self.source_registry.all_sources()),
            "merge_stats": merge_stats,
            "report_markdown": md_report,
            "report_json": json_report,
        }

    def _parse_syllabus(self) -> Optional[ParsedSyllabus]:
        syllabus_dir = self.subject_dir / "syllabus"
        if not syllabus_dir.exists():
            return None
            
        syllabus_files = list(syllabus_dir.glob("*.pdf")) + list(syllabus_dir.glob("*.docx")) + list(syllabus_dir.glob("*.txt"))
        if not syllabus_files:
            return None

        # Parse first available syllabus file
        parser = SyllabusParser()
        try:
            return parser.parse_file(str(syllabus_files[0]), subject_code=self.subject_code)
        except Exception as e:
            logger.error(f"[ACB Pipeline] Failed to parse syllabus file {syllabus_files[0]}: {e}")
            return None

    def _scan_sources(self) -> List[Tuple[Path, str]]:
        sources = []
        mappings = {
            "textbooks": SourceType.TEXTBOOK,
            "notes": SourceType.NOTES,
            "question_bank": SourceType.QUESTION_BANK,
            "previous_papers": SourceType.PREVIOUS_PAPER,
            "answer_keys": SourceType.ANSWER_KEY,
            "images": SourceType.IMAGES,
        }
        for subfolder, stype in mappings.items():
            folder = self.subject_dir / subfolder
            if folder.exists():
                for ext in ["*.pdf", "*.docx", "*.txt"]:
                    for file in folder.glob(ext):
                        sources.append((file, stype))
        return sources

    def _extract_text(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            try:
                import fitz
                try:
                    doc = fitz.open(str(file_path))
                    text = "\n".join(page.get_text("text") for page in doc)
                    doc.close()
                    return text
                except Exception:
                    # Fallback to plain text read (useful in mock unit tests)
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        return f.read()
            except ImportError:
                return ""
        elif suffix == ".docx":
            try:
                import docx as python_docx
                doc = python_docx.Document(str(file_path))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return ""
        else:
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()
            except Exception:
                return ""

    def _split_into_blocks(self, text: str) -> List[Dict[str, Any]]:
        blocks = []
        lines = text.splitlines()
        current_block = []
        current_kind = "text"
        page_num = 1
        line_num = 0

        for line in lines:
            line_num += 1
            line_strip = line.strip()
            if not line_strip:
                continue

            # Detect page marker heuristics
            if "page" in line_strip.lower() or re.match(r"^\d+\s*$", line_strip):
                page_num += 1

            # Detect heading heuristics
            if len(line_strip) < 80 and (
                line_strip.isupper() or 
                line_strip.endswith(":") or 
                re.match(r"^\d+(\.\d+)*\s+[A-Z]", line_strip)
            ):
                if current_block:
                    blocks.append({
                        "text": "\n".join(current_block),
                        "kind": current_kind,
                        "page": page_num,
                        "location": f"page {page_num}, line {line_num}"
                    })
                    current_block = []
                blocks.append({
                    "text": line_strip,
                    "kind": "heading",
                    "page": page_num,
                    "location": f"page {page_num}, line {line_num}"
                })
                current_kind = "text"
            else:
                current_block.append(line_strip)

        if current_block:
            blocks.append({
                "text": "\n".join(current_block),
                "kind": current_kind,
                "page": page_num,
                "location": f"page {page_num}, line {line_num}"
            })
        return blocks
