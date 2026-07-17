# AION-Trainer/tests/test_acb.py
"""
Unit tests for the Academic Course Builder (ACB) subsystem.
"""

import pytest
import tempfile
import json
from pathlib import Path

from acb.concept import Concept, ConceptStore, ConceptScope, ConceptStatus
from acb.source_registry import SourceRegistry, SourceQualityProfile, SourceType
from acb.syllabus_parser import SyllabusParser, SyllabusModule, ParsedSyllabus
from acb.concept_discoverer import ConceptDiscoverer, ConceptCandidate
from acb.concept_merger import ConceptMerger
from acb.confidence_engine import ConfidenceEngine, CONFIDENCE_THRESHOLD
from acb.importance_scorer import ImportanceScorer
from acb.knowledge_graph import KnowledgeGraph
from acb.completeness_analyzer import CompletenessAnalyzer
from acb.course_intelligence_report import CourseIntelligenceReport
from acb.acb_pipeline import ACBPipeline

from conftest import write_fake_file


def test_syllabus_parser():
    parser = SyllabusParser()
    text = (
        "Module 1: Introduction to Search Algorithms (10 Hours)\n"
        "Introduction to AI, uninformed search, DFS, BFS.\n"
        "CO1: Understand basic search methods.\n"
        "Module 2: Heuristic Search (8 Hours)\n"
        "Heuristic functions, A* Search, greedy search.\n"
        "CO2: Apply heuristic search to problems.\n"
    )
    syllabus = parser.parse_text(text, subject_code="BAI404", subject_name="Intro to AI")
    
    assert syllabus.subject_code == "BAI404"
    assert syllabus.subject_name == "Intro to AI"
    assert len(syllabus.modules) == 2
    assert syllabus.modules[0].module_number == 1
    assert syllabus.modules[0].hours == 10
    assert "uninformed search" in syllabus.modules[0].topics
    assert "Understand basic search methods." in syllabus.modules[0].learning_outcomes


def test_concept_store(tmp_path):
    store_file = tmp_path / "concepts.json"
    store = ConceptStore(str(store_file))

    c1 = Concept(name="A* Search", aliases=["AStar"])
    c1.add_module_link("BAI404", 2, is_primary=True)
    store.add(c1)

    assert store.size() == 1
    assert store.find_by_name("A* Search") == c1
    assert store.find_by_name("AStar") == c1
    
    # Save and reload
    store.save()
    assert store_file.exists()

    new_store = ConceptStore(str(store_file))
    new_store.load()
    assert new_store.size() == 1
    loaded_c = new_store.get(c1.concept_id)
    assert loaded_c.name == "A* Search"
    assert loaded_c.primary_module() == 2


def test_concept_merger_and_discoverer():
    store = ConceptStore()
    registry = SourceRegistry()
    
    profile = SourceQualityProfile(source_id="src-1", source_type=SourceType.TEXTBOOK)
    registry.register(profile)

    discoverer = ConceptDiscoverer()
    merger = ConceptMerger(store, registry, similarity_threshold=0.5)

    from document_intelligence.document_model import AcademicDocument, Section
    doc = AcademicDocument(source_path="src-1")
    doc.sections = [
        Section(
            title="A* Search",
            content="A* Search is defined as a heuristic search algorithm. It is used to find shortest paths. Applications include pathfinding in games.",
            page_range=(1,1)
        )
    ]

    cands = discoverer.discover_from_document(doc, "src-1", SourceType.TEXTBOOK)
    assert len(cands) == 1
    assert cands[0].name == "A* Search"
    assert "heuristic search algorithm" in cands[0].definition
    assert "pathfinding in games" in cands[0].applications

    # Merge candidates
    stats = merger.merge_candidates(cands)
    assert stats["created"] == 1
    assert store.size() == 1

    # Merge identical candidate (should enrich, not duplicate)
    stats2 = merger.merge_candidates(cands)
    assert stats2["merged"] == 1
    assert store.size() == 1


def test_knowledge_graph():
    store = ConceptStore()
    kg = KnowledgeGraph(store)

    c1 = Concept(name="Search Basics")
    c2 = Concept(name="Heuristic Functions", prerequisites=[c1.concept_id])
    c3 = Concept(name="A* Search", prerequisites=[c2.concept_id])

    store.add(c1)
    store.add(c2)
    store.add(c3)

    # Prerequisite paths
    prereqs = kg.get_prerequisites_recursive(c3.concept_id)
    assert prereqs == [c1.concept_id, c2.concept_id]

    # Topological sorting
    sorted_concepts = kg.topological_sort()
    assert [c.name for c in sorted_concepts] == ["Search Basics", "Heuristic Functions", "A* Search"]

    # Cycle detection
    c1.prerequisites.append(c3.concept_id) # create cycle: c1 -> c3 -> c2 -> c1
    cycles = kg.find_cycles()
    assert len(cycles) > 0


def test_completeness_and_report(tmp_path):
    store = ConceptStore()
    
    # 2 core concepts
    c1 = Concept(name="DFS")
    c1.add_module_link("BAI404", 1)
    c1.confidence = 0.9
    c2 = Concept(name="A* Search")
    c2.add_module_link("BAI404", 2)
    c2.confidence = 0.95

    store.add(c1)
    store.add(c2)

    syllabus = ParsedSyllabus(
        subject_code="BAI404",
        subject_name="Intro to AI",
        semester=4,
        department="AIML",
        university="VTU",
    )
    syllabus.modules = [
        SyllabusModule(module_number=1, title="Uninformed", topics=["DFS", "BFS"]),
        SyllabusModule(module_number=2, title="Heuristic", topics=["A* Search"]),
    ]

    analyzer = CompletenessAnalyzer(store)
    profile = analyzer.analyze(syllabus)

    assert profile.total_syllabus_topics == 3
    assert profile.covered_syllabus_topics == 2
    assert profile.overall_completeness == pytest.approx(2/3)

    assert profile.modules[0].coverage_ratio == 0.5 # DFS covered, BFS missing
    assert profile.modules[1].coverage_ratio == 1.0 # A* Search covered

    # Report generation
    engine = ConfidenceEngine(syllabus)
    reasonings = engine.compute_all(store.all_concepts())

    reporter = CourseIntelligenceReport(str(tmp_path))
    md, js = reporter.generate(profile, reasonings, subject_name="Intro to AI", semester=4)

    assert "Course Intelligence Report" in md
    assert js["overall_completeness"] == pytest.approx(2/3)
    assert (tmp_path / "course_intelligence_report.md").exists()
    assert (tmp_path / "course_intelligence_report.json").exists()


def test_acb_pipeline(academic_root):
    # Setup test workspace under the academic_root fixture
    subject_code = "BAI404"
    subject_dir = academic_root / "AIML" / "semester_4" / subject_code
    
    # Write mock syllabus file
    write_fake_file(
        subject_dir / "syllabus" / "syllabus.txt",
        "Module 1: Basic AI (5 Hours)\nDFS topic, BFS topic\nCO1: Learn basic search\n"
    )

    # Write mock notes files
    write_fake_file(
        subject_dir / "notes" / "notes_dfs.txt",
        "DFS topic\nDFS topic is defined as depth first search. It uses a stack.\n"
    )

    pipeline = ACBPipeline(
        subject_code=subject_code,
        academic_root=str(academic_root),
        department="AIML",
        semester=4,
    )
    
    res = pipeline.run()
    assert res["status"] == "success"
    assert res["concepts_count"] > 0
    assert res["sources_count"] > 0
