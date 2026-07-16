# AION-Trainer/acb/__init__.py
"""
Academic Course Builder (ACB) — Subsystem for syllabus extraction,
concept modeling, source quality registry, knowledge graph topological
analysis, content completeness analysis, and generation of Course
Intelligence Reports.
"""

from acb.concept import Concept, ConceptStore, ConceptScope, ConceptStatus, ModuleLink, ConceptSource, BloomProgression
from acb.source_registry import SourceRegistry, SourceQualityProfile, SourceType
from acb.syllabus_parser import SyllabusParser, ParsedSyllabus, SyllabusModule
from acb.concept_discoverer import ConceptDiscoverer, ConceptCandidate
from acb.concept_merger import ConceptMerger
from acb.confidence_engine import ConfidenceEngine, ConceptReasoning
from acb.importance_scorer import ImportanceScorer
from acb.knowledge_graph import KnowledgeGraph
from acb.completeness_analyzer import CompletenessAnalyzer
from acb.course_intelligence_report import CourseIntelligenceReport
from acb.acb_pipeline import ACBPipeline
