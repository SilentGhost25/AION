# AION-Trainer/acb/completeness_analyzer.py
"""
Syllabus Completeness Analyzer — evaluates how well the extracted and verified
knowledge base covers the authoritative syllabus.

Calculates module-by-module coverage ratios, identifies missing syllabus topics,
evaluates Bloom level ceiling coverage, and flags content gaps.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional

from acb.concept import Concept, ConceptStore
from acb.syllabus_parser import ParsedSyllabus, SyllabusModule

logger = logging.getLogger("aion.acb.completeness")


@dataclass
class ModuleCoverage:
    module_number: int
    title: str
    total_topics: int
    covered_topics: List[str] = field(default_factory=list)
    missing_topics: List[str] = field(default_factory=list)
    coverage_ratio: float = 0.0
    average_confidence: float = 0.0
    highest_bloom_level: str = "L1"
    concepts_count: int = 0


@dataclass
class CompletenessProfile:
    subject_code: str
    overall_completeness: float            # 0.0 to 1.0
    total_syllabus_topics: int
    covered_syllabus_topics: int
    modules: List[ModuleCoverage] = field(default_factory=list)
    unassigned_concepts: List[str] = field(default_factory=list)   # concepts with no module link
    stubs_count: int = 0
    needs_verification_count: int = 0


class CompletenessAnalyzer:
    def __init__(self, store: ConceptStore):
        self.store = store

    def analyze(self, syllabus: ParsedSyllabus) -> CompletenessProfile:
        profile = CompletenessProfile(
            subject_code=syllabus.subject_code,
            overall_completeness=0.0,
            total_syllabus_topics=0,
            covered_syllabus_topics=0,
        )

        all_concepts = self.store.concepts_for_subject(syllabus.subject_code)
        if not all_concepts:
            # Fallback to all concepts if subject code links are not populated
            all_concepts = self.store.all_concepts()

        # Track unassigned
        for c in all_concepts:
            if not c.module_links:
                profile.unassigned_concepts.append(c.name)
            if c.status == "stub":
                profile.stubs_count += 1
            if c.status == "needs_verification":
                profile.needs_verification_count += 1

        total_topics_all_modules = 0
        total_covered_topics_all_modules = 0

        # Normalise concept names and aliases for matching
        concept_names = {}
        for c in all_concepts:
            norm_c = self._norm(c.name)
            concept_names[norm_c] = c
            for alias in c.aliases:
                concept_names[self._norm(alias)] = c

        for mod in syllabus.modules:
            mod_cov = ModuleCoverage(
                module_number=mod.module_number,
                title=mod.title,
                total_topics=len(mod.topics),
            )

            mod_concepts = [
                c for c in all_concepts
                if any(ml.module == mod.module_number for ml in c.module_links)
            ]
            mod_cov.concepts_count = len(mod_concepts)

            # Match syllabus topics to concepts
            for topic in mod.topics:
                norm_topic = self._norm(topic)
                # Check for direct match or word overlap match
                matched = False
                for c_norm, concept in concept_names.items():
                    if norm_topic == c_norm or norm_topic in c_norm or c_norm in norm_topic:
                        matched = True
                        # Ensure concept is linked to this module if not already
                        concept.add_module_link(syllabus.subject_code, mod.module_number)
                        break
                
                if matched:
                    mod_cov.covered_topics.append(topic)
                else:
                    mod_cov.missing_topics.append(topic)

            total_topics_all_modules += mod_cov.total_topics
            total_covered_topics_all_modules += len(mod_cov.covered_topics)

            if mod_cov.total_topics > 0:
                mod_cov.coverage_ratio = len(mod_cov.covered_topics) / mod_cov.total_topics
            
            # Avg confidence and highest bloom
            if mod_concepts:
                mod_cov.average_confidence = sum(c.confidence for c in mod_concepts) / len(mod_concepts)
                # Highest bloom
                blooms = [c.bloom_progression.highest_level() for c in mod_concepts]
                bloom_order = ["L1", "L2", "L3", "L4", "L5", "L6"]
                highest = "L1"
                for b in blooms:
                    if b in bloom_order and bloom_order.index(b) > bloom_order.index(highest):
                        highest = b
                mod_cov.highest_bloom_level = highest

            profile.modules.append(mod_cov)

        profile.total_syllabus_topics = total_topics_all_modules
        profile.covered_syllabus_topics = total_covered_topics_all_modules
        
        if total_topics_all_modules > 0:
            profile.overall_completeness = total_covered_topics_all_modules / total_topics_all_modules
        else:
            profile.overall_completeness = 0.0

        logger.info(
            f"[CompletenessAnalyzer] Subject {syllabus.subject_code} "
            f"completeness calculated at {profile.overall_completeness * 100:.2f}%"
        )
        return profile

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()
