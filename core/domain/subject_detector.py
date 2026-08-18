"""
Subject Detector — Canonical Subject Profile after extraction
Per audit: Document -> Subject Detector -> Subject Profile -> Knowledge Units
Every downstream component uses that profile. Greatly reduces cross-domain contamination.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SubjectProfile:
    code: str  # e.g., "CSE", "ECE", "ME", "CV", "EE", "AU"
    name: str
    permitted_vocabulary: set
    reasoning_styles: List[str]
    diagram_types: List[str]
    numerical_patterns: List[str]
    question_formats: List[str]
    forbidden_cross_terms: set

# Permitted vocabulary per subject — isolated, never intersect unless explicitly cross-domain
SUBJECT_PROFILES: Dict[str, SubjectProfile] = {
    "CSE": SubjectProfile(
        code="CSE", name="Computer Science & Engineering",
        permitted_vocabulary={"binary tree", "bst", "avl", "heap", "graph", "hash", "sorting", "searching", "algorithm", "complexity", "recursion", "stack", "queue", "linked list", "tree", "traversal", "rotation", "balance factor", "insertion", "deletion", "search", "dequeue", "enqueue", "pointer", "node", "edge", "vertex", "big o", "time complexity", "space complexity"},
        reasoning_styles=["algorithm tracing", "complexity analysis", "case analysis", "optimization", "debugging", "proof"],
        diagram_types=["binary tree", "graph", "flowchart", "memory diagram"],
        numerical_patterns=["array generation", "tree insertion sequence", "graph edge weights"],
        question_formats=["trace algorithm", "analyse complexity", "design data structure", "debug code"],
        forbidden_cross_terms={"ecu", "o2 sensor", "maf", "dlc", "carburetor", "forging", "lathe", "thermodynamics", "satellite", "antenna"}
    ),
    "ECE": SubjectProfile(
        code="ECE", name="Electronics & Communication",
        permitted_vocabulary={"diode", "transistor", "mosfet", "op-amp", "amplifier", "modulation", "antenna", "satellite", "tdma", "fdma", "cdma", "signal", "filter", "circuit", "frequency", "bandwidth", "transponder", "orbit", "uplink", "downlink"},
        reasoning_styles=["circuit analysis", "signal flow", "comparison", "design", "case study"],
        diagram_types=["circuit diagram", "block diagram", "signal flow", "constellation"],
        numerical_patterns=["signal calculation", "link budget", "frequency allocation"],
        question_formats=["analyse circuit", "design filter", "compare modulation", "case study"],
        forbidden_cross_terms={"binary tree", "forging", "lathe", "beam", "column"}
    ),
    "ME": SubjectProfile(
        code="ME", name="Mechanical Engineering",
        permitted_vocabulary={"forging", "casting", "welding", "lathe", "milling", "thermodynamics", "entropy", "enthalpy", "otto", "diesel", "engine", "brake", "clutch", "gear", "cam", "thermodynamics", "heat transfer", "fluid mechanics"},
        reasoning_styles=["process analysis", "design", "numerical calculation", "case study"],
        diagram_types=["mechanical assembly", "p-v diagram", "t-s diagram", "machine drawing"],
        numerical_patterns=["thermodynamic cycle", "force calculation", "material stress"],
        question_formats=["numerical problem", "explain process", "design component"],
        forbidden_cross_terms={"binary tree", "hash", "satellite", "antenna"}
    ),
    "AU": SubjectProfile(
        code="AU", name="Automobile Engineering",
        permitted_vocabulary={"obd", "obd-ii", "dtc", "ecu", "maf", "map", "o2 sensor", "dlc", "can", "mil", "p0171", "p0300", "p0420", "misfire", "catalyst", "crankshaft", "camshaft", "fuel injection", "ignition", "scan tool", "freeze frame", "live data", "actuator", "egr", "evap"},
        reasoning_styles=["diagnosis", "case study", "procedure sequencing", "cause analysis"],
        diagram_types=["block diagram", "sensor circuit", "engine diagram"],
        numerical_patterns=["sensor threshold", "dtc analysis", "actuator test"],
        question_formats=["diagnose case", "outline procedure", "interpret data", "evaluate decision"],
        forbidden_cross_terms={"binary tree", "forging", "antenna", "beam"}
    ),
    "CV": SubjectProfile(
        code="CV", name="Civil Engineering",
        permitted_vocabulary={"beam", "column", "slab", "foundation", "surveying", "concrete", "cement", "aggregate", "levelling", "contour", "benchmark", "load", "stress", "strain"},
        reasoning_styles=["structural analysis", "design", "surveying", "estimation"],
        diagram_types=["structural diagram", "survey plot", "cross-section"],
        numerical_patterns=["load calculation", "levelling", "estimation"],
        question_formats=["numerical", "design", "explain procedure"],
        forbidden_cross_terms={"binary tree", "o2 sensor", "satellite"}
    ),
}

# Keyword to subject mapping for detection
SUBJECT_KEYWORDS = {
    "CSE": ["binary tree", "bst", "avl", "algorithm", "data structure", "sorting", "searching", "recursion", "stack", "queue", "tree traversal", "balance factor", "insertion", "deletion"],
    "ECE": ["satellite", "transponder", "antenna", "modulation", "tdma", "fdma", "orbit", "uplink", "downlink", "geostationary"],
    "AU": ["obd", "dtc", "ecu", "maf", "o2 sensor", "dlc", "misfire", "catalyst", "crankshaft", "scan tool", "freeze frame", "p0171", "p0300", "p0420"],
    "ME": ["forging", "casting", "lathe", "milling", "thermodynamics", "otto cycle", "heat transfer"],
    "CV": ["beam", "column", "surveying", "concrete", "foundation", "levelling"],
}

class SubjectDetector:
    """Detects subject from clean_text, returns isolated SubjectProfile."""

    def detect(self, clean_text: str) -> SubjectProfile:
        text_low = clean_text.lower()
        scores = {code: 0 for code in SUBJECT_PROFILES}
        for code, kws in SUBJECT_KEYWORDS.items():
            for kw in kws:
                # Word boundary for short terms, substring for phrases
                if " " in kw:
                    if kw in text_low:
                        scores[code] += 2
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_low):
                        scores[code] += 1
        # Fallback: count vocabulary hits
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            # Default to CSE for generic
            return SUBJECT_PROFILES["CSE"]
        return SUBJECT_PROFILES[best]

    def detect_with_confidence(self, clean_text: str) -> tuple[SubjectProfile, float, Dict[str,int]]:
        text_low = clean_text.lower()
        scores = {code: sum(1 for kw in kws if kw in text_low or re.search(r"\b"+re.escape(kw)+r"\b", text_low)) for code, kws in SUBJECT_KEYWORDS.items()}
        total = sum(scores.values())
        best = max(scores, key=scores.get)
        conf = scores[best] / max(total, 1)
        return SUBJECT_PROFILES[best], round(conf, 2), scores
