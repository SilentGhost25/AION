"""
Subject Detector — Canonical Subject Profile after extraction
Per audit: Document -> Subject Detector -> Subject Profile -> Knowledge Units
Every downstream component uses that profile. Greatly reduces cross-domain contamination.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

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

@dataclass
class DocumentKnowledgeGraph:
    subject_name: str = "General Course"
    subject_code: str = "GEN"
    discipline_type: str = "COMPUTATIONAL_ENGINEERING"
    modules: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    vocabulary: set = field(default_factory=set)
    equations: List[Dict[str, Any]] = field(default_factory=list)
    parameter_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    scenario_whitelist: set = field(default_factory=set)


class DynamicPedagogicalProfiler:
    """
    Infers subject character, discipline classification, and vocabulary
    dynamically from document text in <350ms without hardcoded domain tables.
    """

    DISCIPLINE_PATTERNS = {
        "THEORETICAL_QUALITATIVE": [
            r"\b(?:law|legal|jurisprudence|contract|constitution|tort|statute|ethics|governance|corporate governance)\b",
            r"\b(?:management|marketing|human resources|organizational behavior|strategy|business administration)\b",
            r"\b(?:philosophy|epistemology|ethics|sociology|political science|history|literature)\b",
        ],
        "EMPIRICAL_ANALYTICAL": [
            r"\b(?:machine learning|deep learning|neural network|classification|clustering|supervised|unsupervised|dataset)\b",
            r"\b(?:genetics|molecular biology|biotechnology|pharmacology|cellular|enzyme|protein|dna|rna)\b",
            r"\b(?:microeconomics|macroeconomics|econometrics|gdp|inflation|market equilibrium|fiscal)\b",
        ],
        "MATHEMATICAL_FORMAL": [
            r"\b(?:differential equation|laplace transform|fourier series|eigenvalues?|eigenvectors?|linear algebra)\b",
            r"\b(?:calculus|integration|derivative|matrix|vector space|determinant|probability density)\b",
            r"\b(?:theorem|lemma|corollary|proof|axiom|group theory|complex analysis)\b",
        ],
        "COMPUTATIONAL_ENGINEERING": [
            r"\b(?:algorithm|data structure|complexity|operating system|database|compiler|networking)\b",
            r"\b(?:circuit|voltage|current|transistor|signal|amplifier|impedance|electromagnetics)\b",
            r"\b(?:thermodynamics|fluid mechanics|heat transfer|stress|strain|structural|kinematics)\b",
        ],
    }

    GENERIC_SCENARIOS = {
        "system", "case", "scenario", "study", "organization", "company", "firm", "industry",
        "client", "user", "customer", "patient", "hospital", "bank", "account", "transaction",
        "factory", "warehouse", "robot", "sensor", "device", "vehicle", "network", "server",
        "process", "model", "parameter", "data", "application", "environment", "implementation"
    }

    def profile(self, clean_text: str, title_hint: str = "") -> DocumentKnowledgeGraph:
        text_low = clean_text.lower()
        title_low = title_hint.lower()
        combined = f"{title_low}\n{text_low[:10000]}"

        # 1. Infer Subject Code
        code_match = re.search(r"\b(?:course\s*code|sub\s*code|code)?[:\s-]*([0-9]{2}[A-Z]{2,4}[0-9]{2,3}|[A-Z]{2,4}[0-9]{2,3}|BCS[0-9]{3})\b", combined, re.I)
        subj_code = code_match.group(1).upper() if code_match else "GEN"

        # 2. Infer Subject Name
        subj_name = ""
        name_match = re.search(r"^(?:course\s*title|subject(?:\s*name)?|title)[:\s-]*([^\n\r]{4,80})", combined, re.M | re.I)
        if name_match:
            subj_name = name_match.group(1).strip()
        elif title_hint:
            subj_name = title_hint.strip()
        else:
            first_line = clean_text.strip().splitlines()[0][:80].strip() if clean_text.strip() else "Universal Subject"
            first_line = re.sub(r'^[#*\-\s\d\.]+', '', first_line).strip()
            subj_name = first_line if len(first_line) >= 4 else "Universal Subject"

        # 3. Classify Discipline Type
        disc_scores = {dtype: 0 for dtype in self.DISCIPLINE_PATTERNS}
        for dtype, patterns in self.DISCIPLINE_PATTERNS.items():
            for pat in patterns:
                matches = re.findall(pat, combined)
                disc_scores[dtype] += len(matches)

        best_disc = max(disc_scores, key=disc_scores.get)
        if disc_scores[best_disc] == 0:
            best_disc = "COMPUTATIONAL_ENGINEERING"

        # 4. Extract Dynamic Vocabulary (n-grams and technical keyphrases)
        words = re.findall(r"\b[a-zA-Z]{4,}\b", clean_text)
        vocab = {w.lower() for w in words[:2000]}
        # Add common multi-word capitalized entities
        phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", clean_text[:15000])
        for p in phrases[:150]:
            vocab.add(p.lower())

        # 5. Extract Equations
        equations = []
        eq_matches = re.findall(r"(\b[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^;\n\r$]{3,50})", clean_text)
        for eq in eq_matches[:25]:
            equations.append({"raw": eq.strip()})

        # 6. Extract Parameter Ranges
        parameter_ranges = {}
        param_candidates = re.findall(r"\b([a-zA-Z_]{2,15})\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z%Ωμ]+)?\b", clean_text)
        for name, val_str, unit in param_candidates[:50]:
            val = float(val_str)
            name_low = name.lower()
            if name_low not in parameter_ranges:
                parameter_ranges[name_low] = (val * 0.5, val * 2.0)
            else:
                low, high = parameter_ranges[name_low]
                parameter_ranges[name_low] = (min(low, val * 0.8), max(high, val * 1.2))

        return DocumentKnowledgeGraph(
            subject_name=subj_name,
            subject_code=subj_code,
            discipline_type=best_disc,
            vocabulary=vocab,
            equations=equations,
            parameter_ranges=parameter_ranges,
            scenario_whitelist=set(self.GENERIC_SCENARIOS)
        )

    def create_subject_profile(self, dkg: DocumentKnowledgeGraph) -> SubjectProfile:
        disc = dkg.discipline_type
        if disc == "THEORETICAL_QUALITATIVE":
            reasoning = ["case analysis", "statutory interpretation", "comparative evaluation", "ethical critique"]
            diagrams = ["flowchart", "process hierarchy", "framework diagram"]
            numeric = []
            formats = ["case study analysis", "evaluate doctrine", "critique policy", "distinguish concepts"]
        elif disc == "EMPIRICAL_ANALYTICAL":
            reasoning = ["empirical deduction", "data interpretation", "model evaluation", "comparative benchmark"]
            diagrams = ["scatter plot", "ROC curve", "block diagram", "confusion matrix"]
            numeric = ["parameter estimation", "confusion matrix calculation", "performance metric evaluation"]
            formats = ["analyze model performance", "interpret empirical results", "formulate hypothesis"]
        elif disc == "MATHEMATICAL_FORMAL":
            reasoning = ["formal derivation", "algebraic proof", "step-by-step evaluation", "symbolic computation"]
            diagrams = ["function curve", "vector field", "geometric projection"]
            numeric = ["differential equation solution", "eigenvalue calculation", "integral computation"]
            formats = ["derive equation", "evaluate integral", "prove theorem", "compute exact values"]
        else:
            reasoning = ["algorithmic tracing", "architectural design", "parametric calculation", "trade-off analysis"]
            diagrams = ["block diagram", "schematic diagram", "state chart", "timing diagram"]
            numeric = ["performance calculation", "resource estimation", "state transition trace"]
            formats = ["design system", "trace algorithm", "calculate parameters", "analyze trade-offs"]

        return SubjectProfile(
            code=dkg.subject_code or "GEN",
            name=dkg.subject_name or "General Course Profile",
            permitted_vocabulary=dkg.vocabulary,
            reasoning_styles=reasoning,
            diagram_types=diagrams,
            numerical_patterns=numeric,
            question_formats=formats,
            forbidden_cross_terms=set()
        )


class SubjectDetector:
    """Detects subject from clean_text, returns isolated SubjectProfile (dynamically or from baseline)."""

    def __init__(self):
        self.profiler = DynamicPedagogicalProfiler()

    def detect(self, clean_text: str) -> SubjectProfile:
        text_low = clean_text.lower()
        scores = {code: 0 for code in SUBJECT_PROFILES}
        for code, kws in SUBJECT_KEYWORDS.items():
            for kw in kws:
                if " " in kw:
                    if kw in text_low:
                        scores[code] += 2
                else:
                    if re.search(r"\b" + re.escape(kw) + r"\b", text_low):
                        scores[code] += 1

        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return SUBJECT_PROFILES[best]

        # Zero hardcoded hits: dynamically profile document without defaulting to CSE
        dkg = self.profiler.profile(clean_text)
        return self.profiler.create_subject_profile(dkg)

    def detect_with_confidence(self, clean_text: str) -> tuple[SubjectProfile, float, Dict[str,int]]:
        text_low = clean_text.lower()
        scores = {code: sum(1 for kw in kws if kw in text_low or re.search(r"\b"+re.escape(kw)+r"\b", text_low)) for code, kws in SUBJECT_KEYWORDS.items()}
        total = sum(scores.values())
        best = max(scores, key=scores.get)
        if total > 0 and scores[best] > 0:
            conf = scores[best] / max(total, 1)
            return SUBJECT_PROFILES[best], round(conf, 2), scores

        dkg = self.profiler.profile(clean_text)
        profile = self.profiler.create_subject_profile(dkg)
        return profile, 1.0, scores
