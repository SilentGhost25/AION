"""
Knowledge Unit Builder — Canonical Representation
Per audit: Concept is not enough. Need Knowledge Unit with
Concept + Definition + Formula + Procedure + Diagram + Applications
+ Relationships + Common mistakes + Numerical templates + Expected answer + Difficulty + Evidence

This becomes the single source of truth before grounding/reasoning/planning.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from core.concepts.extractor import ExtractedConcept


def _normalize_concept_name(raw: str) -> str:
    """Normalize raw fragment like 'The Obd-II Diagnostic Link Connector Dlc' → 'OBD-II Diagnostic Link Connector (DLC)'."""
    # Remove leading articles
    s = re.sub(r"^(The|A|An)\s+", "", raw.strip(), flags=re.I)
    # Fix casing for known acronyms
    acronyms = {
        "obd": "OBD", "obd-ii": "OBD-II", "dlc": "DLC", "dtc": "DTC",
        "ecu": "ECU", "mil": "MIL", "maf": "MAF", "map": "MAP",
        "tdma": "TDMA", "fdma": "FDMA", "can": "CAN", "pid": "PID",
        "rpm": "RPM", "o2": "O2", "egr": "EGR", "evap": "EVAP",
    }
    parts = s.split()
    out_parts = []
    for p in parts:
        core = re.sub(r"[^A-Za-z0-9\-]", "", p).lower()
        if core in acronyms:
            p = re.sub(re.escape(core), acronyms[core], p, flags=re.I)
        out_parts.append(p)
    s = " ".join(out_parts)
    # Clean verbose suffixes
    s = re.sub(r"\s+Are\s+Five.*$", "", s, flags=re.I)
    s = re.sub(r"\s+Is\s+Defined\s+As.*$", "", s, flags=re.I)
    # Remove trailing verb phrases like "Involve Monitoring Switching", "Are Five..." etc.
    s = re.sub(r"\s+Involve\s+.*$", "", s, flags=re.I)
    s = re.sub(r"\s+Should\s+.*$", "", s, flags=re.I)
    # If DLC appears without parentheses, add them if OBD-II present and DLC at end
    if "OBD-II" in s and "DLC" in s and "(DLC)" not in s:
        s = re.sub(r"\s+DLC\s*$", " (DLC)", s)
    if "DLC" in s and "Diagnostic Link Connector" in s and "(DLC)" not in s:
        s = s.replace("DLC", "(DLC)")
        s = re.sub(r"\(+", "(", s)
        s = re.sub(r"\)+", ")", s)
    # Remove trailing "where" clauses
    s = re.sub(r"\s+where\s+.*$", "", s, flags=re.I)
    # For oxygen sensor etc, keep first 2-3 words as concept if verbose
    # e.g., "Oxygen Sensor Diagnostics Involve Monitoring Switching" -> "Oxygen Sensor Diagnostics"
    if len(s.split()) > 3 and re.search(r"\b(Involve|Should|Monitoring|Diagnostics)\b", s, re.I):
        # Keep first 3 words for sensor diagnostics
        if "Oxygen Sensor" in s:
            s = "Oxygen Sensor Diagnostics"
        elif "Sensor Diagnostics" in s:
            s = "Sensor Diagnostics"
    # Truncate verbose to 6 words max for clean titles
    if len(s.split()) > 6:
        s = " ".join(s.split()[:6])
    s = re.sub(r"\s{2,}", " ", s).strip(" -–")
    if len(s) < 3:
        return raw.strip()
    return s


def _extract_procedure(evidence: str) -> Optional[str]:
    # Look for step-like patterns
    if re.search(r"\b(step|procedure|process|diagnos|test|scan tool|actuator|balance test)\b", evidence, re.I):
        # Extract sentences with procedure verbs
        sents = re.split(r"(?<=[.!?])\s+", evidence)
        proc_sents = [s for s in sents if re.search(r"\b(should|must|allow|command|monitor|compare|retrieve|capture|diagnos)\b", s, re.I)]
        if proc_sents:
            return " ".join(proc_sents[:2])[:400]
    return None

def _extract_applications(evidence: str) -> List[str]:
    apps = []
    for pat in [r"used in ([^.]{10,80})", r"application[s]? ([^.]{10,80})", r"predominantly use ([^.]{10,80})"]:
        for m in re.finditer(pat, evidence, re.I):
            apps.append(m.group(1).strip()[:80])
            if len(apps) >= 2:
                break
    return apps

def _extract_misconceptions(concept_name: str, evidence: str) -> List[str]:
    # Heuristic misconceptions per domain — only if contrast term is actually grounded in evidence or broader doc will be checked later
    misc = []
    low = (concept_name + " " + evidence).lower()
    # Only add TDMA vs FDMA if both terms are in evidence (to avoid hallucination)
    if "tdma" in low and "fdma" in low:
        misc.append("Confuse TDMA time slots with FDMA frequency bands")
    elif "tdma" in low and "fdma" not in low:
        # Generic TDMA misconception without FDMA hallucination
        misc.append("Think TDMA guard time is optional — without it, propagation delays cause slot overlap")
    if "p0171" in low:
        misc.append("Students confuse P0171 (lean) with rich condition — opposite fuel trim interpretation")
    if "p0300" in low and "misfire" in low:
        misc.append("Confuse random misfire (P0300) with specific cylinder misfire (P0301-P0308)")
    if "p0420" in low:
        misc.append("Misattribute P0420 to O2 sensor failure vs catalyst oxygen storage degradation")
    if "oxygen sensor" in low:
        misc.append("Think O2 switching slowly is normal — healthy sensor should switch 8+ times/10s at 2500 RPM")
    if "maf" in low:
        misc.append("Confuse MAF under-reporting (lean) with fuel pressure issue")
    if not misc:
        misc.append("Generic misconception: recall without understanding mechanism")
    return misc[:2]

def _extract_numerical_template(evidence: str) -> Optional[Dict[str, Any]]:
    # Only for truly numerical contexts: explicit arrays/calculations or O2 sensor threshold diagnostics
    # DTC codes like P0171 alone should NOT trigger — they are identifiers, not calculations
    # Exclude header-like evidence
    if re.match(r"^\s*MODULE\s*\d+", evidence, re.I):
        return None
    low = evidence.lower()
    # Strict calculation contexts: array, matrix, quick sort, or explicit calculate/compute
    has_explicit_calculation = bool(re.search(r"\barray\s*\[|\bquick\s*sort\b|\bmatrix\s*\[|calculate|compute.*\d|complexity\s*O\(", evidence, re.I))
    has_sensor_threshold = bool(re.search(r"0\.1V.*0\.9V.*8\s*times.*10\s*seconds.*2500\s*RPM", evidence, re.I | re.S))
    if has_sensor_threshold:
        return {
            "type": "sensor_threshold",
            "params": {
                "o2_voltage_low": "0.1V", "o2_voltage_high": "0.9V",
                "switch_rate": "8/10s", "rpm": "2500",
            },
            "verifiable": True
        }
    if has_explicit_calculation:
        return {"type": "calculation", "params": {}, "verifiable": True}
    return None

@dataclass
class KnowledgeUnit:
    """Canonical academic knowledge representation."""
    ku_id: str
    concept: str  # canonical normalized
    raw_concept: str
    definition: str
    evidence: str
    source_chunk_id: str
    source_hash: str
    confidence: float
    concept_type: str

    # Enriched fields
    formula: Optional[str] = None
    procedure: Optional[str] = None
    diagram_ref: Optional[str] = None
    applications: List[str] = field(default_factory=list)
    relationships: List[Dict[str, str]] = field(default_factory=list)  # [{"target": "ECU", "relation": "monitors"}]
    prerequisites: List[str] = field(default_factory=list)
    misconceptions: List[str] = field(default_factory=list)
    numerical_template: Optional[Dict[str, Any]] = None
    expected_answer_canonical: str = ""  # not copied, distilled
    difficulty: str = "medium"
    bloom_suggestions: List[str] = field(default_factory=list)
    word_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ku_id": self.ku_id, "concept": self.concept, "definition": self.definition,
            "formula": self.formula, "procedure": self.procedure, "diagram_ref": self.diagram_ref,
            "applications": self.applications, "relationships": self.relationships,
            "misconceptions": self.misconceptions, "numerical_template": self.numerical_template,
            "expected_answer_canonical": self.expected_answer_canonical, "difficulty": self.difficulty,
            "confidence": self.confidence, "source_hash": self.source_hash
        }


class KnowledgeUnitBuilder:
    """Builds KnowledgeUnit from ExtractedConcept + document context."""

    def build(self, concept: ExtractedConcept) -> KnowledgeUnit:
        canonical = _normalize_concept_name(concept.concept_name)
        # Distill definition: first definitional sentence, not whole evidence
        defn = concept.canonical_definition
        if not defn or len(defn) < 20:
            # Extract first sentence with definition signal from evidence
            sents = re.split(r"(?<=[.!?])\s+", concept.supporting_evidence)
            for s in sents:
                if re.search(r"is defined|is a|are|indicates|is a standardized", s, re.I):
                    defn = s.strip()
                    break
            if not defn:
                defn = sents[0][:200] if sents else concept.supporting_evidence[:200]

        # Formula / equations
        formula = concept.equations[0] if concept.equations else None
        # Procedure
        procedure = _extract_procedure(concept.supporting_evidence)
        # Diagram
        diagram_ref = concept.diagram_refs[0] if concept.diagram_refs else None
        # Applications
        apps = _extract_applications(concept.supporting_evidence)
        # Relationships: heuristic entity linking
        relationships = []
        ev_low = concept.supporting_evidence.lower()
        entities = ["ecu", "o2 sensor", "maf sensor", "dlc", "can", "mil", "dtc", "catalyst", "crankshaft"]
        for ent in entities:
            if ent in ev_low and ent not in canonical.lower():
                relationships.append({"target": ent.upper(), "relation": "related"})
        # Misconceptions
        miscon = _extract_misconceptions(canonical, concept.supporting_evidence)
        # Numerical template
        num_tmpl = _extract_numerical_template(concept.supporting_evidence)
        # Canonical expected answer: distilled, not copied — 2 sentences max, key facts
        canon_ans = self._distill_canonical(defn, procedure, formula)

        # Difficulty from word count and type
        if concept.concept_type == "numerical" or num_tmpl:
            diff = "hard"
        elif len(concept.supporting_evidence.split()) > 250:
            diff = "medium"
        else:
            diff = "easy" if concept.confidence < 0.6 else "medium"

        source_hash = hashlib.sha256(concept.supporting_evidence.encode()).hexdigest()[:12]

        return KnowledgeUnit(
            ku_id=f"KU_{concept.concept_id}",
            concept=canonical,
            raw_concept=concept.concept_name,
            definition=defn[:300],
            evidence=concept.supporting_evidence,
            source_chunk_id=concept.source_chunk_id,
            source_hash=source_hash,
            confidence=concept.confidence,
            concept_type=concept.concept_type,
            formula=formula,
            procedure=procedure,
            diagram_ref=diagram_ref,
            applications=apps,
            relationships=relationships[:3],
            prerequisites=concept.prerequisites[:2],
            misconceptions=miscon,
            numerical_template=num_tmpl,
            expected_answer_canonical=canon_ans,
            difficulty=diff,
            bloom_suggestions=concept.bloom_suggestions,
            word_count=concept.word_count
        )

    def build_batch(self, concepts: List[ExtractedConcept]) -> List[KnowledgeUnit]:
        return [self.build(c) for c in concepts]

    def _distill_canonical(self, definition: str, procedure: Optional[str], formula: Optional[str]) -> str:
        # Canonical answer should be concise, not verbatim copy
        parts = [definition.strip().rstrip(".")]
        if procedure:
            # Add procedure summary one sentence
            proc_sent = re.split(r"(?<=[.!?])\s+", procedure)[0]
            parts.append(proc_sent.strip())
        ans = ". ".join(parts) + "."
        if formula:
            ans += f" Relevant: {formula}."
        # Ensure not >300 chars
        if len(ans) > 350:
            ans = ans[:340] + "..."
        return ans
