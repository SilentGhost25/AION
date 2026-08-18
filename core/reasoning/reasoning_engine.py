"""
Reasoning Engine — Decides what to ask and how to think
Per audit: Text -> Concept -> Question works for definitions, not engineering.
Engineering needs: Concepts + Relationships + Procedures + Calculations + Figures + Constraints

Reasoning Engine sits between Knowledge Unit Builder and Planner:
  Extraction -> Concept -> Knowledge Unit Builder -> Grounding -> Reasoning Engine -> Planner -> Composer -> Self-Critic -> Auditor
"""

import re
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from core.knowledge.knowledge_unit import KnowledgeUnit

@dataclass
class ReasoningIntent:
    ku_id: str
    intent_type: str  # scenario | numerical | misconception | relationship | procedure | diagram | recall
    scenario_prompt: Optional[str] = None  # e.g., "Vehicle reports P0171 after intake manifold replacement"
    misconception_target: Optional[str] = None
    numerical_transform: Optional[Dict[str, Any]] = None
    relationship_focus: Optional[Dict[str, str]] = None
    procedure_step_focus: Optional[str] = None
    bloom_target: int = 2
    examiner_pattern: str = "standard"  # e.g., DTC case study, sensor analysis
    reasoning_operations: List[str] = field(default_factory=list)  # e.g., ["identify", "justify", "sequence"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ku_id": self.ku_id, "intent_type": self.intent_type,
            "scenario": self.scenario_prompt, "misconception": self.misconception_target,
            "bloom": self.bloom_target, "examiner_pattern": self.examiner_pattern,
            "operations": self.reasoning_operations
        }

class ReasoningEngine:
    """Deterministic reasoning over Knowledge Units."""

    SCENARIO_TEMPLATES = {
        "DTC": "A {vehicle} exhibits DTC {code} after {trigger}. The technician {action}. Evaluate the decision and suggest a diagnostic sequence. Justify each test.",
        "sensor": "During live data monitoring at {rpm} RPM, the O2 sensor shows {voltage} with switching rate {rate}. Diagnose the sensor condition and recommend the next verification test.",
        "actuator": "An actuator test on {actuator} shows {result}. Interpret the result and identify the probable cause.",
        "misfire": "A vehicle shows rough idle and MIL with DTC P0302. Crankshaft sensor data indicates lack of acceleration on cylinder 2. List probable causes in order of likelihood and justify the diagnostic order.",
        "procedure": "Outline the diagnostic sequence for {concept} using a scan tool. Include PID checks and expected values. Justify the order.",
    }

    def reason(self, kus: List[KnowledgeUnit]) -> List[ReasoningIntent]:
        # If only one KU in doc, force recall/definition to avoid hallucinated relationship
        if len(kus) == 1:
            ku = kus[0]
            # For single-concept, always do recall/definition, not relationship
            return [ReasoningIntent(ku_id=ku.ku_id, intent_type="recall", bloom_target=2, examiner_pattern="definition", reasoning_operations=["recall", "explain"])]
        intents = []
        for ku in kus:
            intent = self._reason_single(ku)
            intents.append(intent)
        return intents

    def _reason_single(self, ku: KnowledgeUnit) -> ReasoningIntent:
        low = (ku.concept + " " + ku.definition + " " + (ku.procedure or "")).lower()
        # Choose intent type based on KU enrichment — numerical if template exists (any difficulty, deterministic generation)
        if ku.numerical_template:
            return self._numerical_intent(ku)
        if ku.misconceptions and "confuse" in ku.misconceptions[0].lower():
            return self._misconception_intent(ku)
        if ku.procedure and len(ku.procedure.split()) > 20:
            return self._procedure_intent(ku)
        if ku.diagram_ref or "diagram" in ku.concept_type:
            return ReasoningIntent(ku_id=ku.ku_id, intent_type="diagram", bloom_target=3, examiner_pattern="diagram_interpretation", reasoning_operations=["interpret", "explain"], scenario_prompt=f"With reference to the {ku.diagram_ref} for {ku.concept}, explain the signal flow.")
        if ku.relationships and len(ku.relationships) > 0:
            # Only relationship if target is meaningful and not generic like "node" or "satellite" for single-concept
            # For single-concept docs, prefer recall over generic relationship — also filter signal, traversal, insertion etc. for single-concept
            targets = [r.get("target","").lower() for r in ku.relationships]
            generic = ["node", "satellite", "ecu", "bst", "tree", "signal", "deletion", "insertion", "traversal", "time complexity", "search"]
            if any(t not in generic for t in targets):
                return self._relationship_intent(ku)
            # Generic relationship with node/satellite/signal is not meaningful for single concept — fallback to recall
        if "p0171" in low or "p0300" in low or "p0420" in low:
            return self._scenario_intent(ku, low)
        # Default recall but add misconception if available
        return ReasoningIntent(ku_id=ku.ku_id, intent_type="recall", bloom_target=2 if ku.difficulty=="easy" else 3, examiner_pattern="definition", reasoning_operations=["recall", "explain"])

    def _scenario_intent(self, ku: KnowledgeUnit, low: str) -> ReasoningIntent:
        if "p0171" in low:
            return ReasoningIntent(
                ku_id=ku.ku_id, intent_type="scenario", bloom_target=5,
                scenario_prompt="A vehicle exhibits DTC P0171 (System Too Lean Bank 1) after replacing the intake manifold. The technician suspects fuel injectors and proposes replacement. Evaluate this decision, identify probable causes (vacuum leak, MAF under-reporting, fuel pressure), and suggest a diagnostic sequence. Justify each test.",
                misconception_target=ku.misconceptions[0] if ku.misconceptions else None,
                examiner_pattern="DTC_case_study", reasoning_operations=["identify", "evaluate", "sequence", "justify"]
            )
        if "p0300" in low:
            return ReasoningIntent(
                ku_id=ku.ku_id, intent_type="scenario", bloom_target=5,
                scenario_prompt="Engine shows DTC P0302 (Cylinder 2 misfire) with rough idle. The scan tool balance test shows even RPM drop on all cylinders except cylinder 2. Identify probable causes and justify the next diagnostic step.",
                examiner_pattern="misfire_case", reasoning_operations=["diagnose", "prioritize", "justify"]
            )
        if "p0420" in low:
            return ReasoningIntent(
                ku_id=ku.ku_id, intent_type="scenario", bloom_target=4,
                scenario_prompt="Downstream O2 sensor shows switching amplitude similar to upstream during steady cruise. DTC P0420 is stored. Differentiate between catalyst failure and O2 sensor fault. What additional test confirms?",
                examiner_pattern="catalyst_diagnosis", reasoning_operations=["compare", "differentiate", "propose_test"]
            )
        # Generic scenario
        scenario = self.SCENARIO_TEMPLATES["sensor"].format(rpm="2500", voltage="0.45V stuck", rate="2/10s")
        return ReasoningIntent(ku_id=ku.ku_id, intent_type="scenario", bloom_target=4, scenario_prompt=scenario, examiner_pattern="sensor_diagnosis", reasoning_operations=["interpret", "diagnose"])

    def _misconception_intent(self, ku: KnowledgeUnit) -> ReasoningIntent:
        return ReasoningIntent(
            ku_id=ku.ku_id, intent_type="misconception", bloom_target=5,
            scenario_prompt=f"Students often confuse: {ku.misconceptions[0]}. Design a question that exposes this misconception. Vehicle scenario: {ku.concept} — ask to predict or explain the error if misconception applied.",
            misconception_target=ku.misconceptions[0],
            examiner_pattern="misconception_probe", reasoning_operations=["predict_mistake", "explain_correct", "justify"]
        )

    def _numerical_intent(self, ku: KnowledgeUnit) -> ReasoningIntent:
        return ReasoningIntent(
            ku_id=ku.ku_id, intent_type="numerical", bloom_target=3,
            numerical_transform=ku.numerical_template,
            examiner_pattern="numerical_application", reasoning_operations=["calculate", "interpret", "verify"]
        )

    def _procedure_intent(self, ku: KnowledgeUnit) -> ReasoningIntent:
        return ReasoningIntent(
            ku_id=ku.ku_id, intent_type="procedure", bloom_target=4,
            procedure_step_focus=ku.procedure[:120] if ku.procedure else None,
            scenario_prompt=f"For {ku.concept}, outline the scan-tool diagnostic sequence. Include PID checks ({ku.procedure[:80]}...) and expected values. Justify the order.",
            examiner_pattern="procedure_sequencing", reasoning_operations=["sequence", "justify", "predict"]
        )

    def _relationship_intent(self, ku: KnowledgeUnit) -> ReasoningIntent:
        rel = ku.relationships[0] if ku.relationships else {"target": "ECU", "relation": "monitors"}
        return ReasoningIntent(
            ku_id=ku.ku_id, intent_type="relationship", bloom_target=4,
            relationship_focus=rel,
            scenario_prompt=f"Explain the relationship between {ku.concept} and {rel['target']} ({rel['relation']}). How does failure of {ku.concept} manifest in {rel['target']} data?",
            examiner_pattern="relationship_analysis", reasoning_operations=["explain", "relate", "predict"]
        )
