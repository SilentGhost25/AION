"""
Educational Reasoner — Infers assessment opportunities from Knowledge Unit
Per audit: Knowledge Unit -> Educational Reasoner -> Planner (not KU -> Planner)
Decides: Can this concept become Definition, Comparison, Case Study, Numerical, Diagram, Proof, Procedure, Derivation, Design, Evaluation?

Example BST can generate: Definition, Insertion, Deletion, Traversal, Complexity, AVL comparison, Real world scenario, Implementation, Numerical tracing
"""

from typing import List, Dict, Any
import re

class EducationalReasoner:
    """Determines all possible assessment opportunities for a Knowledge Unit."""
    
    OPPORTUNITY_TYPES = [
        "Definition", "Comparison", "Case Study", "Numerical", "Diagram",
        "Proof", "Procedure", "Derivation", "Design", "Evaluation",
        "Implementation", "Complexity", "Real-world", "Debugging", "Optimization"
    ]
    
    def infer_opportunities(self, knowledge_unit, subject_profile=None) -> List[Dict[str, Any]]:
        """Infer all assessment opportunities for a KU — subject-aware."""
        concept_low = knowledge_unit.concept.lower()
        definition_low = knowledge_unit.definition.lower()
        evidence_low = knowledge_unit.evidence.lower() if hasattr(knowledge_unit, 'evidence') else ""
        
        opportunities = []
        
        # Always possible: Definition/Recall
        opportunities.append({
            "type": "Definition",
            "bloom": 1,
            "description": f"Define {knowledge_unit.concept}",
            "student_ability": f"can recall definition of {knowledge_unit.concept}",
            "operations": ["recall", "define"],
        })
        
        # Check for numerical potential
        if knowledge_unit.numerical_template or re.search(r"\barray\b|keys\s*\[|complexity|O\(|2\^|insertion", evidence_low, re.I):
            opportunities.append({
                "type": "Numerical",
                "bloom": 3,
                "description": f"Perform numerical tracing for {knowledge_unit.concept}",
                "student_ability": "can trace algorithm with fresh values",
                "operations": ["trace", "calculate", "verify"],
            })
        
        # Check for procedure
        if knowledge_unit.procedure or re.search(r"\bstep\b|procedure|process|insertion|deletion|traversal", evidence_low, re.I):
            opportunities.append({
                "type": "Procedure",
                "bloom": 3,
                "description": f"Outline procedure for {knowledge_unit.concept}",
                "student_ability": "can sequence steps and justify order",
                "operations": ["sequence", "justify", "execute"],
            })
        
        # Check for comparison (if relationships exist)
        if knowledge_unit.relationships:
            for rel in knowledge_unit.relationships:
                if rel.get("relation") in ["comparison", "prerequisite", "related"]:
                    opportunities.append({
                        "type": "Comparison",
                        "bloom": 4,
                        "description": f"Compare {knowledge_unit.concept} with {rel['target']}",
                        "student_ability": "can differentiate and contrast",
                        "operations": ["compare", "differentiate", "contrast"],
                    })
                    break
        
        # Check for case study (if applications exist or is AU/ECE)
        if knowledge_unit.applications or (subject_profile and subject_profile.code in ["AU", "ECE"]):
            opportunities.append({
                "type": "Case Study",
                "bloom": 5,
                "description": f"Diagnose case for {knowledge_unit.concept}",
                "student_ability": "can diagnose and propose solution",
                "operations": ["diagnose", "propose", "justify"],
            })
        
        # Check for diagram
        if knowledge_unit.diagram_ref or re.search(r"\btree\b|graph|diagram|circuit", concept_low, re.I):
            opportunities.append({
                "type": "Diagram",
                "bloom": 3,
                "description": f"Draw and explain {knowledge_unit.concept} diagram",
                "student_ability": "can visualize and interpret structure",
                "operations": ["draw", "interpret", "explain"],
            })
        
        # Check for complexity/proof
        if re.search(r"\bcomplexity\b|O\(|proof|theorem", evidence_low, re.I):
            opportunities.append({
                "type": "Complexity",
                "bloom": 4,
                "description": f"Analyse time/space complexity for {knowledge_unit.concept}",
                "student_ability": "can analyse complexity and prove bounds",
                "operations": ["analyse", "prove", "compare"],
            })
        
        # Check for design
        if re.search(r"\bdesign\b|implementation|construct", evidence_low, re.I):
            opportunities.append({
                "type": "Design",
                "bloom": 6,
                "description": f"Design {knowledge_unit.concept} for given constraints",
                "student_ability": "can design and justify choices",
                "operations": ["design", "justify", "evaluate"],
            })
        
        return opportunities[:4]  # Return top 4 diverse opportunities
    
    def choose_best_opportunity(self, knowledge_unit, subject_profile=None, used_archetypes=None) -> Dict[str, Any]:
        """Choose best opportunity considering pedagogical variation."""
        opportunities = self.infer_opportunities(knowledge_unit, subject_profile)
        if not opportunities:
            return {
                "type": "Definition", "bloom": 1,
                "description": f"Define {knowledge_unit.concept}",
                "student_ability": "can recall",
                "operations": ["recall"],
            }
        
        # Filter out already used archetypes if provided
        if used_archetypes:
            unused = [op for op in opportunities if op["type"] not in used_archetypes]
            if unused:
                return unused[0]
        
        # Default: pick first non-definition for variety, or definition if only one
        for op in opportunities:
            if op["type"] != "Definition":
                return op
        return opportunities[0]
