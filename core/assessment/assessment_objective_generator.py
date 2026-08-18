"""
Assessment Objective Generator — Learning Objective -> Student Ability -> Scenario -> Question
Per audit: Planner thinks like parser (concept -> relationship). Should be: Learning Objective -> Student Ability -> Scenario -> Question

This layer sits after Knowledge Unit, before Planner:
Extraction -> Concept -> Knowledge Unit -> Assessment Objective -> Planner -> Composer
"""

from typing import List, Dict
from dataclasses import dataclass
import re

@dataclass
class AssessmentObjective:
    objective: str  # e.g., "Analyse why heap gives O(log n) insertion while maintaining priority order"
    student_ability: str  # e.g., "can trace heap insertion and compare with array"
    scenario_type: str  # e.g., "Hospital emergency queue", "BST insertion", "Heat exchanger"
    question_type: str  # Algorithmic, Analytical, Numerical, Case Study, Design, etc.
    bloom_level: int
    bloom_label: str
    marks: int
    requires_numerical: bool
    requires_diagram: bool
    learning_outcome: str

# Mapping from KU concept patterns to assessment objectives
OBJECTIVE_TEMPLATES = {
    "BST": {
        "insertion": {
            "objective": "Perform BST insertion for a given key sequence and justify the resulting tree structure",
            "ability": "can trace BST insertion algorithm step-by-step and compare balanced vs skewed outcomes",
            "scenario": "BST initially contains {existing_keys}; Insert {new_keys}; Show tree after each insertion and justify final structure",
            "type": "Algorithmic", "bloom": 3, "marks": 10, "numerical": True, "diagram": True,
        },
        "search": {
            "objective": "Analyse BST search efficiency and compare with linear search",
            "ability": "can analyse time complexity O(h) and worst-case O(n)",
            "scenario": "Search for key {key} in BST with root {root}; Trace comparison path and count comparisons",
            "type": "Analytical", "bloom": 4, "marks": 8, "numerical": False, "diagram": False,
        },
    },
    "stack": {
        "conversion": {
            "objective": "Demonstrate stack use for infix to postfix conversion and evaluation",
            "ability": "can apply stack precedence and evaluate postfix expression",
            "scenario": "A compiler uses a stack while evaluating expressions. Demonstrate conversion of {infix} into postfix and evaluate with stack, explaining every intermediate step",
            "type": "Algorithmic", "bloom": 3, "marks": 10, "numerical": False, "diagram": False,
        },
        "overflow": {
            "objective": "Analyse stack overflow and underflow conditions for array implementation",
            "ability": "can identify boundary conditions and prevent errors",
            "scenario": "A stack of capacity {capacity} has top={top}; Evaluate push/pop sequence {ops} and identify overflow/underflow",
            "type": "Analytical", "bloom": 4, "marks": 8, "numerical": True, "diagram": False,
        },
    },
    "queue": {
        "circular": {
            "objective": "Analyse circular queue full/empty conditions and distinguish from linear queue",
            "ability": "can trace circular queue pointers and explain why one slot is wasted",
            "scenario": "A circular queue has capacity {capacity}, front={front}, rear={rear}; Insert {keys} and show queue after each operation, explain overflow condition",
            "type": "Numerical", "bloom": 4, "marks": 10, "numerical": True, "diagram": True,
        },
    },
    "priority queue": {
        "heap": {
            "objective": "Analyse why heap implementation gives O(log n) insertion while maintaining priority order",
            "ability": "can analyse heap insertion vs array implementation and compare time complexity",
            "scenario": "Hospital emergency queue with priorities {priorities}; Show heap after each insertion and compare with array implementation",
            "type": "Analytical", "bloom": 4, "marks": 10, "numerical": True, "diagram": True,
        }
    },
    "binary tree": {
        "traversal": {
            "objective": "Perform inorder, preorder, and postorder traversals and explain their applications",
            "ability": "can trace recursion and distinguish traversal orders",
            "scenario": "Given binary tree with root {root} and structure {structure}; Perform inorder, preorder, postorder and explain when each is used",
            "type": "Algorithmic", "bloom": 3, "marks": 8, "numerical": False, "diagram": True,
        }
    },
}

class AssessmentObjectiveGenerator:
    """Generates assessment objectives from Knowledge Units — subject-aware."""
    
    def generate(self, knowledge_units, subject_profile=None) -> List[AssessmentObjective]:
        objectives = []
        for ku in knowledge_units:
            obj = self._generate_single(ku, subject_profile)
            objectives.append(obj)
        return objectives
    
    def _generate_single(self, ku, subject_profile=None):
        concept_low = ku.concept.lower()
        definition_low = ku.definition.lower()
        
        # Try to match objective templates
        for key, variants in OBJECTIVE_TEMPLATES.items():
            if key in concept_low or key in definition_low:
                # Pick first variant or match by evidence keywords
                variant_key = list(variants.keys())[0]
                # Check for more specific match
                for v_key, template in variants.items():
                    if v_key in concept_low or v_key in definition_low or v_key in (ku.procedure or "").lower():
                        variant_key = v_key
                        break
                template = variants[variant_key]
                return AssessmentObjective(
                    objective=template["objective"],
                    student_ability=template["ability"],
                    scenario_type=template["scenario"],
                    question_type=template["type"],
                    bloom_level=template["bloom"],
                    bloom_label={1:"Remember",2:"Understand",3:"Apply",4:"Analyse",5:"Evaluate",6:"Create"}[template["bloom"]],
                    marks=template["marks"],
                    requires_numerical=template["numerical"],
                    requires_diagram=template["diagram"],
                    learning_outcome=f"Student should {template['objective'].lower()}"
                )
        
        # Fallback: create objective from KU definition and misconception
        # Determine type from KU content
        if ku.numerical_template:
            q_type = "Numerical"
            bloom = 3
        elif ku.procedure:
            q_type = "Procedure"
            bloom = 4
        elif ku.diagram_ref:
            q_type = "Diagram"
            bloom = 3
        else:
            q_type = "Conceptual"
            bloom = 2 if ku.difficulty == "easy" else 3
        
        return AssessmentObjective(
            objective=f"Explain {ku.concept} and its significance in {ku.concept} applications",
            student_ability=f"can recall and explain {ku.concept}",
            scenario_type=f"Explain {ku.concept} where {ku.definition[:80]}",
            question_type=q_type,
            bloom_level=bloom,
            bloom_label={1:"Remember",2:"Understand",3:"Apply",4:"Analyse",5:"Evaluate",6:"Create"}[bloom],
            marks=10 if bloom >= 4 else 8,
            requires_numerical=bool(ku.numerical_template),
            requires_diagram=bool(ku.diagram_ref),
            learning_outcome=f"Student should understand {ku.concept}"
        )
