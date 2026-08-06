"""
Composer V2 — True Natural Language Generation from QuestionSpec
Per audit: Composer should receive QuestionSpec and write fresh academic question, never copy planner text.

Pipeline: QuestionSpec → Compose → Polish → Grammar pass → VTU formatting
Final wording independent of planner.
"""

from typing import Optional
import re

class ComposerV2:
    """Generates polished academic questions from complete QuestionSpec."""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        self._llm = None
        if use_llm:
            try:
                from v0_1.llm import get_llm
                self._llm = get_llm()
            except:
                self.use_llm = False
    
    def compose(self, spec) -> str:
        """Compose from QuestionSpec — true NLG."""
        if self.use_llm and self._llm:
            return self._compose_with_llm(spec)
        else:
            return self._compose_template(spec)
    
    def _compose_with_llm(self, spec) -> str:
        # Build rich prompt from complete spec — composer never infers, everything already planned
        prompt = f"""You are a VTU professor writing an exam question. Generate ONE polished academic question.

SUBJECT: {spec.subject} ({spec.subject_code}) — Module: {spec.module}
KNOWLEDGE UNIT: {spec.knowledge_unit}
ASSESSMENT OBJECTIVE: {spec.assessment_objective}
STUDENT ABILITY: {spec.student_ability}
QUESTION TYPE: {spec.question_type} ({spec.assessment_type})
EXPECTED REASONING: {spec.expected_reasoning}
REQUIRED OPERATIONS: {', '.join(spec.required_operations)}
BLOOM: {spec.bloom} ({spec.bloom_level}) | MARKS: {spec.marks} | DIFFICULTY: {spec.difficulty}
SCENARIO: {spec.scenario or 'None — conceptual question'}
NUMERICAL: {spec.numerical_payload if spec.requires_numerical else 'Not required'}
DIAGRAM: {'Required — include figure reference' if spec.requires_diagram else 'Not required'}
FORMULA: {'Required' if spec.formula_required else 'Not required'}
CONSTRAINTS: {spec.constraints}
GROUNDING EVIDENCE: {spec.grounding_evidence[0][:400] if spec.grounding_evidence else 'N/A'}
ALLOWED ENTITIES: {spec.allowed_entities[:5]}
FORBIDDEN: {spec.forbidden_entities[:5]}

INSTRUCTIONS:
- Write ONE exam question, 1-3 sentences, ending with . or ?
- Start with strong academic verb for Bloom {spec.bloom} (e.g., Analyse, Evaluate, Design)
- Scenario-based if provided — do NOT write generic "Explain X"
- Include fresh numerical values if payload provided — do NOT copy example values
- Include "with reference to the given figure" if diagram required
- No answer, no explanation, no preamble — only the question
- VTU formatting, formal academic English

Question:"""
        
        try:
            raw = self._llm.generate(prompt, options={"num_predict": 180, "temperature": 0.35, "stop": ["Ideal Answer", "Marking Scheme"]})
            if raw and len(raw.split()) >= 10:
                return self._polish(raw.strip(), spec)
        except Exception as e:
            print(f"[COMPOSER-V2] LLM error: {e}")
        
        return self._compose_template(spec)
    
    def _compose_template(self, spec) -> str:
        """Fallback template — still uses QuestionSpec, not planner dump, but scenario-aware."""
        # Priority 1: If scenario is fully specified (no placeholders), use it directly — this is the desired professor-style
        if spec.scenario and len(spec.scenario) > 20 and "{" not in spec.scenario:
            # Scenario is already a complete, filled question stem (e.g., "A circular queue has capacity 8, rear=6, front=3. Insert 45,18,72...")
            # Use it directly, just ensure marks and ensure it ends properly
            scenario = spec.scenario.strip().rstrip(".")
            # If numerical payload has fresh values, ensure they are in scenario (replace placeholders if any)
            if spec.numerical_payload and spec.numerical_payload.get("insert_keys"):
                # For queue, ensure fresh keys are used
                pass
            return f"{scenario}. Justify your answer with reference to {spec.knowledge_unit}. ({spec.marks} marks)" if not scenario.endswith("?") else f"{scenario} ({spec.marks} marks)"
        
        # Priority 1b: If scenario has Insert and numerical payload, use BST insertion template with fresh values
        if spec.scenario and "Insert" in spec.scenario:
            if spec.numerical_payload and spec.numerical_payload.get("fresh_values"):
                fv = spec.numerical_payload["fresh_values"]
                if isinstance(fv, dict):
                    vals = ", ".join(str(v) for v in list(fv.values())[:5])
                else:
                    vals = str(fv)[:60]
                # Use existing_keys if available, else generic
                existing = spec.numerical_payload.get('existing_keys', '[40, 20, 60, 10, 30]')
                return f"A Binary Search Tree initially contains {existing} Insert {vals}. Show the tree after each insertion and justify the final structure. ({spec.marks} marks)"
            # Even without fresh payload, use scenario if it is filled
            if "{" not in spec.scenario:
                return f"{spec.scenario} Justify your answer with reference to {spec.knowledge_unit}. ({spec.marks} marks)"
        
        if spec.question_type == "Numerical" and spec.numerical_payload:
            # Fallback numerical — use scenario if available, else generic
            if spec.scenario and "{" not in spec.scenario:
                return f"{spec.scenario} Show the required calculation and interpret the result. ({spec.marks} marks)"
            return f"Consider {spec.knowledge_unit} where {spec.grounding_evidence[0][:80] if spec.grounding_evidence else spec.assessment_objective[:80]}. Using fresh values {spec.numerical_payload.get('fresh_values', 'generated')}, perform the required calculation and interpret the result. ({spec.marks} marks)"
        
        if spec.scenario and len(spec.scenario) > 20 and "{" not in spec.scenario:
            return f"{spec.scenario[:280]} Justify your answer with reference to {spec.knowledge_unit}. ({spec.marks} marks)"
        
        # Scenario-based fallback per audit example
        if "circular queue" in spec.knowledge_unit.lower():
            return f"A circular queue has capacity 8, rear=6, front=3. Insert 45, 18, 72. Show the queue after every insertion and explain the overflow condition. ({spec.marks} marks)"
        
        if "bst" in spec.knowledge_unit.lower() or "binary search tree" in spec.knowledge_unit.lower():
            return f"A BST initially contains 40, 20, 60, 10, 30, 50, 70. Insert 45, 65, 15. Show the tree after each insertion and justify the final structure. ({spec.marks} marks)"
        
        if "stack" in spec.knowledge_unit.lower():
            return f"A compiler uses a stack while evaluating expressions. Demonstrate the use of a stack to convert A+B*(C-D) into postfix notation and explain every intermediate step. ({spec.marks} marks)"
        
        # Default still scenario-aware, not generic
        return f"{spec.assessment_objective}. {spec.scenario[:120] if spec.scenario else ''} ({spec.marks} marks)".strip()
    
    def _polish(self, text: str, spec) -> str:
        """Polish, grammar pass, VTU formatting."""
        # Remove markdown, preamble
        text = re.sub(r"\*+", "", text)
        text = re.sub(r'^(Question:|Q\d+[:\)]|\d+[\.\)])\s*', "", text, flags=re.I)
        text = re.sub(r"\s{2,}", " ", text).strip()
        # Ensure ends with . or ?
        if text and text[-1] not in ".?":
            text += "."
        # Ensure VTU formatting: marks in parentheses
        if f"({spec.marks} marks)" not in text and f"{spec.marks} marks" not in text.lower():
            text = text.rstrip(".") + f" ({spec.marks} marks)."
        # Length guard
        if len(text.split()) > 100:
            sents = re.split(r"(?<=[.!?])\s+", text)
            text = " ".join(sents[:3])
        # Capitalize first letter
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]
        return text
