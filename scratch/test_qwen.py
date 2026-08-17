# scratch/test_qwen.py

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from v0_1.llm import RobustLLMCaller
from core.contracts.question_slot import QuestionSlot
from core.contracts.budgets import AnswerBudget, QuestionBudget
from core.contracts.task_signature import TaskSignature
from core.generation.orchestrator import SlotOrchestrator

def main():
    answer_budget = AnswerBudget.from_marks_and_bloom(6, "L2")
    question_budget = QuestionBudget.from_bloom("L2", 6)
    task_signature = TaskSignature.from_bloom_marks_type("L2", 6, "descriptive")

    slot = QuestionSlot(
        slot_id="module_1_Q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="mod1_OR_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L2",
        bloom_verb="Explain",
        bloom_operation="UNDERSTAND",
        co="CO1",
        difficulty="MIXED",
        question_type="descriptive",
        topic="Transmission Control Protocol (TCP)",
        evidence_ids=("chk1",),
        answer_budget=answer_budget,
        question_budget=question_budget,
        task_signature=task_signature,
        math_required=False,
        visual_required=False,
        generation_seed=123
    )

    evidence_text = "Transmission Control Protocol (TCP) is a connection-oriented transport layer protocol. TCP provides reliable, ordered, and error-checked delivery of a stream of octets."
    
    orchestrator = SlotOrchestrator(artifact=None)
    prompt = orchestrator._format_prompt(slot, evidence_pack=evidence_text, extra_hints="")
    
    print("--- PROMPT ---")
    print(prompt)
    print("--------------")
    
    caller = RobustLLMCaller()
    res = caller.call(prompt, max_tokens=1024)
    print("\n--- RESPONSE ---")
    print(res)
    print("----------------")
    
    if res:
        try:
            data = orchestrator._parse_json(res)
            print("Parsed Data:", data)
        except Exception as e:
            print("Parsing Error:", e)

if __name__ == "__main__":
    main()
