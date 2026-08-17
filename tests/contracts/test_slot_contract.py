# tests/contracts/test_slot_contract.py

import pytest
from core.contracts.question_slot import QuestionSlot
from core.contracts.task_signature import TaskSignature
from core.contracts.budgets import AnswerBudget, QuestionBudget


def make_slot(**kwargs) -> QuestionSlot:
    defaults = dict(
        slot_id="Q5a", question_no=5, sub_label="a",
        or_pair_id="OR_3", is_alternative=False,
        module_id=3, marks=6,
        bloom_level="L4", bloom_verb="Analyze", bloom_operation="ANALYZE",
        co="CO3", difficulty="MEDIUM", question_type="ANALYTICAL",
        topic="test", evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L4"),
        question_budget=QuestionBudget.from_bloom("L4", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L4", 6, "ANALYTICAL"),
        math_required=False, visual_required=False, generation_seed=42,
    )
    defaults.update(kwargs)
    return QuestionSlot(**defaults)


class TestSlotImmutability:

    def test_slot_is_frozen(self):
        slot = make_slot()
        with pytest.raises(Exception):
            slot.co = "CO5"

    def test_make_attempt_slot_returns_new_object(self):
        slot  = make_slot(generation_seed=100)
        retry = slot.make_attempt_slot(1)
        assert retry.generation_seed == 101
        assert slot.generation_seed  == 100  # original unchanged

    def test_make_attempt_slot_preserves_contract(self):
        slot  = make_slot(co="CO3", marks=6, bloom_level="L4")
        retry = slot.make_attempt_slot(2)
        assert retry.co          == "CO3"
        assert retry.marks       == 6
        assert retry.bloom_level == "L4"


class TestSlotValidation:

    def test_marks_zero_raises(self):
        with pytest.raises(ValueError, match="marks"):
            make_slot(marks=0)

    def test_invalid_bloom_level_raises(self):
        with pytest.raises(ValueError, match="bloom"):
            make_slot(bloom_level="L9")

    def test_co_without_prefix_raises(self):
        with pytest.raises(ValueError, match="CO"):
            make_slot(co="3")
