"""
AION Structural Architecture v2 — Automated Pytest Suite
=========================================================
Verifies all 11 parts of AION Structural Architecture v2:
  - Immutable contracts & structure locking
  - MDE policies (BALANCED, PRIMARY_HEAVY, PROGRESSIVE with safety borrowing, CUSTOM)
  - JBMCS joint Bloom x Mark constraint solver
  - OR Pair Equivalence Profile symmetry
  - SeedManager HMAC-SHA256 seed reproducibility
  - ContentRandomizer multi-register deduplication
  - QwenAdapter linguistic realization boundaries
  - PaperContractVerifier 38 quantitative checks (C01-C38) and SLOT_REGEN recovery
"""

import pytest
from v0_1.structural_v2.contracts import (
    Alternative,
    AlternativeEquivalenceProfile,
    BloomLevel,
    DifficultyBand,
    DistributionPolicy,
    ORPair,
    QuestionSlot,
    RecoveryAction,
    SlotStatus,
    StructuralSignature,
    VisualDecision,
    VisualPrior,
)
from v0_1.structural_v2.mde import DistributionError, MarkDistributionEngine
from v0_1.structural_v2.jbmcs import JointBloomMarkConstraintSolver
from v0_1.structural_v2.equivalence import (
    DifficultyResolver,
    DomainQuestionTypeMatrix,
    ORPairEquivalenceBuilder,
)
from v0_1.structural_v2.visual_pipeline import VisualDecisionPipeline
from v0_1.structural_v2.seed_manager import SeedManager
from v0_1.structural_v2.content_randomizer import ContentRandomizer
from v0_1.structural_v2.qwen_adapter import QwenAdapter, build_context
from v0_1.structural_v2.verifier import PaperContractVerifier


def test_structural_signature_invariants():
    sig1 = StructuralSignature(
        total_marks=10,
        sub_question_count=2,
        mark_distribution=(6, 4),
        distribution_policy=DistributionPolicy.PRIMARY_HEAVY,
    )
    sig2 = StructuralSignature(
        total_marks=10,
        sub_question_count=2,
        mark_distribution=(6, 4),
        distribution_policy=DistributionPolicy.BALANCED,
    )

    # Hashability and equality ignoring policy
    assert sig1 == sig2
    assert hash(sig1) == hash(sig2)
    assert repr(sig1) == "σ(marks=10, n=2, D=[6, 4])"

    # Precondition assertion failure checks
    with pytest.raises(AssertionError):
        StructuralSignature(total_marks=10, sub_question_count=2, mark_distribution=(5, 4))

    with pytest.raises(AssertionError):
        StructuralSignature(total_marks=10, sub_question_count=3, mark_distribution=(6, 4))

    with pytest.raises(AssertionError):
        StructuralSignature(total_marks=10, sub_question_count=2, mark_distribution=(10, 0))


def test_mark_distribution_engine_policies():
    # 1. BALANCED (10, 4) -> (3, 3, 2, 2)
    d_bal = MarkDistributionEngine.compute(10, 4, DistributionPolicy.BALANCED)
    assert d_bal == (3, 3, 2, 2)

    # 2. PRIMARY_HEAVY (10, 2) -> (6, 4)
    d_ph = MarkDistributionEngine.compute(10, 2, DistributionPolicy.PRIMARY_HEAVY)
    assert d_ph == (6, 4)

    # 3. PROGRESSIVE (10, 3) -> raw [5, 3, 1] + deficit front-load -> [6, 3, 1] -> Safety borrowing -> (5, 3, 2)
    d_prog = MarkDistributionEngine.compute(10, 3, DistributionPolicy.PROGRESSIVE)
    assert d_prog == (5, 3, 2)
    assert d_prog[2] >= 2  # Non-L1 position minimum 2 marks safety enforced

    # 4. CUSTOM (10, 3, custom=[5, 3, 2])
    d_cust = MarkDistributionEngine.compute(10, 3, DistributionPolicy.CUSTOM, custom=[5, 3, 2])
    assert d_cust == (5, 3, 2)

    with pytest.raises(DistributionError):
        MarkDistributionEngine.compute(10, 3, DistributionPolicy.CUSTOM, custom=[6, 3])


def test_joint_bloom_mark_constraint_solver():
    rng = SeedManager.get_rng(12345)
    bloom_levels = [BloomLevel.L2, BloomLevel.L3, BloomLevel.L4]

    # Slot distribution (6, 4)
    P = JointBloomMarkConstraintSolver.solve((6, 4), bloom_levels, sub_question_count=2, rng=rng)
    assert len(P) == 2
    assert P[0] in (BloomLevel.L3, BloomLevel.L4)
    assert P[1] in (BloomLevel.L2, BloomLevel.L3)


def test_equivalence_builder_and_difficulty():
    rng = SeedManager.get_rng(54321)
    sig = StructuralSignature(total_marks=10, sub_question_count=2, mark_distribution=(6, 4))
    P = (BloomLevel.L3, BloomLevel.L2)

    profile = ORPairEquivalenceBuilder.build(sig, P, domain="CSE", rng=rng)
    assert profile.bloom_profile == (BloomLevel.L3, BloomLevel.L2)
    assert len(profile.difficulty_profile) == 2
    assert len(profile.question_type_profile) == 2
    assert abs(sum(profile.cognitive_weights) - 1.0) < 1e-5


def test_seed_manager_reproducibility():
    master_seed = 987654321
    seeds1 = SeedManager.derive_seeds(master_seed)
    seeds2 = SeedManager.derive_seeds(master_seed)
    assert seeds1 == seeds2

    slot_seed1 = SeedManager.slot_seed(master_seed, "Q1a")
    slot_seed2 = SeedManager.slot_seed(master_seed, "Q1a")
    assert slot_seed1 == slot_seed2


def test_question_slot_structure_locking():
    slot = QuestionSlot(
        slot_id="Q1a",
        question_number=1,
        sub_label="a",
        module_id=1,
        marks=6,
        bloom=BloomLevel.L3,
        co="CO1",
        question_type="GRAPH_SHORTEST_PATH",
        difficulty_band=DifficultyBand.MEDIUM,
        visual_prior=VisualPrior.PREFERRED,
    )

    # Before locking, assertion fails
    with pytest.raises(AssertionError):
        slot.assert_structure_locked()

    # Lock structure
    slot.lock_structure()
    assert slot.status == SlotStatus.STRUCTURE_LOCKED
    slot.assert_structure_locked()  # Must pass cleanly


def test_content_randomizer_and_qwen_adapter():
    slot = QuestionSlot(
        slot_id="Q1a",
        question_number=1,
        sub_label="a",
        module_id=1,
        marks=6,
        bloom=BloomLevel.L3,
        co="CO1",
        question_type="GRAPH_SHORTEST_PATH",
        difficulty_band=DifficultyBand.MEDIUM,
        visual_prior=VisualPrior.PREFERRED,
    )
    slot.lock_structure()

    dedup = {}
    ContentRandomizer.fill_slot(slot, corpus=None, seed=42, dedup_registers=dedup)
    assert slot.status == SlotStatus.CONTENT_READY
    assert slot.concept is not None
    assert slot.grounding_score >= 0.70

    # Realize text via QwenAdapter
    QwenAdapter.generate_slot(slot)
    assert slot.status == SlotStatus.GENERATED
    assert "GRAPH_SHORTEST_PATH" in slot.question_text
    assert slot.marks == 6  # Enforced invariant


def test_paper_contract_verifier_38_checks():
    master_seed = 101010
    derived = SeedManager.derive_seeds(master_seed)
    rng = SeedManager.get_rng(derived["structure"])

    or_pairs = []

    for m_id in range(1, 6):
        sig = StructuralSignature(total_marks=10, sub_question_count=2, mark_distribution=(6, 4))
        P = (BloomLevel.L3, BloomLevel.L2)
        profile = ORPairEquivalenceBuilder.build(sig, P, domain="CSE", rng=rng)

        alternatives = []
        for alt_idx in range(2):
            q_num = (m_id - 1) * 2 + 1 if alt_idx == 0 else (m_id - 1) * 2 + 2
            slots = []

            for sub_idx, (letter, marks, b_level, q_type, diff) in enumerate(
                zip(["a", "b"], [6, 4], P, profile.question_type_profile, profile.difficulty_profile)
            ):
                slot_id = f"Q{q_num}{letter}"
                slot = QuestionSlot(
                    slot_id=slot_id,
                    question_number=q_num,
                    sub_label=letter,
                    module_id=m_id,
                    marks=marks,
                    bloom=b_level,
                    co=f"CO{m_id}",
                    question_type=q_type,
                    difficulty_band=diff,
                    visual_prior=VisualPrior.OPTIONAL,
                )
                slot.lock_structure()
                ContentRandomizer.fill_slot(slot, corpus=None, seed=SeedManager.slot_seed(master_seed, slot_id), dedup_registers={})
                QwenAdapter.generate_slot(slot)
                slots.append(slot)

            alt = Alternative(
                question_id=f"Q{q_num}",
                slots=slots,
                signature=sig,
                profile=profile,
            )
            alternatives.append(alt)

        pair = ORPair(module_id=m_id, signature=sig, profile=profile, alternatives=alternatives)
        or_pairs.append(pair)

    report = PaperContractVerifier.verify(or_pairs, max_marks=50)

    for res in report.results:
        if not res.passed:
            print(f"FAILED CHECK: {res.check_id} ({res.category}) - {res.message} - {res.failed_slots}")

    assert report.total_checks == 38
    assert report.passed_count == 38
    assert report.failed_count == 0
    assert report.exportable is True


def test_recovery_protocol_invariants():
    slot = QuestionSlot(
        slot_id="Q1a",
        question_number=1,
        sub_label="a",
        module_id=1,
        marks=6,
        bloom=BloomLevel.L3,
        co="CO1",
        question_type="GRAPH_SHORTEST_PATH",
        difficulty_band=DifficultyBand.MEDIUM,
        visual_prior=VisualPrior.PREFERRED,
    )
    slot.lock_structure()

    act1 = PaperContractVerifier.execute_recovery(slot, attempt=1)
    assert act1 == RecoveryAction.NEW_EVIDENCE

    act2 = PaperContractVerifier.execute_recovery(slot, attempt=2)
    assert act2 == RecoveryAction.NEW_CONCEPT

    act3 = PaperContractVerifier.execute_recovery(slot, attempt=3)
    assert act3 == RecoveryAction.SLOT_FAILED
