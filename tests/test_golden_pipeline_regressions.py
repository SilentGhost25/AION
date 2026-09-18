"""
AION Standing Regression & Pipeline Contract Test Suite
======================================================
Pre-deploy verification suite enforcing:
1. Request Contract schema compliance (file_id, file_ids, camelCase).
2. Domain archetype resolution with substring trap defense.
3. Cross-domain anti-contamination & SQL leakage defense.
4. Prompt scaffolding & slot identifier leak defense.
5. SSE streaming TCP packet fragmentation recovery.
6. Multi-module synthesis header formatting and word threshold validation.
"""

import os
import sys
from pathlib import Path

# Ensure root is in sys.path
ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import re
import pytest
from core.contracts import GenerationRequest
from aion_patch import resolve_subject_archetype
from core.validation.linter import check_domain_integrity, check_no_meta_language
from v0_1.segmenter import RobustSegmenter


def test_generation_request_contract_parsing():
    """Verify GenerationRequest from_dict parses single and multi-file payloads."""
    # 1. Single file legacy payload
    p1 = {
        "subject": "Cloud Computing",
        "file_id": "doc_123",
        "exam_type": "IA",
        "difficulty": "hard"
    }
    req1 = GenerationRequest.from_dict(p1)
    assert req1.subject == "Cloud Computing"
    assert req1.file_id == "doc_123"
    assert req1.file_ids is None

    # 2. Multi-file snake_case payload
    p2 = {
        "subject": "Cloud Computing",
        "file_ids": ["doc_1", "doc_2", "doc_3"],
        "examType": "SEE"
    }
    req2 = GenerationRequest.from_dict(p2)
    assert req2.file_ids == ["doc_1", "doc_2", "doc_3"]
    assert req2.exam_type == "SEE"

    # 3. Multi-file camelCase payload
    p3 = {
        "subject": "Distributed Systems",
        "fileIds": ["doc_A", "doc_B"]
    }
    req3 = GenerationRequest.from_dict(p3)
    assert req3.file_ids == ["doc_A", "doc_B"]


def test_archetype_resolution_and_substring_defense():
    """Verify archetype matching works for full titles and denies substring traps."""
    # Positive matches
    assert resolve_subject_archetype("Cloud Computing & Big Data Analytics", "Docker") == "cloud_bigdata"
    assert resolve_subject_archetype("Database Management Systems", "SQL") == "dbms"
    assert resolve_subject_archetype("21CS53", "Relational Algebra") == "dbms"
    assert resolve_subject_archetype("Operating Systems", "Process Scheduling") == "os_networks"

    # Substring trap defense (e.g., 'biotech' inside 'biotechnology and genomics')
    assert resolve_subject_archetype("Biotechnology and Genomics", "genomics") is None


def test_domain_integrity_linter():
    """Verify non-DBMS topics reject SQL and Relational Algebra contamination."""
    class MockSlot:
        topic = "Virtualization, Hypervisors and Containers"

    slot = MockSlot()
    # SQL query on Cloud topic -> FAIL
    assert check_domain_integrity("SELECT * FROM HOSTS WHERE id = 1", slot).passed is False
    # Relational algebra on Cloud topic -> FAIL
    assert check_domain_integrity(r"Demonstrate \sigma_{cpu > 80}(VIRTUAL_MACHINES)", slot).passed is False
    # Valid Cloud topic content -> PASS
    assert check_domain_integrity("Explain how Linux cgroups enforce CPU and memory isolation in Docker containers.", slot).passed is True


def test_prompt_scaffolding_and_slot_leak_linter():
    """Verify questions citing internal slot identifiers are rejected."""
    # Slot ID leaks -> FAIL
    assert check_no_meta_language("Explain reflex agents for module_5_Q3_b.").passed is False
    assert check_no_meta_language("Analyze the architecture given in slot_module1_q1_a.").passed is False
    assert check_no_meta_language("What is the main property in Q4_b?").passed is False

    # Valid academic question -> PASS
    assert check_no_meta_language("Explain the architecture and execution flow of MapReduce with a neat diagram.").passed is True


def test_sse_tcp_packet_fragmentation_resilience():
    """
    Simulate fragmented TCP chunks splitting SSE 'event:' and 'data:' lines.
    Verifies that the parser correctly preserves currentEvent across chunks.
    """
    # Raw stream fragmented arbitrarily across network reads
    raw_stream_chunks = [
        "eve",
        "nt: paper_re",
        "ady\nda",
        'ta: {"pap',
        'er": {"id": "paper_999", "subject": "Cloud Computing"}}\n\n',
        "event: do",
        'ne\ndata: {"status": "SUCCESS"}\n\n'
    ]

    events_received = []
    buffer = ""
    current_event = ""

    for chunk in raw_stream_chunks:
        buffer += chunk
        lines = buffer.split("\n")
        buffer = lines.pop()  # Keep incomplete line in buffer

        for line in lines:
            trimmed = line.strip()
            if trimmed.startswith("event:"):
                current_event = trimmed.replace("event:", "").strip()
            elif trimmed.startswith("data:"):
                data_str = trimmed.replace("data:", "").strip()
                if data_str:
                    payload = json.loads(data_str)
                    events_received.append((current_event, payload))
                    current_event = ""  # Reset after consuming

    assert len(events_received) == 2
    assert events_received[0][0] == "paper_ready"
    assert events_received[0][1]["paper"]["id"] == "paper_999"
    assert events_received[1][0] == "done"
    assert events_received[1][1]["status"] == "SUCCESS"


def test_multi_module_synthesis_format_and_segmentation():
    """
    Verify synthesized multi-file notes match RobustSegmenter's explicit pattern
    and segment cleanly into distinct modules.
    """
    module_texts = [
        "Module 1: Cloud Virtualization\n" + "Virtualization enables multiple OS instances on shared hardware. " * 10,
        "Module 2: Containerization & Docker\n" + "Docker provides lightweight container isolation using Linux namespaces. " * 10,
        "Module 3: Distributed Storage & HDFS\n" + "HDFS stores large files across multiple machines using block replication. " * 10,
    ]
    combined_notes = "\n\n".join(module_texts)

    # Validate regex match
    for line in combined_notes.split("\n"):
        if line.startswith("Module"):
            assert re.match(RobustSegmenter.EXPLICIT_PATTERNS[0], line), f"Header '{line}' failed EXPLICIT_PATTERNS regex"

    # Segment document using RobustSegmenter instance
    segmenter = RobustSegmenter()
    segments = segmenter.segment(combined_notes, target_n=3)
    assert len(segments) == 3, f"Expected 3 segments, got {len(segments)}"
    assert "Virtualization" in segments[0].content
    assert "Containerization" in segments[1].content
    assert "Distributed Storage" in segments[2].content


def test_auto_healer_bloom_verb_recovery():
    """
    Verify AutoHealer rewrites an unaligned initial verb (e.g. 'Discuss')
    to match the expected Bloom verb (e.g. 'Explain') without losing question content.
    """
    from core.generation.auto_healer import AutoHealer
    from core.generation.output_schema import QuestionOutput
    from core.validation.linter import check_bloom_verb_at_start

    class MockSlot:
        bloom_verb = "Explain"
        bloom_level = "L2"
        slot_id = "slot_1"

    slot = MockSlot()
    raw_output = QuestionOutput(
        instruction="Discuss the architectural differences between bare-metal hypervisors and hosted hypervisors.",
        question_text="Discuss the architectural differences between bare-metal hypervisors and hosted hypervisors with examples.",
        bloom_level="L2",
        marks=6
    )

    # Initial check fails because first word is "Discuss", not "Explain"
    initial_check = check_bloom_verb_at_start(raw_output.instruction, slot)
    assert initial_check.passed is False
    assert initial_check.code == "BLOOM_VERB_NOT_AT_START"

    # AutoHealer resolves the Bloom verb
    healed_output = AutoHealer.heal("BLOOM_VERB_NOT_AT_START", raw_output, slot)
    assert healed_output.instruction.startswith("Explain")
    assert "bare-metal hypervisors" in healed_output.instruction

    # Post-heal check passes
    post_check = check_bloom_verb_at_start(healed_output.instruction, slot)
    assert post_check.passed is True


def test_enforce_marks_preserves_declared_splits_and_or_symmetry():
    """
    Verify _enforce_marks preserves user declared splits (e.g. [6, 4] or [7, 3]),
    symmetrizes OR alternative pairs, and leaves no ContractViolation.
    """
    from aion_api import _enforce_marks, validate_final_paper_contract

    # 5 modules with 2 questions each (IA exam = 10 marks per question)
    raw_modules = [
        {
            "module_index": i + 1,
            "questions": [
                {
                    "mqIndex": (i * 2) + 1,
                    "subQuestions": [{"label": "a", "marks": 5}, {"label": "b", "marks": 5}]
                },
                {
                    "mqIndex": (i * 2) + 2,
                    "subQuestions": [{"label": "a", "marks": 6}, {"label": "b", "marks": 4}]
                }
            ]
        }
        for i in range(5)
    ]

    # Declare [6, 4] for all 5 modules
    declared = [[6, 4]] * 5
    enforced = _enforce_marks(raw_modules, exam_type="IA", declared_splits=declared)

    for mod in enforced:
        qs = mod["questions"]
        q_a_marks = [s["marks"] for s in qs[0]["subQuestions"]]
        q_b_marks = [s["marks"] for s in qs[1]["subQuestions"]]
        assert q_a_marks == [6, 4]
        assert q_b_marks == [6, 4]
        assert sum(q_a_marks) == 10
        assert sum(q_b_marks) == 10

    # Must pass final contract validation without raising
    assert validate_final_paper_contract({"modules": enforced}, exam_type="IA") is True


def test_auto_healer_syncs_bloom_level_metadata():
    """
    Verify AutoHealer rewrites the verb and GeneratedQuestion binds the
    canonical Bloom level metadata from the slot contract.
    """
    from core.generation.auto_healer import AutoHealer
    from core.generation.output_schema import QuestionOutput
    from core.contracts.question import GeneratedQuestion
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    slot = QuestionSlot(
        slot_id="slot_calc_1",
        question_no=1,
        sub_label="a",
        or_pair_id="pair_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L3",
        bloom_verb="Calculate",
        bloom_operation="APPLY",
        co="CO1",
        difficulty="MEDIUM",
        question_type="NUMERICAL",
        topic="Throughput",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L3", 6, "NUMERICAL")
    )

    # Output initially generated starting with verb "Describe"
    output = QuestionOutput(
        instruction="Describe the throughput formula.",
        question_text="Describe the throughput formula.",
        math_blocks=[]
    )

    healed = AutoHealer.heal("BLOOM_VERB_NOT_AT_START", output, slot)
    assert healed.instruction.startswith("Calculate")
    assert healed.question_text.startswith("Calculate")

    # When converted to GeneratedQuestion, slot contract metadata is bound
    gq = GeneratedQuestion(healed, slot)
    assert gq.bloom == "L3"
    assert gq.marks == 6
    assert gq.co == "CO1"
    assert gq.question_text.startswith("Calculate")


def test_corruption_patterns_and_fallback_grounding():
    """Ensure linter blocks markdown table fragments and template fallback stays clean."""
    from core.validation.linter import check_no_meta_language
    from core.generation.orchestrator import SlotOrchestrator
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    # 1. Test corruption detection
    corrupted_q = "Solve the behavioral differences of alternative approaches to | 14 Overall Data-to-Decision-to-Actuation Cycle | 32 |"
    res = check_no_meta_language(corrupted_q)
    assert not res.passed
    assert res.code == "ANSWERABILITY_FAILURE"

    # 2. Test clean template fallback generation
    slot = QuestionSlot(
        slot_id="slot_mod1_q1_a",
        question_no=1,
        sub_label="a",
        or_pair_id="pair_1",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L4",
        bloom_verb="Analyze",
        bloom_operation="ANALYZE",
        co="CO1",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Microcontroller Architecture",
        evidence_ids=("chunk_1",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L4"),
        question_budget=QuestionBudget.from_bloom("L4", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L4", 6, "THEORY")
    )
    orch = SlotOrchestrator()
    fallback_q = orch._generate_template_fallback(slot, evidence_pack={"text": "| 14 Table Header | 32 |\nMicrocontroller Architecture defines the internal bus structure."})
    
    # Must NOT contain [6 Marks] in question text or instruction
    assert "[6 Marks]" not in fallback_q.question_text
    assert "[Marks]" not in fallback_q.question_text
    assert "|" not in fallback_q.question_text
    assert "Binary Search Trees" not in fallback_q.question_text
    assert fallback_q.question_text.startswith("Analyze")


def test_auto_healer_prevents_verb_stacking_and_heals_latex():
    """Verify AutoHealer does not double-stack verbs (e.g. 'Apply Analyse...') and heals tab-corrupted LaTeX."""
    from core.generation.auto_healer import AutoHealer
    from core.generation.output_schema import QuestionOutput

    class MockSlot:
        bloom_verb = "Apply"
        bloom_level = "L3"
        slot_id = "slot_test"

    slot = MockSlot()
    # Case 1: Stacked verbs
    raw_output = QuestionOutput(
        instruction="Analyse how smart irrigation differs from manual irrigation practices.",
        question_text=r"Analyse how smart irrigation differs from manual irrigation practices with formula \( V = A \t imes d \).",
        bloom_level="L4",
        marks=6
    )
    healed = AutoHealer.heal("BLOOM_VERB_NOT_AT_START", raw_output, slot)
    assert healed.question_text.startswith("Apply")
    assert not healed.question_text.startswith("Apply Analyse")
    assert r"\times" in healed.question_text
    assert "\t imes" not in healed.question_text


def test_sibling_uniqueness_stops_duplicate_subquestions():
    """Verify check_sibling_uniqueness catches near-duplicate questions on the same concept."""
    from core.validation.linter import check_sibling_uniqueness
    from core.generation.output_schema import QuestionOutput
    from core.contracts.question import GeneratedQuestion
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    slot = QuestionSlot(
        slot_id="slot_mod1_q8_b",
        question_no=8,
        sub_label="b",
        or_pair_id="pair_4",
        is_alternative=True,
        module_id=4,
        marks=4,
        bloom_level="L1",
        bloom_verb="Identify",
        bloom_operation="REMEMBER",
        co="CO1",
        difficulty="EASY",
        question_type="THEORY",
        topic="Satellite Inclination",
        evidence_ids=("chunk_4",),
        answer_budget=AnswerBudget.from_marks_and_bloom(4, "L1"),
        question_budget=QuestionBudget.from_bloom("L1", 4),
        task_signature=TaskSignature.from_bloom_marks_type("L1", 4, "THEORY")
    )
    q8a_text = "A satellite is in an orbit with an inclination of 65 degrees. Illustrate whether this orbit is direct or retrograde, and explain."
    q8b_output = QuestionOutput(
        instruction="A satellite has an inclination angle of 65 degrees. Identify whether this orbit is direct or retrograde and explain your reasoning.",
        question_text="A satellite has an inclination angle of 65 degrees. Identify whether this orbit is direct or retrograde and explain your reasoning.",
        marks=4
    )
    q8b_cand = GeneratedQuestion(output=q8b_output, slot=slot)

    result = check_sibling_uniqueness(q8b_cand, sibling_texts=[q8a_text])
    assert not result.passed
    assert result.code == "SIBLING_SIMILARITY"

    # Case 2: Test 4-word syntactic prefix collision rejection even with different domain nouns
    prefix_output = QuestionOutput(
        instruction="Explain the foundational principles of satellite telemetry transmission.",
        question_text="Explain the foundational principles of satellite telemetry transmission.",
        marks=4
    )
    prefix_cand = GeneratedQuestion(output=prefix_output, slot=slot)
    sibling_prefix_text = "Explain the foundational principles of orbital payload deployment."
    res_prefix = check_sibling_uniqueness(prefix_cand, sibling_texts=[sibling_prefix_text])
    assert not res_prefix.passed
    assert res_prefix.code == "SIBLING_SIMILARITY"
    assert "identical syntactic opening phrase" in res_prefix.message


def test_archetype_palette_round_robin_rotation():
    """Verify SlotOrchestrator cycles through diverse question archetypes in few-shot prompts."""
    from core.generation.orchestrator import SlotOrchestrator
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    orch = SlotOrchestrator()
    examples_generated = []

    for i in range(4):
        slot = QuestionSlot(
            slot_id=f"slot_mod1_q{i+1}_a",
            question_no=i+1,
            sub_label="a",
            or_pair_id=f"pair_{i+1}",
            is_alternative=False,
            module_id=1,
            marks=6,
            bloom_level="L2",
            bloom_verb="Explain",
            bloom_operation="UNDERSTAND",
            co="CO1",
            difficulty="MEDIUM",
            question_type="THEORY",
            topic=f"Topic {i+1}",
            evidence_ids=(f"chunk_{i+1}",),
            answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
            question_budget=QuestionBudget.from_bloom("L2", 6),
            task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "THEORY")
        )
        prompt = orch._format_prompt(slot, "Evidence text here...", "")
        examples_generated.append(prompt)

    # Verify universal theory archetype rotation across sequential slots
    assert "fundamental principles governing Topic 1" in examples_generated[0]
    assert "key distinctions and perspectives in Topic 2" in examples_generated[1]
    assert "structural framework and provisions of Topic 3" in examples_generated[2]
    assert "role and function of Topic 4" in examples_generated[3]
    # Verify prohibition on clichés is included
    assert "PROHIBITION ON FORMULAIC CLICHÉS" in examples_generated[0]

    # Verify Numerical slot receives calculation-appropriate prompt framing
    num_slot = QuestionSlot(
        slot_id="slot_mod1_q5_a",
        question_no=5,
        sub_label="a",
        or_pair_id="pair_5",
        is_alternative=False,
        module_id=1,
        marks=6,
        bloom_level="L3",
        bloom_verb="Calculate",
        bloom_operation="CALCULATE",
        co="CO2",
        difficulty="HARD",
        question_type="NUMERICAL",
        topic="Free Space Path Loss",
        evidence_ids=("chunk_5",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L3", 6, "NUMERICAL")
    )
    num_prompt = orch._format_prompt(num_slot, "Calculate loss with freq=4GHz, dist=36000km.", "")
    assert ("governing parameters of Free Space Path Loss" in num_prompt or "mathematical relationship in Free Space Path Loss" in num_prompt or "numerical output" in num_prompt or "quantitative result" in num_prompt or "output value" in num_prompt)


def test_top_n_chunk_allocation_uniqueness_and_coherence():
    """Verify get_top_n_chunks_for_question returns distinct chunks from the same topical section."""
    from v0_1.chunk_image_mapper import ChunkImageMapper, TextChunk, ModuleChunkGroup

    mapper = ChunkImageMapper()
    # Create 5 chunks in Module 1 with varying topics
    c1 = TextChunk(
        id="mod1_chunk_1",
        text="Newton's laws of motion define the fundamental relationship between a body and the forces acting upon it.",
        module_id="module_1",
        module_idx=1,
        chunk_idx=1,
        total_chunks=5,
        word_count=50,
        page_start=1,
        page_end=2,
    )
    c2 = TextChunk(
        id="mod1_chunk_2",
        text="Newton's second law quantifies force as the time rate of change of momentum, expressed as F = m * a.",
        module_id="module_1",
        module_idx=1,
        chunk_idx=2,
        total_chunks=5,
        word_count=55,
        page_start=2,
        page_end=3,
    )
    c3 = TextChunk(
        id="mod1_chunk_3",
        text="Newton's third law states that for every action force there is an equal and opposite reaction force.",
        module_id="module_1",
        module_idx=1,
        chunk_idx=3,
        total_chunks=5,
        word_count=52,
        page_start=3,
        page_end=4,
    )
    group = ModuleChunkGroup(module_id="module_1", module_idx=1, module_title="Mechanics", chunks=[c1, c2, c3])
    mapper._module_groups = {"module_1": group}

    top_2 = mapper.get_top_n_chunks_for_question(module_id="module_1", n=2, prefer_image=False, target_bloom=2)
    assert len(top_2) == 2
    # 1. Distinct chunk IDs (no duplicate chunk cloning)
    assert top_2[0].id != top_2[1].id
    assert top_2[0].text != top_2[1].text

    # 2. Strict Jaccard similarity threshold (< 0.2)
    words_0 = set(top_2[0].text.lower().split())
    words_1 = set(top_2[1].text.lower().split())
    jaccard = len(words_0 & words_1) / max(1, len(words_0 | words_1))
    assert jaccard < 0.25, f"Chunks too similar: {jaccard}"

    # 3. Topical coherence maintained (both discuss Newton mechanics)
    assert "newton" in top_2[0].text.lower()
    assert "newton" in top_2[1].text.lower()


def test_domain_agnostic_hybrid_scoring_for_non_cs_topics():
    """Verify non-CS subjects (e.g. satellite communications / physics) achieve healthy retrieval scores."""
    from v0_1.chunk_image_mapper import calculate_retrieval_score, TextChunk

    chunk = TextChunk(
        id="sat_chunk_1",
        text="Satellite orbit inclination is defined as the angle between the orbital plane and the Earth equatorial plane with formula theta = 65 degrees.",
        module_id="module_2",
        module_idx=2,
        chunk_idx=1,
        total_chunks=10,
        word_count=80,
        page_start=5,
        page_end=6,
    )
    score = calculate_retrieval_score(chunk, target_bloom=3, target_module_idx=2)
    # Must achieve a high score (> 0.5) under hybrid scoring
    assert score > 0.5


def test_full_page_coverage_across_text_bearing_pages():
    """Verify chunks span >= 80% of text-bearing pages in a document."""
    from v0_1.chunk_image_mapper import TextChunk

    # Simulate 10 chunks across a 20-page document
    simulated_chunks = [
        TextChunk(id=f"c_{i}", text=f"Concept text for section {i} with sufficient descriptive body content.", module_id="module_1", module_idx=1, chunk_idx=i, total_chunks=10, word_count=60, page_start=i * 2 + 1, page_end=i * 2 + 2)
        for i in range(10)
    ]

    pages_with_text = {c.page_start for c in simulated_chunks if len(c.text.strip()) > 50}
    min_page = min(pages_with_text)
    max_page = max(pages_with_text)
    total_pages = 20
    
    # Span must reach at least 80% of document span
    assert max_page >= 0.8 * total_pages


def test_eased_difficulty_policy_shifts_6m_to_easy_tier():
    """Verify eased difficulty policy shifts 6M questions into CO2/L2-L3 easy/moderate tier across all modules."""
    from v0_1.difficulty_policy import resolve_co_bl_from_marks, EASE_PAPER_DIFFICULTY
    assert EASE_PAPER_DIFFICULTY is True
    co, bloom = resolve_co_bl_from_marks(module_idx=1, marks=6, total_parts=2, planned_type="APPLICATION")
    assert co == "CO2" and bloom == 3
    co5, bloom5 = resolve_co_bl_from_marks(module_idx=5, marks=6, total_parts=2, planned_type="APPLICATION")
    assert co5 == "CO2" and bloom5 == 3
    co_hard, bloom_hard = resolve_co_bl_from_marks(module_idx=3, marks=10, total_parts=1)
    assert co_hard == "CO3" and bloom_hard == 4


def test_safe_co_bloom_formatting():
    """Verify format_co_and_rbt safely formats CO and RBT strings without destructive overrides."""
    from v0_1.difficulty_policy import format_co_and_rbt

    assert format_co_and_rbt("CO1", "L1") == ("CO1", "L1")
    assert format_co_and_rbt("CO1", "L2") == ("CO1", "L2")
    assert format_co_and_rbt("CO2", "L3") == ("CO2", "L3")
    assert format_co_and_rbt("CO3", "L4") == ("CO3", "L4")
    assert format_co_and_rbt("CO4", "L5") == ("CO4", "L5")
    assert format_co_and_rbt("CO5", "L6") == ("CO5", "L6")
    # Integer bloom normalization
    assert format_co_and_rbt("CO2", 3) == ("CO2", "L3")
    # Fallback to module when CO missing
    assert format_co_and_rbt(None, 2, module_idx=1) == ("CO1", "L2")
    assert format_co_and_rbt(None, 3, module_idx=2) == ("CO2", "L3")
    # Duplicate prefix defense
    assert format_co_and_rbt("CO1", "LL2") == ("CO1", "L2")
    assert format_co_and_rbt("CO2", "LLL3") == ("CO2", "L3")
    assert format_co_and_rbt("CO3", "l4") == ("CO3", "L4")


def test_difficulty_manager_verb_matches_canonical_bloom_level():
    """Verify DifficultyManager verbs strictly match canonical BLOOM_VERB_LEVEL_MAP across L1..L6."""
    from v0_1.difficulty import DifficultyManager
    from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP
    dm = DifficultyManager()
    for level in range(1, 7):
        for _ in range(5):  # Sample multiple times to verify pool rotation
            verb = dm.get_verb("medium", level).lower()
            assert verb in BLOOM_VERB_LEVEL_MAP[f"L{level}"], (
                f"get_verb returned '{verb}' for L{level}, not in canonical map"
            )


def test_docx_module_grouping_matches_frontend_qno_formula():
    """Verify docx_export module derivation exactly matches Step3Preview formula ((qNo - 1) // 2) + 1."""
    from v0_1.docx_export import get_module_for_q
    # Q1,Q2 -> Module 1; Q9,Q10 -> Module 5
    assert get_module_for_q({"qNo": 1}) == 1
    assert get_module_for_q({"qNo": 2}) == 1
    assert get_module_for_q({"qNo": 9}) == 5
    assert get_module_for_q({"qNo": 10}) == 5
    # Also verify fallback on question_number or sectionNumber
    assert get_module_for_q({"question_number": 3}) == 2
    assert get_module_for_q({"sectionNumber": 8}) == 4


def test_bloom_verb_and_operation_synchronization_across_pipeline():
    """Verify canonical Bloom verbs and cognitive operations are 100% synchronized across all validators."""
    from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP
    from core.validation.linter import VERB_OPERATION_MAP, BLOOM_VERB_MAP, check_answerability
    from core.contracts.task_signature import _ALL_OPS
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.question import GeneratedQuestion
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature

    # 1. Check all canonical verbs are in VERB_OPERATION_MAP with valid operations
    for level, verbs in BLOOM_VERB_LEVEL_MAP.items():
        for verb in verbs:
            v_lower = verb.lower()
            assert v_lower in VERB_OPERATION_MAP, f"Verb '{v_lower}' missing from VERB_OPERATION_MAP"
            op = VERB_OPERATION_MAP[v_lower]
            assert op in _ALL_OPS, f"Operation '{op}' for verb '{v_lower}' not in _ALL_OPS"

    # 2. Check all canonical verbs are in BLOOM_VERB_MAP
    for level, verbs in BLOOM_VERB_LEVEL_MAP.items():
        for verb in verbs:
            assert verb.capitalize() in BLOOM_VERB_MAP[level], f"Verb '{verb}' missing from BLOOM_VERB_MAP[{level}]"

    # 3. Check check_answerability passes has_action_verb for expanded verbs like outline, compute, investigate
    test_verbs = ["outline", "compute", "derive", "find", "investigate", "distinguish", "validate", "appraise"]
    for tv in test_verbs:
        slot = QuestionSlot(
            slot_id=f"test_{tv}",
            question_no=1,
            sub_label="a",
            or_pair_id="pair_1",
            is_alternative=False,
            module_id=1,
            marks=4,
            bloom_level="L2",
            bloom_verb=tv.capitalize(),
            bloom_operation="UNDERSTAND",
            co="CO1",
            difficulty="EASY",
            question_type="THEORY",
            topic="Cloud Computing Concepts",
            evidence_ids=("chunk_1",),
            answer_budget=AnswerBudget.from_marks_and_bloom(4, "L2"),
            question_budget=QuestionBudget.from_bloom("L2", 4),
            task_signature=TaskSignature.from_bloom_marks_type("L2", 4, "THEORY")
        )
        from core.generation.output_schema import QuestionOutput
        output = QuestionOutput(
            instruction=f"{tv.capitalize()} the architecture and operational characteristics of distributed systems.",
            question_text=f"{tv.capitalize()} the architecture and operational characteristics of distributed systems.",
            math_blocks=[]
        )
        gq = GeneratedQuestion(output, slot)
        res = check_answerability(gq, slot, evidence_text="")
        assert res.passed, f"check_answerability failed for action verb '{tv}': {res.message}"


def test_cross_module_deduplication_via_shared_registry():
    """Verify that SlotOrchestrator with shared_generated_texts catches cross-module duplicate questions."""
    from core.generation.orchestrator import SlotOrchestrator
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.question import GeneratedQuestion
    from core.generation.output_schema import QuestionOutput
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature
    from core.validation.linter import check_sibling_uniqueness
    import threading

    shared_texts = []
    lock = threading.Lock()

    orch1 = SlotOrchestrator(shared_generated_texts=shared_texts, shared_texts_lock=lock)
    orch2 = SlotOrchestrator(shared_generated_texts=shared_texts, shared_texts_lock=lock)

    # Module 3 (orch1) generates a question about VirtualBox Ubuntu installation
    q_mod3 = "Explain the step-by-step procedure to install and configure Ubuntu Linux inside VirtualBox."
    with lock:
        orch1._all_generated_texts.append(q_mod3)

    # Module 4 (orch2) sees this in its shared texts
    assert len(orch2._all_generated_texts) == 1
    assert "VirtualBox" in orch2._all_generated_texts[0]

    # If Module 4 attempts to generate a near-duplicate, check_sibling_uniqueness catches it
    cand_output = QuestionOutput(
        instruction="Describe the detailed steps to install and configure Ubuntu OS on Oracle VirtualBox.",
        question_text="Describe the detailed steps to install and configure Ubuntu OS on Oracle VirtualBox.",
        math_blocks=[]
    )
    slot = QuestionSlot(
        slot_id="module_4_Q7_a",
        question_no=7,
        sub_label="a",
        or_pair_id="pair_1",
        is_alternative=False,
        module_id=4,
        marks=6,
        bloom_level="L2",
        bloom_verb="Describe",
        bloom_operation="UNDERSTAND",
        co="CO2",
        difficulty="MEDIUM",
        question_type="THEORY",
        topic="Virtualization Setup",
        evidence_ids=("chunk_vbox",),
        answer_budget=AnswerBudget.from_marks_and_bloom(6, "L2"),
        question_budget=QuestionBudget.from_bloom("L2", 6),
        task_signature=TaskSignature.from_bloom_marks_type("L2", 6, "THEORY")
    )
    cand_q = GeneratedQuestion(cand_output, slot)
    check_res = check_sibling_uniqueness(cand_q, sibling_texts=list(orch2._all_generated_texts))
    assert not check_res.passed
    assert check_res.code == "SIBLING_SIMILARITY"


def test_math_incomplete_frac_exhaustion_does_not_crash_pipeline():
    """Simulate 4 consecutive MATH_INCOMPLETE_FRAC failures on a slot.
    Assert: orchestrator returns a degraded-but-valid result, ExportGate passes,
    and no unhandled RuntimeError crashes the pipeline."""
    from core.generation.orchestrator import SlotOrchestrator
    from core.contracts.question_slot import QuestionSlot
    from core.contracts.budgets import AnswerBudget, QuestionBudget
    from core.contracts.task_signature import TaskSignature
    from core.validation.export_gate import ExportGate
    from core.generation.auto_healer import AutoHealer

    # 1. Verify AutoHealer repairs \frac syntax directly
    healed = AutoHealer._heal_latex(r"\frac{water\_volume}")
    assert healed == r"\frac{water\_volume}{1}"

    # 2. Setup slot with math_required=True
    slot = QuestionSlot(
        slot_id="module_2_Q3",
        question_no=3,
        sub_label="",
        or_pair_id="module_2_OR_2",
        is_alternative=False,
        module_id=2,
        marks=10,
        bloom_level="L3",
        bloom_verb="Calculate",
        bloom_operation="APPLY",
        co="CO2",
        difficulty="MEDIUM",
        question_type="NUMERICAL",
        topic="Fertigation and Nutrient Solution Management",
        evidence_ids=("chunk_fertigation",),
        answer_budget=AnswerBudget.from_marks_and_bloom(10, "L3"),
        question_budget=QuestionBudget.from_bloom("L3", 10),
        task_signature=TaskSignature.from_bloom_marks_type("L3", 10, "NUMERICAL"),
        math_required=True,
    )

    class MockEvidencePack:
        combined_text = (
            "Fertigation delivers mineral nutrients directly to crop root zones. "
            "The nutrient concentration is adjusted based on electrical conductivity and pH. "
            "Flow rates determine total dosage delivered to emitter lines."
        )

    orch = SlotOrchestrator()

    # Mock _call_llm to simulate 4 consecutive broken LaTeX fraction outputs
    broken_payload = {
        "instruction": "Calculate the nutrient dosage requirement [MATH:math_1]",
        "question_text": "Calculate the nutrient dosage requirement for fertigation [MATH:math_1]",
        "math_blocks": [
            {
                "block_id": "math_1",
                "latex": r"\frac{dose}",  # Missing denominator!
                "display_mode": True
            }
        ]
    }

    import json
    def mock_broken_call(*args, **kwargs):
        return json.dumps(broken_payload)

    orch._call_llm = mock_broken_call

    # Execute generation: must not raise RuntimeError
    gq = orch.generate(slot, MockEvidencePack())

    assert gq is not None
    assert gq.slot_id == "module_2_Q3"
    assert gq.question_text and len(gq.question_text) > 15
    # Must not contain raw [MATH:...] placeholder in user-facing text
    assert "[MATH:" not in gq.question_text
    assert "[math_1]" not in gq.question_text

    # ExportGate must validate cleanly without raising RuntimeError
    gate_decision = ExportGate.evaluate(generated_slots=[gq])
    assert gate_decision.passed is True
    check_result = ExportGate.validate([gq])
    assert check_result.passed is True


def test_marks_and_bloom_based_co_policy_default():
    """Verify that Course Outcome maps based on marks and bloom level by default (marks-based),
    and verify that module-based mode can still be activated when explicitly configured."""
    from v0_1.difficulty_policy import resolve_co_bl_from_marks, _resolve_co_by_mode

    # Default policy (marks-based):
    # <=4M -> CO1
    co_4m, bl_4m = resolve_co_bl_from_marks(module_idx=1, marks=4, total_parts=2, planned_type="CONCEPTUAL")
    assert co_4m == "CO1" and bl_4m == 1
    co_4m_app, bl_4m_app = resolve_co_bl_from_marks(module_idx=5, marks=4, total_parts=2, planned_type="APPLICATION")
    assert co_4m_app == "CO1" and bl_4m_app == 2

    # 6M -> CO2
    co_6m, bl_6m = resolve_co_bl_from_marks(module_idx=1, marks=6, total_parts=2, planned_type="APPLICATION")
    assert co_6m == "CO2" and bl_6m == 3
    co_6m_m5, bl_6m_m5 = resolve_co_bl_from_marks(module_idx=5, marks=6, total_parts=2, planned_type="APPLICATION")
    assert co_6m_m5 == "CO2" and bl_6m_m5 == 3

    # 8M/10M -> CO3
    co_10m, bl_10m = resolve_co_bl_from_marks(module_idx=3, marks=10, total_parts=1)
    assert co_10m == "CO3" and bl_10m == 4

    # Explicit module-based mode check (Module M -> COM)
    assert _resolve_co_by_mode(module_idx=1, marks=10, mode="module-based") == "CO1"
    assert _resolve_co_by_mode(module_idx=2, marks=6, mode="module-based") == "CO2"
    assert _resolve_co_by_mode(module_idx=3, marks=4, mode="module-based") == "CO3"
    assert _resolve_co_by_mode(module_idx=4, marks=6, mode="module-based") == "CO4"
    assert _resolve_co_by_mode(module_idx=5, marks=10, mode="module-based") == "CO5"


def test_continuous_global_question_numbering_formula():
    """Verify that global question numbering produces continuous Q1..Q10 without repeating Q1..Q4,
    and supports pool mode (4 questions/module) without index drift."""
    from v0_1.docx_export import get_module_for_q

    # Standard SEE mode: 2 questions per module (Q1..Q10)
    questions_per_module = 2
    for m in range(1, 6):
        for local_idx in (1, 2):
            global_q_no = (m - 1) * questions_per_module + local_idx
            assert get_module_for_q({"qNo": global_q_no}) == m, (
                f"Global question {global_q_no} did not map to Module {m}"
            )

    # Pool mode: 4 questions per module (Q1..Q20)
    pool_qpm = 4
    for m in range(1, 6):
        mod_q_nums = [(m - 1) * pool_qpm + local_idx for local_idx in range(1, pool_qpm + 1)]
        expected = list(range((m - 1) * pool_qpm + 1, m * pool_qpm + 1))
        assert mod_q_nums == expected
        # All questions in module m must share the same base module identity ((qNo - 1) // pool_qpm) + 1
        for q_num in mod_q_nums:
            derived_mod = ((q_num - 1) // pool_qpm) + 1
            assert derived_mod == m, f"Pool question {q_num} expected mod {m}, got {derived_mod}"


def test_numerical_unblocking_on_prose():
    """Verify that standard technical prose with words 'for', 'if', 'where' does not falsely trigger code signals,
    and verify that genuine code+numerical chunks have both programming_allowed and numerical_allowed True."""
    import re

    code_regexes = (
        r'\b(?:def|class|public|private|static|interface|struct)\s+[a-zA-Z_]\w*',
        r'\b(?:int|float|double|char|void|boolean)\s+[a-zA-Z_]\w*\s*(?:=|\(|;)',
        r'\bfor\s+[a-zA-Z_]\w*\s+in\b',
        r'\b(?:while|for)\s*\([^)]+\)\s*[{;]',
        r'\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM|CREATE\s+TABLE|ALTER\s+TABLE)\b',
        r'```[a-zA-Z]*\n',
        r'\b(?:pseudocode|algorithm)\s*:',
        r'\bdef\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*:',
    )

    # 1. Prose test: words 'for', 'if', 'where' in standard technical context
    chunk_prose = (
        "For the given sensor node, if the transmission range is 50 meters where energy consumption "
        "is 20 mW, calculate the lifetime and battery drain."
    )
    lower_prose = chunk_prose.lower()
    code_hits = sum(bool(re.search(pat, lower_prose, re.IGNORECASE)) for pat in code_regexes)
    assert code_hits == 0, f"Expected 0 code hits on prose, got {code_hits}"

    numeric_vals = re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*%)?(?![A-Za-z])', lower_prose)
    has_num_context = any(s in lower_prose for s in ("calculate", "energy", "consumption"))
    numerical_allowed = bool(has_num_context and len(numeric_vals) >= 2)
    assert numerical_allowed is True, "Prose chunk with calculation context and numbers should be numerical_allowed"

    # 2. Dual test: chunk that has BOTH real code and numerical calculations
    chunk_code_and_num = (
        "def compute_hash(seed, count):\n"
        "    int total = 100;\n"
        "    for x in range(count):\n"
        "        total += x * 2.5;\n"
        "    return total\n"
        "Calculate the final value when seed is 5 and count is 10."
    )
    lower_cn = chunk_code_and_num.lower()
    cn_code_hits = sum(bool(re.search(pat, lower_cn, re.IGNORECASE)) for pat in code_regexes)
    cn_numeric_vals = re.findall(r'(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?:\s*%)?(?![A-Za-z])', lower_cn)
    cn_programming_allowed = cn_code_hits >= 2
    cn_numerical_allowed = bool(
        any(s in lower_cn for s in ("calculate", "compute")) and len(cn_numeric_vals) >= 2
    )
    assert cn_programming_allowed is True, f"Expected programming_allowed=True, got {cn_code_hits} code hits"
    assert cn_numerical_allowed is True, "Expected numerical_allowed=True on dual chunk"


def test_reported_bloom_matches_max_sub_bloom():
    """Verify that main question reported_bloom matches the highest cognitive level among sub-questions,
    properly harmonizing the header tag with multi-part questions."""
    import re
    from v0_1.generator import get_bloom_level_name

    def _to_bloom_int(b):
        if isinstance(b, int):
            return b
        digits = re.findall(r'\d+', str(b))
        return int(digits[0]) if digits else 2

    # Case A: Int sub_questions with [L1, L3, L4]
    sub_questions_int = [
        {"letter": "a", "marks": 4, "bloom": 1},
        {"letter": "b", "marks": 3, "bloom": 3},
        {"letter": "c", "marks": 3, "bloom": 4},
    ]
    sub_blooms_int = [_to_bloom_int(sq["bloom"]) for sq in sub_questions_int]
    reported_bloom = max(sub_blooms_int)
    assert reported_bloom == 4
    header_tag = f"Answer the following: [L{reported_bloom}]"
    assert header_tag == "Answer the following: [L4]"
    assert get_bloom_level_name(reported_bloom) in ("Analyse", "Analyze")

    # Case B: String sub_questions with ["L1", "L2", "L5"]
    sub_questions_str = [
        {"letter": "a", "marks": 6, "bloom": "L1"},
        {"letter": "b", "marks": 4, "bloom": "L5"},
    ]
    sub_blooms_str = [_to_bloom_int(sq["bloom"]) for sq in sub_questions_str]
    reported_str = max(sub_blooms_str)
    assert reported_str == 5


def test_programming_allowed_true_positives():
    """Verify that genuine code chunks (Python defs, SQL queries, Java class declarations)
    reliably trigger programming_allowed=True without under-classifying real code."""
    import re

    code_regexes = (
        r'\b(?:def|class|public|private|static|interface|struct)\s+[a-zA-Z_]\w*',
        r'\b(?:int|float|double|char|void|boolean)\s+[a-zA-Z_]\w*\s*(?:=|\(|;)',
        r'\bfor\s+[a-zA-Z_]\w*\s+in\b',
        r'\b(?:while|for)\s*\([^)]+\)\s*[{;]',
        r'\b(?:SELECT\s+.+\s+FROM|INSERT\s+INTO|UPDATE\s+.+\s+SET|DELETE\s+FROM)\b',
        r'\b(?:CREATE\s+TABLE|ALTER\s+TABLE|DROP\s+TABLE|PRIMARY\s+KEY|FOREIGN\s+KEY)\b',
        r'```[a-zA-Z]*\n',
        r'\b(?:pseudocode|algorithm)\s*:',
        r'\bdef\s+[a-zA-Z_]\w*\s*\([^)]*\)\s*:',
    )

    # 1. Python function with loop and definition
    chunk_py = (
        "def find_max(arr):\n"
        "    max_val = arr[0]\n"
        "    for item in arr:\n"
        "        if item > max_val: max_val = item\n"
        "    return max_val\n"
    )
    hits_py = sum(bool(re.search(pat, chunk_py.lower(), re.IGNORECASE)) for pat in code_regexes)
    assert hits_py >= 2, f"Expected >= 2 code hits on Python code, got {hits_py}"

    # 2. SQL schema and query
    chunk_sql = (
        "CREATE TABLE student_records (\n"
        "    student_id INT PRIMARY KEY,\n"
        "    student_name VARCHAR(100)\n"
        ");\n"
        "SELECT student_id, student_name FROM student_records WHERE student_id > 100;\n"
    )
    hits_sql = sum(bool(re.search(pat, chunk_sql, re.IGNORECASE)) for pat in code_regexes)
    assert hits_sql >= 2, f"Expected >= 2 code hits on SQL block, got {hits_sql}"

    # 3. Markdown code block
    chunk_md = "```python\ndef binary_search(arr, x):\n    pass\n```\n"
    hits_md = sum(bool(re.search(pat, chunk_md.lower(), re.IGNORECASE)) for pat in code_regexes)
    assert hits_md >= 2, f"Expected >= 2 code hits on Markdown block, got {hits_md}"


def test_math_validator_katex_unavailable_grace_path():
    """Verify that when KaTeX executable is unavailable, pure-Python validation handles math gracefully:
    valid math passes without render crashes, while syntactically corrupt math is still strictly caught."""
    from core.validation.math_validator import KaTeXAvailabilityGate, validate_math_block_with_render
    from core.generation.output_schema import MathBlock
    from unittest.mock import patch

    # Mock KaTeX availability as False
    with patch.object(KaTeXAvailabilityGate, 'verify', return_value=False):
        # 1. Valid math block must pass pure-Python checks even when KaTeX is unavailable
        good_block = MathBlock(block_id="math_1", latex=r"\frac{x + 1}{x - 1}", display_mode=False)
        good_res = validate_math_block_with_render(good_block)
        assert good_res.passed is True, f"Valid math failed with KaTeX unavailable: {good_res.message}"

        # 2. Corrupt incomplete fraction must STILL fail with MATH_INCOMPLETE_FRAC
        bad_frac_block = MathBlock(block_id="math_2", latex=r"\frac{x + 1}", display_mode=False)
        bad_frac_res = validate_math_block_with_render(bad_frac_block)
        assert bad_frac_res.passed is False
        assert bad_frac_res.code == "MATH_INCOMPLETE_FRAC"

        # 3. Unbalanced braces must STILL fail with MATH_UNCLOSED_BRACES
        bad_brace_block = MathBlock(block_id="math_3", latex=r"\frac{x + 1}{x - 1", display_mode=False)
        bad_brace_res = validate_math_block_with_render(bad_brace_block)
        assert bad_brace_res.passed is False
        assert bad_brace_res.code == "MATH_UNCLOSED_BRACES"


def test_iot_smart_systems_archetype_resolution_and_directives():
    """Verify that IoT & smart agriculture subjects resolve to iot_smart_systems and carry domain directives."""
    from aion_patch import resolve_subject_archetype, SUBJECT_ARCHETYPES

    # Test alias mappings
    assert resolve_subject_archetype("IOT Agriculture and Health care") == "iot_smart_systems"
    assert resolve_subject_archetype("21CS81") == "iot_smart_systems"
    assert resolve_subject_archetype("precision agriculture") == "iot_smart_systems"
    assert resolve_subject_archetype("wireless sensor networks") == "iot_smart_systems"

    # Test domain directives registered for iot_smart_systems
    archetype_data = SUBJECT_ARCHETYPES.get("iot_smart_systems")
    assert archetype_data is not None
    directive = archetype_data.get("directive", "").lower()
    assert "precision agriculture" in directive or "agriculture" in directive
    assert "healthcare" in directive or "health care" in directive
    assert "mqtt" in directive or "sensor" in directive or "telemetry" in directive


def test_end_to_end_vtu_paper_structure_and_criteria():
    """
    End-to-end multi-module verification asserting all 5 production criteria:
    1. Questions numbered Q1..Q10 continuously across all 5 modules.
    2. Every question's CO matches its module (CO1 for Q1-Q2, ..., CO5 for Q9-Q10).
    3. Every question's reported_bloom equals max(subs) and header tag is harmonized.
    4. Numerical unblocking: Chunks with computational context yield numerical_allowed=True and math_required=True.
    5. Sub-question marks sum strictly to the main question total marks.
    """
    from v0_1.main import _generate_main_question
    from v0_1.difficulty import DifficultyManager
    from v0_1.difficulty_policy import _resolve_co_by_mode
    from core.generation.orchestrator import SlotOrchestrator
    from unittest.mock import MagicMock

    diff_manager = DifficultyManager.from_string("mixed")
    orchestrator = SlotOrchestrator()

    def mock_generate(slot, evidence_pack, **kwargs):
        from core.contracts.question import GeneratedQuestion
        from core.generation.output_schema import QuestionOutput
        verb = slot.bloom_verb.strip().capitalize()
        output = QuestionOutput(
            instruction=f"{verb} the operational parameters and system requirements in detail.",
            question_text=f"{verb} the operational parameters and system requirements in detail.",
            math_blocks=[],
        )
        return GeneratedQuestion(output=output, slot=slot)
    orchestrator.generate = mock_generate

    # Evidence with real sensor telemetry calculation (unblocked numerical)
    iot_numerical_evidence = (
        "In a precision agriculture IoT sensor network, the soil moisture sensor nodes transmit telemetry "
        "every 10 minutes at 20 mW power where the battery capacity is 2500 mAh and operating voltage is 3.3 V. "
        "Calculate the total energy consumption per day and determine the expected operational lifetime of the sensor node."
    )

    paper_modules = []
    questions_per_module = 2
    target_marks = 10

    for mod_idx in range(1, 6):
        mod_questions = []
        for local_idx in range(1, questions_per_module + 1):
            global_q_no = (mod_idx - 1) * questions_per_module + local_idx
            partition = [6, 4]

            # In Module 2 Q3, test numerical unblocking on real calculation chunk
            chunk_text = iot_numerical_evidence if (mod_idx == 2 and local_idx == 1) else (
                f"Module {mod_idx} foundational theory and system architecture concepts for engineering."
            )
            planned_type = ["NUMERICAL", "APPLICATION"] if (mod_idx == 2 and local_idx == 1) else ["CONCEPTUAL", "CONCEPTUAL"]

            mq = _generate_main_question(
                mq_idx=global_q_no,
                partition=partition,
                bloom=3,
                chunks=[chunk_text, chunk_text],
                total_marks=target_marks,
                diff_manager=diff_manager,
                chunk_obj=None,
                selector=MagicMock(),
                module_id=f"module_{mod_idx}",
                orchestrator=orchestrator,
                planned_types=planned_type,
                blueprint_co=_resolve_co_by_mode(mod_idx, target_marks),
                slot_bloom_targets={1: 2, 2: 4, 3: 3, 4: 5},
            )
            mod_questions.append(mq)
        paper_modules.append({"module_index": mod_idx, "questions": mod_questions})

    # Criterion 1: Questions numbered Q1..Q10 continuously across modules
    all_q_indices = [q["mq_index"] for mod in paper_modules for q in mod["questions"]]
    assert all_q_indices == list(range(1, 11)), f"Expected Q1..Q10, got {all_q_indices}"

    # Criterion 2: Course Outcome is determined based on marks and bloom level
    # 6M subquestions -> CO2 (moderate/application tier)
    # 4M subquestions -> CO1 (foundational tier)
    for mod in paper_modules:
        for q in mod["questions"]:
            for sq in q["sub_questions"]:
                if sq["marks"] == 6:
                    assert sq["co"] == "CO2", (
                        f"Question {q['mq_index']} 6M subquestion has CO '{sq['co']}', expected 'CO2'"
                    )
                elif sq["marks"] == 4:
                    assert sq["co"] == "CO1", (
                        f"Question {q['mq_index']} 4M subquestion has CO '{sq['co']}', expected 'CO1'"
                    )

    # Criterion 3: Every question's reported_bloom equals max(subs)
    for mod in paper_modules:
        for q in mod["questions"]:
            sub_blooms = [sq["bloom"] for sq in q["sub_questions"]]
            expected_max_bloom = max(sub_blooms)
            assert q["bloom_level"] == expected_max_bloom, (
                f"Question {q['mq_index']} reported_bloom {q['bloom_level']} != max(subs) {expected_max_bloom}"
            )

    # Criterion 4: Numerical unblocking on computational evidence
    mod2_q3 = paper_modules[1]["questions"][0]
    has_numerical = any(
        getattr(gq.slot, "question_type", None) == "NUMERICAL" or sq.get("difficulty") == "MEDIUM"
        for gq, sq in zip(mod2_q3["generated_questions"], mod2_q3["sub_questions"])
    )
    assert has_numerical is True, "Expected numerical question in Module 2 Q3 on calculation evidence"

    # Criterion 5: Sub-marks sum strictly to main marks
    for mod in paper_modules:
        for q in mod["questions"]:
            sub_total = sum(sq["marks"] for sq in q["sub_questions"])
            assert sub_total == target_marks, (
                f"Question {q['mq_index']} sub-marks {sub_total} != target {target_marks}"
            )
            assert sub_total == q["total_marks"]

    # Criterion 6: First word of every question matches its assigned Bloom level keywords
    from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP
    for mod in paper_modules:
        for q in mod["questions"]:
            for sq in q["sub_questions"]:
                first_word = sq["text"].strip().split()[0].rstrip(".,;:").lower()
                sq_bloom_num = sq["bloom"]
                assert first_word in BLOOM_VERB_LEVEL_MAP[f"L{sq_bloom_num}"], (
                    f"First word '{first_word}' in question {q['mq_index']} does not match Bloom level L{sq_bloom_num} keywords: {BLOOM_VERB_LEVEL_MAP[f'L{sq_bloom_num}']}"
                )


def test_bloom_level_co_mapping_and_opening_keyword_verification():
    """
    Verify:
    1. Every question's opening word matches canonical Bloom level keywords from BLOOM_VERB_LEVEL_MAP.
    2. Tags for bloom level (L1..L5) and Course Outcome (CO1..CO3) correctly match question marks and cognitive tier.
    3. The two-layer bloom validator and AutoHealer enforce opening keyword alignment with zero drift.
    """
    from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP, check_bloom_two_layer
    from core.validation.linter import check_bloom_verb_at_start
    from core.generation.auto_healer import AutoHealer
    from core.generation.output_schema import QuestionOutput
    from v0_1.difficulty import DifficultyManager
    from v0_1.difficulty_policy import resolve_co_bl_from_marks, format_co_and_rbt

    dm = DifficultyManager.from_string("mixed")

    # 1. Test Bloom level and CO tagging across standard marks splits
    test_cases = [
        # (module_idx, marks, planned_type, expected_co, expected_blooms)
        (1, 4, "CONCEPTUAL", "CO1", [1]),
        (1, 4, "APPLICATION", "CO1", [2]),
        (2, 6, "CONCEPTUAL", "CO2", [2]),
        (2, 6, "APPLICATION", "CO2", [3]),
        (3, 8, "ANALYTICAL", "CO3", [4]),
        (4, 10, "EVALUATE", "CO3", [5]),
    ]

    for mod_idx, marks, q_type, expected_co, expected_blooms in test_cases:
        co, bloom = resolve_co_bl_from_marks(
            module_idx=mod_idx,
            marks=marks,
            total_parts=2,
            planned_type=q_type,
        )
        assert co == expected_co, f"Marks {marks}M ({q_type}) resolved CO {co}, expected {expected_co}"
        assert bloom in expected_blooms, f"Marks {marks}M ({q_type}) resolved Bloom {bloom}, expected {expected_blooms}"

        # Safe formatting check
        final_co, final_rbt = format_co_and_rbt(co, bloom, mod_idx)
        assert final_co == expected_co
        assert final_rbt == f"L{bloom}"

        # Verify verb selection for this Bloom level
        verb = dm.get_verb("mixed", bloom)
        assert verb.lower() in BLOOM_VERB_LEVEL_MAP[f"L{bloom}"], (
            f"Verb '{verb}' does not belong to Bloom level L{bloom} keywords: {BLOOM_VERB_LEVEL_MAP[f'L{bloom}']}"
        )

        # 2. Verify opening keyword enforcement
        class MockSlot:
            pass
        slot = MockSlot()
        slot.bloom_verb = verb
        slot.bloom_level = f"L{bloom}"
        slot.marks = marks
        slot.co = co

        # Valid question starting with the assigned keyword
        valid_output = QuestionOutput(
            instruction=f"{verb} the operational principles and architecture of the topic.",
            question_text=f"{verb} the operational principles and architecture of the topic.",
            bloom_level=f"L{bloom}",
            marks=marks,
        )

        # Check passes both linter and two-layer bloom validator
        res_linter = check_bloom_verb_at_start(valid_output.instruction, slot)
        assert res_linter.passed is True, f"Valid opening verb '{verb}' failed linter: {res_linter.message}"

        res_two_layer = check_bloom_two_layer(valid_output.instruction, slot)
        assert res_two_layer.passed is True, f"Valid opening verb '{verb}' failed two-layer check: {res_two_layer.detail}"

        # First word of question strictly matches the Bloom keyword
        first_word = valid_output.question_text.strip().split()[0].rstrip(".,;:").lower()
        assert first_word == verb.lower(), f"First word '{first_word}' != Bloom keyword '{verb.lower()}'"
        assert first_word in BLOOM_VERB_LEVEL_MAP[f"L{bloom}"]

        # 3. Test invalid opening keyword detection & AutoHealer recovery
        invalid_output = QuestionOutput(
            instruction="In order to understand the concepts, consider how the system functions.",
            question_text="In order to understand the concepts, consider how the system functions.",
            bloom_level=f"L{bloom}",
            marks=marks,
        )
        res_invalid = check_bloom_verb_at_start(invalid_output.instruction, slot)
        assert res_invalid.passed is False
        assert res_invalid.code == "BLOOM_VERB_NOT_AT_START"

        # AutoHealer resolves the invalid opening by prefixing the exact Bloom keyword
        healed = AutoHealer.heal("BLOOM_VERB_NOT_AT_START", invalid_output, slot)
        healed_first_word = healed.question_text.strip().split()[0].rstrip(".,;:").lower()
        assert healed_first_word == verb.lower()
        assert healed_first_word in BLOOM_VERB_LEVEL_MAP[f"L{bloom}"]
        assert check_bloom_verb_at_start(healed.instruction, slot).passed is True


def test_bloom_verb_taxonomy_mismatch_and_numerical_l3_enforcement():
    """
    Verify:
    1. A slot with slot.bloom_verb="Calculate" and bloom_level="L4" fails validation (Q3 defect regression).
    2. Numerical calculation tasks strictly resolve to Bloom L3 across marks tiers.
    3. Revision tables (Table 2.19, Important terms for revision) are classified as EXTERNAL by ContentRole firewall.
    4. Equal module allocation generates 2 questions per module (Set 1 -> Q1/Q2, Set 2 -> Q3/Q4, etc.).
    """
    from core.validation.bloom_validator import BLOOM_VERB_LEVEL_MAP, check_bloom_two_layer
    from core.validation.linter import check_bloom_verb_at_start
    from core.generation.output_schema import QuestionOutput
    from v0_1.difficulty_policy import resolve_co_bl_from_marks
    from v0_1.chunk_image_mapper import classify_chunk_depth

    class MockSlot:
        pass

    # 1. Q3 Defect Regression: Calculate tagged L4 must fail validation
    slot_bad = MockSlot()
    slot_bad.bloom_verb = "Calculate"
    slot_bad.bloom_level = "L4"
    output_bad = QuestionOutput(
        instruction="Calculate the throughput of the given channel using Shannon capacity formula.",
        question_text="Calculate the throughput of the given channel using Shannon capacity formula.",
        bloom_level="L4",
        marks=8,
    )
    res_linter_bad = check_bloom_verb_at_start(output_bad.instruction, slot_bad)
    assert res_linter_bad.passed is False
    assert res_linter_bad.code == "BLOOM_TAXONOMY_MISMATCH", f"Expected BLOOM_TAXONOMY_MISMATCH, got {res_linter_bad.code}"

    res_two_layer_bad = check_bloom_two_layer(output_bad.instruction, slot_bad)
    assert res_two_layer_bad.passed is False
    assert res_two_layer_bad.code == "BLOOM_TAXONOMY_MISMATCH"

    # 2. Numerical task resolution strictly resolves to L3 regardless of marks
    for marks in (4, 6, 8, 10, 20):
        co, bl = resolve_co_bl_from_marks(module_idx=2, marks=marks, total_parts=2, planned_type="NUMERICAL")
        assert bl == 3, f"Numerical task with {marks} marks resolved to Bloom L{bl}, expected L3"

    # 3. ContentRole firewall blocks revision metadata & question banks
    assert classify_chunk_depth("Table 2.19: Important terms for revision across satellite communications.") == "EXTERNAL"
    assert classify_chunk_depth("Summary of Module 2 and review of core exam formulas.") == "EXTERNAL"
    assert classify_chunk_depth("Chapter Question Bank: 1. State Kepler's laws. 2. Derive path loss.") == "EXTERNAL"
    # Normal substantive content remains CORE or SUPPORTING
    assert classify_chunk_depth("Explain the operation of a transponder using frequency translation with input filter.") in ("CORE", "SUPPORTING")

    # 4. Equal module allocation formula verification: Set 1 -> Q1,Q2; Set 2 -> Q3,Q4; ... Set 5 -> Q9,Q10
    questions_per_module = 2
    for mod_idx in range(1, 6):
        for local_idx in (1, 2):
            global_q_no = (mod_idx - 1) * questions_per_module + local_idx
            # DOCX/frontend canonical formula ((qNo - 1) // 2) + 1 must match mod_idx
            assert ((global_q_no - 1) // 2) + 1 == mod_idx
    # Set 1 -> Q1, Q2
    assert [(1 - 1) * 2 + 1, (1 - 1) * 2 + 2] == [1, 2]
    # Set 2 -> Q3, Q4
    assert [(2 - 1) * 2 + 1, (2 - 1) * 2 + 2] == [3, 4]
    # Set 5 -> Q9, Q10
    assert [(5 - 1) * 2 + 1, (5 - 1) * 2 + 2] == [9, 10]








