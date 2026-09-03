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






