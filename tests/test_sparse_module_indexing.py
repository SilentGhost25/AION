"""
Parametrized test suite for sparse module indexing, segmentation, chunk mapping,
and question-to-module assignment across all sparse permutations.
Addresses Issue 3 from the Critical Analysis & Problem Audit.
"""

import pytest
from typing import List, Set
from v0_1.segmenter import RobustSegmenter, ModuleSegment
from v0_1.chunk_image_mapper import ChunkImageMapper
from v0_1.docx_export import get_module_for_q


def _generate_synthetic_module_text(module_num: int, word_count: int = 60) -> str:
    """Generates synthetic module content with an explicit header and sufficient words."""
    words = [
        f"concept_{i}" for i in range(word_count)
    ]
    body = " ".join(words)
    return f"Module {module_num}: Advanced Study of Topic {module_num}\n\n{body}."


@pytest.mark.parametrize("module_slots,expected_question_modules", [
    ([1, 3], {1, 3}),           # Q1-Q2 -> Mod 1, Q5-Q6 -> Mod 3
    ([1, 2, 4, 5], {1, 2, 4, 5}), # Q1-Q4 -> Mod 1-2, Q7-Q10 -> Mod 4-5
    ([2, 5], {2, 5}),           # Q3-Q4 -> Mod 2, Q9-Q10 -> Mod 5
    ([3], {3}),                 # Single module -> Q5-Q6 (Mod 3)
])
def test_sparse_module_segmentation(module_slots: List[int], expected_question_modules: Set[int]):
    """Verify RobustSegmenter preserves exact module indices for all sparse permutations."""
    text = "\n\n".join([_generate_synthetic_module_text(m) for m in module_slots])
    segmenter = RobustSegmenter()
    segments = segmenter.segment(text)

    found_indices = {s.module_index for s in segments}
    assert found_indices == expected_question_modules, (
        f"Segmenter returned indices {found_indices}, expected {expected_question_modules}"
    )


@pytest.mark.parametrize("module_slots,expected_question_modules", [
    ([1, 3], {1, 3}),
    ([1, 2, 4, 5], {1, 2, 4, 5}),
    ([2, 5], {2, 5}),
    ([3], {3}),
])
def test_sparse_module_chunk_image_mapper(module_slots: List[int], expected_question_modules: Set[int]):
    """Verify ChunkImageMapper preserves module identity for sparse module segments."""
    segments = [
        ModuleSegment(
            title=f"Module {m}: Specialized Discipline {m}",
            content=f"Detailed syllabus and reference content for domain topic {m}. " * 10,
            word_count=80,
            module_index=m,
        )
        for m in module_slots
    ]

    mapper = ChunkImageMapper()
    mapper.build(segments)
    
    # Assert each module_id exists and has chunks mapped
    for m in expected_question_modules:
        mod_id = f"module_{m}"
        chunks = mapper.get_chunks_for_module(mod_id)
        assert len(chunks) > 0, f"Expected chunks for {mod_id}, but found none"
        for chunk in chunks:
            assert chunk.module_id == mod_id
            assert f"topic {m}" in chunk.text.lower()


@pytest.mark.parametrize("module_slots,expected_question_modules", [
    ([1, 3], {1, 3}),
    ([1, 2, 4, 5], {1, 2, 4, 5}),
    ([2, 5], {2, 5}),
    ([3], {3}),
])
def test_sparse_module_question_mapping(module_slots: List[int], expected_question_modules: Set[int]):
    """
    Verify questions derived from sparse modules map to the expected module numbers
    both via explicit field and via the canonical VTU formula ((q_num - 1) // 2) + 1.
    """
    generated_questions = []
    for m in module_slots:
        # Each module produces a pair of main questions
        q1_num = (m - 1) * 2 + 1
        q2_num = (m - 1) * 2 + 2
        
        q1 = {
            "qNo": q1_num,
            "module": m,
            "module_index": m,
            "subQuestions": [
                {"letter": "a", "marks": 6, "co": f"CO{min(m, 5)}", "bloom": "L2", "module": m},
                {"letter": "b", "marks": 4, "co": f"CO{min(m, 5)}", "bloom": "L3", "module": m},
            ]
        }
        q2 = {
            "qNo": q2_num,
            "module": m,
            "module_index": m,
            "subQuestions": [
                {"letter": "a", "marks": 6, "co": f"CO{min(m, 5)}", "bloom": "L2", "module": m},
                {"letter": "b", "marks": 4, "co": f"CO{min(m, 5)}", "bloom": "L3", "module": m},
            ]
        }
        generated_questions.extend([q1, q2])

    for q in generated_questions:
        derived_mod = get_module_for_q(q)
        assert derived_mod in expected_question_modules, (
            f"Question {q['qNo']} resolved to module {derived_mod}, not in {expected_question_modules}"
        )
        assert derived_mod == q["module"]
        # Subquestion check
        for sq in q["subQuestions"]:
            assert sq["module"] in expected_question_modules
