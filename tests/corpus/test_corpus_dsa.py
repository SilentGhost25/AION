"""
Real-Document Corpus Test: Data Structures and Algorithms (DSA)
"""

import pytest
from v0_1.unified_pipeline import run_unified
from v0_1.question_completeness import QuestionCompletenessValidator


@pytest.fixture
def dsa_corpus_file(tmp_path) -> str:
    import fitz
    pdf_path = tmp_path / "dsa_full_textbook_chapter.pdf"
    content = (
        "MODULE 1: Linear Data Structures\n\n"
        "Arrays, Stacks, and Queues are fundamental linear data structures. "
        "A stack is a Last-In First-Out (LIFO) data structure supporting push and pop operations.\n\n"
        "MODULE 2: Trees and Graphs\n\n"
        "Binary Search Trees maintain left child less than root and right child greater. "
        "AVL trees perform LL, RR, LR, and RL rotations to ensure balance factor remains between -1 and +1.\n\n"
        "MODULE 3: Graph Algorithms\n\n"
        "Dijkstra's algorithm finds single-source shortest paths in weighted graphs with non-negative edge weights. "
        "Prim's and Kruskal's algorithms construct Minimum Spanning Trees (MST) for connected weighted graphs."
    )
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


def test_dsa_corpus_pipeline_run(dsa_corpus_file):
    paper = run_unified(
        file_path=dsa_corpus_file,
        exam_type="IA",
        difficulty="Mixed",
        subject="Data Structures",
        max_questions=5,
    )

    assert paper is not None
    assert paper.exportable is True
    assert len(paper.modules) > 0

    # Verify all generated sub-questions pass QuestionCompletenessValidator
    for module in paper.modules:
        for q in module.get("questions", []):
            for sub in q.get("subQuestions", []):
                text = sub["text"]
                valid, errors = QuestionCompletenessValidator.validate(text)
                assert valid is True, f"Incomplete question detected: {errors} in '{text}'"
