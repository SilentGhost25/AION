import pytest
from pathlib import Path

from server.examiner_similarity import ExaminerSimilarityScorer
from server.candidate_generator import NullCandidateGenerator
from conftest import write_fake_file


def test_scorer_handles_empty_previous_papers(academic_root):
    generator = NullCandidateGenerator()
    scorer = ExaminerSimilarityScorer(generator)
    
    # Empty papers
    res = scorer.compute([], ["concept text"])
    assert res["examiner_similarity_score"] == 0.0
    assert res["note"] == "no_reference_questions"


def test_scorer_handles_empty_knowledge_samples(academic_root):
    generator = NullCandidateGenerator()
    scorer = ExaminerSimilarityScorer(generator)
    
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    paper_file = write_fake_file(subject_dir / "previous_papers" / "2024.pdf", "Explain CSP concept (10 Marks)\nDefine search method (5 Marks)")

    res = scorer.compute([str(paper_file)], [])
    assert res["examiner_similarity_score"] == 0.0
    assert res["note"] == "no_knowledge_samples"


def test_scorer_computes_style_metrics_successfully(academic_root):
    generator = NullCandidateGenerator()
    scorer = ExaminerSimilarityScorer(generator)
    
    subject_dir = academic_root / "AIML" / "semester_4" / "BAI401"
    paper_file = write_fake_file(subject_dir / "previous_papers" / "2024.pdf", "Explain planning process (10 Marks)\nDefine heuristics approach (5 Marks)")

    res = scorer.compute([str(paper_file)], ["AI searches", "planning logic"], sample_size=5)
    
    assert "examiner_similarity_score" in res
    assert "verb_similarity" in res
    assert "bloom_similarity" in res
    assert "marks_similarity" in res
    assert res["examiner_similarity_score"] >= 0.0
