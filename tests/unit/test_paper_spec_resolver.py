"""
Tests for PaperSpec and paper_spec_resolver.
Validates catalog resolution, precedence rules, and error handling.
"""

import pytest
from core.contracts.paper_spec import PaperSpec
from core.generation.paper_spec_resolver import resolve_paper_spec, load_exam_specs_catalog


def test_resolver_from_catalog():
    """Verify standard exam specs resolve correctly from exam_specs.json."""
    spec_iat = resolve_paper_spec("IAT1")
    assert spec_iat.module_count == 5
    assert spec_iat.questions_per_module == 2
    assert spec_iat.marks_per_question == 10
    assert spec_iat.co_count == 5
    assert spec_iat.total_questions == 10
    assert spec_iat.total_marks == 50

    spec_see = resolve_paper_spec("SEE")
    assert spec_see.module_count == 5
    assert spec_see.questions_per_module == 2
    assert spec_see.marks_per_question == 20
    assert spec_see.total_questions == 10
    assert spec_see.total_marks == 100

    spec_elec = resolve_paper_spec("ELECTIVE_3MOD")
    assert spec_elec.module_count == 3
    assert spec_elec.questions_per_module == 3
    assert spec_elec.marks_per_question == 10
    assert spec_elec.total_questions == 9
    assert spec_elec.total_marks == 30


def test_resolver_override_precedence():
    """Payload overrides take precedence over catalog defaults."""
    override = {
        "co_count": 4,
        "marks_per_question": 15,
        "depth_threshold_ratio": 0.75,
    }
    spec = resolve_paper_spec("IAT1", override=override)
    assert spec.co_count == 4
    assert spec.marks_per_question == 15
    assert spec.depth_threshold_ratio == 0.75
    # Non-overridden fields retain catalog values
    assert spec.module_count == 5
    assert spec.questions_per_module == 2


def test_resolver_unknown_exam_type_fail_fast():
    """Unknown exam types without full specifications must fail fast."""
    with pytest.raises(ValueError, match="Unknown exam_type"):
        resolve_paper_spec("UNKNOWN_EXAM_FORMAT_XYZ")


def test_resolver_custom_ad_hoc_exam():
    """Custom exam types succeed if full overrides are provided."""
    custom_override = {
        "module_count": 4,
        "questions_per_module": 3,
        "marks_per_question": 12,
        "co_count": 4,
    }
    spec = resolve_paper_spec("CUSTOM_4X3", override=custom_override)
    assert spec.module_count == 4
    assert spec.questions_per_module == 3
    assert spec.marks_per_question == 12
    assert spec.total_questions == 12
    assert spec.total_marks == 48
