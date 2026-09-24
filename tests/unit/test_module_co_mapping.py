"""
Unit tests for subject-agnostic Course Outcome (CO) mapping.
Validates parameterized module/CO combinations, custom mapping overrides,
and default module-based distribution ending the 80% CO3 skew.
"""

import pytest
from core.contracts.paper_spec import PaperSpec
from v0_1.difficulty_policy import build_co_map, _resolve_co_by_mode


@pytest.mark.parametrize("module_count, co_count, expected", [
    (5, 5, {"1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO5"}),
    (4, 4, {"1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4"}),
    (5, 4, {"1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO4"}),
    (6, 6, {"1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO5", "6": "CO6"}),
    (3, 3, {"1": "CO1", "2": "CO2", "3": "CO3"}),
])
def test_build_co_map_generalized(module_count, co_count, expected):
    """Verify build_co_map computes correct mappings for diverse syllabi."""
    mapping = build_co_map(module_count, co_count)
    assert mapping == expected


def test_custom_co_map_override():
    """Verify course-specific custom CO mappings override computed defaults."""
    custom = {"1": "CO1", "2": "CO1", "3": "CO2", "4": "CO3", "5": "CO4"}
    mapping = build_co_map(5, 4, custom_map=custom)
    assert mapping == custom


def test_resolve_co_by_mode_module_based_default(monkeypatch):
    """
    Verify _resolve_co_by_mode defaults to module-based and produces
    balanced distribution across modules rather than skewing to CO3.
    """
    monkeypatch.delenv("AION_CO_MODE", raising=False)

    # 10M questions in module 1, 2, 3, 4, 5
    co_mod1 = _resolve_co_by_mode(module_idx=1, marks=10)
    co_mod2 = _resolve_co_by_mode(module_idx=2, marks=10)
    co_mod3 = _resolve_co_by_mode(module_idx=3, marks=10)
    co_mod4 = _resolve_co_by_mode(module_idx=4, marks=10)
    co_mod5 = _resolve_co_by_mode(module_idx=5, marks=10)

    assert co_mod1 == "CO1"
    assert co_mod2 == "CO2"
    assert co_mod3 == "CO3"
    assert co_mod4 == "CO4"
    assert co_mod5 == "CO5"

    # All 5 COs represented equally (20% each), not 80% CO3
    assert len({co_mod1, co_mod2, co_mod3, co_mod4, co_mod5}) == 5


def test_paperspec_build_co_map_integration():
    """Verify PaperSpec builds identical CO maps with custom override validation."""
    spec_5x5 = PaperSpec("IAT1", 5, 2, 10, 5)
    assert spec_5x5.build_co_map() == {
        "1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO5"
    }

    custom_spec = PaperSpec(
        "CUSTOM", 5, 2, 10, 4,
        custom_co_map={"1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO4"}
    )
    assert custom_spec.build_co_map() == {
        "1": "CO1", "2": "CO2", "3": "CO3", "4": "CO4", "5": "CO4"
    }
