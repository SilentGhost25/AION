"""
Acceptance Test Suite for Stage G2-C:
Assessment Blueprint, Composable Question Archetypes & Task Contracts.

Covers Acceptance Criteria AC-C1 through AC-C14:
- AC-C1: Universal Archetype Enumeration & Composition
- AC-C2: Cognitive Capability Enumeration
- AC-C3: Artifact Modality Enumeration
- AC-C4: Parameter Requirement Provenance & State Validation
- AC-C5: Input vs. Requested Student Artifact Separation
- AC-C6: Expected Student Response Contract & Time Estimation
- AC-C7: Marking Rubric Allocation & Mark Conservation
- AC-C8: SubPart and Question Mark Conservation Invariants
- AC-C9: Alternative Group Structure & Internal Parity
- AC-C10: Task Completeness Assessment (Answerability Gate)
- AC-C11: PlanningAgent Blueprint Generation (50M IA & 100M SEE)
- AC-C12: Blueprint-to-QuestionPlans Projection & Specification Preservation
- AC-C13: WritingAgent Archetype Prompt Generation Integration
- AC-C14: Backward Compatibility & Zero-Regression
"""

import pytest
from typing import Any, Dict, List, Optional

from core.contracts.assessment_blueprint import (
    AlternativeGroup,
    ArtifactModality,
    AssessmentBlueprint,
    AssessmentSlot,
    CognitiveCapability,
    ExpectedResponseContract,
    InputArtifactRequirement,
    MarkingRubricContract,
    ParameterRequirement,
    ParameterState,
    QuestionSpecification,
    RubricCriterion,
    SubPartSpecification,
    TaskCompleteness,
    UniversalArchetype,
    evaluate_task_completeness,
)
from core.generation.agents.base import AgentContext
from core.generation.assessment_config import AssessmentConfig, AssessmentRegistry
from core.generation.agents.planning_agent import (
    PlanningAgent,
    QuestionPlan,
    SubPartPlan,
)
from core.generation.agents.prompts import (
    build_archetype_subpart_prompt,
    build_subpart_prompt,
)
from core.generation.agents.writing_agent import WritingAgent


@pytest.fixture(autouse=True)
def setup_test_registry():
    """Register synthetic and VTU baseline assessment configs for tests."""
    AssessmentRegistry._reset_for_tests()
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (4, 6), (3, 7), (10,)),
        max_subparts=2,
        reference_source="VTU 2022 Scheme AEC Regulations (2024), Annexure-II, Clause 4(a)",
    ))
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="CIE_IAT",
        blueprint_id="BP_10M",
        total_marks_per_question=10,
        permitted_partitions=((5, 5), (6, 4), (10,)),
        max_subparts=2,
        reference_source="Test Specs 10M",
    ))
    AssessmentRegistry.register(AssessmentConfig(
        scheme_id="SYNTH",
        course_type="SYNTHETIC_TEST",
        assessment_type="SEE_EXAM",
        blueprint_id="BP_20M",
        total_marks_per_question=20,
        permitted_partitions=((8, 6, 6), (10, 10), (20,)),
        max_subparts=3,
        reference_source="Test Specs 20M",
    ))
    yield
    AssessmentRegistry._reset_for_tests()


# =============================================================================
# Test Mocks
# =============================================================================


class MockLLMCaller:
    def __init__(self, responses: Optional[List[Any]] = None):
        self.responses = list(responses) if responses is not None else []
        self.calls: List[Dict[str, Any]] = []

    def call(self, prompt: str, schema=None, image_path=None, seed=None):
        self.calls.append({
            "prompt": prompt,
            "schema": schema,
            "image_path": image_path,
            "seed": seed,
        })
        if self.responses:
            resp = self.responses.pop(0)
            if isinstance(resp, Exception):
                raise resp
            return resp
        import re
        m_bloom = re.search(r"Target Bloom Level:\s*(L\d)", prompt) or re.search(r"Target Bloom level \(RBT\):\s*(L\d)", prompt) or re.search(r"Bloom:\s*(L\d)", prompt)
        bloom = m_bloom.group(1) if m_bloom else "L2"
        verb = "Explain" if bloom == "L2" else ("Calculate" if bloom == "L3" else "Describe")
        text = f"{verb} the operating principles and characteristics of {bloom} systems."
        m = re.search(r"Marks:\s*(\d+)", prompt) or re.search(r"Total Subpart Marks:\s*(\d+)", prompt)
        marks = int(m.group(1)) if m else 5
        m1 = max(1, marks // 2)
        m2 = marks - m1
        return {
            "text": text,
            "solution": "Solution steps and governing principles...",
            "marking_scheme": [
                {"criterion": "Formula identification and boundary setup", "marks": m1},
                {"criterion": "Numerical substitution and computation", "marks": m2},
            ],
            "diagram_request": None,
            "references_image": False,
            "bloom_verb_used": verb,
        }


# =============================================================================
# Helper Fixtures
# =============================================================================


def make_valid_subpart_spec(label: str = "a", marks: int = 6, bloom: str = "L3") -> SubPartSpecification:
    p1 = ParameterRequirement(
        name="mass",
        state=ParameterState.GIVEN,
        value=10.5,
        unit="kg",
        source_reference="Module 1 Section 2.1",
    )
    art = InputArtifactRequirement(
        artifact_id="art1",
        artifact_type="System Parameters",
        modality=ArtifactModality.STRUCTURED_TABLE,
        caption_or_reference="Initial boundary conditions",
        parameters=(p1,),
    )
    m1 = max(1, marks // 2)
    m2 = marks - m1
    rubric = MarkingRubricContract(
        criteria=(
            RubricCriterion(criterion="Problem statement setup", marks=m1),
            RubricCriterion(criterion="Final derivation & steps", marks=m2),
        ),
    )
    expected_resp = ExpectedResponseContract(
        response_type="Analytical Solution",
        estimated_time_minutes=12,
        expected_milestones=("State governing equation", "Substitute parameters and compute"),
        requested_artifacts=("Free body diagram",),
    )
    return SubPartSpecification(
        label=label,
        marks=marks,
        bloom_level=bloom,
        primary_archetype=UniversalArchetype.WORKED_PROBLEM_CALCULATION,
        secondary_archetypes=(UniversalArchetype.CONCEPTUAL_EXPLANATION,),
        required_capabilities=(CognitiveCapability.NUMERICAL_SOLVING,),
        input_artifacts=(art,),
        expected_response=expected_resp,
        marking_rubric=rubric,
    )


def make_valid_question_spec(question_id: str = "q1", marks: int = 10, bloom: str = "L3") -> QuestionSpecification:
    sp_a = make_valid_subpart_spec("a", 6, bloom)
    sp_b = make_valid_subpart_spec("b", 4, bloom)
    return QuestionSpecification(
        question_id=question_id,
        syllabus_topic="Fluid Dynamics and Viscosity",
        course_outcome="CO1",
        bloom_level=bloom,
        marks=marks,
        subparts=(sp_a, sp_b),
        primary_archetype=UniversalArchetype.WORKED_PROBLEM_CALCULATION,
        secondary_archetypes=(UniversalArchetype.CONCEPTUAL_EXPLANATION,),
        required_capabilities=(CognitiveCapability.NUMERICAL_SOLVING,),
    )


# =============================================================================
# Tests
# =============================================================================


def test_ac_c1_archetypes_enum():
    """AC-C1: Universal Archetype Enumeration & Composition."""
    expected = {
        "CONCEPTUAL_EXPLANATION",
        "WORKED_PROBLEM_CALCULATION",
        "ALGORITHM_PROCEDURAL_TRACE",
        "COMPARATIVE_ANALYSIS",
        "SCHEMATIC_SYSTEM_SYNTHESIS",
        "EMPIRICAL_MATERIAL_DESIGN",
        "LOGICAL_FORMULATION_INFERENCE",
        "TECHNO_ECONOMIC_EVALUATION",
    }
    actual = {a.name for a in UniversalArchetype}
    assert actual == expected

    # Verify composition in SubPartSpecification
    spec = make_valid_subpart_spec()
    assert spec.primary_archetype == UniversalArchetype.WORKED_PROBLEM_CALCULATION
    assert UniversalArchetype.CONCEPTUAL_EXPLANATION in spec.secondary_archetypes


def test_ac_c2_cognitive_capabilities_enum():
    """AC-C2: Cognitive Capability Enumeration."""
    expected = {
        "NUMERICAL_SOLVING",
        "DERIVATION",
        "SCHEMATIC_ANALYSIS",
        "ALGORITHM_TRACE",
        "TRADE_OFF_ANALYSIS",
        "CODE_COMPLIANCE",
        "OPTIMIZATION",
        "DIAGRAMMATIC_SYNTHESIS",
    }
    actual = {c.name for c in CognitiveCapability}
    assert actual == expected


def test_ac_c3_artifact_modalities_enum():
    """AC-C3: Artifact Modality Enumeration."""
    expected = {
        "VISUAL_IMAGE",
        "STRUCTURED_TABLE",
        "TEXTUAL_INSTANCE",
    }
    actual = {m.name for m in ArtifactModality}
    assert actual == expected


def test_ac_c4_parameter_state_validation():
    """AC-C4: Parameter requirement provenance & state validation invariants."""
    # 1. GIVEN parameter with value succeeds
    p_valid = ParameterRequirement(
        name="voltage",
        state=ParameterState.GIVEN,
        value=230,
        unit="V",
        source_reference="Textbook Sec 3.2",
    )
    assert p_valid.value == 230

    # 2. GIVEN parameter with value=None MUST raise ValueError
    with pytest.raises(ValueError, match="must have an explicit value, not None"):
        ParameterRequirement(
            name="voltage",
            state=ParameterState.GIVEN,
            value=None,
        )

    # 3. STUDENT_SELECTED parameter without constraints MUST raise ValueError
    with pytest.raises(ValueError, match="must declare selection_constraints"):
        ParameterRequirement(
            name="resistor_choice",
            state=ParameterState.STUDENT_SELECTED,
            selection_constraints="",
        )

    # 4. STUDENT_SELECTED with constraints succeeds
    p_sel = ParameterRequirement(
        name="resistor_choice",
        state=ParameterState.STUDENT_SELECTED,
        selection_constraints="Choose standard E12 resistor value between 1k and 10k ohm",
    )
    assert p_sel.state == ParameterState.STUDENT_SELECTED

    # 5. DERIVABLE parameter without relationship or source MUST raise ValueError
    with pytest.raises(ValueError, match="must declare governing_relationship or source_reference"):
        ParameterRequirement(
            name="current",
            state=ParameterState.DERIVABLE,
        )

    # 6. DERIVABLE parameter with governing relationship succeeds with value=None
    p_der = ParameterRequirement(
        name="current",
        state=ParameterState.DERIVABLE,
        governing_relationship="I = V / R",
    )
    assert p_der.value is None


def test_ac_c5_input_vs_requested_artifacts_separation():
    """AC-C5: Input problem artifacts distinct from requested student artifacts."""
    spec = make_valid_subpart_spec()
    # Input artifact represents given problem stimulus
    assert len(spec.input_artifacts) == 1
    assert spec.input_artifacts[0].artifact_type == "System Parameters"
    assert spec.input_artifacts[0].modality == ArtifactModality.STRUCTURED_TABLE

    # Requested student artifacts are what student must produce
    assert "Free body diagram" in spec.expected_response.requested_artifacts
    assert isinstance(spec.expected_response.requested_artifacts, tuple)


def test_ac_c6_expected_response_contract():
    """AC-C6: Expected student response contract and time estimation."""
    # Valid contract
    resp = ExpectedResponseContract(
        response_type="Analytical Derivation",
        estimated_time_minutes=15,
        expected_milestones=("Declare pointer structure", "Implement traversal algorithm"),
    )
    assert resp.estimated_time_minutes == 15

    # Non-positive time must fail
    with pytest.raises(ValueError, match="estimated_time_minutes must be positive int"):
        ExpectedResponseContract(
            response_type="Analytical Derivation",
            estimated_time_minutes=0,
            expected_milestones=("Step 1",),
        )

    # Empty response_type must fail
    with pytest.raises(ValueError, match="response_type must be a non-empty string"):
        ExpectedResponseContract(
            response_type="",
            estimated_time_minutes=10,
            expected_milestones=("Step 1",),
        )


def test_ac_c7_marking_rubric_conservation():
    """AC-C7: Marking rubric allocation conservation invariant."""
    # Valid rubric
    rubric = MarkingRubricContract(
        criteria=(
            RubricCriterion("Setup", 3),
            RubricCriterion("Execution", 3),
        ),
    )
    assert rubric.total_marks == 6

    # Empty criteria must raise ValueError
    with pytest.raises(ValueError, match="criteria must be a non-empty tuple"):
        MarkingRubricContract(criteria=())

    # Non-positive criterion marks must raise ValueError
    with pytest.raises(ValueError, match="marks must be strictly positive int"):
        RubricCriterion("Setup", 0)


def test_ac_c8_mark_conservation_invariants():
    """AC-C8: Subpart, question, and slot mark conservation invariants."""
    # 1. SubPartSpecification marks must match rubric total marks
    rubric_4m = MarkingRubricContract(
        criteria=(RubricCriterion("Step 1", 4),),
    )
    with pytest.raises(ValueError, match=r"marking_rubric marks sum .* != subpart marks"):
        SubPartSpecification(
            label="a",
            marks=6,  # mismatch with rubric (4)
            bloom_level="L2",
            primary_archetype=UniversalArchetype.CONCEPTUAL_EXPLANATION,
            marking_rubric=rubric_4m,
        )

    # 2. QuestionSpecification marks must equal sum of subparts
    sp_a = make_valid_subpart_spec("a", 6)
    sp_b = make_valid_subpart_spec("b", 4)
    with pytest.raises(ValueError, match=r"subparts marks sum .* != question marks"):
        QuestionSpecification(
            question_id="q1",
            syllabus_topic="Fluid Dynamics",
            course_outcome="CO1",
            bloom_level="L3",
            marks=12,  # mismatch: 6 + 4 = 10 != 12
            subparts=(sp_a, sp_b),
        )

    # 3. AssessmentSlot marks must match alternative options' marks
    q_spec = make_valid_question_spec("q1", 10)
    alt_group = AlternativeGroup(group_id="grp1", alternatives=(q_spec,))
    with pytest.raises(ValueError, match=r"marks .* != slot marks"):
        AssessmentSlot(
            slot_id="Q1",
            marks=12,  # mismatch with alt_group (10)
            course_outcome="CO1",
            bloom_level="L3",
            groups=(alt_group,),
        )


def test_ac_c9_alternative_group_parity():
    """AC-C9: AlternativeGroup structure and internal parity."""
    q_opt_a = make_valid_question_spec("q1", 10)
    q_opt_b = make_valid_question_spec("q2", 10)

    # Equal marks -> valid
    group = AlternativeGroup(group_id="grp_mod1", alternatives=(q_opt_a, q_opt_b))
    assert len(group.alternatives) == 2

    # Mismatched option marks -> raises ValueError
    sp_c1 = make_valid_subpart_spec("a", 6)
    sp_c2 = make_valid_subpart_spec("b", 6)
    q_12m = QuestionSpecification(
        question_id="q3",
        syllabus_topic="Dynamics",
        course_outcome="CO1",
        bloom_level="L3",
        marks=12,
        subparts=(sp_c1, sp_c2),
    )
    with pytest.raises(ValueError, match="mark mismatch between alternatives"):
        AlternativeGroup(group_id="grp_invalid", alternatives=(q_opt_a, q_12m))

    # Empty alternatives -> raises ValueError
    with pytest.raises(ValueError, match="alternatives must not be empty"):
        AlternativeGroup(group_id="grp_empty", alternatives=())


def test_ac_c10_task_completeness_gate():
    """AC-C10: Task completeness assessment (answerability gate)."""
    # 1. Complete specification
    valid_spec = make_valid_subpart_spec()
    status, reason = evaluate_task_completeness(valid_spec)
    assert status == TaskCompleteness.COMPLETE
    assert reason is None

    # 2. Incomplete specification: WORKED_PROBLEM without input artifacts
    incomplete_spec = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L3",
        primary_archetype=UniversalArchetype.WORKED_PROBLEM_CALCULATION,
        input_artifacts=(),  # empty artifacts
    )
    status, reason = evaluate_task_completeness(incomplete_spec)
    assert status == TaskCompleteness.INCOMPLETE
    assert "requires input artifacts" in reason

    # 3. Indeterminate specification: SCHEMATIC_SYSTEM_SYNTHESIS without artifacts
    schematic_spec = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L4",
        primary_archetype=UniversalArchetype.SCHEMATIC_SYSTEM_SYNTHESIS,
        input_artifacts=(),
    )
    status, reason = evaluate_task_completeness(schematic_spec)
    assert status == TaskCompleteness.INDETERMINATE


def test_ac_c11_planning_agent_blueprint_generation():
    """AC-C11: PlanningAgent blueprint generation for 50M IA & 100M SEE."""
    agent = PlanningAgent()

    # 1. VTU 50M IA Blueprint
    bp_50m = agent.create_assessment_blueprint(scheme="VTU_50M_IA")
    assert isinstance(bp_50m, AssessmentBlueprint)
    assert bp_50m.total_marks == 50
    assert len(bp_50m.slots) == 5  # 5 modules
    for slot in bp_50m.slots:
        assert slot.marks == 10
        assert len(slot.groups) == 1
        grp = slot.groups[0]
        assert len(grp.alternatives) == 2  # Main + Alternative
        for opt in grp.alternatives:
            assert opt.marks == 10
            # Check valid partition summing to 10M
            assert sum(sp.marks for sp in opt.subparts) == 10
            assert len(opt.subparts) in (1, 2)
            assert all(sp.marks > 0 for sp in opt.subparts)

    # 2. VTU 100M SEE Blueprint
    bp_100m = agent.create_assessment_blueprint(scheme="VTU_100M_SEE")
    assert isinstance(bp_100m, AssessmentBlueprint)
    assert bp_100m.total_marks == 100
    assert len(bp_100m.slots) == 5
    for slot in bp_100m.slots:
        assert slot.marks == 20
        assert len(slot.groups) == 1
        grp = slot.groups[0]
        assert len(grp.alternatives) == 2
        for opt in grp.alternatives:
            assert opt.marks == 20
            # Sum of subparts equals 20M
            assert sum(sp.marks for sp in opt.subparts) == 20


def test_ac_c12_blueprint_projection_to_plans():
    """AC-C12: Blueprint-to-QuestionPlans projection & specification preservation."""
    agent = PlanningAgent()
    bp = agent.create_assessment_blueprint(scheme="VTU_50M_IA")
    plans = agent.project_blueprint_to_plans(bp)

    # 5 slots * 2 options = 10 projected plans
    assert len(plans) == 10

    for plan in plans:
        assert isinstance(plan, QuestionPlan)
        assert plan.specification is not None
        assert isinstance(plan.specification, QuestionSpecification)
        assert plan.constraints.get("alternative_group_id")
        assert plan.constraints.get("option_id")

        # Verify each SubPartPlan has specification attached
        assert len(plan.sub_part_plans) == len(plan.specification.subparts)
        for sp_plan, sp_spec in zip(plan.sub_part_plans, plan.specification.subparts):
            assert sp_plan.specification is sp_spec
            assert sp_plan.marks == sp_spec.marks
            assert sp_plan.label == sp_spec.label


def test_ac_c13_writing_agent_archetype_prompt_routing():
    """AC-C13: WritingAgent routes to archetype prompt when specification present."""
    mock_caller = MockLLMCaller()
    writing_agent = WritingAgent(llm_caller=mock_caller, enable_irt=False)

    # 1. Plan WITH specification
    agent = PlanningAgent()
    bp = agent.create_assessment_blueprint(
        scheme="VTU_50M_IA",
        modules=[{
            "module_id": "module_1",
            "topics": ["Calculate Stress and Deflection in Beams", "Compute Shear Force Distribution"],
        }],
    )
    plans = agent.project_blueprint_to_plans(bp)
    plan_with_spec = plans[0]

    ctx = AgentContext(plans=[plan_with_spec])
    result = writing_agent.run(ctx)

    assert result.success is True
    assert len(result.payload.questions) == 1
    assert len(result.payload.failures) == 0
    assert len(mock_caller.calls) == len(plan_with_spec.sub_part_plans)

    # Verify that the generated prompt contains task contract sections
    first_call_prompt = mock_caller.calls[0]["prompt"]
    assert "[ASSESSMENT TASK CONTRACT]" in first_call_prompt
    assert "Primary Archetype:" in first_call_prompt
    assert "[INPUT PROBLEM DATA CONTRACT]" in first_call_prompt
    assert "[EXPECTED STUDENT RESPONSE CONTRACT]" in first_call_prompt
    assert "[MARKING RUBRIC ALLOCATION]" in first_call_prompt

    # 2. Plan WITHOUT specification (legacy subpart plan)
    mock_caller_legacy = MockLLMCaller()
    writing_agent_legacy = WritingAgent(llm_caller=mock_caller_legacy, enable_irt=False)

    # Create a plan identical to plan_with_spec but without specification
    plans_legacy = agent.project_blueprint_to_plans(bp)
    plan_legacy = plans_legacy[1]
    plan_legacy.specification = None
    for sp in plan_legacy.sub_part_plans:
        sp.specification = None

    ctx_legacy = AgentContext(plans=[plan_legacy])
    result_legacy = writing_agent_legacy.run(ctx_legacy)

    assert result_legacy.success is True
    assert len(result_legacy.payload.questions) == 1
    assert len(result_legacy.payload.failures) == 0
    legacy_prompt = mock_caller_legacy.calls[0]["prompt"]
    # Legacy prompt must NOT contain task contract section
    assert "[ASSESSMENT TASK CONTRACT]" not in legacy_prompt
    assert "[SUBPART CONTRACT]" in legacy_prompt


def test_ac_c14_backward_compatibility():
    """AC-C14: Backward compatibility of QuestionPlan and SubPartPlan without specification."""
    # Instantiating SubPartPlan without specification defaults to None
    sp = SubPartPlan(
        label="a",
        marks=10,
        topic="Memory Hierarchy",
        bloom="L2",
        co="CO1",
    )
    assert sp.specification is None

    # Instantiating QuestionPlan without specification defaults to None
    qp = QuestionPlan(
        slot_id="Q1",
        module_id="mod1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        topic="Memory Hierarchy",
        bloom="L2",
        co="CO1",
        total_marks=10,
        sub_part_plans=[sp],
    )
    assert qp.specification is None
