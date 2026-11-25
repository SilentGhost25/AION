"""
Acceptance Test Suite for Stage G2-C:
Assessment Blueprint, Composable Question Archetypes & Task Contracts.

Covers Acceptance Criteria AC-C1 through AC-C14 and Corrective Closure Requirements:
- AC-C1: Universal Archetype Enumeration & Composition
- AC-C2: Cognitive Capability Enumeration
- AC-C3: Artifact Modality Enumeration
- AC-C4: Parameter Requirement Provenance & State Validation (GIVEN, STANDARD_REFERENCED, STUDENT_SELECTED, DERIVABLE)
- AC-C5: Input vs. Requested Student Artifact Separation
- AC-C6: Expected Student Response Contract & Time Estimation
- AC-C7: Marking Rubric Allocation & Mark Conservation
- AC-C8: SubPart and Question Mark Conservation Invariants
- AC-C9: Alternative Group Structure & Internal Parity
- AC-C10: Task Completeness Assessment (Answerability Gate: COMPLETE, INCOMPLETE, INDETERMINATE)
- AC-C11: PlanningAgent Blueprint Generation (50M IA & 100M SEE)
- AC-C12: Blueprint-to-QuestionPlans Projection & Specification Preservation
- AC-C13: WritingAgent Archetype Prompt Generation Integration
- AC-C14: Backward Compatibility & Zero-Regression (Positional, Keyword, Type Safety)
- AC-C15: Integrated End-to-End Pipeline (50M IA)
- AC-C16: Integrated End-to-End Pipeline (100M SEE)
"""

import re
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
    project_blueprint_to_plans,
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
        scheme_id="VTU_2022_NEP",
        course_type="ENGINEERING_THEORY",
        assessment_type="SEE",
        blueprint_id="VTU_SEE_100M",
        total_marks_per_question=20,
        permitted_partitions=((8, 6, 6), (10, 10), (20,)),
        max_subparts=3,
        reference_source="VTU 2022 Scheme SEE Regulations (2024), 20M per question",
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

        # Dynamically match requested Bloom level and mark allocation
        m_bloom = (
            re.search(r"Bloom level:\s*(L\d)", prompt)
            or re.search(r"Target Bloom Level:\s*(L\d)", prompt)
            or re.search(r"Target Bloom level \(RBT\):\s*(L\d)", prompt)
            or re.search(r"Bloom:\s*(L\d)", prompt)
        )
        bloom = m_bloom.group(1) if m_bloom else "L2"
        if bloom == "L1":
            verb = "Define"
            text = "Define the basic operational concepts and fundamental terminology for the system."
        elif bloom == "L2":
            verb = "Explain"
            text = "Explain the operating principles and mechanisms of the system architecture."
        elif bloom == "L3":
            verb = "Calculate"
            text = "Calculate the output values and execution parameters using the given input values."
        elif bloom == "L4":
            verb = "Analyze"
            text = "Analyze the differences and performance trade-offs between the two operational mechanisms."
        elif bloom == "L5":
            verb = "Evaluate"
            text = "Evaluate the design trade-offs against performance criteria under the specified architectural requirements."
        elif bloom == "L6":
            verb = "Design"
            text = "Design a custom architecture satisfying system specifications and timing constraints."
        else:
            verb = "Explain"
            text = "Explain the system parameters under the operational requirements."

        m_marks = re.search(r"Marks:\s*(\d+)", prompt) or re.search(r"Total Subpart Marks:\s*(\d+)", prompt)
        marks = int(m_marks.group(1)) if m_marks else 5
        m1 = max(1, marks // 2)
        m2 = marks - m1

        return {
            "text": text,
            "solution": "Solution steps and governing derivations.",
            "marking_scheme": [
                {"criterion": "Formulation and principle identification", "marks": m1},
                {"criterion": "Analytical execution and solution steps", "marks": m2},
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
# Unit Tests (AC-C1 through AC-C14)
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
    """AC-C4: Parameter requirement provenance & state validation invariants across all states."""
    # 1. GIVEN parameter with value and source reference succeeds
    p_valid = ParameterRequirement(
        name="voltage",
        state=ParameterState.GIVEN,
        value=230,
        unit="V",
        source_reference="Textbook Sec 3.2",
    )
    assert p_valid.value == 230
    assert p_valid.source_reference == "Textbook Sec 3.2"

    # 2. GIVEN parameter with value=None MUST raise ValueError
    with pytest.raises(ValueError, match="must have an explicit value, not None or empty"):
        ParameterRequirement(
            name="voltage",
            state=ParameterState.GIVEN,
            value=None,
            source_reference="Textbook Sec 3.2",
        )

    # 3. GIVEN parameter with empty string value MUST raise ValueError
    with pytest.raises(ValueError, match="must have an explicit value, not None or empty"):
        ParameterRequirement(
            name="voltage",
            state=ParameterState.GIVEN,
            value="   ",
            source_reference="Textbook Sec 3.2",
        )

    # 4. GIVEN parameter without source reference MUST raise ValueError
    with pytest.raises(ValueError, match="must declare a nonempty source_reference"):
        ParameterRequirement(
            name="voltage",
            state=ParameterState.GIVEN,
            value=230,
            source_reference="",
        )

    # 5. STANDARD_REFERENCED parameter with standard reference succeeds
    p_std = ParameterRequirement(
        name="permissible_shear_stress",
        state=ParameterState.STANDARD_REFERENCED,
        source_reference="IS 456:2000 Table 19 Clause 40.2.1",
    )
    assert p_std.state == ParameterState.STANDARD_REFERENCED
    assert "IS 456" in p_std.source_reference

    # 6. STANDARD_REFERENCED parameter without source reference MUST raise ValueError
    with pytest.raises(ValueError, match="must declare a standard code/reference in source_reference"):
        ParameterRequirement(
            name="permissible_shear_stress",
            state=ParameterState.STANDARD_REFERENCED,
            source_reference="",
        )

    # 7. STUDENT_SELECTED parameter without constraints MUST raise ValueError
    with pytest.raises(ValueError, match="must declare selection_constraints"):
        ParameterRequirement(
            name="resistor_choice",
            state=ParameterState.STUDENT_SELECTED,
            selection_constraints="",
        )

    # 8. STUDENT_SELECTED with constraints succeeds
    p_sel = ParameterRequirement(
        name="resistor_choice",
        state=ParameterState.STUDENT_SELECTED,
        selection_constraints="Choose standard E12 resistor value between 1k and 10k ohm",
    )
    assert p_sel.state == ParameterState.STUDENT_SELECTED

    # 9. DERIVABLE parameter without relationship or source MUST raise ValueError
    with pytest.raises(ValueError, match="must declare governing_relationship or source_reference"):
        ParameterRequirement(
            name="current",
            state=ParameterState.DERIVABLE,
        )

    # 10. DERIVABLE parameter with governing relationship succeeds with value=None
    p_der = ParameterRequirement(
        name="current",
        state=ParameterState.DERIVABLE,
        governing_relationship="I = V / R",
    )
    assert p_der.value is None


def test_ac_c5_input_vs_requested_artifacts_separation():
    """AC-C5: Input problem artifacts distinct from requested student artifacts."""
    spec = make_valid_subpart_spec()
    assert len(spec.input_artifacts) == 1
    assert spec.input_artifacts[0].artifact_type == "System Parameters"
    assert spec.input_artifacts[0].modality == ArtifactModality.STRUCTURED_TABLE

    assert "Free body diagram" in spec.expected_response.requested_artifacts
    assert isinstance(spec.expected_response.requested_artifacts, tuple)


def test_ac_c6_expected_response_contract():
    """AC-C6: Expected student response contract and time estimation."""
    resp = ExpectedResponseContract(
        response_type="Analytical Derivation",
        estimated_time_minutes=15,
        expected_milestones=("Declare pointer structure", "Implement traversal algorithm"),
    )
    assert resp.estimated_time_minutes == 15

    with pytest.raises(ValueError, match="estimated_time_minutes must be positive int"):
        ExpectedResponseContract(
            response_type="Analytical Derivation",
            estimated_time_minutes=0,
            expected_milestones=("Step 1",),
        )

    with pytest.raises(ValueError, match="response_type must be a non-empty string"):
        ExpectedResponseContract(
            response_type="",
            estimated_time_minutes=10,
            expected_milestones=("Step 1",),
        )


def test_ac_c7_marking_rubric_conservation():
    """AC-C7: Marking rubric allocation conservation invariant."""
    rubric = MarkingRubricContract(
        criteria=(
            RubricCriterion("Setup", 3),
            RubricCriterion("Execution", 3),
        ),
    )
    assert rubric.total_marks == 6

    with pytest.raises(ValueError, match="criteria must be a non-empty tuple"):
        MarkingRubricContract(criteria=())

    with pytest.raises(ValueError, match="marks must be strictly positive int"):
        RubricCriterion("Setup", 0)


def test_ac_c8_mark_conservation_invariants():
    """AC-C8: Subpart, question, and slot mark conservation invariants."""
    rubric_4m = MarkingRubricContract(
        criteria=(RubricCriterion("Step 1", 4),),
    )
    with pytest.raises(ValueError, match=r"marking_rubric marks sum .* != subpart marks"):
        SubPartSpecification(
            label="a",
            marks=6,
            bloom_level="L2",
            primary_archetype=UniversalArchetype.CONCEPTUAL_EXPLANATION,
            marking_rubric=rubric_4m,
        )

    sp_a = make_valid_subpart_spec("a", 6)
    sp_b = make_valid_subpart_spec("b", 4)
    with pytest.raises(ValueError, match=r"subparts marks sum .* != question marks"):
        QuestionSpecification(
            question_id="q1",
            syllabus_topic="Fluid Dynamics",
            course_outcome="CO1",
            bloom_level="L3",
            marks=12,
            subparts=(sp_a, sp_b),
        )

    q_spec = make_valid_question_spec("q1", 10)
    alt_group = AlternativeGroup(group_id="grp1", alternatives=(q_spec,))
    with pytest.raises(ValueError, match=r"marks .* != slot marks"):
        AssessmentSlot(
            slot_id="Q1",
            marks=12,
            course_outcome="CO1",
            bloom_level="L3",
            groups=(alt_group,),
        )


def test_ac_c9_alternative_group_parity():
    """AC-C9: AlternativeGroup structure and internal parity."""
    q_opt_a = make_valid_question_spec("q1", 10)
    q_opt_b = make_valid_question_spec("q2", 10)

    group = AlternativeGroup(group_id="grp_mod1", alternatives=(q_opt_a, q_opt_b))
    assert len(group.alternatives) == 2

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

    with pytest.raises(ValueError, match="alternatives must not be empty"):
        AlternativeGroup(group_id="grp_empty", alternatives=())


def test_ac_c10_task_completeness_gate():
    """AC-C10: Task completeness assessment (answerability gate: COMPLETE, INCOMPLETE, INDETERMINATE)."""
    # 1. Complete specification -> COMPLETE
    valid_spec = make_valid_subpart_spec()
    status, reason = evaluate_task_completeness(valid_spec)
    assert status == TaskCompleteness.COMPLETE
    assert reason is None

    # 2. Incomplete specification: WORKED_PROBLEM without input artifacts -> INCOMPLETE
    incomplete_spec = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L3",
        primary_archetype=UniversalArchetype.WORKED_PROBLEM_CALCULATION,
        input_artifacts=(),
    )
    status, reason = evaluate_task_completeness(incomplete_spec)
    assert status == TaskCompleteness.INCOMPLETE
    assert "requires input artifacts" in reason

    # 3. Incomplete specification: Rubric criteria missing -> INCOMPLETE
    spec_no_rubric = make_valid_subpart_spec()
    object.__setattr__(spec_no_rubric.marking_rubric, "criteria", ())
    status, reason = evaluate_task_completeness(spec_no_rubric)
    assert status == TaskCompleteness.INCOMPLETE
    assert "lacks marking rubric criteria" in reason

    # 4. Incomplete specification: GIVEN parameter missing source_reference -> INCOMPLETE
    p_given_no_src = ParameterRequirement(
        name="flow_rate",
        state=ParameterState.GIVEN,
        value=5.0,
        source_reference="Ref 1",
    )
    object.__setattr__(p_given_no_src, "source_reference", "")
    art_corrupt = InputArtifactRequirement(
        artifact_id="art_corrupt",
        artifact_type="System Data",
        modality=ArtifactModality.STRUCTURED_TABLE,
        parameters=(p_given_no_src,),
    )
    spec_corrupt_given = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L3",
        primary_archetype=UniversalArchetype.WORKED_PROBLEM_CALCULATION,
        input_artifacts=(art_corrupt,),
        marking_rubric=MarkingRubricContract(criteria=(RubricCriterion("Step", 6),)),
    )
    status, reason = evaluate_task_completeness(spec_corrupt_given)
    assert status == TaskCompleteness.INCOMPLETE
    assert "lacks source reference" in reason

    # 5. Indeterminate specification: None specification -> INDETERMINATE
    status, reason = evaluate_task_completeness(None)
    assert status == TaskCompleteness.INDETERMINATE
    assert "None" in reason

    # 6. Indeterminate specification: SCHEMATIC_SYSTEM_SYNTHESIS without artifacts -> INDETERMINATE
    schematic_spec = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L4",
        primary_archetype=UniversalArchetype.SCHEMATIC_SYSTEM_SYNTHESIS,
        input_artifacts=(),
    )
    status, reason = evaluate_task_completeness(schematic_spec)
    assert status == TaskCompleteness.INDETERMINATE

    # 7. Indeterminate specification: Unsupported/unknown archetype -> INDETERMINATE
    class FakeArchetype:
        value = "UNSUPPORTED_CUSTOM_ARCHETYPE"
    fake_spec = SubPartSpecification(
        label="a",
        marks=6,
        bloom_level="L2",
        primary_archetype=UniversalArchetype.CONCEPTUAL_EXPLANATION,
    )
    object.__setattr__(fake_spec, "primary_archetype", FakeArchetype())
    status, reason = evaluate_task_completeness(fake_spec)
    assert status == TaskCompleteness.INDETERMINATE
    assert "Available evidence cannot establish completeness" in reason


def test_ac_c11_planning_agent_blueprint_generation():
    """AC-C11: PlanningAgent blueprint generation for 50M IA & 100M SEE."""
    agent = PlanningAgent()

    # 1. VTU 50M IA Blueprint
    bp_50m = agent.create_assessment_blueprint(scheme="VTU_50M_IA")
    assert isinstance(bp_50m, AssessmentBlueprint)
    assert bp_50m.total_marks == 50
    assert len(bp_50m.slots) == 5
    for slot in bp_50m.slots:
        assert slot.marks == 10
        assert len(slot.groups) == 1
        grp = slot.groups[0]
        assert len(grp.alternatives) == 2
        for opt in grp.alternatives:
            assert opt.marks == 10
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
            assert sum(sp.marks for sp in opt.subparts) == 20


def test_ac_c12_blueprint_projection_to_plans():
    """AC-C12: Blueprint-to-QuestionPlans projection & specification preservation."""
    agent = PlanningAgent()
    bp = agent.create_assessment_blueprint(scheme="VTU_50M_IA")
    plans = agent.project_blueprint_to_plans(bp)

    assert len(plans) == 10

    for plan in plans:
        assert isinstance(plan, QuestionPlan)
        assert plan.specification is not None
        assert isinstance(plan.specification, QuestionSpecification)
        assert plan.constraints.get("alternative_group_id")
        assert plan.constraints.get("option_id")

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

    first_call_prompt = mock_caller.calls[0]["prompt"]
    assert "[ASSESSMENT TASK CONTRACT]" in first_call_prompt
    assert "Primary Archetype:" in first_call_prompt
    assert "[INPUT PROBLEM DATA CONTRACT]" in first_call_prompt
    assert "[EXPECTED STUDENT RESPONSE CONTRACT]" in first_call_prompt
    assert "[MARKING RUBRIC ALLOCATION]" in first_call_prompt

    # 2. Plan WITHOUT specification (legacy subpart plan)
    mock_caller_legacy = MockLLMCaller()
    writing_agent_legacy = WritingAgent(llm_caller=mock_caller_legacy, enable_irt=False)

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
    assert "[ASSESSMENT TASK CONTRACT]" not in legacy_prompt
    assert "[SUBPART CONTRACT]" in legacy_prompt


def test_ac_c14_backward_compatibility():
    """AC-C14: Positional and keyword constructor compatibility and type validation."""
    # 1. Positional construction of SubPartPlan without specification
    sp_pos = SubPartPlan(
        "a",
        5,
        "Memory Hierarchy",
        "L2",
        "CO1",
    )
    assert sp_pos.label == "a"
    assert sp_pos.marks == 5
    assert sp_pos.specification is None

    # 2. Keyword construction of SubPartPlan without specification
    sp_kw = SubPartPlan(
        label="b",
        marks=5,
        topic="Cache Coherence",
        bloom="L3",
        co="CO2",
    )
    assert sp_kw.label == "b"
    assert sp_kw.specification is None

    # 3. Positional construction of QuestionPlan without specification
    qp_pos = QuestionPlan(
        "Q1",
        "module_1",
        1,
        0,
        1,
    )
    assert qp_pos.slot_id == "Q1"
    assert qp_pos.specification is None

    # 4. Keyword construction of QuestionPlan without specification
    qp_kw = QuestionPlan(
        slot_id="Q2",
        module_id="module_1",
        module_idx=1,
        slot_idx=1,
        global_q_idx=2,
        topic="Virtual Memory",
        bloom="L2",
        co="CO1",
        total_marks=10,
        sub_part_plans=[sp_pos, sp_kw],
    )
    assert qp_kw.slot_id == "Q2"
    assert qp_kw.specification is None

    # 5. Construction with valid typed specification contracts
    sp_spec = make_valid_subpart_spec("a", 5)
    sp_with_spec = SubPartPlan(
        label="a",
        marks=5,
        topic="Stress Calculation",
        specification=sp_spec,
    )
    assert sp_with_spec.specification is sp_spec

    q_spec = make_valid_question_spec("q1", 10)
    qp_with_spec = QuestionPlan(
        slot_id="Q1",
        module_id="module_1",
        module_idx=1,
        slot_idx=0,
        global_q_idx=1,
        specification=q_spec,
    )
    assert qp_with_spec.specification is q_spec

    # 6. Type safety rejection: invalid specification types MUST raise TypeError
    with pytest.raises(TypeError, match="SubPartPlan specification must be SubPartSpecification or None"):
        SubPartPlan(
            label="a",
            marks=5,
            topic="Test",
            specification="invalid_string_specification",
        )

    with pytest.raises(TypeError, match="QuestionPlan specification must be QuestionSpecification or None"):
        QuestionPlan(
            slot_id="Q1",
            module_id="module_1",
            module_idx=1,
            slot_idx=0,
            global_q_idx=1,
            specification={"not": "a_question_spec"},
        )


# =============================================================================
# Integration Tests (AC-C15 & AC-C16: Real Registry-to-Writing Execution)
# =============================================================================


def test_mandatory_integrated_flow_50m_ia():
    """
    AC-C15: Integrated Pipeline for 50M IA:
    AssessmentRegistry -> PlanningAgent -> Blueprint -> Projection -> WritingAgent -> SubParts -> Validation
    """
    # 1. Real AssessmentRegistry resolution
    config = AssessmentRegistry.get(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
    )
    assert config.total_marks_per_question == 10

    # 2. PlanningAgent blueprint synthesis (5 modules x 10M = 50M)
    agent = PlanningAgent()
    bp = agent.create_assessment_blueprint(
        scheme_id="VTU_2022_NEP",
        course_type="AEC_GENERAL_THEORY",
        assessment_type="CIE_IAT",
        blueprint_id="VTU_AEC_ANNEXURE_II",
        course_code="22ETC15",
        title="Emerging Technology CIE-1",
        modules=[
            {"module_id": f"module_{i}", "topics": [f"Calculate Stress in Beam {i}", f"Algorithm for Scheduling {i}"]}
            for i in range(1, 6)
        ],
    )
    assert bp.total_marks == 50
    assert len(bp.slots) == 5

    # 3. Projection into QuestionPlan instances with preserved specifications
    plans = project_blueprint_to_plans(bp)
    assert len(plans) == 10  # 5 slots * 2 options (Main & Alternative)
    for p in plans:
        assert p.specification is not None
        assert isinstance(p.specification, QuestionSpecification)
        for sp in p.sub_part_plans:
            assert sp.specification is not None
            assert isinstance(sp.specification, SubPartSpecification)

    # 4. WritingAgent generation over all projected plans
    mock_caller = MockLLMCaller()
    writing_agent = WritingAgent(llm_caller=mock_caller, enable_irt=False)
    ctx = AgentContext(plans=plans)
    result = writing_agent.run(ctx)

    assert result.success is True
    assert len(result.payload.questions) == 10
    assert len(result.payload.failures) == 0

    # 5. Verification of generated SubParts and existing validation path
    for gq in result.payload.questions:
        valid_q, err_q = gq.validate_structure()
        assert valid_q, f"Generated question failed validation: {err_q}"
        assert len(gq.sub_parts) >= 1

        for sp in gq.sub_parts:
            valid_sp, err_sp = sp.validate()
            assert valid_sp, f"Generated subpart ({sp.label}) failed validation: {err_sp}"
            # Subpart retains its specification contract
            assert hasattr(sp, "specification")
            assert isinstance(sp.specification, SubPartSpecification)
            assert sp.specification.label == sp.label
            assert sp.specification.marks == sp.marks


def test_mandatory_integrated_flow_100m_see():
    """
    AC-C16: Integrated Pipeline for 100M SEE:
    AssessmentRegistry -> PlanningAgent -> Blueprint -> Projection -> WritingAgent -> SubParts -> Validation
    """
    # 1. Real AssessmentRegistry resolution for SEE
    config = AssessmentRegistry.get(
        scheme_id="VTU_2022_NEP",
        course_type="ENGINEERING_THEORY",
        assessment_type="SEE",
        blueprint_id="VTU_SEE_100M",
    )
    assert config.total_marks_per_question == 20

    # 2. PlanningAgent blueprint synthesis (5 modules x 20M = 100M)
    agent = PlanningAgent()
    bp = agent.create_assessment_blueprint(
        scheme_id="VTU_2022_NEP",
        course_type="ENGINEERING_THEORY",
        assessment_type="SEE",
        blueprint_id="VTU_SEE_100M",
        course_code="22CS51",
        title="Semester End Examination (SEE)",
        modules=[
            {"module_id": f"module_{i}", "topics": [f"Design Circuit Architecture {i}", f"Compute Operating Characteristics {i}"]}
            for i in range(1, 6)
        ],
    )
    assert bp.total_marks == 100
    assert len(bp.slots) == 5

    # 3. Projection into QuestionPlans
    plans = project_blueprint_to_plans(bp)
    assert len(plans) == 10
    for p in plans:
        assert p.total_marks == 20
        assert p.specification is not None
        assert isinstance(p.specification, QuestionSpecification)

    # 4. WritingAgent generation
    mock_caller = MockLLMCaller()
    writing_agent = WritingAgent(llm_caller=mock_caller, enable_irt=False)
    ctx = AgentContext(plans=plans)
    result = writing_agent.run(ctx)

    assert result.success is True
    assert len(result.payload.questions) == 10
    assert len(result.payload.failures) == 0

    # 5. Validation path verification
    for gq in result.payload.questions:
        valid_q, err_q = gq.validate_structure()
        assert valid_q, f"SEE Question failed validation: {err_q}"
        assert gq.marks == 20
        assert len(gq.sub_parts) >= 1
        for sp in gq.sub_parts:
            valid_sp, err_sp = sp.validate()
            assert valid_sp, f"SEE SubPart ({sp.label}) failed validation: {err_sp}"
            assert hasattr(sp, "specification")
            assert isinstance(sp.specification, SubPartSpecification)
            assert sp.specification.label == sp.label
            assert sp.specification.marks == sp.marks
