# AION-Trainer/tests/test_ese.py
import pytest
import shutil
from pathlib import Path
from fastapi.testclient import TestClient

from acb.concept import Concept, ModuleLink, ConceptStore
from ese.answer_blueprint import AnswerBlueprint, AnswerBlueprintBuilder
from ese.exam_blueprint import ExamBlueprint, ExamBlueprintBuilder, QuestionSlot
from ese.question_planner import QuestionPlanner
from ese.question_discoverer import QuestionDiscoverer, QuestionCandidate
from ese.question_ranker import QuestionRanker, RankingScore
from ese.language_realizer import LanguageRealizer
from ese.grammar_validator import GrammarValidator, GrammarIssue
from ese.vtu_validator import VTUValidator
from ese.chief_examiner import ChiefExaminer, ChiefExaminerReport
from ese.examiner_simulation_engine import ExaminerSimulationEngine
from ese.question_metadata import QuestionMetadata
from server.api import app


class DummyLLM:
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 100) -> str:
        if "CANDIDATE QUESTIONS" in prompt:
            return "1. Trace the working of DFS for the given graph.\n2. Trace DFS step-by-step."
        return "Trace the working of DFS for the given graph."


@pytest.fixture
def mock_concept_store():
    # Create temp store
    store_file = Path("scratch/test_ese_concepts.json")
    if store_file.exists():
        store_file.unlink()
        
    store = ConceptStore(str(store_file))
    
    # Add dummy concepts for BAI404
    c1 = Concept(
        concept_id="c_dfs",
        name="DFS",
        definition="Depth first search is a graph traversal algorithm using a stack.",
        explanation="DFS visits nodes as deep as possible before backtracking.",
        key_points=["Uses LIFO stack structure", "Time complexity is O(V+E)"],
        algorithms=["Push start node to stack", "Pop and visit", "Push unvisited neighbors"],
        applications=["Cycle detection", "Topological sorting"],
        formulas=[],
        requires_diagram=True,
        diagram_description="Stack structure and tree graph illustration",
        importance=0.9,
        previous_paper_frequency=5,
        syllabus_mentions=1,
    )
    c1.module_links.append(ModuleLink(subject_code="BAI404", module=1, is_primary=True))
    
    c2 = Concept(
        concept_id="c_bfs",
        name="BFS",
        definition="Breadth first search is a graph traversal algorithm using a queue.",
        explanation="BFS visits nodes layer by layer using a FIFO queue.",
        key_points=["Uses FIFO queue structure", "Finds shortest path in unweighted graphs"],
        algorithms=["Enqueue start node", "Dequeue and visit", "Enqueue unvisited neighbors"],
        applications=["Shortest path detection", "Web crawling"],
        formulas=[],
        requires_diagram=False,
        importance=0.8,
        previous_paper_frequency=3,
        syllabus_mentions=1,
    )
    c2.module_links.append(ModuleLink(subject_code="BAI404", module=1, is_primary=True))

    c3 = Concept(
        concept_id="c_astar",
        name="A* Search",
        definition="A* is an informed search algorithm using heuristics.",
        explanation="A* minimizes f(n) = g(n) + h(n).",
        key_points=["Uses admissible heuristics", "Guarantees optimal path if heuristic is consistent"],
        algorithms=["Maintain open and closed lists", "Choose node with lowest f(n)", "Expand"],
        applications=["Map navigation", "Pathfinding in games"],
        formulas=["f(n) = g(n) + h(n)"],
        requires_diagram=True,
        diagram_description="Open list and closed list tracing",
        importance=0.95,
        previous_paper_frequency=12,
        syllabus_mentions=2,
    )
    c3.module_links.append(ModuleLink(subject_code="BAI404", module=2, is_primary=True))

    c4 = Concept(
        concept_id="c_minimax",
        name="Minimax",
        definition="Minimax is a decision rule used in two-player games.",
        explanation="It minimizes the possible loss for a maximum loss scenario.",
        key_points=["Backtracks values from leaf nodes", "Assumes optimal play from opponent"],
        algorithms=["Traverse game tree", "Maximize at agent turn", "Minimize at opponent turn"],
        applications=["Chess", "Tic-Tac-Toe"],
        formulas=[],
        requires_diagram=True,
        diagram_description="Game search tree with minimax levels",
        importance=0.75,
        previous_paper_frequency=2,
        syllabus_mentions=1,
    )
    c4.module_links.append(ModuleLink(subject_code="BAI404", module=2, is_primary=True))

    c5 = Concept(
        concept_id="c_alpha_beta",
        name="Alpha-Beta Pruning",
        definition="Alpha-beta pruning is an optimization technique for minimax.",
        explanation="It prunes branches that cannot influence the final decision.",
        key_points=["Maintains alpha and beta thresholds", "Prunes when alpha >= beta"],
        algorithms=["Calculate minimax with alpha-beta cuts", "Prune search tree branches"],
        applications=["Game playing optimizations"],
        formulas=["alpha >= beta"],
        requires_diagram=True,
        diagram_description="Pruned branches shown with cuts",
        importance=0.85,
        previous_paper_frequency=8,
        syllabus_mentions=1,
    )
    c5.module_links.append(ModuleLink(subject_code="BAI404", module=3, is_primary=True))

    c6 = Concept(
        concept_id="c_csp",
        name="Constraint Satisfaction Problems",
        definition="CSP are mathematical problems defined by constraints.",
        explanation="Variables must be assigned values satisfying all constraints.",
        key_points=["Consists of Variables, Domains, Constraints", "Backtracking search with MRV heuristic"],
        algorithms=["Backtracking search", "Forward checking", "Arc consistency AC-3"],
        applications=["Sudoku solving", "Map coloring"],
        formulas=[],
        requires_diagram=False,
        importance=0.70,
        previous_paper_frequency=4,
        syllabus_mentions=1,
    )
    c6.module_links.append(ModuleLink(subject_code="BAI404", module=3, is_primary=True))

    c7 = Concept(
        concept_id="c_expert_systems",
        name="Expert Systems",
        definition="Expert systems mimic human decision-making processes.",
        explanation="Consists of an inference engine and rule database.",
        key_points=["Forward chaining starts with facts", "Backward chaining starts with goals"],
        algorithms=["Inference loop matching rules", "Forward chaining algorithm", "Backward chaining algorithm"],
        applications=["Medical diagnosis", "Financial credit scoring"],
        formulas=[],
        requires_diagram=True,
        diagram_description="Knowledge base, working memory and inference engine block diagram",
        importance=0.65,
        previous_paper_frequency=1,
        syllabus_mentions=1,
    )
    c7.module_links.append(ModuleLink(subject_code="BAI404", module=4, is_primary=True))

    c8 = Concept(
        concept_id="c_nlp",
        name="Natural Language Processing",
        definition="NLP processes natural human language.",
        explanation="Covers parsing, syntax analysis, and semantics.",
        key_points=["Tokenisation and POS tagging", "Syntactic parsing using context-free grammars"],
        applications=["Machine translation", "Sentiment analysis"],
        formulas=[],
        requires_diagram=False,
        importance=0.80,
        previous_paper_frequency=6,
        syllabus_mentions=1,
    )
    c8.module_links.append(ModuleLink(subject_code="BAI404", module=4, is_primary=True))

    c9 = Concept(
        concept_id="c_ann",
        name="Artificial Neural Networks",
        definition="ANN are models inspired by biological neural structures.",
        explanation="Consists of input, hidden, and output layers.",
        key_points=["Backpropagation adjusts connection weights", "Activation functions introduce non-linearity"],
        algorithms=["Feedforward pass", "Calculate error loss", "Backpropagate errors via gradient descent"],
        applications=["Image recognition", "Time-series forecasting"],
        formulas=["y = f(w.x + b)"],
        requires_diagram=True,
        diagram_description="Multilayer perceptron architecture diagram",
        importance=0.95,
        previous_paper_frequency=15,
        syllabus_mentions=2,
    )
    c9.module_links.append(ModuleLink(subject_code="BAI404", module=5, is_primary=True))

    c10 = Concept(
        concept_id="c_clustering",
        name="Clustering",
        definition="Clustering groups similar data objects together.",
        explanation="Unsupervised learning method.",
        key_points=["K-means algorithm partitions into K groups", "Hierarchical clustering builds trees"],
        algorithms=["K-means centroid assignment", "Centroid update loop"],
        applications=["Customer segmentation", "Anomaly detection"],
        formulas=[],
        requires_diagram=False,
        importance=0.75,
        previous_paper_frequency=3,
        syllabus_mentions=1,
    )
    c10.module_links.append(ModuleLink(subject_code="BAI404", module=5, is_primary=True))

    for c in [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10]:
        store.add(c)
        
    store.save()
    yield store
    
    if store_file.exists():
        store_file.unlink()


def test_answer_blueprint_builder(mock_concept_store):
    concept = mock_concept_store.get("c_dfs")
    builder = AnswerBlueprintBuilder()
    
    # Fake intent
    from server.prompt.assessment_intent import AssessmentIntent
    intent = AssessmentIntent(
        topic=concept.name,
        bloom_level="L3",
        marks=10,
        question_type="algorithm",
        difficulty="medium",
    )
    
    bp = builder.build(concept, intent)
    assert bp.concept_id == "c_dfs"
    assert bp.bloom_level == "L3"
    assert bp.marks == 10
    assert bp.diagram_required is True
    assert bp.blueprint_confidence > 0.5
    
    # Check components distribution
    marks_dist = bp.marks_per_component()
    assert len(marks_dist) > 0
    assert sum(marks_dist.values()) > 0


def test_exam_blueprint_builder(mock_concept_store):
    builder = ExamBlueprintBuilder()
    blueprint = builder.build(
        subject_code="BAI404",
        subject_name="Introduction to AI",
        semester=4,
        concept_store=mock_concept_store,
        include_optional=True,
    )
    
    assert blueprint.subject_code == "BAI404"
    assert len(blueprint.slots) == 20  # 5 modules x 4 slots (2 required + 2 optional each)
    assert blueprint.coverage_score > 0.0
    assert blueprint.diversity_score > 0.0
    
    # Distribution check
    assert blueprint.bloom_distribution["L3"] > 0
    assert blueprint.module_distribution[1] == 4


def test_question_planner(mock_concept_store):
    planner = QuestionPlanner(mock_concept_store)
    
    slot = QuestionSlot(
        slot_id="1a",
        module=1,
        concept_id="c_dfs",
        concept_name="DFS",
        bloom_level="L3",
        marks=10,
        difficulty="medium",
        question_type="algorithm",
    )
    
    output = planner.plan_slot(slot)
    assert output is not None
    assert output.intent.action_verb in ("Illustrate", "Trace")
    assert output.blueprint.marks == 10


def test_question_discoverer_and_ranker(mock_concept_store):
    llm = DummyLLM()
    discoverer = QuestionDiscoverer(llm)
    ranker = QuestionRanker()
    
    concept = mock_concept_store.get("c_dfs")
    from server.prompt.assessment_intent import AssessmentIntent
    intent = AssessmentIntent(
        topic=concept.name,
        bloom_level="L3",
        action_verb="Illustrate",
        marks=10,
        question_type="algorithm",
        difficulty="medium",
    )
    
    bp = AnswerBlueprintBuilder().build(concept, intent)
    
    candidates = discoverer.discover(bp, intent)
    assert len(candidates) > 0
    
    scores = ranker.rank(candidates, bp, intent)
    assert len(scores) == len(candidates)
    
    best = ranker.best(candidates, bp, intent)
    assert best is not None
    assert isinstance(best[0], QuestionCandidate)


def test_validators():
    gv = GrammarValidator()
    vv = VTUValidator()
    
    # Casual/bad grammar
    issues_grammar = gv.validate("hey solve this dfs please gonna do it")
    assert any(i.rule_name == "slang_detected" for i in issues_grammar)
    
    # Correct sentence
    issues_ok = gv.validate("Explain Depth First Search (DFS) with an illustration.")
    assert len([i for i in issues_ok if i.severity == "error"]) == 0

    # VTU cognitive verbs check
    issues_vtu = vv.validate("Define Depth First Search", "L3", 10, False)
    assert any(i.rule_name == "bloom_verb_misalignment" for i in issues_vtu)


def test_chief_examiner():
    ce = ChiefExaminer()
    
    # Create empty blueprint
    bp = ExamBlueprint(blueprint_id="test", subject_code="BAI404")
    report = ce.evaluate_paper(bp)
    assert report.passed is False
    assert any(f.rule == "Empty Paper" for f in report.flags)


def test_engine_end_to_end(mock_concept_store):
    engine = ExaminerSimulationEngine(mock_concept_store, llm_client=DummyLLM())
    
    blueprint, report, metadata = engine.generate_paper(
        subject_code="BAI404",
        subject_name="Introduction to AI",
        semester=4,
    )
    
    assert blueprint.subject_code == "BAI404"
    assert len(blueprint.slots) > 0
    assert len(metadata) == len(blueprint.slots)
    
    # All slots filled
    for slot in blueprint.slots:
        assert slot.filled is True
        assert slot.question_text != ""


def test_ese_api_endpoints(mock_concept_store):
    client = TestClient(app)
    headers = {"x-aion-token": "test-token-0000"}

    # Mock ACBPipeline to return our fixture concept store
    # Since acb_api/ese_api initialize ACBPipeline directly, we can override its concept store
    # by writing the JSON file to scratch/test_academic/AIML/semester_4/BAI404/concepts.json
    db_file_1 = Path("scratch/test_academic/AIML/semester_4/BAI404/concepts.json")
    db_file_2 = Path("scratch/test_academic/AIML/semester_4/BAI404/db/concepts.json")
    db_file_1.parent.mkdir(parents=True, exist_ok=True)
    db_file_2.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy mock concepts to these target locations
    shutil.copy("scratch/test_ese_concepts.json", db_file_1)
    shutil.copy("scratch/test_ese_concepts.json", db_file_2)
    
    # Also write dummy sources files to pass SourceRegistry loading
    db_file_1.parent.joinpath("sources.json").write_text("{}", encoding="utf-8")
    db_file_2.parent.joinpath("sources.json").write_text("{}", encoding="utf-8")

    # 1. POST /ese/generate
    payload = {
        "subject_code": "BAI404",
        "subject_name": "Introduction to AI",
        "academic_root": "scratch/test_academic",
        "semester": 4,
        "department": "AIML",
        "include_optional": True,
    }
    
    res = client.post("/ese/generate", json=payload, headers=headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    blueprint_id = data["blueprint"]["blueprint_id"]
    
    # 2. GET /ese/paper/{subject_code}/{blueprint_id}
    res_get = client.get(
        f"/ese/paper/BAI404/{blueprint_id}?academic_root=scratch/test_academic",
        headers=headers
    )
    assert res_get.status_code == 200
    paper_data = res_get.json()
    assert paper_data["blueprint"]["blueprint_id"] == blueprint_id

    # 3. POST /ese/paper/{subject_code}/{blueprint_id}/override
    override_payload = {
        "academic_root": "scratch/test_academic",
        "slot_id": "1a",
        "overridden_text": "Explain Breadth First Search (BFS) with a neat graph illustration.",
    }
    res_override = client.post(
        f"/ese/paper/BAI404/{blueprint_id}/override",
        json=override_payload,
        headers=headers
    )
    assert res_override.status_code == 200
    assert res_override.json()["blueprint"]["slots"][0]["question_text"] == override_payload["overridden_text"]

    # 4. POST /ese/paper/{subject_code}/{blueprint_id}/regenerate-slot
    regen_payload = {
        "academic_root": "scratch/test_academic",
        "slot_id": "1b",
    }
    res_regen = client.post(
        f"/ese/paper/BAI404/{blueprint_id}/regenerate-slot",
        json=regen_payload,
        headers=headers
    )
    assert res_regen.status_code == 200
    assert res_regen.json()["blueprint"]["slots"][1]["question_text"] != ""

    # 5. POST /ese/paper/{subject_code}/{blueprint_id}/promote
    res_promote = client.post(
        f"/ese/paper/BAI404/{blueprint_id}/promote?academic_root=scratch/test_academic",
        headers=headers
    )
    assert res_promote.status_code == 200
    assert res_promote.json()["is_promoted"] is True
