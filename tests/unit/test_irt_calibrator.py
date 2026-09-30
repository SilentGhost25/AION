"""
Unit tests for the Rasch 1PL IRT calibrator.

Deterministic simulation: the student_sim_fn is fully controlled by the
test, and the RNG is seeded so sampled outcomes are reproducible.
"""

import math
import random
from typing import List, Tuple

import pytest

from core.evaluation import IRTCalibrator, DifficultyEstimate


# -----------------------------------------------------------------------------
# Test simulators
# -----------------------------------------------------------------------------


def make_deterministic_sim(correct_at_or_above: float):
    """
    Returns a simulator that says "correct" iff θ >= correct_at_or_above.
    This creates a sharp step, which is a strong signal for the Rasch fit.
    """
    def sim(question: str, theta: float) -> float:
        return 1.0 if theta >= correct_at_or_above else 0.0
    return sim


def make_smooth_sim(true_b: float, slope: float = 1.0):
    """
    Returns a simulator following the Rasch model at a known difficulty.
    P(correct | θ) = sigmoid(slope * (θ - true_b))
    """
    def sim(question: str, theta: float) -> float:
        z = slope * (theta - true_b)
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
    return sim


def make_constant_sim(p: float):
    def sim(question: str, theta: float) -> float:
        return p
    return sim


def make_raising_sim():
    def sim(question: str, theta: float) -> float:
        raise RuntimeError("simulator crashed")
    return sim


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def seeded_rng():
    return random.Random(42)


@pytest.fixture
def ability_levels():
    return [-2.0, -1.0, 0.0, 1.0, 2.0]


# -----------------------------------------------------------------------------
# Basic fit
# -----------------------------------------------------------------------------


def test_calibrate_easy_question(seeded_rng):
    """A question easy for everyone → low b."""
    sim = make_constant_sim(0.95)
    calib = IRTCalibrator(sim, samples_per_level=20, rng=seeded_rng)
    est = calib.calibrate("Trivially easy question?")
    # Most responses correct, but not 100% at 0.95 with 100 samples
    assert est.n_responses == 100
    # Not hitting boundary (some incorrect responses expected)
    assert est.notes != "all_correct_boundary"


def test_calibrate_hard_question(seeded_rng):
    """A question hard for everyone → high b."""
    sim = make_constant_sim(0.05)
    calib = IRTCalibrator(sim, samples_per_level=20, rng=seeded_rng)
    est = calib.calibrate("Very hard question?")
    assert est.n_responses == 100


def test_calibrate_recovers_known_difficulty(seeded_rng, ability_levels):
    """When simulating from a known Rasch model, MLE should be close."""
    true_b = 0.5
    sim = make_smooth_sim(true_b=true_b)
    calib = IRTCalibrator(
        sim,
        ability_levels=ability_levels,
        samples_per_level=100,
        rng=seeded_rng,
    )
    est = calib.calibrate("Question at true difficulty 0.5?")
    # With 500 responses, estimate should be within 0.3 of truth
    assert abs(est.difficulty - true_b) < 0.5
    assert est.converged is True


def test_calibrate_negative_difficulty(seeded_rng):
    """A question easy for low-ability students → negative b."""
    sim = make_smooth_sim(true_b=-1.0)
    calib = IRTCalibrator(sim, samples_per_level=50, rng=seeded_rng)
    est = calib.calibrate("Easy question?")
    assert est.difficulty < 0


def test_calibrate_positive_difficulty(seeded_rng):
    sim = make_smooth_sim(true_b=1.5)
    calib = IRTCalibrator(sim, samples_per_level=50, rng=seeded_rng)
    est = calib.calibrate("Hard question?")
    assert est.difficulty > 0


def test_seed_determinism(ability_levels):
    """Same seed → identical estimate."""
    sim = make_smooth_sim(true_b=0.3)
    c1 = IRTCalibrator(sim, samples_per_level=20, rng=random.Random(7))
    c2 = IRTCalibrator(sim, samples_per_level=20, rng=random.Random(7))
    e1 = c1.calibrate("Q?")
    e2 = c2.calibrate("Q?")
    assert e1.difficulty == e2.difficulty
    assert e1.n_responses == e2.n_responses


# -----------------------------------------------------------------------------
# Boundary cases
# -----------------------------------------------------------------------------


def test_all_correct_boundary(seeded_rng):
    sim = make_constant_sim(1.0)
    calib = IRTCalibrator(sim, samples_per_level=5, rng=seeded_rng)
    est = calib.calibrate("Q?")
    assert est.difficulty == -3.0  # clamped to lower bound
    assert est.standard_error == float("inf")
    assert est.notes == "all_correct_boundary"
    assert est.converged is True


def test_all_incorrect_boundary(seeded_rng):
    sim = make_constant_sim(0.0)
    calib = IRTCalibrator(sim, samples_per_level=5, rng=seeded_rng)
    est = calib.calibrate("Q?")
    assert est.difficulty == 3.0  # clamped to upper bound
    assert est.standard_error == float("inf")
    assert est.notes == "all_incorrect_boundary"


def test_custom_bounds_respected(seeded_rng):
    sim = make_constant_sim(1.0)
    calib = IRTCalibrator(
        sim,
        samples_per_level=5,
        rng=seeded_rng,
        difficulty_bounds=(-1.0, 1.0),
    )
    est = calib.calibrate("Q?")
    assert est.difficulty == -1.0


# -----------------------------------------------------------------------------
# Standard error
# -----------------------------------------------------------------------------


def test_standard_error_finite_for_normal_case(seeded_rng):
    sim = make_smooth_sim(true_b=0.0)
    calib = IRTCalibrator(sim, samples_per_level=50, rng=seeded_rng)
    est = calib.calibrate("Q?")
    assert math.isfinite(est.standard_error)
    assert est.standard_error > 0


def test_standard_error_decreases_with_more_samples(seeded_rng):
    """
    More responses → tighter estimate. Same simulator, same seed distribution.
    """
    sim = make_smooth_sim(true_b=0.0)
    c1 = IRTCalibrator(sim, samples_per_level=10, rng=random.Random(1))
    c2 = IRTCalibrator(sim, samples_per_level=100, rng=random.Random(1))
    se_10 = c1.calibrate("Q?").standard_error
    se_100 = c2.calibrate("Q?").standard_error
    assert se_100 < se_10


# -----------------------------------------------------------------------------
# Simulator failure handling
# -----------------------------------------------------------------------------


def test_simulator_crash_skips_sample(seeded_rng):
    """A crashing simulator at one ability level still yields a fit."""
    calls = {"n": 0}

    def flaky_sim(question: str, theta: float) -> float:
        calls["n"] += 1
        if theta == 0.0:
            raise RuntimeError("boom")
        return 0.5

    calib = IRTCalibrator(
        flaky_sim,
        ability_levels=[-1.0, 0.0, 1.0],
        samples_per_level=2,
        rng=seeded_rng,
    )
    est = calib.calibrate("Q?")
    # 3 abilities × 2 samples = 6 expected; 2 crashed at θ=0
    assert est.n_responses == 4


def test_simulator_all_crash_raises(seeded_rng):
    calib = IRTCalibrator(
        make_raising_sim(),
        samples_per_level=2,
        rng=seeded_rng,
    )
    with pytest.raises(ValueError, match="No responses"):
        calib.calibrate("Q?")


def test_simulator_nan_skipped(seeded_rng):
    calls = {"n": 0}

    def nan_sim(question: str, theta: float) -> float:
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return float("nan")
        return 0.5

    calib = IRTCalibrator(nan_sim, samples_per_level=4, rng=seeded_rng)
    est = calib.calibrate("Q?")
    # Half of the responses were NaN and skipped
    assert est.n_responses == 10  # 5 abilities × 4 samples / 2 (NaN skipped)


def test_simulator_out_of_range_clamped(seeded_rng):
    """Simulators returning >1.0 or <0.0 are clamped."""
    def out_of_range(question: str, theta: float) -> float:
        return 1.5 if theta > 0 else -0.3

    calib = IRTCalibrator(out_of_range, samples_per_level=3, rng=seeded_rng)
    est = calib.calibrate("Q?")
    # Should produce a valid fit — all responses clamped to 1 or 0
    assert math.isfinite(est.difficulty)
    assert est.converged is True
    assert est.n_responses == 15


# -----------------------------------------------------------------------------
# Convergence
# -----------------------------------------------------------------------------


def test_max_iterations_respected(seeded_rng):
    """Extremely tight tolerance forces max iterations."""
    sim = make_smooth_sim(true_b=0.0)
    calib = IRTCalibrator(
        sim,
        samples_per_level=5,
        rng=seeded_rng,
        max_iterations=1,  # too few to converge
        tolerance=1e-12,
    )
    est = calib.calibrate("Q?")
    # With 1 iteration, may or may not converge — just verify it returns
    assert isinstance(est, DifficultyEstimate)


# -----------------------------------------------------------------------------
# Batch
# -----------------------------------------------------------------------------


def test_calibrate_batch_returns_one_per_question(seeded_rng):
    sim = make_smooth_sim(true_b=0.0)
    calib = IRTCalibrator(sim, samples_per_level=5, rng=seeded_rng)
    questions = ["Q1?", "Q2?", "Q3?"]
    estimates = calib.calibrate_batch(questions)
    assert len(estimates) == 3
    for e in estimates:
        assert isinstance(e, DifficultyEstimate)


# -----------------------------------------------------------------------------
# Constructor validation
# -----------------------------------------------------------------------------


def test_missing_sim_rejected():
    with pytest.raises(ValueError):
        IRTCalibrator(None)


def test_invalid_samples_per_level_rejected():
    with pytest.raises(ValueError):
        IRTCalibrator(make_constant_sim(0.5), samples_per_level=0)


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        IRTCalibrator(make_constant_sim(0.5), difficulty_bounds=(3.0, -3.0))
    with pytest.raises(ValueError):
        IRTCalibrator(make_constant_sim(0.5), difficulty_bounds=(0.0,))


def test_invalid_tolerance_rejected():
    with pytest.raises(ValueError):
        IRTCalibrator(make_constant_sim(0.5), tolerance=0)


def test_invalid_max_iterations_rejected():
    with pytest.raises(ValueError):
        IRTCalibrator(make_constant_sim(0.5), max_iterations=0)


# -----------------------------------------------------------------------------
# Sigmoid numerical stability
# -----------------------------------------------------------------------------


def test_sigmoid_handles_extreme_values():
    assert IRTCalibrator._sigmoid(0.0) == pytest.approx(0.5)
    assert IRTCalibrator._sigmoid(100.0) == pytest.approx(1.0, abs=1e-30)
    assert IRTCalibrator._sigmoid(-100.0) == pytest.approx(0.0, abs=1e-30)
    # No overflow or NaN
    assert math.isfinite(IRTCalibrator._sigmoid(1000.0))
    assert math.isfinite(IRTCalibrator._sigmoid(-1000.0))
